"""Conservative grayscale-proxy quality proposals, never human approval.

Worker entry point: propose(rows). Only this function imports numpy/sklearn and
fits models; importing this module and using its helpers needs only the stdlib.
The caller must supply a consistent ordered vector of standard scalar quality
features (e.g. blur, exposure, detection/pose summaries). Feature names cannot be
verified from anonymous vectors: session/time, raw landmarks and identity
embeddings MUST NOT be placed in features. Metadata is never appended to X.

Exact returned schema (JSON-compatible):
  proposals: {unlabelled_id: {source_sha256, suggested_label: str|null,
    vote_score: float|null, margin: float|null, decision: auto_proposal|review,
    needs_review: bool, reasons: [str]}}
  validation: {status: str, training_counts: {label: int}, folds: [
    {fold: int, train_count: int, test_count: int, purged_count: int,
     training_counts: {label: int}, status: str}],
    oof_predictions: {id: {source_sha256, session_id, true_label,
      suggested_label, vote_score, margin, eligible: bool, reasons: [str]}},
    gates: {session_id: {label: {passed: bool,
      global: {predictions: int, correct: int, precision: float|null},
      session: {predictions: int, correct: int, precision: float|null},
      reasons: [str]}}}, limitations: [str]}
  policy: static settings/constraints returned by policy_spec().
Invalid/duplicate IDs, missing provenance/session or unknown human labels raise
ValueError. Bad measurements/errors abstain. Input rows are never mutated.
"""

from collections import defaultdict
import math

LABELS = ('identity', 'motion', 'reject')
SEED = 42
MIN_CLASS_SAMPLES = 10
LIMITATIONS = [
    'Within-session temporally contiguous purged OOF; NOT independent capture generalization.',
    'Grayscale proxy input; not full-color/full-clip verification or identity recognition.',
    'Forest class scores (mean leaf class fractions) are not calibrated probabilities; empirical precision is not a guarantee.',
    'All suggestions are separate from human labels and are NOT approval.',
    'Anonymous feature vectors require caller-enforced scalar quality feature semantics.',
]


def policy_spec():
    return {
        'schema_version': 1,
        'model': 'RandomForestClassifier',
        'model_parameters': {'n_estimators': 200, 'max_depth': 6,
                             'min_samples_leaf': 5, 'class_weight': 'balanced',
                             'random_state': SEED, 'n_jobs': 2},
        'labels': list(LABELS), 'min_training_samples_per_class': MIN_CLASS_SAMPLES,
        'folds': 4, 'purge_seconds': 30.0,
        'min_vote': 0.8, 'min_margin': 0.25,
        'global_gate': {'min_predictions': 5, 'min_precision': 0.9},
        'session_gate': {'min_predictions': 3, 'min_precision': 0.8},
        'min_detection_fraction': 0.8,
        'support': {'percentiles': [1, 99],
                    'distance': 'nearest training Euclidean distance in percentile-range units',
                    'distance_limit': '95th percentile of training leave-one-out nearest distances'},
        'predictors': 'Only caller-supplied ordered standard scalar quality features',
        'excluded_predictors': ['session', 'time', 'raw landmarks', 'identity embeddings'],
        'limitations': list(LIMITATIONS),
    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _measurement_reasons(row, dimension=None):
    reasons = []
    if row.get('error'):
        reasons.append('row_error')
    values = row.get('features')
    if not isinstance(values, list) or not values or not all(_finite(v) for v in values):
        reasons.append('invalid_features')
    elif dimension is not None and len(values) != dimension:
        reasons.append('feature_dimension_mismatch')
    start, end = row.get('start_seconds'), row.get('end_seconds')
    if not (_finite(start) and _finite(end) and 0 <= start < end):
        reasons.append('invalid_time_interval')
    return reasons


def _detection_ok(row):
    value = row.get('detection_fraction')
    return _finite(value) and 0.8 <= value <= 1.0


def temporal_folds(rows, purge_seconds=30.0):
    """Four deterministic contiguous rank blocks per session; purge interval gaps <=30s.

    Input is the valid labelled training population. Test rows from every session
    share each fold. Interval overlap (including long clips) has zero gap.
    """
    if not _finite(purge_seconds) or purge_seconds < 0:
        raise ValueError('purge_seconds must be finite and nonnegative')
    groups = defaultdict(list)
    for row in rows:
        if _measurement_reasons(row):
            raise ValueError('temporal_folds requires valid measurements')
        groups[row['session_id']].append(row)
    blocks = [[] for _ in range(4)]
    for session in sorted(groups):
        ordered = sorted(groups[session], key=lambda r: (r['start_seconds'], r['end_seconds'], r['id']))
        for i in range(4):
            blocks[i].extend(ordered[i * len(ordered) // 4:(i + 1) * len(ordered) // 4])
    result = []
    for i, heldout in enumerate(blocks):
        test_ids = {r['id'] for r in heldout}
        train, purged = [], []
        for row in rows:
            if row['id'] in test_ids:
                continue
            near = any(
                row['session_id'] == test['session_id'] and
                max(0.0, row['start_seconds'] - test['end_seconds'],
                    test['start_seconds'] - row['end_seconds']) <= purge_seconds
                for test in heldout)
            (purged if near else train).append(row['id'])
        result.append({'fold': i, 'train_ids': sorted(train),
                       'test_ids': sorted(test_ids), 'purged_ids': sorted(purged)})
    return result


def _percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def build_support(vectors):
    """Fit only on training data; no heldout/unlabelled range information."""
    if len(vectors) < 2 or not vectors[0]:
        raise ValueError('support requires at least two nonempty vectors')
    dimension = len(vectors[0])
    if any(len(v) != dimension or not all(_finite(x) for x in v) for v in vectors):
        raise ValueError('invalid support vectors')
    low = [_percentile([v[j] for v in vectors], 1) for j in range(dimension)]
    high = [_percentile([v[j] for v in vectors], 99) for j in range(dimension)]
    scales = [hi - lo if hi > lo else 1.0 for lo, hi in zip(low, high)]
    def distance(a, b):
        return math.sqrt(sum(((x - y) / scale) ** 2 for x, y, scale in zip(a, b, scales)))
    nearest = [min(distance(v, other) for j, other in enumerate(vectors) if i != j)
               for i, v in enumerate(vectors)]
    return {'low': low, 'high': high, 'scales': scales,
            'distance_limit': _percentile(nearest, 95),
            'vectors': [list(v) for v in vectors]}


def feature_supported(vector, support):
    if len(vector) != len(support['low']) or not all(_finite(v) for v in vector):
        return False
    if any(v < lo or v > hi for v, lo, hi in zip(vector, support['low'], support['high'])):
        return False
    distance = min(math.sqrt(sum(((x - y) / scale) ** 2
                                for x, y, scale in zip(vector, other, support['scales'])))
                   for other in support['vectors'])
    return distance <= support['distance_limit'] + 1e-12


def validation_gate(oof, label, session_id):
    """Empirical precision of eligible high-vote predictions of this class, not recall."""
    qualified = [r for r in oof if r.get('eligible') is True and
                 r.get('suggested_label') == label and r.get('true_label') in LABELS and
                 _finite(r.get('vote_score')) and 0.8 <= r['vote_score'] <= 1 and
                 _finite(r.get('margin')) and 0.25 <= r['margin'] <= 1]
    def stats(records):
        correct = sum(r['true_label'] == label for r in records)
        return {'predictions': len(records), 'correct': correct,
                'precision': correct / len(records) if records else None}
    global_stats = stats(qualified)
    session_stats = stats([r for r in qualified if r['session_id'] == session_id])
    reasons = []
    for name, counts, minimum, precision in [('global', global_stats, 5, 0.9),
                                             ('session', session_stats, 3, 0.8)]:
        if counts['predictions'] < minimum:
            reasons.append(name + '_insufficient_oof_predictions')
        if counts['precision'] is None or counts['precision'] < precision:
            reasons.append(name + '_oof_precision_below_threshold')
    return {'passed': not reasons, 'global': global_stats, 'session': session_stats,
            'reasons': reasons}


def finalize(row, suggested_label=None, vote_score=None, margin=None, *,
             gate=None, supported=False, reasons=()):
    """Pure abstention gate. Human labels cannot be converted into proposals."""
    if row.get('label') is not None:
        raise ValueError('human-labelled rows must never receive proposals')
    failures = list(reasons) + _measurement_reasons(row)
    if not _detection_ok(row):
        failures.append('failed_or_missing_detection_review_not_automatic_reject')
    if suggested_label not in LABELS:
        failures.append('no_model_suggestion')
    if not _finite(vote_score) or not 0.8 <= vote_score <= 1:
        failures.append('insufficient_vote')
    if not _finite(margin) or not 0.25 <= margin <= 1:
        failures.append('insufficient_margin')
    if not supported:
        failures.append('outside_or_unavailable_training_feature_support')
    if gate is None or gate.get('passed') is not True:
        failures.extend(gate['reasons'] if gate and gate.get('reasons') else ['validation_gate_unavailable'])
    failures = list(dict.fromkeys(failures))
    return {'source_sha256': row['source_sha256'],
            'suggested_label': None if failures else suggested_label,
            'vote_score': float(vote_score) if _finite(vote_score) and 0 <= vote_score <= 1 else None,
            'margin': float(margin) if _finite(margin) and 0 <= margin <= 1 else None,
            'decision': 'review' if failures else 'auto_proposal',
            'needs_review': bool(failures),
            'reasons': failures or ['empirical_oof_gates_passed_suggestion_only_not_approval']}


def propose(rows):
    """Fit on an HF CPU worker only. No network, persistence or approval side effects."""
    seen = set()
    for row in rows:
        for key in ('id', 'source_sha256', 'session_id'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError('missing or invalid ' + key)
        if row['id'] in seen:
            raise ValueError('duplicate id: ' + row['id'])
        seen.add(row['id'])
        if row.get('label') is not None and row['label'] not in LABELS:
            raise ValueError('unknown human label')
    labelled = [r for r in rows if r.get('label') is not None and not _measurement_reasons(r)]
    dimension = len(labelled[0]['features']) if labelled else None
    training = [r for r in labelled if not _measurement_reasons(r, dimension)]
    pending = [r for r in rows if r.get('label') is None]
    counts = lambda samples: {label: sum(r['label'] == label for r in samples) for label in LABELS}
    sufficient = lambda samples: all(n >= MIN_CLASS_SAMPLES for n in counts(samples).values())
    validation = {'status': 'insufficient_training', 'training_counts': counts(training),
                  'folds': [], 'oof_predictions': {}, 'gates': {}, 'limitations': list(LIMITATIONS)}
    result = {'proposals': {}, 'validation': validation, 'policy': policy_spec()}
    def review_all(reason):
        result['proposals'] = {r['id']: finalize(r, reasons=[reason]) for r in pending}
        return result
    if not sufficient(training):
        return review_all('insufficient_training_classes')
    if not pending:
        validation['status'] = 'no_unlabelled_rows'
        return result

    # Heavy dependencies and all fitting stay inside the worker entry point.
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        validation['status'] = 'dependency_error'
        return review_all('worker_dependencies_unavailable')

    def fit(samples):
        model = RandomForestClassifier(**policy_spec()['model_parameters'])
        model.fit(np.asarray([r['features'] for r in samples], dtype=float),
                  [r['label'] for r in samples])
        return model, build_support([r['features'] for r in samples])

    def predict(model, samples):
        votes = model.predict_proba(np.asarray([r['features'] for r in samples], dtype=float))
        predictions = []
        for scores in votes:
            ranked = sorted(zip(model.classes_, scores), key=lambda pair: (-pair[1], pair[0]))
            predictions.append((str(ranked[0][0]), float(ranked[0][1]),
                                float(ranked[0][1] - ranked[1][1])))
        return predictions

    by_id = {r['id']: r for r in training}
    try:
        for fold in temporal_folds(training):
            train = [by_id[i] for i in fold['train_ids']]
            test = [by_id[i] for i in fold['test_ids']]
            summary = {'fold': fold['fold'], 'train_count': len(train), 'test_count': len(test),
                       'purged_count': len(fold['purged_ids']), 'training_counts': counts(train),
                       'status': 'insufficient_training' if not sufficient(train) else 'empty_test' if not test else 'complete'}
            validation['folds'].append(summary)
            if summary['status'] != 'complete':
                continue
            model, support = fit(train)
            for row, (label, vote, margin) in zip(test, predict(model, test)):
                reasons = []
                if not _detection_ok(row):
                    reasons.append('failed_or_missing_detection')
                if not feature_supported(row['features'], support):
                    reasons.append('outside_training_feature_support')
                if vote < 0.8 or margin < 0.25:
                    reasons.append('insufficient_vote_or_margin')
                validation['oof_predictions'][row['id']] = {
                    'source_sha256': row['source_sha256'], 'session_id': row['session_id'],
                    'true_label': row['label'], 'suggested_label': label,
                    'vote_score': vote, 'margin': margin, 'eligible': not reasons,
                    'reasons': reasons}
        oof = list(validation['oof_predictions'].values())
        for session in sorted({r['session_id'] for r in pending}):
            validation['gates'][session] = {label: validation_gate(oof, label, session) for label in LABELS}
        validation['status'] = 'complete' if all(f['status'] == 'complete' for f in validation['folds']) else 'partial'
        model, support = fit(training)
        valid_pending = [r for r in pending if not _measurement_reasons(r, dimension)]
        predictions = dict(zip((r['id'] for r in valid_pending),
                               predict(model, valid_pending) if valid_pending else []))
        for row in pending:
            label, vote, margin = predictions.get(row['id'], (None, None, None))
            gate = validation['gates'][row['session_id']].get(label)
            result['proposals'][row['id']] = finalize(
                row, label, vote, margin, gate=gate,
                supported=label is not None and feature_supported(row['features'], support),
                reasons=_measurement_reasons(row, dimension))
        return result
    except Exception:
        # A worker/model failure must never leave partial automatic proposals.
        validation['status'] = 'worker_model_error'
        return review_all('worker_model_error')
