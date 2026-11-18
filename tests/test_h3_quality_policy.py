"""Stdlib-only policy tests: no local model fitting or inference."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from research.h3_personal.quality_policy import (
    LABELS, build_support, feature_supported, finalize, policy_spec,
    propose, temporal_folds, validation_gate,
)


def row(identifier='pending', **changes):
    result = {'id': identifier, 'source_sha256': 'a' * 64, 'session_id': 'capture-a',
              'start_seconds': 0.0, 'end_seconds': 5.0, 'features': [0.5, 0.5],
              'detection_fraction': 1.0, 'label': None}
    result.update(changes)
    return result


def oof(count=5, correct=5, session='capture-a', label='identity', **changes):
    records = []
    for i in range(count):
        record = {'session_id': session, 'true_label': label if i < correct else 'motion',
                  'suggested_label': label, 'vote_score': 0.8, 'margin': 0.25,
                  'eligible': True}
        record.update(changes)
        records.append(record)
    return records


class ValidationGateTests(unittest.TestCase):
    def test_exact_global_precision_threshold(self):
        gate = validation_gate(oof(10, 9), 'identity', 'capture-a')
        self.assertTrue(gate['passed'])
        self.assertEqual(gate['global'], {'predictions': 10, 'correct': 9, 'precision': 0.9})
        self.assertFalse(validation_gate(oof(9, 8), 'identity', 'capture-a')['passed'])

    def test_minimum_counts_and_target_session(self):
        self.assertFalse(validation_gate(oof(4), 'identity', 'capture-a')['passed'])
        records = oof(3) + oof(2, 2, session='capture-b')
        self.assertTrue(validation_gate(records, 'identity', 'capture-a')['passed'])
        gate = validation_gate(records, 'identity', 'capture-b')
        self.assertFalse(gate['passed'])
        self.assertIn('session_insufficient_oof_predictions', gate['reasons'])
        self.assertFalse(validation_gate(records, 'identity', 'new-capture')['passed'])

    def test_target_session_precision_threshold(self):
        records = oof(5, 4) + oof(5, 5, session='capture-b')
        self.assertTrue(validation_gate(records, 'identity', 'capture-a')['passed'])
        # Global 18/20 passes, target 3/5 fails.
        records = oof(5, 3) + oof(15, 15, session='capture-b')
        gate = validation_gate(records, 'identity', 'capture-a')
        self.assertEqual(gate['global']['precision'], 0.9)
        self.assertIn('session_oof_precision_below_threshold', gate['reasons'])

    def test_only_eligible_high_vote_high_margin_predictions_count(self):
        for changes in ({'vote_score': 0.799}, {'margin': 0.249}, {'eligible': False},
                        {'vote_score': float('nan')}, {'margin': float('inf')},
                        {'vote_score': 1.01}, {'true_label': None}):
            with self.subTest(changes=changes):
                gate = validation_gate(oof(**changes), 'identity', 'capture-a')
                self.assertFalse(gate['passed'])
                self.assertEqual(gate['global']['predictions'], 0)
                self.assertIsNone(gate['global']['precision'])

    def test_class_specific_predicted_precision_not_true_class_accuracy(self):
        records = oof() + oof(20, 20, label='reject')
        gate = validation_gate(records, 'motion', 'capture-a')
        self.assertEqual(gate['global']['predictions'], 0)
        self.assertFalse(gate['passed'])


class TemporalFoldTests(unittest.TestCase):
    def test_four_contiguous_blocks_deterministic_and_cover_once(self):
        rows = [row(str(i), start_seconds=i * 10, end_seconds=i * 10 + 5)
                for i in range(16)]
        before = copy.deepcopy(rows)
        folds = temporal_folds(rows)
        self.assertEqual(folds, temporal_folds(list(reversed(rows))))
        self.assertEqual(rows, before)
        self.assertEqual(len(folds), 4)
        for i, fold in enumerate(folds):
            self.assertEqual(set(fold['test_ids']), {str(j) for j in range(i * 4, i * 4 + 4)})
        self.assertEqual(sorted(i for f in folds for i in f['test_ids']), sorted(r['id'] for r in rows))

    def test_interval_purge_including_exact_30_seconds_and_overlap(self):
        rows = [row('heldout', start_seconds=0, end_seconds=100),
                row('overlap', start_seconds=80, end_seconds=90),
                row('boundary', start_seconds=130, end_seconds=131),
                row('far', start_seconds=130.01, end_seconds=131)]
        fold = temporal_folds(rows)[0]
        self.assertEqual(fold['test_ids'], ['heldout'])
        self.assertEqual(fold['purged_ids'], ['boundary', 'overlap'])
        self.assertEqual(fold['train_ids'], ['far'])

    def test_only_same_session_is_purged(self):
        rows = [row('a0', start_seconds=0, end_seconds=5),
                row('a1', start_seconds=10, end_seconds=15),
                row('a2', start_seconds=20, end_seconds=25),
                row('a3', start_seconds=30, end_seconds=35),
                row('b0', session_id='b', start_seconds=0, end_seconds=5),
                row('b1', session_id='b', start_seconds=200, end_seconds=205),
                row('b2', session_id='b', start_seconds=300, end_seconds=305),
                row('b3', session_id='b', start_seconds=400, end_seconds=405)]
        fold = temporal_folds(rows)[0]
        self.assertEqual(fold['test_ids'], ['a0', 'b0'])
        self.assertEqual(fold['purged_ids'], ['a1', 'a2', 'a3'])
        self.assertEqual(fold['train_ids'], ['b1', 'b2', 'b3'])

    def test_all_folds_have_no_leakage_and_partition_rows(self):
        rows = [row(str(i), session_id=str(i % 2),
                    start_seconds=i * 12, end_seconds=i * 12 + 20) for i in range(40)]
        by_id = {r['id']: r for r in rows}
        for fold in temporal_folds(rows):
            sets = [set(fold[k]) for k in ('train_ids', 'test_ids', 'purged_ids')]
            self.assertEqual(set.union(*sets), set(by_id))
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            for train_id in fold['train_ids']:
                train = by_id[train_id]
                for test_id in fold['test_ids']:
                    test = by_id[test_id]
                    if train['session_id'] == test['session_id']:
                        gap = max(0, train['start_seconds'] - test['end_seconds'],
                                  test['start_seconds'] - train['end_seconds'])
                        self.assertGreater(gap, 30)

    def test_invalid_time_and_purge_rejected(self):
        for bad in (row(end_seconds=0), row(start_seconds=float('nan'))):
            with self.assertRaises(ValueError):
                temporal_folds([bad])
        for seconds in (-1, float('inf')):
            with self.assertRaises(ValueError):
                temporal_folds([], seconds)
        self.assertEqual(len(temporal_folds([])), 4)


class FeatureSupportTests(unittest.TestCase):
    def test_percentile_bounds_fail_closed(self):
        support = build_support([[float(i)] for i in range(101)])
        self.assertEqual(support['low'], [1.0])
        self.assertEqual(support['high'], [99.0])
        self.assertTrue(feature_supported([50.5], support))
        for vector in ([0], [100], [float('nan')], [50, 50]):
            self.assertFalse(feature_supported(vector, support))

    def test_distance_rejects_unseen_feature_combinations_inside_ranges(self):
        support = build_support([[i / 100, i / 100] for i in range(101)])
        self.assertTrue(feature_supported([0.5, 0.5], support))
        self.assertFalse(feature_supported([0.2, 0.8], support))

    def test_constant_features_require_exact_match(self):
        support = build_support([[1, 2]] * 10)
        self.assertTrue(feature_supported([1, 2], support))
        self.assertFalse(feature_supported([1.00001, 2], support))

    def test_invalid_training_vectors_rejected(self):
        for vectors in ([], [[1]], [[], []], [[1], [1, 2]], [[1], [float('inf')]]):
            with self.assertRaises(ValueError):
                build_support(vectors)


class FinalizeTests(unittest.TestCase):
    def qualify(self, candidate, **changes):
        kwargs = {'gate': validation_gate(oof(), 'identity', 'capture-a'), 'supported': True}
        kwargs.update(changes)
        return finalize(candidate, 'identity', 0.8, 0.25, **kwargs)

    def test_passing_suggestion_is_separate_not_approval(self):
        candidate = row(detection_fraction=0.8)
        before = copy.deepcopy(candidate)
        record = self.qualify(candidate)
        self.assertEqual(candidate, before)
        self.assertEqual(record['decision'], 'auto_proposal')
        self.assertEqual(record['suggested_label'], 'identity')
        self.assertFalse(record['needs_review'])
        self.assertEqual(record['source_sha256'], candidate['source_sha256'])
        self.assertIn('not_approval', record['reasons'][0])
        self.assertEqual(set(record), {'source_sha256', 'suggested_label', 'vote_score',
                                      'margin', 'decision', 'needs_review', 'reasons'})

    def test_failed_detection_abstains_even_for_reject(self):
        for detection in (None, 0.79, -1, 1.01, float('nan'), True):
            candidate = row(detection_fraction=detection)
            record = finalize(candidate, 'reject', 0.99, 0.9,
                              gate={'passed': True}, supported=True)
            self.assertIsNone(record['suggested_label'])
            self.assertEqual(record['decision'], 'review')
            self.assertTrue(record['needs_review'])
            self.assertIn('failed_or_missing_detection_review_not_automatic_reject', record['reasons'])

    def test_errors_invalid_features_and_times_force_abstention(self):
        for changes in ({'error': 'landmark extraction failed'}, {'features': []},
                        {'features': [float('nan')]}, {'features': [True]},
                        {'features': ['0.5']}, {'features': (0.5,)},
                        {'end_seconds': 0}, {'start_seconds': -1}):
            with self.subTest(changes=changes):
                record = self.qualify(row(**changes))
                self.assertTrue(record['needs_review'])
                self.assertIsNone(record['suggested_label'])

    def test_support_validation_and_low_votes_fail_closed(self):
        self.assertTrue(self.qualify(row(), supported=False)['needs_review'])
        self.assertTrue(self.qualify(row(), gate=None)['needs_review'])
        for vote, margin in ((0.79, 0.5), (0.9, 0.24), (float('nan'), 0.5), (1.1, 0.5)):
            record = finalize(row(), 'identity', vote, margin,
                              gate={'passed': True}, supported=True)
            self.assertTrue(record['needs_review'])
            self.assertIsNone(record['suggested_label'])
            json.dumps(record, allow_nan=False)

    def test_human_labels_never_finalize(self):
        for label in LABELS:
            with self.assertRaises(ValueError):
                self.qualify(row(label=label))


class EntryPointTests(unittest.TestCase):
    def test_import_and_small_training_never_import_heavy_dependencies(self):
        script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'sklearn'):
        raise AssertionError('heavy dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from research.h3_personal.quality_policy import propose
result = propose([{'id':'a','source_sha256':'hash','session_id':'s',
                   'start_seconds':0,'end_seconds':1,'features':[1.0],'label':None}])
assert result['proposals']['a']['decision'] == 'review'
assert not any(n.split('.')[0] in ('numpy', 'sklearn') for n in sys.modules)
'''
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                   text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_insufficient_classes_review_all_only_unlabelled(self):
        rows = [row('human-' + label, label=label) for label in LABELS]
        rows += [row('p1'), row('p2', error='decode failed')]
        before = copy.deepcopy(rows)
        result = propose(rows)
        self.assertEqual(rows, before)
        self.assertEqual(set(result), {'proposals', 'validation', 'policy'})
        self.assertEqual(set(result['proposals']), {'p1', 'p2'})
        self.assertEqual(result['validation']['training_counts'], dict.fromkeys(LABELS, 1))
        for record in result['proposals'].values():
            self.assertEqual(record['decision'], 'review')
            self.assertIsNone(record['suggested_label'])
            self.assertIn('insufficient_training_classes', record['reasons'])
        json.dumps(result, allow_nan=False)

    def test_bad_measurements_do_not_count_as_training(self):
        rows = [row(str(i), label='identity', error='bad') for i in range(12)]
        rows += [row('ok', label='motion'), row('wrong-dim', label='reject', features=[1]), row()]
        result = propose(rows)
        self.assertEqual(result['validation']['training_counts'], {'identity': 0, 'motion': 1, 'reject': 0})
        self.assertEqual(set(result['proposals']), {'pending'})

    def test_single_class_even_with_many_labels_is_insufficient(self):
        result = propose([row(str(i), label='identity') for i in range(40)] + [row()])
        self.assertEqual(result['validation']['status'], 'insufficient_training')
        self.assertTrue(result['proposals']['pending']['needs_review'])

    def test_invalid_labels_duplicate_ids_and_missing_provenance_raise(self):
        for rows in ([row(label='approved')], [row(), row()],
                     [row(source_sha256='')], [row(session_id=None)], [row(id=None)]):
            with self.assertRaises(ValueError):
                propose(rows)

    def test_empty_input(self):
        self.assertEqual(propose([])['proposals'], {})

    def test_policy_exact_model_configuration_and_disclosures(self):
        spec = policy_spec()
        self.assertEqual(spec['model_parameters'], {
            'n_estimators': 200, 'max_depth': 6, 'min_samples_leaf': 5,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 2})
        self.assertEqual(spec['folds'], 4)
        self.assertEqual(spec['purge_seconds'], 30)
        self.assertIn('NOT independent capture generalization', ' '.join(spec['limitations']))
        self.assertIn('Grayscale proxy input', ' '.join(spec['limitations']))
        self.assertIn('not calibrated probabilities', ' '.join(spec['limitations']))


if __name__ == '__main__':
    unittest.main()
