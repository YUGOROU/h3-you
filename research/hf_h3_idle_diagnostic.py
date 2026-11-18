"""Finite isolated generated-only idle diagnostic; never launches cloud resources."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

SECTION = {'schema': 'h3-idle-drift-diagnostic-v1', 'windows': 14}
ANCHOR_POLICY = 'first-listening-tail-every3-v1'
ANCHOR_SECTION = dict(SECTION, anchor_policy=ANCHOR_POLICY)
ANCHOR_ENV = 'H3_IDLE_ANCHOR_DIAGNOSTIC'
LIMIT = 200 * 1024 * 1024


def validate_spec(spec):
    from research.h3_personal.fast_ref2va import config_for_layout
    from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA
    from research.hf_h3_continuation_diagnostic import strict_equal
    if type(spec) is not dict or not any(
            strict_equal(spec.get('idle_diagnostic'), section)
            for section in (SECTION, ANCHOR_SECTION)):
        raise ValueError('exact idle diagnostic section required')
    for key in ('locked_speech', 'streaming_trial', 'conversation_scenario',
                'live_voice_session', 'live_direct_session', 'live_relay_session',
                'continuation_diagnostic', 'continuation_candidate', 'continuation_comparison',
                'decode_parity_diagnostic', 'conditioning_exploration', 'efficiency_comparison',
                'utilization_comparison', 'late_listening_diagnostic'):
        if key in spec:
            raise ValueError('exclusive idle diagnostic required')
    if not strict_equal(spec.get('fast_config'), asdict(config_for_layout('portrait'))):
        raise ValueError('accepted BF16 SMD4 portrait configuration required')
    if (spec.get('layout') != 'portrait' or spec.get('attention_mode') != 'fa4-packed'
            or spec.get('runtime_policy') != 'resident-visual-cached'
            or spec.get('identity_prefix') != IDENTITY_PREFIX
            or spec.get('reference_sha256') != REFERENCE_SHA
            or spec.get('face_crop_box') is not None):
        raise ValueError('accepted reference1024 resident runtime required')
    for key in ('regional_compile', 'warm_stage_diagnostic', 'incremental_transfer'):
        if key in spec and spec[key] is not False:
            raise ValueError('experimental runtime forbidden')
    config = spec.get('config', {})
    if type(config.get('timeout_seconds')) is not int or config['timeout_seconds'] != 1200:
        raise ValueError('finite 1200 second configuration required')
    return spec['idle_diagnostic'].copy()


def write_anchor_evidence(coverage, directory):
    """Validate actual per-window rank receipts; publish only scalar evidence."""
    def digest(value):
        if (type(value) is not str or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('invalid anchor digest')
        return value

    # Retirement moves receipts as a set; never stitch windows across owners.
    import re
    coverage = Path(coverage)
    if any(p.is_symlink() for p in (coverage, *coverage.parents)) or not coverage.is_dir():
        raise ValueError('regular anchor coverage directory required')
    candidates = [coverage]
    for child in coverage.iterdir():
        if child.name.startswith('retired-'):
            if (re.fullmatch(r'retired-[0-9a-f]{24}', child.name) is None
                    or child.is_symlink() or not child.is_dir()):
                raise ValueError('invalid retired anchor directory')
            candidates.append(child)
    names = {f'streaming-rank-{rank}-window-{sequence}.json'
             for sequence in range(14) for rank in range(2)}
    complete = []
    for candidate in candidates:
        receipts = {p.name for p in candidate.iterdir()
                    if p.name.startswith('streaming-rank-')}
        if not receipts:
            continue
        if receipts != names or any(
                (candidate/name).is_symlink() or not (candidate/name).is_file()
                for name in names):
            raise ValueError('incomplete or nonregular anchor receipt set')
        complete.append(candidate)
    if len(complete) != 1:
        raise ValueError('exactly one complete anchor receipt set required')
    receipt_root = complete[0]
    windows = []
    for sequence in range(14):
        rows = []
        expected = 124 + sequence * 102
        for rank in range(2):
            path = receipt_root/f'streaming-rank-{rank}-window-{sequence}.json'
            if path.is_symlink() or not path.is_file():
                raise ValueError('missing regular anchor rank receipt')
            row = json.loads(path.read_text())
            streaming = row['streaming']
            if (type(row.get('idle_anchor_interval')) is not int
                    or row['idle_anchor_interval'] != 3
                    or type(streaming.get('sequence')) is not int
                    or streaming['sequence'] != sequence
                    or streaming.get('logical_index') != sequence
                    or streaming.get('role') != 'listening'
                    or type(row.get('committed_frame')) is not int
                    or row['committed_frame'] != expected):
                raise ValueError('anchor rank interval/sequence/frame drift')
            canonical = digest(row['canonical_sha256'])
            guides = row['guide_digests']
            if type(guides) is not list or len(guides) != 3:
                raise ValueError('anchor guide digest list required')
            for value in guides:
                if value is not None:
                    digest(value)
            # Locked audio is the current canonical tensor, not a previous window.
            if guides[2] != canonical or (sequence > 0 and None in guides):
                raise ValueError('anchor canonical/guide audio evidence missing')
            rows.append(dict(sequence=sequence, committed_frame=expected,
                             canonical_sha256=canonical, guide_digests=guides))
        if rows[0] != rows[1]:
            raise ValueError('anchor both-rank digest disagreement')
        windows.append(rows[0])
    for sequence in (3, 6, 9, 12):
        if windows[sequence]['guide_digests'][0] != windows[1]['guide_digests'][0]:
            raise ValueError('anchor visual digest drift')
    evidence = dict(anchor_policy=ANCHOR_POLICY, interval=3, ranks=[0, 1], windows=windows)
    with (Path(directory)/'anchor-evidence.json').open('x') as stream:
        json.dump(evidence, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return evidence


def generated_files(report, *, partial=False):
    """Explicit harness outputs only, never enumerate driver/coverage/input trees."""
    if not partial and (report.get('windows') != 14 or report.get('native_samples') != 42
            or report.get('encoded_samples') != 42 or report.get('final') is not True):
        raise ValueError('incomplete diagnostic report')
    # Current harness returns scalar counts rather than a file manifest.
    names = ['samples.json', 'REVIEW.txt', 'native-manifest.json']
    for i in range(14):
        count = 119 if i == 0 else (107 if i == 13 else 102)
        names += [f'segments/segment-{i:03d}.mp4', f'segments/native-{i:03d}.mkv',
                  f'{i:02d}-native-contact.png']
        names += [f'{i:02d}-{pick:03d}-{kind}.png'
                  for pick in (0, count//2, count-1) for kind in ('native', 'encoded')]
    # Honor an explicit returned manifest when the harness supplies one, but
    # never let it widen the known generated-artifact contract.
    if not partial and 'files' in report:
        if type(report['files']) is not list or sorted(report['files']) != sorted(names):
            raise ValueError('unexpected diagnostic file manifest')
        return report['files']
    return names


def publish(api, deployment, directory, report, *, partial=False, anchor_policy=None):
    from huggingface_hub import CommitOperationAdd
    directory = Path(directory).resolve()
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('private diagnostic deployment required')
    names = list(generated_files(report, partial=partial))
    if anchor_policy == ANCHOR_POLICY and not partial:
        names.append('anchor-evidence.json')
    fields = ('windows', 'native_samples', 'encoded_samples', 'final', 'committed_frames',
              'emitted_frames', 'final_packet_frames', 'mae_mean', 'max_error',
              'sample_manifest_sha256', 'bytes', 'merged', 'silence')
    clean = {}
    if not partial:
        for key in fields:
            if key not in report:
                continue
            value = report[key]
            if key == 'sample_manifest_sha256':
                if (type(value) is not str or len(value) != 64
                        or any(c not in '0123456789abcdef' for c in value)):
                    raise ValueError('invalid diagnostic digest')
            elif type(value) not in (bool, int, float) or not math.isfinite(value):
                raise ValueError('scalar diagnostic evidence required')
            clean[key] = value
    else:
        clean = {'completed': False, 'diagnostic_only': True}
    if anchor_policy is not None:
        if anchor_policy != ANCHOR_POLICY:
            raise ValueError('unknown idle anchor policy')
        clean['anchor_policy'] = anchor_policy
    data = (json.dumps(clean, sort_keys=True, allow_nan=False)+'\n').encode()
    total = len(data)
    metadata = 'failure.json' if partial else 'report.json'
    operations = [CommitOperationAdd(path_in_repo='idle-diagnostic/'+metadata, path_or_fileobj=data)]
    for name in names:
        path = directory/name
        if partial and not path.exists() and not path.is_symlink():
            continue
        if (any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
                or not path.resolve().is_relative_to(directory.resolve())):
            raise ValueError('regular generated diagnostic output required')
        size = path.stat().st_size
        total += size
        if size <= 0 or total > LIMIT:
            raise ValueError('bounded diagnostic publication required')
        operations.append(CommitOperationAdd(path_in_repo='idle-diagnostic/'+name, path_or_fileobj=str(path)))
    api.create_commit(deployment, repo_type='dataset', operations=operations,
                      commit_message='Private silent idle diagnostic generated artifacts')


def run_session(root, paths, spec, deployment, deadline):
    from huggingface_hub import HfApi
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha
    from research.h3_personal.ref2va_runtime import sha
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.idle_drift_diagnostic import run_diagnostic
    section = validate_spec(spec)
    anchor_policy = section.get('anchor_policy')
    previous_anchor = os.environ.get(ANCHOR_ENV)
    if anchor_policy is None and ANCHOR_ENV in os.environ:
        raise ValueError('baseline refuses preexisting idle anchor environment')
    if not math.isfinite(deadline) or deadline-time.monotonic() < 360:
        raise ValueError('finite diagnostic deadline required')
    root = Path(root)
    paths = {k: Path(v) for k, v in paths.items()}
    api = HfApi(token=os.environ['HF_TOKEN'])
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('private diagnostic deployment required')
    reference = paths['reference']/'production-inputs/references/secondary.jpeg'
    profile = choose_profile(ordered_source_sha256=(sha(reference),), prompt_sha256=prompt_sha('speech'))
    if profile.name != 'reference1024':
        raise ValueError('no quality fallback')
    output = root.with_name(root.name+'-idle-generated')
    identity = paths['personal']/spec['identity_prefix']
    runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors', reference=reference,
        archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True,
        quality_profile=profile, conversation_roles=True)
    try:
        try:
            if anchor_policy is not None:
                os.environ[ANCHOR_ENV] = anchor_policy
            runtime.start()
            report = run_diagnostic(runtime, output/'diagnostic', deadline=min(deadline-60, time.monotonic()+900))
            if anchor_policy is not None:
                write_anchor_evidence(runtime.coverage, output/'diagnostic')
        finally:
            try:
                runtime.close()
            finally:
                if previous_anchor is None:
                    os.environ.pop(ANCHOR_ENV, None)
                else:
                    os.environ[ANCHOR_ENV] = previous_anchor
    except Exception as original:
        # Only explicit generated names; never driver/input/latent trees or error text.
        if time.monotonic() < deadline:
            try:
                publish(api, deployment, output/'diagnostic', {}, partial=True,
                        anchor_policy=anchor_policy)
            except Exception:
                original.add_note('Sanitized idle diagnostic failure publication failed.')
        raise
    if time.monotonic() >= deadline:
        raise TimeoutError('idle diagnostic publication deadline')
    if anchor_policy is not None:
        report = dict(report, anchor_policy=anchor_policy)
    publish(api, deployment, output/'diagnostic', report, anchor_policy=anchor_policy)
    return report
