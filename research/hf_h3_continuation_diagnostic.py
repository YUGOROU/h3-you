"""Single fixed-source Jobs diagnostic; no launch, TTS, retries or input uploads.

Bootstrap runs this module in a fresh, deadline-bounded process group. The spec's
continuation_diagnostic is the unmodified validate_metadata/preflight receipt;
locked_speech and the speech asset must bind its fixed source. Publication goes
only to the bootstrap deployment dataset, under outputs/continuation/.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from research.h3_personal import continuation_diagnostic as contract
from research.h3_personal.continuation_plan import plan
from research.h3_personal.continuation_candidate import normalize
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime, POLICY, REFERENCE_SHA

VIDEOS = ('segment-000.mp4', 'segment-001.mp4', 'segment-002.mp4',
          'joined.mp4', 'review-with-speech.mp4')
RECEIPTS = ('profile.json', 'window-000-receipts.json',
            'window-001-receipts.json', 'window-002-receipts.json')
PREFIX = 'outputs/continuation/'
CONTINUATION_COMPARISON = dict(schema='h3-continuation-comparison-v1',
    order=['independent-supplied-speech', 'latent-tail-v1'],
    same_warm_runtime=True, full_source=True, controlled_noise_equality=False)


DECODE_PARITY_DIAGNOSTIC = dict(schema='h3-same-latent-decode-parity-v1',
    serial=True, atol=0.0, retain_latents_in_memory_only=True,
    total_frames=328, arrivals=[37, 30, 30])


def strict_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(strict_equal(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(strict_equal(a, b) for a, b in zip(value, expected))
    return value == expected


def validate_parity(row, rank, session, *, parallel=False):
    fixed = dict(schema='h3-decode-parity-v1', session_id=session, worker_rank=rank,
        total_frames=328, arrivals=[37, 30, 30], emitted_counts=[119, 102, 107],
        normalized_shape=[1, 24, 97, 28, 16], decoded_shape=[1, 3, 328, 448, 256],
        inputs_unchanged=True, frames_exactly_once=True, closed=True,
        native_parallel_verified=False, diagnostic_only=True,
        quality_verified=False, realtime_verified=False,
        serial=True, atol=0.0,
        source_sha256='e3cec57d51623c8343c3bc25160e863d253997fb4158aa283bba874154f70fa1',
        source_revision='ba371e33ed080e51f1884ae166baf7c4447a66fe',
        cuda_fp16_autocast=True, denormalized_dtype='torch.float32')
    if parallel:
        fixed.update(schema='h3-parallel-decode-parity-v1', serial=False,
                     reference_schedule='alternating-v1', candidate_schedule='two-clip-waves-v1',
                     reference_emitted_counts=[119, 102, 107])
    if not isinstance(row, dict):
        raise ValueError('invalid decode parity evidence object')
    invalid = [k for k, v in fixed.items() if not strict_equal(row.get(k), v)]
    if invalid:
        # Constant schema field names only: no private paths/media in logs.
        raise ValueError('invalid decode parity evidence: ' + ','.join(invalid))
    sha = row.get('input_sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('invalid decode parity input SHA256')
    clean = dict(fixed, input_sha256=sha)
    for name in ('raw_float', 'denormalized_pixels', 'uint8'):
        metric = row.get(name)
        keys = {'max', 'mean', 'per_frame_max', 'per_frame_mean', 'exact', 'within_atol'}
        if not isinstance(metric, dict) or set(metric) != keys:
            raise ValueError('invalid parity metrics')
        scalars = [metric['max'], metric['mean']]
        for key in ('per_frame_max', 'per_frame_mean'):
            if type(metric[key]) is not list or len(metric[key]) != 328:
                raise ValueError('invalid per-frame metrics')
            scalars.extend(metric[key])
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in scalars):
            raise ValueError('unsafe parity scalar')
        if any(type(metric[k]) is not bool for k in ('exact', 'within_atol')):
            raise ValueError('invalid parity boolean')
        zero = metric['max'] == 0
        if (metric['mean'] > metric['max'] or
                metric['within_atol'] != zero or metric['exact'] != zero):
            raise ValueError('inconsistent parity metrics')
        clean[name] = metric
    if (type(row.get('parity')) is not bool or row['parity'] != all(
            clean[name]['within_atol'] for name in ('raw_float', 'denormalized_pixels', 'uint8'))):
        raise ValueError('invalid parity finding')
    clean['parity'] = row['parity']
    return clean


CONDITIONING_EXCLUSIVE = ('continuation_candidate', 'continuation_comparison',
    'decode_parity_diagnostic', 'efficiency_comparison', 'utilization_comparison')


def validate_conditioning_mode(spec):
    if 'conditioning_exploration' not in spec:
        return
    from research.hf_h3_conditioning_exploration import CONTRACT
    if (any(key in spec for key in CONDITIONING_EXCLUSIVE) or
            not strict_equal(spec['conditioning_exploration'], CONTRACT)):
        raise ValueError('invalid or incompatible conditioning exploration')


def validate_spec(spec, assets=None):
    """Metadata-only admission, before downloads/model load. No authorization grant."""
    validate_conditioning_mode(spec)
    if 'continuation_comparison' in spec and json.dumps(
            spec['continuation_comparison'], sort_keys=True) != json.dumps(
                CONTINUATION_COMPARISON, sort_keys=True):
        raise ValueError('invalid continuation comparison contract')
    if 'decode_parity_diagnostic' in spec:
        if ('continuation_comparison' in spec or not strict_equal(
                spec['decode_parity_diagnostic'], DECODE_PARITY_DIAGNOSTIC)):
            raise ValueError('invalid or incompatible decode parity contract')
    if 'continuation_candidate' in spec:
        candidate = normalize(spec['continuation_candidate'])
        if (candidate is None or not strict_equal(candidate, spec['continuation_candidate']) or
                (candidate['audio_context'] == 'full-source-v1' and
                 candidate['progress_receipts'] == 'per-step-v1') or
                any(k in spec for k in ('continuation_comparison', 'decode_parity_diagnostic'))):
            raise ValueError('invalid or incompatible continuation candidate')
    if 'efficiency_comparison' in spec:
        from research.h3_personal.continuation_efficiency import comparison_contract
        if (not strict_equal(spec['efficiency_comparison'], comparison_contract()) or
                any(k in spec for k in ('continuation_candidate', 'continuation_comparison',
                                       'decode_parity_diagnostic'))):
            raise ValueError('invalid or incompatible efficiency comparison')
    if 'utilization_comparison' in spec:
        from research.hf_h3_utilization_comparison import comparison_contract, EXCLUSIVE
        if (not strict_equal(spec['utilization_comparison'], comparison_contract()) or
                any(k in spec for k in EXCLUSIVE)):
            raise ValueError('invalid or incompatible utilization comparison')
    timeout = spec['config']['timeout_seconds']
    source = contract.source_contract()
    planned = dict(schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
        sha256=source['sha256'], source_samples=577920, source_sample_rate=48000,
        source_channels=1, total_frames=328, window_frames=124, overlap_frames=22,
        expected_windows=3, sampler='SMD4', fps=24, seed=314159,
        windows=[w.record() for w in plan(328, 124, 22)])
    expected = contract.validate_metadata(source, planned, timeout_seconds=timeout)
    supplied = spec.get('continuation_diagnostic')
    if not isinstance(supplied, dict) or type(supplied.get('local_source_verified')) is not bool:
        raise ValueError('diagnostic contract required')
    expected['local_source_verified'] = supplied['local_source_verified']
    if json.dumps(supplied, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('diagnostic contract differs')
    speech = {k: source[k] for k in ('repo', 'revision', 'path', 'sha256')}
    speech.update(samples=577920, sample_rate=48000)
    if json.dumps(spec.get('locked_speech'), sort_keys=True) != json.dumps(speech, sort_keys=True):
        raise ValueError('fixed locked speech required')
    if (timeout <= 25 or spec.get('layout') != 'portrait' or
            spec.get('runtime_policy') != POLICY or
            spec.get('attention_mode') != 'fa4-packed' or
            spec.get('reference_sha256') != REFERENCE_SHA or
            spec.get('face_crop_box') is not None or
            any(k in spec for k in ('live_voice_session', 'asr_policy', 'modal_live_approval')) or
            any(spec.get(k, False) is not False for k in
                ('regional_compile', 'warm_stage_diagnostic', 'incremental_transfer'))):
        raise ValueError('incompatible continuation diagnostic runtime')
    if assets is not None:
        roles = [a['role'] for a in assets]
        if len(roles) != len(set(roles)) or set(roles) != {'base', 'personal', 'reference', 'smd', 'speech'}:
            raise ValueError('only fixed diagnostic asset roles allowed')
        rows = [a for a in assets if a['role'] == 'speech']
        if len(rows) != 1:
            raise ValueError('one speech asset required')
        asset = rows[0]
        if (asset.get('repo_type') != 'dataset' or
                any(asset.get(k) != speech[k] for k in ('repo', 'revision')) or
                len(asset['files']) != 1 or
                any(asset['files'][0].get(k) != speech[k] for k in ('path', 'sha256'))):
            raise ValueError('speech asset provenance differs')
    return expected


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError('continuation diagnostic deadline exhausted')
    return seconds


def run(root, paths, spec, deployment, deadline):
    """One start/warmup, optionally two ordered warm renders, always close.

    The bootstrap process-group timeout is the hard stop, including native GPU
    hangs and publication. Checks here never extend that absolute deadline.
    """
    if 'conditioning_exploration' in spec:
        validate_spec(spec)
        from research.hf_h3_conditioning_exploration import run as conditioning_run
        return conditioning_run(root, paths, spec, deployment, deadline)
    if 'utilization_comparison' in spec:
        from research.hf_h3_utilization_comparison import run as utilization_run
        return utilization_run(root, paths, spec, deployment, deadline)
    if 'efficiency_comparison' in spec:
        from research.hf_h3_efficiency_comparison import run as efficiency_run
        return efficiency_run(root, paths, spec, deployment, deadline)
    started = time.monotonic()
    validate_spec(spec)
    remaining(deadline)
    root = Path(root)
    paths = {k: Path(v) for k, v in paths.items()}
    wav = paths['speech'] / spec['locked_speech']['path']
    verified = contract.preflight(wav, timeout_seconds=spec['config']['timeout_seconds'])
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ['HF_TOKEN'])
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('private output dataset required')
    output = root.with_name(root.name + '-continuation-generated')
    turn = output / 'continuation'
    identity = paths['personal'] / spec['identity_prefix']
    comparison = 'continuation_comparison' in spec
    parity = 'decode_parity_diagnostic' in spec
    candidate = normalize(spec.get('continuation_candidate'))
    candidate_kwargs = (dict(
        window_audio_context=candidate['audio_context'] == 'window-context-canonical-v1',
        defer_progress_receipts=candidate['progress_receipts'] == 'bind-and-commit-v1',
        profile_stages=True) if candidate is not None else {})
    runtime = LiveVoiceRuntime(**candidate_kwargs, **({'comparison_diagnostics': True} if comparison else
                                 {'decode_parity': True} if parity else {}), model=paths['base'], model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',
        reference=paths['reference']/'production-inputs/references/secondary.jpeg',
        archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True)
    independent = output / 'independent'
    startup = time.monotonic()
    try:
        remaining(deadline)
        runtime.start()
        ready = time.monotonic()
        remaining(deadline)
        if comparison:
            independent_result = runtime.render(wav, contract.SOURCE_SHA256, independent)
            remaining(deadline)
        continuation_started = time.monotonic() if comparison else ready
        result = runtime.render_continuation(wav, contract.SOURCE_SHA256, turn)
        rendered = time.monotonic()
    finally:
        runtime.close()
    remaining(deadline)
    if Path(result.profile) != turn/'profile.json' or Path(result.review_with_speech) != turn/'review-with-speech.mp4':
        raise ValueError('unexpected runtime artifact paths')
    if candidate is not None:
        profile = json.loads((turn/'profile.json').read_text())
        evidence = profile.get('continuation_plan')
        if not isinstance(evidence, dict) or not strict_equal(evidence.get('candidate'), candidate):
            raise ValueError('continuation candidate profile differs')
    artifacts = []
    exports = [(turn, name, PREFIX+name) for name in (*VIDEOS, *RECEIPTS)]
    if comparison:
        if (Path(independent_result.profile) != independent/'profile.json' or
                Path(independent_result.review_with_speech) != independent/'review-with-speech.mp4'):
            raise ValueError('unexpected independent artifact paths')
        exports += [(independent, name, PREFIX+'independent/'+name)
                    for name in (*VIDEOS, *RECEIPTS)]
    if comparison or candidate is not None:
        stage_dir = output/'stage-profile'
        for rank in range(2):
            for hit in range(1, 7 if comparison else 4):
                name = f'rank-{rank}-warm-{hit}.json'
                path = stage_dir/name
                if not path.is_file() or path.is_symlink() or stage_dir.is_symlink():
                    raise ValueError('missing or unsafe stage timing')
                timing = json.loads(path.read_text())
                if (set(timing) != {'schema', 'worker_rank', 'cache_hit',
                        'synchronization_added', 'benchmark_comparable', 'stages'} or
                        timing['schema'] != 'h3-ref2va-warm-stage-diagnostic-v1' or
                        type(timing['worker_rank']) is not int or timing['worker_rank'] != rank or
                        type(timing['cache_hit']) is not int or timing['cache_hit'] != hit or
                        timing['synchronization_added'] is not True or
                        timing['benchmark_comparable'] is not False or
                        not isinstance(timing['stages'], list) or len(timing['stages']) != 4):
                    raise ValueError('invalid stage timing')
                for stage, expected_name in zip(timing['stages'], (
                        'latent_preparation_stage', 'denoising_stage',
                        'video_decoding_stage', 'audio_decoding_stage')):
                    if (not isinstance(stage, dict) or set(stage) != {
                            'name', 'host_seconds', 'default_stream_elapsed_ms'} or
                            stage['name'] != expected_name or any(
                                type(stage[k]) not in (int, float) or
                                not math.isfinite(stage[k]) or stage[k] < 0
                                for k in ('host_seconds', 'default_stream_elapsed_ms'))):
                        raise ValueError('invalid stage timing scalar')
                exports.append((stage_dir, name, PREFIX+'stage-profile/'+name))
    parity_rows = []
    if parity:
        profile = json.loads((turn/'profile.json').read_text())
        session = profile.get('session_id')
        if type(session) is not str or not session:
            raise ValueError('decode parity requires profile session_id')
        coverage = output/'coverage'
        for rank in range(2):
            name = f'decode-parity-rank-{rank}.json'
            path = coverage/name
            if coverage.is_symlink() or path.is_symlink() or not path.is_file():
                raise ValueError('missing or unsafe decode parity receipt')
            parity_rows.append(validate_parity(json.loads(path.read_text()), rank, session))
        if parity_rows[0]['input_sha256'] != parity_rows[1]['input_sha256']:
            raise ValueError('decode parity rank input hashes differ')
        sanitized = turn/'decode-parity'
        sanitized.mkdir()
        for rank, row in enumerate(parity_rows):
            name = f'decode-parity-rank-{rank}.json'
            (sanitized/name).write_text(json.dumps(row, allow_nan=False))
            exports.append((sanitized, name, PREFIX+'decode-parity/'+name))
    for directory, name, published in exports:
        path = directory/name
        if not path.is_file() or path.is_symlink() or path.resolve().parent != directory.resolve() or directory.is_symlink():
            raise ValueError('missing or unsafe generated artifact')
        if name in VIDEOS:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
                            '-f', 'null', '-'], check=True, timeout=min(120, remaining(deadline)))
        artifacts.append(dict(path=published, bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    receipt = dict(schema='h3-continuation-diagnostic-worker-v1', source=verified,
        startup_seconds=ready-startup, render_seconds=rendered-continuation_started,
        total_seconds=time.monotonic()-started,
        total_scope='worker entry through close and export validation; excludes publication',
        warmups=1, render_calls=2 if comparison else 1, expected_windows=3, full_decode_verified=True,
        artifacts=artifacts, raw_inputs_uploaded=False, latents_uploaded=False,
        identity_quality_verified=False, realtime_verified=False)
    if candidate is not None:
        receipt['continuation_candidate'] = dict(options=candidate, quality_verified=False,
            stage_timing_artifacts=[a for a in artifacts
                                    if a['path'].startswith(PREFIX+'stage-profile/')])
    if parity:
        receipt['decode_parity'] = dict(diagnostic_only=True,
            parity=all(r['parity'] for r in parity_rows), rank_results=parity_rows,
            rank_metrics_equal=all(parity_rows[0][k] == parity_rows[1][k]
                for k in ('raw_float', 'denormalized_pixels', 'uint8')),
            quality_verified=False, realtime_verified=False)
    if comparison:
        profiles = [json.loads((p/'profile.json').read_text()) for p in (independent, turn)]
        for profile in profiles:
            windows = profile.get('windows')
            if (not isinstance(windows, list) or len(windows) != 3 or
                    not isinstance(windows[0], dict) or
                    type(windows[0].get('seed')) is not int or windows[0]['seed'] != 314159):
                raise ValueError('comparison windows or actual first seed differs')
        adapters = [p['derived_adapter_receipt'] for p in profiles]
        if not adapters[0] or adapters[0] != adapters[1]:
            raise ValueError('comparison actual adapter receipts differ')
        adapter_hash = hashlib.sha256(json.dumps(adapters[0], sort_keys=True).encode()).hexdigest()
        receipt['comparison'] = dict(schema=CONTINUATION_COMPARISON['schema'],
            same_warm_runtime=True, full_source=True, controlled_noise_equality=False,
            startup_seconds=ready-startup,
            independent_wall_seconds=continuation_started-ready,
            continuation_wall_seconds=rendered-continuation_started,
            actual_adapter_receipts_equal=True, actual_adapter_receipt_sha256=adapter_hash,
            stage_timing_artifacts=[a for a in artifacts if a['path'].startswith(PREFIX+'stage-profile/')],
            independent_first_seed=profiles[0]['windows'][0]['seed'],
            continuation_seed=profiles[1]['windows'][0]['seed'],
            first_window_comparison_only=True, independent_first_output_frames=124,
            continuation_first_output_frames=119,
            caveats='Later independent seeds progress; audio encoding and video decoding differ; '
                    'equal first seeds do not prove controlled noise equality.')
    receipt_path = turn/'diagnostic-profile.json'
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    operations = [CommitOperationAdd(path_in_repo=published, path_or_fileobj=str(directory/name))
                  for directory, name, published in exports]
    operations.append(CommitOperationAdd(path_in_repo=PREFIX+'diagnostic-profile.json',
                                        path_or_fileobj=str(receipt_path)))
    remaining(deadline)
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('publication must remain private')
    remaining(deadline)
    commit = api.create_commit(deployment, repo_type='dataset', operations=operations,
        commit_message='Fixed-source continuation diagnostic generated artifacts only')
    print('H3_CONTINUATION_PUBLISHED', commit.oid, flush=True)
    return receipt


def main(root, deployment, deadline):
    root = Path(root)
    spec = json.loads((root/'spec.json').read_text())
    assets = json.loads((root/'asset-manifest.json').read_text())['assets']
    validate_spec(spec, assets)
    run(root, {a['role']: root/a['role'] for a in assets}, spec, deployment, float(deadline))


if __name__ == '__main__':
    main(*sys.argv[1:])
