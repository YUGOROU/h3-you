"""Finite, fixed-source, same-residency efficiency comparison; never launches Jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research import hf_h3_continuation_diagnostic as baseworker
from research.h3_personal import continuation_diagnostic as source_contract
from research.h3_personal.continuation_candidate import make_options
from research.h3_personal.continuation_efficiency import comparison_contract, make_options as efficiency_options
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime

PREFIX = 'outputs/efficiency/'
VIDEOS = baseworker.VIDEOS
RECEIPTS = baseworker.RECEIPTS


def load(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, path.parent, path.parent.parent)):
        raise ValueError('missing or unsafe receipt')
    return json.loads(path.read_text())


def validate_spec(spec, assets=None):
    baseworker.validate_spec(spec, assets)
    if not baseworker.strict_equal(spec.get('efficiency_comparison'), comparison_contract()):
        raise ValueError('efficiency comparison contract differs')
    if any(k in spec for k in ('continuation_candidate', 'continuation_comparison', 'decode_parity_diagnostic')):
        raise ValueError('incompatible comparison mode')


def validate_memory(row, rank, hit, acceptance):
    fixed = dict(schema='h3-worker-memory-v1', worker_rank=rank, cache_hit=hit,
                 contains_parallel_acceptance=acceptance)
    if type(row) is not dict or set(row) != set(fixed) | {'peak_allocated_bytes', 'peak_reserved_bytes'}:
        raise ValueError('invalid memory receipt keys')
    if any(not baseworker.strict_equal(row[k], v) for k, v in fixed.items()):
        raise ValueError('invalid memory receipt identity')
    if any(type(row[k]) is not int or row[k] < 0 for k in ('peak_allocated_bytes', 'peak_reserved_bytes')):
        raise ValueError('invalid memory byte count')
    if row['peak_reserved_bytes'] < row['peak_allocated_bytes']:
        raise ValueError('invalid memory reservation')
    return dict(row)


def validate_profile(profile, optimized=False, acceptance=False):
    candidate = make_options(window_audio_context=True, defer_progress_receipts=optimized)
    efficiency = efficiency_options(device_audio_validation=optimized, parallel_decode_waves=optimized)
    planned = profile.get('continuation_plan', {})
    if (not baseworker.strict_equal(planned.get('candidate'), candidate) or
            not baseworker.strict_equal(planned.get('efficiency'), efficiency) or
            not baseworker.strict_equal(planned.get('parallel_acceptance'), acceptance) or
            profile.get('unique_frames') != 328 or profile.get('speech_pcm_exact') is not True or
            profile.get('full_decode_verified') is not True):
        raise ValueError('invalid comparison profile policy/outcome')
    windows = profile.get('windows')
    if (type(windows) is not list or len(windows) != 3 or
            type(windows[0].get('seed')) is not int or windows[0]['seed'] != 314159 or
            not profile.get('derived_adapter_receipt') or not profile.get('input_proof') or
            type(profile.get('session_id')) is not str or not profile['session_id']):
        raise ValueError('missing comparison provenance')
    for window in windows:
        if (type(window) is not dict or type(window.get('seed')) is not int or
                window['seed'] != 314159 or window.get('session_id') != profile['session_id'] or
                window.get('fixed_reference_sha256') != baseworker.REFERENCE_SHA or
                window.get('wav_sha256') != source_contract.SOURCE_SHA256):
            raise ValueError('window source/reference/seed/session differs')
    return profile


def run(root, paths, spec, deployment, deadline):
    started = time.monotonic()
    validate_spec(spec)
    remaining = lambda: baseworker.remaining(deadline)
    remaining()
    root = Path(root)
    paths = {k: Path(v) for k, v in paths.items()}
    wav = paths['speech']/spec['locked_speech']['path']
    verified = source_contract.preflight(wav, timeout_seconds=spec['config']['timeout_seconds'])
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ['HF_TOKEN'])
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('private output dataset required')
    output = root.with_name(root.name+'-efficiency-generated')
    identity = paths['personal']/spec['identity_prefix']
    runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',
        reference=paths['reference']/'production-inputs/references/secondary.jpeg',
        archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True,
        window_audio_context=True, profile_stages=False,
        parallel_decode_acceptance=True, measure_memory=True)
    profiles, walls, parity = {}, {}, []
    startup = time.monotonic()
    try:
        remaining()
        runtime.start()
        startup_seconds = time.monotonic()-startup
        for arm in ('acceptance', 'baseline', 'optimized'):
            if arm != 'acceptance' and not all(r['parity'] for r in parity):
                break
            remaining()
            if arm == 'optimized':
                runtime.configure_efficiency(device_audio_validation=True, parallel_decode_waves=True,
                                             defer_progress_receipts=True)
            before = time.monotonic()
            result = runtime.render_continuation(wav, source_contract.SOURCE_SHA256, output/arm)
            walls[arm] = time.monotonic()-before
            remaining()
            if (Path(result.profile) != output/arm/'profile.json' or
                    Path(result.review_with_speech) != output/arm/'review-with-speech.mp4'):
                raise ValueError('unexpected artifact path')
            profiles[arm] = validate_profile(load(result.profile), arm == 'optimized', arm == 'acceptance')
            if arm == 'acceptance':
                for rank in range(2):
                    row = load(output/'coverage'/f'parallel-parity-rank-{rank}.json')
                    parity.append(baseworker.validate_parity(row, rank, profiles[arm]['session_id'], parallel=True))
                if parity[0]['input_sha256'] != parity[1]['input_sha256']:
                    raise ValueError('parallel parity input hashes differ')
    finally:
        runtime.close()
    remaining()
    first = profiles['acceptance']
    for profile in profiles.values():
        for key in ('input_proof', 'derived_adapter_receipt'):
            if not baseworker.strict_equal(profile[key], first[key]):
                raise ValueError('comparison provenance differs')
    if (output/'stage-profile').exists():
        raise ValueError('unexpected profiling evidence in profile-OFF run')
    exports = []
    for arm, profile in profiles.items():
        for index, window in enumerate(profile['windows']):
            # The immutable receipt precedes the renderer's segment metadata attachment.
            evidence = {key: value for key, value in window.items() if key != 'segment_metadata'}
            if not baseworker.strict_equal(load(output/arm/f'window-{index:03d}-receipts.json'), evidence):
                raise ValueError('window receipt differs from profile')
        exports.extend((output/arm/name, PREFIX+arm+'/'+name) for name in (*VIDEOS, *RECEIPTS))
    sanitized = output/'sanitized'
    sanitized.mkdir(exist_ok=True)
    for rank, row in enumerate(parity):
        name = f'parallel-parity-rank-{rank}.json'
        path = sanitized/name
        path.write_text(json.dumps(row, allow_nan=False))
        exports.append((path, PREFIX+'parity/'+name))
    hits = 9 if len(profiles) == 3 else 3
    for rank in range(2):
        for hit in range(1, hits+1):
            name = f'rank-{rank}-request-hit{hit}.json'
            row = validate_memory(load(output/'memory'/name), rank, hit, hit <= 3)
            path = sanitized/name
            path.write_text(json.dumps(row, allow_nan=False))
            exports.append((path, PREFIX+'memory/'+name))
    artifacts = []
    for path, published in exports:
        remaining()
        if not path.is_file() or any(p.is_symlink() for p in (path, path.parent, path.parent.parent)):
            raise ValueError('unsafe generated artifact')
        if path.name in VIDEOS:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
                            '-f', 'null', '-'], check=True, timeout=min(120, remaining()))
        artifacts.append(dict(path=published, bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    hashes = {a['path']: a['sha256'] for a in artifacts}
    passed = all(r['parity'] for r in parity)
    receipt = dict(schema='h3-efficiency-comparison-worker-v1', comparison=comparison_contract(),
        source=verified, completed=True, parity_gate=passed, parity=parity,
        steady_skipped=not passed, startup_seconds=startup_seconds, arm_wall_seconds=walls,
        acceptance_is_speed_baseline=False, profile_stages=False, warmups=1,
        render_calls=len(profiles), total_seconds=time.monotonic()-started,
        total_scope='Job worker entry through close and export validation; excludes publication and provider startup',
        artifacts=artifacts, raw_inputs_uploaded=False, latents_uploaded=False,
        quality_verified=False, realtime_verified=False,
        exact_video_sha_equal=({name: hashes[PREFIX+'baseline/'+name] == hashes[PREFIX+'optimized/'+name]
                                for name in VIDEOS} if passed else None))
    receipt_path = sanitized/'comparison-profile.json'
    receipt_path.write_text(json.dumps(receipt, allow_nan=False, indent=2))
    exports.append((receipt_path, PREFIX+'comparison-profile.json'))
    remaining()
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('publication must remain private')
    remaining()
    api.create_commit(deployment, repo_type='dataset', operations=[
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for path, name in exports],
        commit_message='Fixed-source efficiency comparison generated artifacts only')
    return receipt


def main(root, deployment, deadline):
    root = Path(root)
    spec = load(root/'spec.json')
    assets = load(root/'asset-manifest.json')['assets']
    validate_spec(spec, assets)
    return run(root, {a['role']: root/a['role'] for a in assets}, spec, deployment, float(deadline))


if __name__ == '__main__':
    main(*sys.argv[1:])
