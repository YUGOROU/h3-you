"""Finite private conditioning candidate exploration; never an adoption or launcher."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from research import hf_h3_continuation_diagnostic as source_worker
from research import hf_h3_efficiency_comparison as efficiency
from research import hf_h3_utilization_comparison as baseworker
from research.h3_personal import continuation_diagnostic as source_contract
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime

SCHEMA = 'h3-conditioning-exploration-v1'
PREFIX = 'outputs/conditioning-exploration/'
ARMS = ('reference1024', 'reference512')
MAX_JSON = 1048576
MAX_VIDEO = 32 * 1024 * 1024
# Canonical wire contract: return fresh containers so callers cannot mutate admission.
_CONTRACT_JSON = json.dumps(dict(schema=SCHEMA, arms=list(ARMS), same_warm_runtime=True,
    full_source_samples=577920, quality_user_review_required=True, auto_adopt=False,
    baseline=dict(version='v14', ref_job_ids=['6adbfe461e112550ce7dad1c'],
        artifact_sha256='a0bebb245bf8aab4abb9802ee5d3936782aa6a37353c20624f04bc26c91f6d87')))


def __getattr__(name):
    if name == 'CONTRACT':
        return json.loads(_CONTRACT_JSON)
    raise AttributeError(name)


def validate_spec(spec, assets=None):
    source_worker.validate_spec(spec, assets)
    plan = spec.get('conditioning_exploration', {})
    if not source_worker.strict_equal(plan, json.loads(_CONTRACT_JSON)):
        raise ValueError('conditioning exploration contract differs')
    for key in ('continuation_candidate', 'continuation_comparison', 'efficiency_comparison',
                'utilization_comparison', 'decode_parity_diagnostic'):
        if key in spec:
            raise ValueError('incompatible exploration mode')
    return plan


def bounded_json(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_JSON:
        raise ValueError('JSON bound exceeded')
    return data


def read_bounded(path, maximum, deadline):
    source_worker.remaining(deadline)
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('unsafe artifact')
    if path.stat().st_size > maximum:
        raise ValueError('artifact bound exceeded')
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    source_worker.remaining(deadline)
    if len(data) > maximum:
        raise ValueError('artifact bound exceeded')
    return data


def validate_evidence(row, rank, edge, session):
    expected = dict(schema='h3-conditioning-experiment-v1', worker_rank=rank,
        session_id=session, window_index=2, cache_hit=True, precomputed_edges=[1024, 512],
        parameter_identity_preserved=True, denoise_stages_ready=True,
        resize_policy='shortedge-lanczos-grid32-v1',
        R=1824 if edge == 1024 else 448, T=1867 if edge == 1024 else 491,
        policy=dict(schema='h3-conditioning-experiment-v1', short_edges=[1024, 512], active_short_edge=edge))
    if any(not source_worker.strict_equal(row.get(k), v) for k, v in expected.items()):
        raise ValueError('conditioning evidence differs')
    sizes = row.get('prepared_sizes')
    if not source_worker.strict_equal(sizes, [[1024, 1824] if edge == 1024 else [512, 896]]):
        raise ValueError('prepared sizes differ')
    digest = row.get('processor_config_sha256', '')
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('missing processor configuration identity')
    return dict(expected, prepared_sizes=sizes, processor_config_sha256=digest)


def finite_supply(ready):
    if (type(ready) is not list or len(ready) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in ready) or
            ready != sorted(ready)):
        raise ValueError('invalid finite supply ready times')
    deadlines = [ready[0]+119/24, ready[0]+(119+102)/24]
    latenesses = [max(0., value-deadline) for value, deadline in zip(ready[1:], deadlines)]
    return dict(segment_ready_elapsed_seconds=list(ready), first_wait_seconds=ready[0],
        middle_ready_interval_seconds=ready[1]-ready[0], middle_generated_supply_seconds=102/24,
        sequential_playback_deadlines_seconds=deadlines, lateness_seconds=latenesses,
        max_stall_seconds=max(latenesses), segment_frames=[119, 102, 107], fps=24,
        full_source_seconds=12.04, rendered_seconds=328/24, padding_seconds=328/24-12.04,
        padding_is_new_conversation=False, finite_buffer_only=True, causal=False,
        long_run_acceptance=False, perceptual_acceptance=False)


def run(root, paths, spec, deployment, deadline):
    started = time.monotonic()
    plan = validate_spec(spec)
    root = Path(root)
    paths = {k: Path(v) for k, v in paths.items()}
    remaining = lambda: source_worker.remaining(deadline)
    remaining()
    wav = paths['speech'] / spec['locked_speech']['path']
    timeout = min(spec['config']['timeout_seconds'], int(math.floor(remaining())))
    if type(timeout) is not int or timeout < 1:
        raise ValueError('insufficient integer preflight timeout')
    source_contract.preflight(wav, timeout_seconds=timeout)
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ['HF_TOKEN'])

    def publish(exports):
        remaining()
        if api.repo_info(deployment, repo_type='dataset').private is not True:
            raise ValueError('private deployment dataset required')
        remaining()
        api.create_commit(deployment, repo_type='dataset', operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in exports],
            commit_message='Private conditioning exploration validated outputs')

    if api.repo_info(deployment, repo_type='dataset').private is not True:
        raise ValueError('private deployment dataset required')
    output = root.with_name(root.name + '-conditioning-generated')
    if output.exists():
        raise ValueError('fresh private output directory required')
    identity = paths['personal'] / spec['identity_prefix']
    runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',
        reference=paths['reference']/'production-inputs/references/secondary.jpeg',
        archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True,
        conditioning_experiment=dict(schema='h3-conditioning-experiment-v1', short_edges=[1024, 512], active_short_edge=1024),
        window_audio_context=True, defer_progress_receipts=True, device_audio_validation=True,
        parallel_decode_waves=True, measure_memory=True, profile_stages=False,
        parallel_decode_acceptance=False)
    completed, exports, sessions, processor = {}, [], set(), None
    try:
        startup = time.monotonic()
        runtime.start()
        startup_seconds = time.monotonic()-startup
        for edge, arm in zip((1024, 512), ARMS):
            remaining()
            if edge == 512:
                runtime.configure_conditioning(short_edge=edge)
            before = time.monotonic()
            result = runtime.render_continuation(wav, source_contract.SOURCE_SHA256, output/arm)
            wall = time.monotonic()-before
            if Path(result.profile) != output/arm/'profile.json' or Path(result.review_with_speech) != output/arm/'review-with-speech.mp4':
                raise ValueError('unexpected output paths')
            profile = efficiency.validate_profile(json.loads(read_bounded(result.profile, MAX_JSON, deadline)), optimized=True, acceptance=False)
            session = profile['session_id']
            if session in sessions:
                raise ValueError('each arm requires new Session')
            sessions.add(session)
            evidence = []
            for rank in range(2):
                row = validate_evidence(json.loads(read_bounded(Path(runtime.coverage)/f'conditioning-experiment-rank-{rank}.json', MAX_JSON, deadline)), rank, edge, session)
                digest = row['processor_config_sha256']
                if processor is not None and digest != processor:
                    raise ValueError('processor changed across ranks/arms')
                processor = digest
                evidence.append(row)
            first_hit = 1 if edge == 1024 else 4
            memory = [efficiency.validate_memory(json.loads(read_bounded(
                output/'memory'/f'rank-{rank}-request-hit{hit}.json', MAX_JSON, deadline)),
                rank, hit, False) for hit in range(first_hit, first_hit+3) for rank in range(2)]
            peaks = [dict(worker_rank=rank, **{key: max(row[key] for row in memory
                if row['worker_rank'] == rank) for key in ('peak_allocated_bytes', 'peak_reserved_bytes')})
                for rank in range(2)]
            media = [baseworker.verify_media(output/arm/name, wav, deadline) for name in source_worker.VIDEOS]
            timing = baseworker.timing_scalars(profile)
            if 'pipeline_timing' not in timing:
                raise ValueError('encode/validation timing required')
            data = read_bounded(result.review_with_speech, MAX_VIDEO, deadline)
            completed[arm] = dict(conditioning=evidence, timing=timing, media=media,
                window_memory=memory, rank_peak_memory=peaks,
                finite_supply=finite_supply(timing['segment_ready_elapsed_seconds']),
                wall_seconds=wall, review_sha256=hashlib.sha256(data).hexdigest())
            exports.append((PREFIX+arm+'-review.mp4', data))
            publish([(PREFIX+arm+'-progress.json', bounded_json(dict(schema=SCHEMA, stage='arm-validated', arm=arm, **completed[arm])))])
    finally:
        runtime.close()
    if (output/'stage-profile').exists():
        raise ValueError('profiler forbidden')
    report = dict(schema=SCHEMA, completed=True, arms=completed, startup_seconds=startup_seconds,
        total_seconds=time.monotonic()-started, same_warm_runtime=True, baseline=dict(plan['baseline'],
        comparison='historical-not-paired', generation_seconds=20.317147144, render_seconds=22.059349014),
        candidate_adopted=False, auto_adopt=False, perceptual_quality='USER-review-pending',
        quality_user_review_required=True, hashes_are_quality_scores=False, realtime_verified=False,
        realtime_counting=dict(frames=328, fps=24, first_seconds=119/24, middle_seconds=102/24,
            last_seconds=107/24, video_seconds=328/24, source_seconds=12.04,
            source_samples=577920, padding_seconds=328/24-12.04, causal=False),
        raw_inputs_uploaded=False)
    exports.append((PREFIX+'report.json', bounded_json(report)))
    publish(exports)
    return report


def main(root, deployment, deadline):
    root = Path(root)
    spec = json.loads(read_bounded(root/'spec.json', MAX_JSON, float(deadline)))
    assets = json.loads(read_bounded(root/'asset-manifest.json', MAX_JSON, float(deadline)))['assets']
    validate_spec(spec, assets)
    return run(root, {a['role']: root/a['role'] for a in assets}, spec, deployment, float(deadline))
