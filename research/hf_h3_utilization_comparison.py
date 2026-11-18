"""Finite same-runtime utilization experiment. No launcher; metadata-only publication.

Runtime integration contract: configure_utilization(trace_policy=..., denoise_policy=...)
requires the previous Session owner to be complete. Each returned profile must carry
utilization={owner_complete: True, trace: ..., denoise: ...}; trace/denoise evidence
must be scalar receipts from BOTH ranks. Ranks are ordered [0,1] with rank,
trace_available, coverage_proven, truncated, event_count, trace_bytes, trace_summary
(the device_activity aggregate), denoise_installed, candidate_rejected,
parameter_identity_preserved, representative_preflight_complete, backend_compiles,
compile_seconds, regions and optional peak_allocated_bytes/peak_reserved_bytes.
Safe rejection requires both ranks to reject with a faulted, incomplete owner; native faults
propagate and prohibit publication. Unsupported instrumentation is unavailable,
never successful coverage. Constructor policy defaults are explicitly None.

Pending integration: current fast_continuation_render does not emit this evidence;
worker intentionally refuses that source instead of manufacturing GPU proof.
External bootstrap process-group deadline remains the hard bound for native hangs.
"""
import hashlib
import json
import math
from itertools import combinations
import os
from pathlib import Path
import re
import subprocess
import time

from research import hf_h3_continuation_diagnostic as base
from research import hf_h3_efficiency_comparison as efficiency
from research.h3_personal import continuation_diagnostic as source
from research.h3_personal.utilization_policy import normalize
from research.h3_personal.denoise_execution import CandidateRejected

TRACE_POLICY = dict(schema='h3-worker-trace-v1', selected_window=1,
                    max_events=200000, max_bytes=262144)
DENOISE_POLICY = dict(schema='h3-local-post-compile-v1', max_backend_compiles=8,
                      max_compile_seconds=180, max_regions=50)
EXCLUSIVE = ('continuation_candidate', 'continuation_comparison',
             'decode_parity_diagnostic', 'efficiency_comparison')


def comparison_contract():
    normalize(TRACE_POLICY, 'trace')
    normalize(DENOISE_POLICY, 'denoise')
    return dict(schema='h3-utilization-comparison-v1', trace_policy=dict(TRACE_POLICY),
                denoise_policy=dict(DENOISE_POLICY), total_frames=328,
                source_samples=577920, arms=['baseline', 'trace', 'candidate_warmup', 'candidate'],
                profile_stages=False, baseline='v7-device-waves-deferred-window-audio',
                candidate_warmup_in_steady=False, numerical_tolerance=None,
                raw_uploads=False)


def validate_spec(spec, assets=None):
    base.validate_spec(spec, assets)
    if (not base.strict_equal(spec.get('utilization_comparison'), comparison_contract()) or
            any(k in spec for k in EXCLUSIVE)):
        raise ValueError('invalid or incompatible utilization comparison')


def parse_topology(text):
    """Discard all identifiers, affinity, hostnames, and unknown topology tokens."""
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ('GPU0', 'GPU1') and len(fields) >= 3:
            tokens = fields[1:3]
            if all(re.fullmatch(r'X|NV[1-9][0-9]?|SYS|NODE|PHB|PXB|PIX', t) for t in tokens):
                rows[fields[0]] = tokens
    return dict(available=set(rows) == {'GPU0', 'GPU1'}, links=rows,
                bandwidth_measured=False, runtime_country_verified=False)


def topology(deadline):
    # Never shell, never publish raw stdout/stderr (which may include host identifiers).
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True,
                                timeout=min(10, base.remaining(deadline)), check=True)
        if len(result.stdout) > 65536:
            return dict(available=False, reason='output-bound')
        return parse_topology(result.stdout.decode('utf-8', errors='replace'))
    except (OSError, subprocess.SubprocessError):
        return dict(available=False, reason='unavailable')


def device_mapping(deadline):
    # A bounded subprocess avoids initializing CUDA in the parent model owner.
    import sys
    code = ('import torch,json; n=torch.cuda.device_count(); '
            'print(json.dumps([{"logical_device":i,"kind":torch.cuda.get_device_name(i),'
            '"total_bytes":torch.cuda.get_device_properties(i).total_memory,'
            '"free_bytes":torch.cuda.mem_get_info(i)[0]} for i in range(min(n,3))]))')
    try:
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, check=True,
                           timeout=min(20, base.remaining(deadline)))
        if len(p.stdout) > 4096:
            raise ValueError('device output bound')
        rows = json.loads(p.stdout)
        if len(rows) != 2 or any(not re.fullmatch(r'(?:NVIDIA )?H200(?: NVL)?', r['kind']) for r in rows):
            raise ValueError('same H200x2 required')
        for i, row in enumerate(rows):
            if row['logical_device'] != i or any(type(row[k]) is not int or row[k] < 0
                                                for k in ('total_bytes', 'free_bytes')):
                raise ValueError('invalid device mapping')
        return rows
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('device mapping unavailable; no hardware fallback') from exc


def validate_profile(profile, arm):
    efficiency.validate_profile(profile, optimized=True, acceptance=False)
    evidence = profile.get('utilization')
    if type(evidence) is not dict or evidence.get('owner_complete') is not True:
        raise ValueError('runtime owner completion evidence missing')
    # Only the explicitly selected arm may trace; ordinary stage profiling stays OFF.
    if not base.strict_equal(evidence.get('trace_policy'), TRACE_POLICY if arm == 'trace' else None):
        raise ValueError('trace policy differs')
    if not base.strict_equal(evidence.get('denoise_policy'), DENOISE_POLICY if arm.startswith('candidate') else None):
        raise ValueError('denoise policy differs')
    return profile


def trace_scalars(report):
    """Allowlist the device_activity aggregate, never event names or tensor shapes."""
    if report is None:
        return dict(available=False, coverage_proven=False)
    categories = ('compute', 'collective', 'copy', 'unknown')
    partitions = {'+'.join(items) for n in range(1, 5) for items in combinations(categories, n)}
    def scalar(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid trace scalar')
        return value
    result = dict(available=True, coverage_proven=False,
                  clock='rank_local_us_no_cross_device_alignment', devices={})
    if 'scope' in report:
        if report['scope'] != 'warm-window-intersection':
            raise ValueError('invalid trace scope')
        result['scope'] = report['scope']
    if 'annotation_filter_applied' in report:
        if report['annotation_filter_applied'] is not True:
            raise ValueError('invalid annotation filter flag')
        result['annotation_filter_applied'] = True
    for key in ('excluded_events', 'clipped_events', 'excluded_gpu_annotations'):
        if key in report:
            value = report[key]
            if type(value) is not int or not 0 <= value <= 200000:
                raise ValueError('invalid trace exclusion count')
            result[key] = value
    for key in ('truncated',):
        if key in report:
            if type(report[key]) is not bool:
                raise ValueError('invalid trace flag')
            result[key] = report[key]
    for device in ('cpu', 'cuda', 'unknown'):
        row = report.get('devices', {}).get(device)
        if row is None:
            continue
        out = {key: scalar(row[key]) for key in ('events', 'busy_us')}
        for key in ('capture_span_us', 'observed_gap_us'):
            out[key] = None if row.get(key) is None else scalar(row[key])
        out['category_union_us'] = {key: scalar(value) for key, value in
            row.get('category_union_us', {}).items() if key in categories}
        out['disjoint_us'] = {key: scalar(value) for key, value in row.get('disjoint_us', {}).items()
                              if key in partitions}
        out['family_event_counts'] = {key: scalar(value) for key, value in
            row.get('family_event_counts', {}).items() if key in
            ('nccl', 'memcpy', 'gemm', 'flash_attention', 'norm', 'triton', 'operator', 'unknown')}
        result['devices'][device] = out
    from research.h3_personal.device_activity import _SEMANTIC
    families = ('nccl', 'memcpy', 'gemm', 'flash_attention', 'norm', 'triton', 'operator', 'unknown')
    for key, bound in (('semantic_intervals', 512), ('gpu_category_intervals', 2048)):
        if key not in report:
            continue
        rows = report[key]
        if type(rows) is not list or len(rows) > bound:
            raise ValueError('invalid timeline bound')
        cleaned = []
        for row in rows:
            labels = ('label',) if key == 'semantic_intervals' else ('category', 'family')
            if type(row) is not dict or set(row) != set(labels + ('start_us', 'duration_us')):
                raise ValueError('invalid timeline fields')
            if labels == ('label',):
                if type(row['label']) is not str or not _SEMANTIC.fullmatch(row['label']):
                    raise ValueError('invalid semantic label')
            elif row['category'] not in categories or row['family'] not in families:
                raise ValueError('invalid GPU label')
            cleaned.append(dict(**{k: row[k] for k in labels},
                                start_us=scalar(row['start_us']), duration_us=scalar(row['duration_us'])))
        result[key] = cleaned
    for key in ('semantic_intervals_truncated', 'gpu_intervals_truncated', 'cuda_timeline_available'):
        if key in report:
            if type(report[key]) is not bool:
                raise ValueError('invalid timeline flag')
            result[key] = report[key]
    if 'shapes' in report:
        shapes = report['shapes']
        if type(shapes) is not list or len(shapes) > 512:
            raise ValueError('invalid shape bound')
        result['shapes'] = []
        for row in shapes:
            if type(row) is not dict or not set(row) <= {'window_index', 'block_index', 'batch', 'tokens', 'channels', 'height', 'width', 'frames', 'forward_index', 'reference_rows', 'text_rows', 'target_video_rows', 'guide_video_rows', 'target_audio_rows', 'guide_audio_rows', 'logical_rows', 'local_rows', 'derived_qkv_payload_bytes'}:
                raise ValueError('untrusted shape fields')
            if any(type(v) is not int or not 0 <= v <= 2147483647 for v in row.values()):
                raise ValueError('invalid shape integer')
            result['shapes'].append(dict(row))
    from research.h3_personal.device_activity import bound_report
    bound_report(result, TRACE_POLICY['max_bytes'])
    # CPU operator activity alone is not a valid CUDA utilization trace.
    result['available'] = bool(result['devices'].get('cuda', {}).get('events', 0))
    return result


def scalar_evidence(profile, arm):
    """Strict allowlist; no recursive copying of worker traces or exception strings."""
    evidence = profile['utilization']
    ranks = evidence.get('ranks')
    if type(ranks) is not list or len(ranks) != 2:
        raise ValueError('both-rank utilization evidence required')
    clean = []
    for rank, row in enumerate(ranks):
        if row.get('rank') != rank:
            raise ValueError('rank identity differs')
        out = dict(rank=rank, trace_summary=trace_scalars(row.get('trace_summary')))
        kind = row.get('rejection_kind')
        if kind is not None:
            if kind not in ('controlled-local-rejection', 'controlled-admission-rejection'):
                raise ValueError('unknown rejection kind')
            out['rejection_kind'] = kind
        if kind == 'controlled-admission-rejection':
            from research.h3_personal.fast_continuation_render import admission_proof
            try:
                out.update(admission_proof(row))
            except RuntimeError as exc:
                raise ValueError('invalid admission proof') from exc
        for key in ('trace_available', 'coverage_proven', 'truncated', 'denoise_installed',
                    'candidate_rejected', 'parameter_identity_preserved', 'representative_preflight_complete'):
            value = row.get(key, False)
            if type(value) is not bool:
                raise ValueError('invalid boolean evidence')
            out[key] = value
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'post_request_free_bytes', 'post_request_total_bytes'):
            value = row.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('invalid memory counter')
            out[key] = value  # Missing telemetry is unavailable, not measured zero.
        for key in ('extra_status_seconds',):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('invalid status timing')
            out[key] = value
        out['extra_status_collectives'] = row.get('extra_status_collectives')
        if out['extra_status_collectives'] is not None and (type(out['extra_status_collectives']) is not int or out['extra_status_collectives'] < 0):
            raise ValueError('invalid status collective count')
        for key in ('event_count', 'trace_bytes', 'backend_compiles', 'regions'):
            value = row.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError('invalid bounded counter')
            out[key] = value
        seconds = row.get('compile_seconds', 0)
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or
                (seconds > 180 and not out['candidate_rejected'])):
            raise ValueError('compile time bound exceeded without diagnosed rejection')
        out['compile_seconds'] = seconds
        out['compile_time_budget_exceeded'] = seconds > DENOISE_POLICY['max_compile_seconds']
        if (out['event_count'] > 200000 or out['trace_bytes'] > 262144 or
                out['backend_compiles'] > 8 or out['regions'] > 50):
            raise ValueError('utilization evidence bound exceeded')
        if (arm == 'trace' and not out['trace_summary']['available'] and
                out['trace_summary'].get('devices')):
            out['trace_available'] = False
            out['coverage_proven'] = False
        if out['coverage_proven'] and (not out['trace_available'] or out['truncated']):
            raise ValueError('unsupported trace success')
        if arm.startswith('candidate') and not out['candidate_rejected'] and not (
                out['denoise_installed'] and out['parameter_identity_preserved'] and
                out['representative_preflight_complete']):
            raise ValueError('candidate installation proof missing')
        metrics = row.get('representative_metrics')
        out['representative_metrics'] = None
        if metrics is not None:
            if type(metrics) is not dict or type(metrics.get('exact')) is not bool:
                raise ValueError('invalid representative metrics')
            safe = dict(exact=metrics['exact'], scope='representative token intermediate, not frame parity')
            for key in ('max', 'mean', 'eager_ms', 'compiled_ms', 'host_seconds'):
                if key in metrics:
                    value = metrics[key]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid representative scalar')
                    safe[key] = value
            if 'shape' in metrics:
                shape = metrics['shape']
                if (type(shape) is not list or len(shape) > 5 or
                        any(type(v) is not int or not 0 < v < 2147483648 for v in shape)):
                    raise ValueError('invalid representative shape')
                safe['shape'] = shape
            out['representative_metrics'] = safe
        out['worker_windows'] = []
        windows = evidence.get('windows', [])
        if windows:
            if (type(windows) is not list or not 1 <= len(windows) <= 3 or
                    (evidence.get('owner_complete') is True and len(windows) != 3)):
                raise ValueError('invalid worker clock windows')
            origin = None
            for i, pair in enumerate(windows):
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError('invalid worker clock pair')
                item = pair[rank]
                start, end = item.get('worker_monotonic_start'), item.get('worker_monotonic_end')
                if (item.get('rank') != rank or item.get('window_index') != i or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or
                        end < start):
                    raise ValueError('invalid worker clock evidence')
                if origin is None:
                    origin = start
                out['worker_windows'].append(dict(window_index=i, clock='rank_local_monotonic',
                    start_seconds=start-origin, end_seconds=end-origin,
                    duration_seconds=end-start))
        clean.append(out)
    if len({row['candidate_rejected'] for row in clean}) != 1:
        raise ValueError('candidate rejection must agree across both ranks')
    return clean


def rejected_evidence(profile, arm):
    """Incomplete renders may publish rejection scalars, never media acceptance.

    Renderer writes structured both-rank receipts before raising CandidateRejected
    (including conversion of an IPC-lost exception). Abort never completes an owner.
    Missing or one-rank evidence remains an ERROR.
    """
    if not arm.startswith('candidate'):
        raise ValueError('non-candidate rejection')
    evidence = profile.get('utilization', {})
    if (evidence.get('owner_complete') is not False or
            evidence.get('owner_terminal') != 'faulted' or
            evidence.get('candidate_rejected') is not True or
            not base.strict_equal(evidence.get('denoise_policy'), DENOISE_POLICY) or
            evidence.get('trace_policy') is not None):
        raise ValueError('rejection owner/policy evidence missing')
    rows = scalar_evidence(profile, arm)
    if not all(row['candidate_rejected'] and row['parameter_identity_preserved'] for row in rows):
        raise ValueError('both-rank identity-preserved diagnosed rejection required')
    raw = evidence['ranks']
    session = raw[0].get('session_id')
    window = raw[0].get('window_index')
    if not isinstance(session, str) or not session or type(window) is not int or window != 0:
        raise ValueError('representative rejection session/window missing')
    for row in raw:
        if (row.get('schema') != 'h3-window-utilization-v1' or
                row.get('session_id') != session or
                type(row.get('window_index')) is not int or row['window_index'] != window or
                row.get('faulted') is not True or
                not base.strict_equal(row.get('denoise_policy'), DENOISE_POLICY) or
                row.get('trace_policy') is not None):
            raise ValueError('rejection rank identity/policy/fault differs')
    kinds = {row.get('rejection_kind') for row in raw}
    if len(kinds) != 1:
        raise ValueError('both-rank rejection kind differs')
    if 'controlled-admission-rejection' in kinds:
        if not all(row['backend_compiles'] <= 8 and
                   row['extra_status_collectives'] is not None and
                   row['extra_status_collectives'] > 0 and
                   row['representative_metrics'] is None for row in rows):
            raise ValueError('invalid admission rejection counters')
    return rows


def compare_rgb(left, right, deadline, *, width=252, height=448, frames=328):
    """Local uint8 RGB comparison; no tolerance and no frame/pixel publication."""
    stride = width * height * 3
    expected = stride * frames
    def decode(path):
        # Fixed output bound enforced by -frames:v and exact decoded byte count.
        result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
            '-map', '0:v:0', '-frames:v', str(frames+1), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
            capture_output=True, check=True, timeout=min(120, base.remaining(deadline)))
        if len(result.stdout) != expected:
            raise ValueError('decoded frame count/geometry differs')
        return result.stdout
    a, b = decode(left), decode(right)
    maxima, means = [0] * frames, [0.0] * frames
    if a != b:
        # Exact integer differences; the stdlib path is authoritative, numpy is not needed.
        for index, start in enumerate(range(0, expected, stride)):
            peak = total = 0
            for x, y in zip(memoryview(a)[start:start+stride], memoryview(b)[start:start+stride]):
                delta = abs(x-y)
                peak, total = max(peak, delta), total + delta
            maxima[index], means[index] = peak, total / stride
    return dict(dtype='uint8', frames=frames, max=max(maxima), mean=sum(means)/frames,
                per_frame_max=maxima, per_frame_mean=means, exact=not any(maxima),
                tolerance=None, quality_accepted=not any(maxima))


def verify_media(path, wav, deadline):
    """Recheck local decoded PTS/geometry and exact PCM, not just receipt assertions."""
    count, offset = {'segment-000.mp4': (119, 0), 'segment-001.mp4': (102, 119),
                     'segment-002.mp4': (107, 221)}.get(path.name, (328, 0))
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:frame=best_effort_timestamp_time',
        '-show_frames', '-of', 'json', str(path)], capture_output=True, check=True,
        timeout=min(60, base.remaining(deadline)))
    if len(probe.stdout) > 1048576:
        raise ValueError('metadata bound exceeded')
    metadata = json.loads(probe.stdout)
    streams, frames = metadata['streams'], metadata['frames']
    if (len(streams) != 1 or streams[0]['width'] != 252 or streams[0]['height'] != 448 or
            len(frames) != count or any(abs(float(f['best_effort_timestamp_time'])-(i+offset)/24) > 0.000002
                                      for i, f in enumerate(frames))):
        raise ValueError('video metadata/PTS differs')
    if path.name == 'review-with-speech.mp4':
        def pcm(p):
            result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(p),
                '-map', '0:a:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-'],
                capture_output=True, check=True, timeout=min(60, base.remaining(deadline)))
            if len(result.stdout) != 577920*2:
                raise ValueError('PCM sample count differs')
            return result.stdout
        if pcm(path) != pcm(wav):
            raise ValueError('source PCM differs')
    return dict(frames=count, pts_start=offset, width=252, height=448, fps=24, pts_continuous=True,
                source_pcm_exact=path.name == 'review-with-speech.mp4')


def timing_scalars(profile):
    metrics = profile['timing_metrics']
    out = {}
    for key in ('first_segment_ready_seconds', 'total_wall_seconds', 'generation_total_seconds',
                'unique_video_seconds', 'unique_throughput_total', 'unique_throughput_generation'):
        value = metrics[key]
        if type(value) not in (int, float) or not 0 < value < 100000:
            raise ValueError('invalid timing scalar')
        out[key] = value
    ready = metrics['segment_ready_elapsed_seconds']
    if (type(ready) is not list or len(ready) != 3 or any(type(v) not in (int, float) or
            not 0 <= v < 100000 for v in ready) or ready != sorted(ready)):
        raise ValueError('invalid suffix ready times')
    out.update(segment_ready_elapsed_seconds=ready, source_speech_seconds=12.04,
               source_speech_throughput_total=12.04/out['total_wall_seconds'],
               later_suffix_deficit_seconds=[max(0, ready[i]-ready[i-1]-4.25) for i in (1, 2)])
    if 'pipeline_timing' in profile:
        rows = profile['pipeline_timing']
        keys = ('request_start', 'request_end', 'encoding_start', 'encoding_end',
                'validation_start', 'validation_end')
        if type(rows) is not list or len(rows) != 3:
            raise ValueError('invalid parent timing rows')
        clean = []
        for index, row in enumerate(rows):
            if (type(row) is not dict or set(row) != set(keys) | {'clock', 'window_index'} or
                    type(row['window_index']) is not int or row['window_index'] != index or
                    row['clock'] != 'parent_monotonic'):
                raise ValueError('invalid parent timing schema')
            values = [row[k] for k in keys]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values) or values != sorted(values):
                raise ValueError('invalid parent timing clock')
            clean.append(dict(row, request_seconds=row['request_end']-row['request_start'],
                encoding_seconds=row['encoding_end']-row['encoding_start'],
                validation_seconds=row['validation_end']-row['validation_start']))
        for index, row in enumerate(clean):
            gap = clean[index+1]['request_start']-row['validation_end'] if index < 2 else None
            if gap is not None and gap < 0:
                raise ValueError('parent windows overlap')
            row['gap_before_next_request_seconds'] = gap
        out['pipeline_timing'] = clean
    return out


PROGRESS_ARMS = ('baseline', 'trace', 'candidate_warmup', 'candidate')
PROGRESS_MAX_BYTES = 1048576
FINAL_MAX_BYTES = PROGRESS_MAX_BYTES


def write_final_report(path, receipt):
    """Bound the complete publication, not merely each rank's trace summary."""
    encoded = json.dumps(receipt, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    if len(encoded) > FINAL_MAX_BYTES:
        raise ValueError('final report byte bound exceeded')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded)


def progress_checkpoint(output, arm, profile, *, max_bytes=PROGRESS_MAX_BYTES):
    """Serialize only validated scalar producers; never copy source receipt dictionaries."""
    if type(arm) is not str or arm not in PROGRESS_ARMS:
        raise ValueError('unknown progress arm')
    if type(max_bytes) is not int or not 0 < max_bytes <= PROGRESS_MAX_BYTES:
        raise ValueError('invalid progress report bound')
    receipt = dict(schema='h3-utilization-progress-v1', arm=arm,
        completed_owner=True, candidate_adopted=False,
        measurement_context='profiled' if arm == 'trace' else 'profileOFF',
        timing=timing_scalars(profile), evidence=scalar_evidence(profile, arm))
    encoded = json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > max_bytes:
        raise ValueError('progress report byte bound exceeded')
    directory = Path(output)/'progress'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('unsafe progress directory')
    path = directory/f'progress-{arm}.json'
    # Exclusive creation prevents following stale files/symlinks or replacing evidence.
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(encoded)
    return path, f'outputs/utilization/progress-{arm}.json'


def run(root, paths, spec, deployment, deadline):
    validate_spec(spec)
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from huggingface_hub import HfApi, CommitOperationAdd
    remaining = lambda: base.remaining(deadline)
    remaining()
    root = Path(root)
    paths = {k: Path(v) for k, v in paths.items()}
    wav = paths['speech']/spec['locked_speech']['path']
    preflight_seconds = math.floor(min(spec['config']['timeout_seconds'], remaining()))
    if preflight_seconds < 1:
        raise TimeoutError('insufficient finite preflight time')
    source.preflight(wav, timeout_seconds=preflight_seconds)
    api = HfApi(token=os.environ['HF_TOKEN'])
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('private publication required')
    topo, mapping = topology(deadline), device_mapping(deadline)
    output = root.with_name(root.name+'-utilization-generated')
    identity = paths['personal']/spec['identity_prefix']
    runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',
        reference=paths['reference']/'production-inputs/references/secondary.jpeg',
        archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True,
        window_audio_context=True, defer_progress_receipts=True, device_audio_validation=True,
        parallel_decode_waves=True, profile_stages=False, measure_memory=True,
        trace_policy=None, denoise_policy=None)
    profiles, walls, evidence = {}, {}, {}
    rejection = None
    started = time.monotonic()
    try:
        runtime.start()
        startup_seconds = time.monotonic()-started
        for arm in comparison_contract()['arms']:
            remaining()
            runtime.configure_utilization(trace_policy=TRACE_POLICY if arm == 'trace' else None,
                denoise_policy=DENOISE_POLICY if arm.startswith('candidate') else None)
            before = time.monotonic()
            try:
                result = runtime.render_continuation(wav, source.SOURCE_SHA256, output/arm)
            except CandidateRejected:
                # No exception-string classification and no eager retry. The renderer
                # owns private per-window evidence and its faulted-owner receipt.
                rejection_path = output/arm/'profile.json'
                evidence[arm] = rejected_evidence(efficiency.load(rejection_path), arm)
                walls[arm] = time.monotonic()-before
                rejection = ('candidate-unavailable' if all(
                    row.get('rejection_kind') == 'controlled-admission-rejection'
                    for row in evidence[arm]) else 'representative-preflight')
                break
            walls[arm] = time.monotonic()-before
            remaining()
            if Path(result.profile) != output/arm/'profile.json':
                raise ValueError('unexpected profile path')
            profiles[arm] = validate_profile(efficiency.load(result.profile), arm)
            evidence[arm] = scalar_evidence(profiles[arm], arm)
            for key in ('input_proof', 'derived_adapter_receipt'):
                if not base.strict_equal(profiles[arm][key], profiles['baseline'][key]):
                    raise ValueError('arm provenance differs')
            if arm.startswith('candidate') and any(r['candidate_rejected'] for r in evidence[arm]):
                raise ValueError('successful owner cannot report candidate abort')
            if len({p['session_id'] for p in profiles.values()}) != len(profiles):
                raise ValueError('arm Session reused')
            path, destination = progress_checkpoint(output, arm, profiles[arm])
            remaining()
            if not api.repo_info(deployment, repo_type='dataset').private:
                raise ValueError('publication must remain private')
            api.create_commit(deployment, repo_type='dataset', operations=[CommitOperationAdd(
                path_in_repo=destination, path_or_fileobj=str(path))],
                commit_message='Bounded per-arm utilization progress only')
    finally:
        runtime.close()
    if len({p['session_id'] for p in profiles.values()}) != len(profiles):
        raise ValueError('arm Session reused')
    numerical, artifacts, media = {}, {}, {}
    for arm in profiles:
        artifacts[arm], media[arm] = {}, {}
        for name in base.VIDEOS:
            path = output/arm/name
            if not path.is_file() or path.is_symlink():
                raise ValueError('unsafe video')
            media[arm][name] = verify_media(path, wav, deadline)
            artifacts[arm][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if arm == 'candidate':
                numerical[name] = compare_rgb(output/'baseline'/name, path, deadline,
                                              frames=media[arm][name]['frames'])
    candidate_exact = bool(numerical) and all(row['exact'] for row in numerical.values())
    candidate_faster = ('candidate' in profiles and
        timing_scalars(profiles['candidate'])['total_wall_seconds'] <
        timing_scalars(profiles['baseline'])['total_wall_seconds'])
    if rejection is None and 'candidate' in profiles:
        rejection = None if candidate_exact and candidate_faster else (
            'numeric-difference' if not candidate_exact else 'performance')
    trace_parts = [dict(rank=row['rank'], devices=row['trace_summary']['devices'])
                   for row in evidence.get('trace', []) if row['trace_summary']['available']]
    receipt = dict(schema='h3-utilization-comparison-worker-v1', comparison=comparison_contract(),
        candidate_adopted=bool(candidate_exact and candidate_faster and rejection is None),
        candidate_rejected=rejection is not None, rejection_reason=rejection,
        compare_trace_parts=trace_parts,
        topology=topo, devices=mapping, evidence=evidence, arm_wall_seconds=walls,
        startup_seconds=startup_seconds, candidate_warmup_in_steady=False,
        video_sha256=artifacts, media=media, numerical=numerical, completed=True,
        timing={arm: timing_scalars(p) for arm, p in profiles.items()},
        total_scope='runtime start through close and local verification; excludes provider startup/publication',
        candidate_measured='candidate' in profiles, realtime_verified=False,
        raw_uploaded=False, trace_uploaded=False, raw_trace_uploaded=False,
        sanitized_timing_metadata_uploaded=True,
        trace_uploaded_scope='raw_only; bounded sanitized timing metadata retained',
        compile_timing_scope='one representative pair; small gate, not a statistical benchmark',
        total_seconds=time.monotonic()-started)
    # Only this allowlisted scalar report is uploaded; videos/PCM/PTSs/traces stay private locally.
    path = output/'utilization-comparison.json'
    write_final_report(path, receipt)
    remaining()
    if not api.repo_info(deployment, repo_type='dataset').private:
        raise ValueError('publication must remain private')
    api.create_commit(deployment, repo_type='dataset', operations=[CommitOperationAdd(
        path_in_repo='outputs/utilization/comparison-profile.json', path_or_fileobj=str(path))],
        commit_message='Bounded utilization scalar evidence only')
    return receipt
