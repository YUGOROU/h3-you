"""Opt-in parent renderer for an already-warmed latent-tail/SMD4 runtime.

No model lifecycle, publication, playback or remote acceptance. The caller owns
serialization and marks the owner degraded on failure; never reload it here.
Import and plan() are stdlib-only. render() requires the worker overlays already
installed and the independent manifest created BEFORE worker startup.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import uuid
import time
import wave

from .continuation_plan import audio_boundary, plan as continuation_plan, video_t
from .live_voice_runtime import (POLICY, RenderResult, _command, _json, _metadata,
                                 _pcm, _require)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plan(wav, digest):
    """Admit the smallest full 124/22 grid covering BOTH video and audio clocks.

    Original integer PCM is never resampled/shortened here. Workers encode the
    whole source once per rank, padding only trailing zeros at 32 kHz.
    """
    wav = Path(wav).resolve()
    _require(_sha(wav) == digest, 'supplied WAV SHA differs')
    with wave.open(str(wav), 'rb') as handle:
        channels, width, rate, samples, compression, _ = handle.getparams()
        _require(compression == 'NONE' and width in (1, 2, 3, 4) and
                 channels in (1, 2) and rate > 0 and samples > 0,
                 'nonempty mono/stereo integer PCM WAV required')
        remaining = samples
        while remaining:
            count = min(remaining, 65536)
            _require(len(handle.readframes(count)) == count * channels * width,
                     'truncated source PCM')
            remaining -= count
    _require(_sha(wav) == digest, 'source changed during planning')
    for n in range(1, 5):
        total = 124 + (n - 1) * 102
        if (Fraction(total, 24) >= Fraction(samples, rate) and
                samples * 32000 <= audio_boundary(total) * 800 * rate):
            windows = continuation_plan(total, 124, 22)
            return dict(schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
                        wav=str(wav), sha256=digest, source_samples=samples,
                        source_sample_rate=rate, source_channels=channels,
                        source_sample_width=width, speech_seconds=str(Fraction(samples, rate)),
                        total_frames=total, window_frames=124, overlap_frames=22,
                        expected_windows=n, seed=314159, sampler='SMD4', fps=24,
                        global_audio_t=audio_boundary(total),
                        audio_capacity_samples_32000=audio_boundary(total)*800,
                        source_padding='trailing zeros only; full source encoded once per rank',
                        windows=[w.record() for w in windows])
    raise ValueError('whole reply exceeds four-window video/audio capacity; never shorten or stretch')


def _equal(receipt, expected, message):
    # bool/int equality must not let malformed scalar evidence through.
    _require(all(type(receipt.get(k)) is type(v) and receipt[k] == v
                 for k, v in expected.items()), message)


def _receipts(runtime, manifest, window, offset, count, per_frame):
    """Verify scalar commit + decoder evidence before exporting any pixels."""
    stages, decodes = [], []
    index = manifest['index']
    from .continuation_candidate import normalize
    options = normalize(manifest.get('candidate'))
    from .continuation_efficiency import normalize as efficiency_policy
    efficiency = efficiency_policy(manifest.get('efficiency'))
    waves = efficiency is not None and efficiency['decode_schedule'] == 'two-clip-waves-v1'
    window_context = options is not None and options['audio_context'] == 'window-context-canonical-v1'
    for rank in (0, 1):
        stage = json.loads((runtime.coverage /
            f'h3-continuation-rank-{rank}-window-{index}.json').read_text())
        _equal(stage, dict(schema='h3-smd-continuation-stage-v1',
            session_id=manifest['session_id'], window_index=index, worker_rank=rank,
            mode='latent-tail-v1', wav_sha256=manifest['sha256'],
            total_frames=manifest['total_frames'], window_frames=124, overlap_frames=22,
            seed=314159, expected_windows=manifest['expected_windows'], final=manifest['final'],
            source_samples=manifest['source_samples'], source_sample_rate=manifest['source_sample_rate'],
            full_source_encodes=0 if window_context else 1,
            global_audio_t=audio_boundary(manifest['total_frames']),
            audio_start=window.audio_start, audio_end=window.audio_end,
            guide_video_t=window.guide_video_t, guide_audio_t=window.guide_audio_t,
            target_audio_rows=2*(window.audio_end-window.audio_start),
            guide_audio_rows=2*window.guide_audio_t, forwards=4, steps_held=4,
            sampler='SMD4', audio_timestep_policy='fixed_clean_1.0',
            audio_velocity_discarded=True, audio_scheduler_step_skipped=True,
            rows_unchanged=True, committed=True, truncation=False,
            suffix_audio_t=window.audio_end-audio_boundary(window.start+window.overlap),
            audio_time_offset=audio_boundary(window.start+window.overlap),
            cumulative_audio_t=window.audio_end), 'continuation commit/source/clock differs')
        if options is not None:
            _equal(stage, dict(candidate_audio_context=options['audio_context'],
                candidate_progress_receipts=options['progress_receipts'],
                source_window_encodes=manifest['expected_windows'] if window_context else 0),
                'candidate audio context/evidence differs')
            if window_context:
                _equal(stage, dict(audio_overlap_policy='prior-canonical-wins'),
                       'candidate canonical overlap differs')
        else:
            _require(not any(k in stage for k in ('candidate_audio_context',
                'candidate_progress_receipts', 'source_window_encodes', 'audio_overlap_policy')),
                'unexpected candidate evidence in legacy path')
        if efficiency is not None:
            _equal(stage, dict(efficiency_audio_validation=efficiency['audio_validation'],
                              efficiency_decode_schedule=efficiency['decode_schedule']),
                   'efficiency validation/scheduling evidence differs')
        else:
            _require(not any(k in stage for k in ('efficiency_audio_validation',
                                                 'efficiency_decode_schedule')),
                     'unexpected efficiency evidence in default path')
        cumulative = stage.get('cumulative_video_rows')
        _require(type(cumulative) is int and cumulative > 0 and
                 cumulative % video_t(window.end) == 0, 'invalid cumulative video rows')
        actual_per_frame = cumulative // video_t(window.end)
        if per_frame is None:
            per_frame = actual_per_frame
        _require(actual_per_frame == per_frame, 'latent canvas changed')
        previous = 0 if index == 0 else video_t(window.start+window.overlap)*per_frame
        _equal(stage, dict(video_row_offset=previous,
                          suffix_video_rows=cumulative-previous), 'suffix video gap/overlap')
        # Resampling can round the original length either way by one sample;
        # require precisely the remaining trailing padding, not truncated speech.
        capacity = audio_boundary(manifest['total_frames'])*800
        resampled = Fraction(manifest['source_samples']*32000, manifest['source_sample_rate'])
        floor = resampled.numerator // resampled.denominator
        ceil = -(-resampled.numerator // resampled.denominator)
        _require(type(stage.get('padded_samples')) is int and
                 stage['padded_samples'] >= 0 and
                 capacity-stage['padded_samples'] in (floor, ceil), 'source padding differs')
        decode = json.loads((runtime.coverage /
            f'h3-continuation-decode-rank-{rank}-window-{index}.json').read_text())
        _equal(decode, dict(session_id=manifest['session_id'], window_index=index,
            worker_rank=rank, total_frames=manifest['total_frames'],
            frame_offset=offset, frame_count=count, latent_received=video_t(window.end),
            retained_latents=0 if manifest['final'] else 2,
            retained_overlap_frames=0 if manifest['final'] else 5, max_clip_latents=7,
            two_latent_lookahead=2, bounded_buffers=True, final=manifest['final'],
            causal=False, execution_policy=('two-rank-two-clip-waves' if waves else
                                           'two-rank-alternating-native-clips'),
            parallel_preserved=True, participation=True, output_worker=rank == 0),
            'two-rank suffix decode/PTS differs')
        if waves:
            _equal(decode, dict(decode_schedule='two-clip-waves-v1',
                               max_buffer_latents=12, max_queued_clips=1),
                   'bounded wave decode evidence differs')
        else:
            _require(not any(k in decode for k in ('decode_schedule', 'max_buffer_latents',
                                                 'max_queued_clips')),
                     'unexpected wave decode in default path')
        stages.append(stage)
        decodes.append(decode)
    _require({k: v for k, v in stages[0].items() if k != 'worker_rank'} ==
             {k: v for k, v in stages[1].items() if k != 'worker_rank'},
             'rank commit evidence differs')
    return stages, decodes, per_frame


def profile_evidence(snapshots, trace, denoise, total_frames):
    """Aggregate only validated per-window receipts; never infer completion from policy."""
    import math
    rows = []
    session = snapshots[0]['session_id'] if snapshots else None
    for index, snapshot in enumerate(snapshots):
        pair = snapshot.get('utilization_receipts', [])
        _require(len(pair) == 2, 'both rank window utilization required')
        for rank, row in enumerate(pair):
            _equal(row, dict(schema='h3-window-utilization-v1', session_id=session,
                window_index=index, rank=rank, trace_policy=trace, denoise_policy=denoise,
                faulted=False), 'utilization identity/policy/fault differs')
            start, end = row.get('worker_monotonic_start'), row.get('worker_monotonic_end')
            _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                         for v in (start, end)) and end >= start, 'invalid worker clock')
        rows.append(pair)
    final = snapshots[-1] if snapshots else {}
    complete = bool(rows) and total_frames == 328 and len(rows) == 3
    complete = complete and all(
        s.get('committed') is True and s.get('final') is True and
        s.get('total_frames') == total_frames
        for s in final.get('continuation_stage_receipts', []))
    complete = complete and len(final.get('continuation_stage_receipts', [])) == 2
    complete = complete and len(final.get('continuation_decode_receipts', [])) == 2 and all(
        d.get('final') is True and d.get('frame_offset', -1) + d.get('frame_count', -1) == total_frames
        for d in final.get('continuation_decode_receipts', []))
    ranks = []
    for rank in (0, 1):
        row = dict(rows[-1][rank]) if rows else dict(rank=rank)
        if trace is not None:
            selected = trace['selected_window']
            _require(selected < len(rows), 'selected trace window missing')
            for key in ('trace_summary', 'trace_available', 'coverage_proven', 'truncated',
                        'event_count', 'trace_bytes'):
                row[key] = rows[selected][rank][key]
        ranks.append(row)
    return dict(owner_complete=complete, trace_policy=trace, denoise_policy=denoise,
                ranks=ranks, windows=rows)


def admission_proof(row):
    """Validate bounded producer metadata; never infer admission from exception text."""
    import re
    _require(row.get('prepare_error_code') in (
        'ffn-configuration-unsupported', 'native-provenance-unsupported',
        'parameter-dtype-unsupported'), 'unknown admission code')
    for key in ('source_pinned_verified', 'main50_verified', 'parameter_identity_preserved',
                'candidate_rejected', 'faulted'):
        _require(row.get(key) is True, 'missing admission proof')
    _require(row.get('state') == row.get('owner_terminal') == 'faulted' and
             row.get('denoise_installed') is False and
             row.get('representative_preflight_complete') is False and
             type(row.get('regions')) is int and row['regions'] == 50,
             'invalid admission terminal state')
    names = {'torch.' + name for name in (
        'bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32',
        'int64', 'uint8', 'bool', 'complex64', 'complex128', 'float8_e4m3fn',
        'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz')}
    histogram = row.get('parameter_dtype_histogram')
    _require(type(histogram) is dict and bool(histogram) and
             all(k in names and type(v) is int and 0 < v < 2147483648
                 for k, v in histogram.items()), 'invalid parameter dtype histogram')
    hashes = row.get('native_type_hashes')
    _require(type(hashes) is list and 1 <= len(hashes) <= 51 and
             all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in hashes),
             'invalid native type hashes')
    return {key: row[key] for key in ('prepare_error_code', 'source_pinned_verified',
        'main50_verified', 'parameter_dtype_histogram', 'native_type_hashes')}


def rejection_evidence(runtime, manifest, denoise):
    """IPC exception text/types are not evidence: require two terminal receipts."""
    import math
    from .fast_ref2va import utilization_name
    from .utilization_policy import normalize
    _require(denoise is not None, 'rejection requires denoise policy')
    normalize(denoise, 'denoise')
    rows = []
    for rank in (0, 1):
        path = runtime.coverage / utilization_name(manifest['session_id'], manifest['index'], rank)
        _require(not path.is_symlink(), 'rejection receipt symlink')
        _require(path.stat().st_size <= 1048576, 'oversized rejection receipt')
        row = json.loads(path.read_text())
        _equal(row, dict(schema='h3-window-utilization-v1',
            session_id=manifest['session_id'], window_index=manifest['index'], rank=rank,
            denoise_policy=denoise, faulted=True, candidate_rejected=True,
            state='faulted', owner_terminal='faulted', denoise_installed=False,
            parameter_identity_preserved=True, representative_preflight_complete=False),
            'invalid terminal candidate rejection')
        admission = row.get('rejection_kind') == 'controlled-admission-rejection'
        if admission:
            admission_proof(row)
        controlled = admission or row.get('rejection_kind') == 'controlled-local-rejection'
        _require('rejection_kind' not in row or controlled, 'unknown rejection kind')
        if controlled:
            _require(type(row.get('regions')) is int and row['regions'] == 50,
                     'controlled rejection requires main50')
        for key, maximum in (('backend_compiles', denoise['max_backend_compiles']),
                             ('regions', denoise['max_regions'])):
            minimum = 0 if controlled and key == 'backend_compiles' else 1
            _require(type(row.get(key)) is int and minimum <= row[key] <= maximum,
                     'invalid rejection backend/region count')
        for key in ('compile_seconds', 'extra_status_seconds'):
            value = row.get(key)
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     'invalid rejection duration')
        _require(controlled or row['compile_seconds'] <= denoise['max_compile_seconds'],
                 'compile budget exceeded')
        _require(type(row.get('extra_status_collectives')) is int and
                 row['extra_status_collectives'] > 0, 'missing rejection agreement')
        _require('representative_metrics' in row and
                 ((controlled and row['representative_metrics'] is None) or
                  (type(row['representative_metrics']) is dict and
                   bool(row['representative_metrics']))), 'missing rejection metrics')
        allowed = ('schema', 'session_id', 'window_index', 'rank', 'trace_policy',
            'denoise_policy', 'faulted', 'candidate_rejected', 'state', 'owner_terminal',
            'denoise_installed', 'parameter_identity_preserved',
            'representative_preflight_complete', 'rejection_kind', 'backend_compiles',
            'regions', 'compile_seconds', 'extra_status_seconds', 'extra_status_collectives',
            'representative_metrics', 'worker_monotonic_start', 'worker_monotonic_end',
            'prepare_error_code', 'source_pinned_verified', 'main50_verified',
            'parameter_dtype_histogram', 'native_type_hashes')
        rows.append({key: row[key] for key in allowed if key in row})
    _require(rows[0].get('rejection_kind') == rows[1].get('rejection_kind'),
             'rejection kinds differ')
    return dict(owner_complete=False, owner_terminal='faulted', candidate_rejected=True,
                trace_policy=None, denoise_policy=denoise, ranks=rows, windows=[rows])


def render(runtime, wav, digest, turnDir, *, on_segment=None):
    """Render on a serialized, ready owner. Callback receives scalar/path dict.

    Each callback follows receipt validation, encoding and full segment metadata
    verification. Failure propagates, restoring independent mode even on an
    encoder/callback failure. This function never launches/closes/reloads models.
    """
    render_started = time.monotonic()
    _require(on_segment is None or callable(on_segment), 'on_segment must be callable')
    mode_file = Path(runtime.continuation_file)
    independent = json.loads(mode_file.read_text())
    _require(independent.get('mode') == 'independent', 'preexisting independent manifest required')
    try:
        return _render(runtime, Path(wav).resolve(), digest, Path(turnDir).resolve(), on_segment,
                       render_started)
    finally:
        _json(mode_file, independent)


def _render(runtime, wav, digest, turn, on_segment, render_started):
    admitted = plan(wav, digest)
    root = Path(__file__).resolve().parents[2]
    _require(not turn.is_relative_to(root), 'private turn output must be outside repository')
    turn.mkdir(mode=0o700, parents=True, exist_ok=False)
    speech = turn / 'speech.wav'
    with speech.open('xb') as target, wav.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            target.write(chunk)
    speech.chmod(0o400)
    planned = plan(speech, digest)
    _require(all(planned[k] == v for k, v in admitted.items() if k != 'wav'),
             'source changed while staging')
    session_id = uuid.uuid4().hex  # fresh even for identical speech on a later turn
    from .continuation_candidate import normalize
    options = normalize(getattr(runtime, '_candidate', None))
    from .continuation_efficiency import normalize as efficiency_policy
    efficiency = efficiency_policy(getattr(runtime, '_efficiency', None))
    if (getattr(runtime, '_parallel_acceptance_enabled', False) or
            getattr(runtime, '_measure_memory', False)):
        planned['parallel_acceptance'] = bool(getattr(runtime, '_parallel_acceptance_pending', False))
    if getattr(runtime, '_measure_memory', False):
        planned['measure_memory'] = True
    if efficiency is not None:
        _require(options is not None and options['audio_context'] == 'window-context-canonical-v1',
                 'efficiency requires v6 quality baseline')
        planned['efficiency'] = efficiency
    if options is not None:
        planned['candidate'] = options
        if options['audio_context'] == 'window-context-canonical-v1':
            planned['source_padding'] = ('trailing zeros only; per-window waveform encodes; '
                                         'prior canonical overlap retained')
    trace, denoise = None, None
    if (getattr(runtime, '_trace_policy', None) is not None or
            getattr(runtime, '_denoise_policy', None) is not None):
        from .utilization_policy import policies
        trace, denoise = policies(runtime._trace_policy, runtime._denoise_policy)
    collect_utilization = (getattr(runtime, '_measure_memory', False) is True or
                           trace is not None or denoise is not None)
    for key, value in (('trace_policy', trace), ('denoise_policy', denoise)):
        if value is not None:
            planned[key] = value
    planned['session_id'] = session_id
    _json(turn / 'speech-plan.json', planned, immutable=True)
    import numpy as np
    from .realtime_segments import SegmentTiming
    from research.run_h3_realtime_first import encode_rgb_segment
    segments, snapshots, intervals = [], [], []
    segment_ready_elapsed = []
    pipeline_timing = []
    offset, per_frame = 0, None
    windows = continuation_plan(planned['total_frames'], 124, 22)
    for index, window in enumerate(windows):
        final = index == len(windows)-1
        manifest = {k: planned[k] for k in ('wav', 'sha256', 'total_frames', 'window_frames',
                    'overlap_frames', 'expected_windows', 'seed', 'source_samples', 'source_sample_rate')}
        manifest.update(mode='latent-tail-v1', session_id=session_id, index=index, final=final)
        if options is not None:
            manifest['candidate'] = dict(options)
        if efficiency is not None:
            manifest['efficiency'] = dict(efficiency)
        for key in ('trace_policy', 'denoise_policy'):
            if key in planned:
                manifest[key] = dict(planned[key])
        for diagnostic in ('parallel_acceptance', 'measure_memory'):
            if diagnostic in planned:
                manifest[diagnostic] = planned[diagnostic]
        # No stale receipt from a previous session can count as this generation.
        for rank in (0, 1):
            for stem in ('h3-continuation', 'h3-continuation-decode'):
                (runtime.coverage / f'{stem}-rank-{rank}-window-{index}.json').unlink(missing_ok=True)
        _json(runtime.continuation_file, manifest)
        requests = runtime._requests
        request_start = time.monotonic()
        try:
            result, interval = runtime._generate(continuation=True)
        except Exception as original:
            try:
                rejected = rejection_evidence(runtime, manifest, denoise)
            except Exception:
                raise original
            _json(turn / 'profile.json', dict(schema='h3-smd-continuation-render-v1',
                session_id=session_id, candidate_rejected=True, utilization=rejected,
                windows=[], files={}, full_decode_verified=False,
                identity_quality_verified=False, realtime_complete=False,
                remote_acceptance=False), immutable=True)
            from .denoise_execution import CandidateRejected
            raise CandidateRejected('both ranks rejected denoise candidate; owner faulted') from original
        request_end = time.monotonic()
        _require(runtime._requests == requests+1 and runtime._last_seed == 314159,
                 'actual continuation seed/request count differs')
        emitted_end = window.end if final else window.end-5
        count = emitted_end-offset
        stages, decodes, per_frame = _receipts(runtime, manifest, window, offset, count, per_frame)
        loaded = runtime._loaded()  # original64 coverage, static merge, BF16 FA4 on BOTH ranks
        cache = runtime._read('cache')
        _require(len(cache) == 2 and all(r == {'hits': runtime._requests-1} for r in cache),
                 'condition-cache hit differs')
        packed = runtime._read('packed')
        _require(len(packed) == 2 and all(r.get('warm_counter_removed') and
                 r.get('packed_route_executed') for r in packed), 'packed FA4 cold route not proved/removed')
        residency = runtime._read('residency')
        _require(len(residency) == 2 and all(r.get('policy') == POLICY and
                 r.get('audio_decode_skipped') for r in residency), 'resident visual policy differs')
        snapshot = dict(window_index=index, seed=314159, session_id=session_id,
            fixed_reference_sha256=runtime.proof['reference_sha256'], wav_sha256=digest,
            continuation_stage_receipts=stages, continuation_decode_receipts=decodes,
            conditioning_cache_receipts=cache, packed_route_receipts=packed,
            residency_receipts=residency, **loaded)
        if collect_utilization:
            from .fast_ref2va import utilization_name
            utilization = []
            for rank in (0, 1):
                receipt_path = runtime.coverage / utilization_name(session_id, index, rank)
                _require(not receipt_path.is_symlink(), 'utilization symlink rejected')
                row = json.loads(receipt_path.read_text())
                _equal(row, dict(schema='h3-window-utilization-v1', session_id=session_id,
                    window_index=index, rank=rank, trace_policy=trace, denoise_policy=denoise,
                    faulted=False), 'per-window utilization identity/policy/fault differs')
                utilization.append(row)
            snapshot['utilization_receipts'] = utilization
        _json(turn / f'window-{index:03d}-receipts.json', snapshot, immutable=True)
        frames = np.stack([np.asarray(f).copy() for f in result.frames])
        _require(frames.dtype == np.uint8 and frames.shape == (count, 448, 256, 3),
                 'native suffix portrait shape/count differs')
        frames = frames[:, :, 2:-2, :].copy()
        _require(frames.shape == (count, 448, 252, 3), 'export shape differs')
        timing = SegmentTiming(index, 0, count, offset, offset+count)
        segment = turn / f'segment-{index:03d}.mp4'
        encoding_start = time.monotonic()
        encode_rgb_segment(frames.tobytes(), segment, timing, width=252, height=448)
        encoding_end = time.monotonic()
        validation_start = time.monotonic()
        metadata = _metadata(segment, count)
        validation_end = time.monotonic()
        pipeline_timing.append(dict(window_index=index, clock='parent_monotonic',
            request_start=request_start, request_end=request_end,
            encoding_start=encoding_start, encoding_end=encoding_end,
            validation_start=validation_start, validation_end=validation_end))
        segment.chmod(0o400)
        segment_ready_elapsed.append(time.monotonic() - render_started)
        segments.append(segment)
        snapshots.append(snapshot)
        intervals.append(interval)
        if on_segment is not None:
            on_segment(dict(path=str(segment), session_id=session_id, index=index,
                frame_count=count, frame_offset=offset, pts_start=offset, pts_end=offset+count,
                time_base_numerator=1, time_base_denominator=24, sha256=_sha(segment),
                video_only=True, remote_acceptance=False))
        snapshot['segment_metadata'] = metadata
        offset += count
    _require(offset == planned['total_frames'], 'final frame grid differs')
    mux_export_started = time.monotonic()
    listing = turn / 'concat.txt'
    listing.write_text(''.join(f"file '{p.name}'\nduration {s['continuation_decode_receipts'][0]['frame_count']/24:.12f}\n"
                               for p, s in zip(segments, snapshots)))
    joined = turn / 'joined.mp4'
    _command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n', '-f', 'concat',
              '-safe', '1', '-i', str(listing), '-map', '0:v:0', '-c:v', 'copy', '-an', str(joined)])
    joined_metadata = _metadata(joined, offset)
    review = turn / 'review-with-speech.mp4'
    _command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n', '-i', str(joined),
              '-i', str(speech), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
              '-c:a', 'alac', str(review)])
    review_metadata = _metadata(review, offset, audio=True)
    _require(_sha(wav) == digest and _sha(speech) == digest and _pcm(speech) == _pcm(review),
             'original speech SHA or lossless mux PCM differs')
    files = {p.name: _sha(p) for p in [speech, *segments, joined, review]}
    artifacts_verified = time.monotonic()
    total_seconds = artifacts_verified - render_started
    generation_seconds = [b-a for a, b in intervals]
    generation_total = sum(generation_seconds)
    unique_video_seconds = offset / planned['fps']
    timing_metrics = dict(
        clock='time.monotonic',
        total_wall_seconds=total_seconds,
        total_wall_scope='render entry through final metadata, PCM/SHA checks and artifact hashes; '
                         'includes callbacks; excludes profile write and manifest restoration',
        first_segment_ready_seconds=segment_ready_elapsed[0],
        segment_ready_elapsed_seconds=segment_ready_elapsed,
        segment_ready_scope='elapsed from render entry after receipt validation, segment encoding, '
                            'metadata verification and chmod, before callback/hash for delivery',
        mux_export_seconds=artifacts_verified-mux_export_started,
        mux_export_scope='post-segment concat list, video concat, speech mux, metadata/PCM/SHA '
                         'verification and artifact hashes; excludes per-segment encoding',
        generation_total_seconds=generation_total,
        generation_scope='sum of runtime-returned generation intervals; not isolated denoise/decode',
        unique_video_seconds=unique_video_seconds,
        unique_throughput_total=unique_video_seconds/total_seconds if total_seconds > 0 else None,
        unique_throughput_generation=(unique_video_seconds/generation_total
                                      if generation_total > 0 else None),
        throughput_units='unique video seconds per wall second; not realtime acceptance')
    profile = dict(schema='h3-smd-continuation-render-v1', mode='latent-tail-v1', sampler='SMD4',
        session_id=session_id, input_proof=runtime.proof, derived_adapter_receipt=runtime.derived,
        continuation_plan=planned, windows=snapshots, generation_intervals=intervals,
        generation_seconds=generation_seconds, unique_frames=offset,
        timing_metrics=timing_metrics, pipeline_timing=pipeline_timing,
        expected_forwards_per_segment=4, total_forwards_per_rank=4*len(windows),
        runtime_policy=POLICY, attention_mode='fa4-packed',
        layout='portrait', video_only=True, generated_audio_discarded=True,
        source_guides=True, two_latent_vae_lookahead=2, zero_lookahead=False, causal=False,
        independent_ref2va_clips=False, shared_head_trim=False, frame_crossfade=False,
        speech_pcm_exact=True, full_decode_verified=True, identity_quality_verified=False,
        lip_sync_verified=False, realtime_complete=False, remote_acceptance=False, uploaded=False,
        latency_scope='resident server render and runtime generation intervals; '
                      'not isolated denoise/decode, playback or remote acceptance',
        joined_metadata=joined_metadata, review_metadata=review_metadata,
        files=files)
    if collect_utilization:
        profile['utilization'] = profile_evidence(snapshots, trace, denoise, offset)
    path = turn / 'profile.json'
    _json(path, profile, immutable=True)
    for p in (joined, review):
        p.chmod(0o400)
    return RenderResult(profile=path, review_with_speech=review)
