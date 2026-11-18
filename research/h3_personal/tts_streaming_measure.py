"""Single-utterance native PCM measurement (not actual playback).

``candidate.stream(text)`` yields 1-D floating numpy arrays for ``kind='float'``
(Vox), or PCM16 little-endian mono bytes for ``kind='pcm16'`` (Qwen).
``candidate.sample_rate`` is an integer. Bounds are checked at iterator boundaries;
a blocking native call cannot be interrupted by this synchronous collector.
CUDA memory accounting belongs to the caller. Output directories must be private
per measurement; fixed filenames are intentionally not derived from input text.
"""

import hashlib
import json
from pathlib import Path
import time
import wave

MAX_WALL_SECONDS = 120
MAX_AUDIO_SECONDS = 30
MAX_CHUNKS = 1024


def measure(candidate, text, output_dir, kind, *, timing_attribution=False):
    """Return bounded timing metadata and artifact names, never input text.

    TTFA is the first nonempty playable chunk's arrival, including stream setup.
    ``max_late_supply_seconds`` models an uninterrupted sink starting at TTFA:
    max(arrival - TTFA - duration supplied before that chunk, 0). It is not a
    playback measurement or evidence that generation is realtime streaming.
    Optional attribution uses an independent perf_counter_ns start: next-call
    durations, consumer intervals from return until the next call (including
    validation/writing), and validated-ready timestamps relative to that start.
    These are host timings, not synchronized GPU execution measurements.
    """
    if kind not in ('float', 'pcm16'):
        raise ValueError('kind must be float or pcm16')
    rate = candidate.sample_rate
    if type(rate) is not int or not 1 <= rate <= 48000:
        raise ValueError('sample_rate must be an integer in 1..48000')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    wav_path = directory / 'native.wav'
    timing_path = directory / 'timing.json'
    # Never overwrite another measurement or delete its artifacts on failure.
    if wav_path.exists() or timing_path.exists():
        raise FileExistsError('measurement artifacts already exist')
    iterator = writer = None
    rows = []
    total = 0
    first = None
    peak = None
    late = 0.0
    digest = hashlib.sha256()
    start = time.monotonic()
    perf_start = time.perf_counter_ns() if timing_attribution else None

    def elapsed():
        value = time.monotonic() - start
        if value > MAX_WALL_SECONDS:
            raise ValueError('wall time limit exceeded')
        return value

    try:
        iterator = iter(candidate.stream(text))
        elapsed()
        while True:
            if timing_attribution:
                next_start = time.perf_counter_ns()
                if rows:
                    rows[-1]['consumer_seconds'] = (next_start - next_end) / 1e9
            try:
                chunk = next(iterator)
            except StopIteration as stop:
                if timing_attribution:
                    final_next_seconds = (time.perf_counter_ns() - next_start) / 1e9
                iterator_return = stop.value
                completion = elapsed()
                break
            if timing_attribution:
                next_end = time.perf_counter_ns()
            arrival = elapsed()
            if len(rows) >= MAX_CHUNKS:
                raise ValueError('chunk limit exceeded')
            if kind == 'pcm16':
                if not isinstance(chunk, bytes) or len(chunk) % 2:
                    raise ValueError('expected aligned PCM16 bytes')
                count = len(chunk) // 2
                payload = chunk
            else:
                import numpy as np
                if not isinstance(chunk, np.ndarray) or chunk.ndim != 1 or chunk.dtype.kind != 'f':
                    raise ValueError('expected 1-D floating numpy array')
                if not np.isfinite(chunk).all():
                    raise ValueError('non-finite PCM')
                count = chunk.size
                payload = chunk.astype('<f4', copy=False).tobytes()
                # Conversion itself may overflow a finite float64 input.
                if not np.isfinite(np.frombuffer(payload, dtype='<f4')).all():
                    raise ValueError('PCM exceeds float32 range')
                if count:
                    value = float(np.max(np.abs(np.frombuffer(payload, dtype='<f4'))))
                    peak = value if peak is None else max(peak, value)
            if total + count > rate * MAX_AUDIO_SECONDS:
                raise ValueError('audio duration limit exceeded')
            row = {'arrival_seconds': arrival, 'samples': count}
            if timing_attribution:
                row.update(
                    next_call_seconds=(next_end - next_start) / 1e9,
                    validated_ready_seconds=(time.perf_counter_ns() - perf_start) / 1e9,
                )
            rows.append(row)
            if not count:
                continue
            if first is None:
                first = arrival
            late = max(late, arrival - first - total / rate)
            if writer is None:
                if kind == 'pcm16':
                    writer = wave.open(str(wav_path), 'wb')
                    writer.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                else:
                    import soundfile as sf
                    writer = sf.SoundFile(str(wav_path), mode='w', samplerate=rate,
                                          channels=1, format='WAV', subtype='FLOAT')
            if kind == 'pcm16':
                writer.writeframesraw(payload)
            else:
                writer.write(np.frombuffer(payload, dtype='<f4'))
            digest.update(payload)
            total += count
            elapsed()
        if not total:
            raise ValueError('stream produced no playable PCM')
        # Snapshot native generation evidence only after exhaustion. Iterator end,
        # PCM arrival and native model EOS are deliberately independent evidence.
        native_completion = json.loads(json.dumps(getattr(candidate, 'completion', None), allow_nan=False))
        natural_eos = (isinstance(native_completion, dict)
                       and native_completion.get('natural_stop') is True
                       and native_completion.get('reason') == 'natural_stop'
                       and native_completion.get('stream_exhausted') is True
                       and type(native_completion.get('decision_monotonic_ns')) is int)
        writer.close()
        writer = None
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
        iterator = None
        elapsed()
        result = {
            'schema_version': 1, 'kind': kind, 'sample_rate': rate,
            'wav_file': wav_path.name, 'timing_file': timing_path.name,
            'ttfa_seconds': first, 'iterator_completion_seconds': completion,
            'iterator_return': iterator_return,
            'last_pcm_arrival_seconds': next(row['arrival_seconds'] for row in reversed(rows) if row['samples']),
            'native_completion': native_completion,
            'formal_completion_accepted': natural_eos,
            'formal_completion_reason': 'native-natural-eos' if natural_eos else 'unknown-or-non-natural-completion',
            'duration_seconds': total / rate, 'total_samples': total,
            'chunk_count': len(rows), 'pcm_sha256': digest.hexdigest(),
            'peak_abs_float': peak, 'max_late_supply_seconds': max(0.0, late),
            'supply_model': 'continuous-from-first-chunk-not-actual-playout',
            'chunks': rows,
        }
        if timing_attribution:
            result['final_next_call_seconds'] = final_next_seconds
            result['attribution_clock'] = 'perf_counter_ns-relative-to-measure-start'
        timing_path.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n')
        return result
    except BaseException:
        # Cleanup must never replace the primary stream/validation/write error.
        for resource in (writer, iterator):
            try:
                close = getattr(resource, 'close', None)
                if close is not None:
                    close()
            except BaseException:
                pass
        for path in (wav_path, timing_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
