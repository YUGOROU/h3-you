"""Offline saved float32 WAV replay; neither playback nor model EOS evidence."""
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
from types import SimpleNamespace
from unittest.mock import patch

from . import relay_native_tts as bridge
from .streaming_pcm import PCMStreamBinding
from .tts_preroll import evaluate_trace

_LIMIT = 8 * 1024 * 1024


def _wav(path):
    # Walk with directory descriptors: no symlinks, including parent components.
    path = Path(os.path.abspath(os.fspath(path)))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, 'rb') as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or not 12 <= info.st_size <= _LIMIT:
                raise ValueError('bounded regular WAV required')
            raw = file.read(_LIMIT)
            if len(raw) != info.st_size:
                raise ValueError('WAV changed during read')
    finally:
        os.close(fd)
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE' or struct.unpack_from('<I', raw, 4)[0] != len(raw)-8:
        raise ValueError('exact RIFF/WAVE length required')
    pos, fmt, data = 12, None, None
    while pos < len(raw):
        if pos + 8 > len(raw):
            raise ValueError('truncated chunk header')
        tag, size = struct.unpack_from('<4sI', raw, pos)
        start = pos + 8
        pos = start + size + (size & 1)
        if pos > len(raw):
            raise ValueError('truncated chunk or padding')
        body = raw[start:start+size]
        if tag == b'fmt ':
            if fmt is not None or len(body) < 16:
                raise ValueError('unique valid fmt required')
            fmt = struct.unpack_from('<HHIIHH', body)
            if len(body) != 16 and (len(body) < 18 or struct.unpack_from('<H', body, 16)[0] != len(body)-18):
                raise ValueError('invalid fmt extension')
        elif tag == b'data':
            if data is not None:
                raise ValueError('unique data required')
            data = body
    if fmt != (3, 1, 48000, 192000, 4, 32) or data is None or len(data) % 4:
        raise ValueError('48k mono IEEE float32 WAV required')
    return raw, data


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 120:
        raise ValueError('finite timing within 120 seconds required')
    return value


def verify_saved_case(row, wav_path, policy):
    """Verify recorded chunks against actual saved bytes, without writing media.

    Row fields: sample_rate, total_samples, duration_seconds, pcm_sha256,
    chunks [{samples, arrival_seconds}], iterator_completion_seconds.
    Completion is recorded iterator completion, not independent source EOS.
    """
    chunks = row['chunks']
    if not isinstance(chunks, (list, tuple)) or not 1 <= len(chunks) <= 1024:
        raise ValueError('one to 1024 chunks required')
    previous, events = 0, []
    for chunk in chunks:
        count = chunk['samples']
        arrival = _seconds(chunk['arrival_seconds'])
        if type(count) is not int or not 0 <= count <= bridge.MAX_NATIVE_CHUNK // 4 or arrival < previous:
            raise ValueError('bounded counts and monotonic arrivals required')
        previous = arrival
        if count:
            events.append(dict(now=arrival, samples=count))
    completion = _seconds(row['iterator_completion_seconds'])
    if completion < previous:
        raise ValueError('completion precedes arrival')
    events.append(dict(now=completion, sealed=True))
    trace = evaluate_trace(dict(sample_rate=row['sample_rate'], source_samples=row['total_samples'],
                                duration_seconds=row['duration_seconds'], events=events), policy)
    if trace['status'] != 'ready':
        raise ValueError('trace rejected reserve policy')
    raw, data = _wav(wav_path)
    digest = hashlib.sha256(data).hexdigest()
    if digest != row['pcm_sha256'] or len(data)//4 != row['total_samples']:
        raise ValueError('saved PCM hash/count mismatch')
    clock = [0.0]

    class Transport:
        def stream(self, reply, epoch, deadline):
            offset = 0
            for chunk in chunks:
                clock[0] = chunk['arrival_seconds']
                end = offset + chunk['samples']*4
                yield data[offset:end]
                offset = end
            clock[0] = completion

        def close(self):
            pass

    published = []

    def publish(stream):
        published.append(dict(time=clock[0], samples=stream.snapshot()['received_samples']))

    binding = PCMStreamBinding('saved-replay', 'saved-float32-wav', 1, digest)
    with patch.object(bridge, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
        receipt = bridge.produce(Transport(), 'Offline saved PCM verification.', 1, binding,
                                 completion+5, publish, lambda: True, preroll=policy)
    try:
        expected_prefix = 0
        for chunk in chunks:
            if chunk['arrival_seconds'] > trace['start_monotonic']:
                break
            remaining = chunk['samples']
            while remaining:
                step = min(32000, remaining)
                expected_prefix += step
                remaining -= step
                if expected_prefix >= policy.reserve_samples:
                    break
            if expected_prefix >= policy.reserve_samples:
                break
        if (len(published) != 1 or published[0] != dict(time=trace['start_monotonic'], samples=expected_prefix)
                or receipt.native_sha256 != digest or receipt.native_samples != row['total_samples']
                or receipt.converted_samples != row['total_samples']):
            raise ValueError('bridge replay does not match trace/native PCM')
        return dict(first_published=published[0], trace=trace,
                    receipt={key: (asdict(value) if key == 'preroll' else value)
                             for key, value in vars(receipt).items() if key != 'stream'},
                    source_pcm_sha256=digest, wav_sha256=hashlib.sha256(raw).hexdigest(),
                    virtual_replay=True, actual_playback=False, actual_model_run=False,
                    model_eos_verified=False)
    finally:
        receipt.stream.cancel()
