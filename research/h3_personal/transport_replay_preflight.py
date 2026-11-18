"""Offline source-integrity preflight, NOT transport/playout acceptance.

No optional media imports, subprocess, network, device access or installation.
Only one saved pair is read at a time. Receiver ACKs below retire a validation
lease after inspection; they never represent a sink or actual playback.
"""
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from .paired_av import Receiver
from .paired_av_wire import parse_envelope

MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_FILES = 144 * 3
MAX_HEADER = 16384


class ReplayPreflightError(ValueError):
    pass


def capabilities():
    # Top-level find_spec only: dotted probes can import parent packages.
    return {name: importlib.util.find_spec(name) is not None
            for name in ('aiortc', 'av', 'gi', 'livekit', 'aiohttp', 'websockets')}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayPreflightError('duplicate JSON field')
        result[key] = value
    return result


def _read(path, limit, expected=None):
    # Reject child symlinks/FIFOs/devices before reading, and recheck the opened
    # descriptor to avoid following a replaced symlink or blocking on a FIFO.
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ReplayPreflightError('regular capture files required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ReplayPreflightError('capture file outside bounds')
        if expected is not None and info.st_size != expected:
            raise ReplayPreflightError('capture body size mismatch')
        body = stream.read(limit + 1)
        if len(body) != info.st_size:
            raise ReplayPreflightError('capture changed during read')
        return body


def inspect_capture(directory, *, timeout_seconds=30):
    """Validate exact source bytes/ancestry, returning only sanitized aggregates.

    Does not decode video, synthesize media, execute a sink, or run WebRTC.
    Source deadlines are not replayed: a fresh finite *validation* lease is used.
    """
    if (type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
        raise ReplayPreflightError('finite validation timeout required')
    end = time.monotonic() + timeout_seconds
    receiver = None
    try:
        root = Path(directory).resolve(strict=True)
        names = set()
        with os.scandir(root) as entries:
            for entry in entries:
                if time.monotonic() >= end:
                    raise ReplayPreflightError('validation deadline expired')
                if len(names) >= MAX_FILES:
                    raise ReplayPreflightError('too many capture files')
                names.add(entry.name)
        count = len(names) // 3
        expected = {f'{index:03d}.{suffix}' for index in range(count)
                    for suffix in ('json', 'mp4', 'pcm')}
        if not 1 <= count <= 144 or names != expected:
            raise ReplayPreflightError('exact contiguous capture triples required')
        total = video_bytes = pcm_bytes = frames = samples = 0
        windows = []
        manifest_hash = hashlib.sha256()
        pcm_hash = hashlib.sha256()
        for index in range(count):
            if time.monotonic() >= end:
                raise ReplayPreflightError('validation deadline expired')
            header = _read(root / f'{index:03d}.json', MAX_HEADER)
            manifest_hash.update(len(header).to_bytes(4, 'big'))
            manifest_hash.update(header)
            value = json.loads(header, object_pairs_hook=_unique)
            envelope = parse_envelope(dict(schema='h3-paired-av-wire-v1', envelope=value))
            # This also prevents a final envelope followed by additional files.
            if envelope.conversation_final != (index == count - 1):
                raise ReplayPreflightError('exact final envelope required')
            total += len(header) + envelope.video_size + envelope.pcm_size
            if total > MAX_CAPTURE_BYTES:
                raise ReplayPreflightError('capture exceeds byte bound')
            video = _read(root / f'{index:03d}.mp4', 16 * 1024 * 1024, envelope.video_size)
            pcm = _read(root / f'{index:03d}.pcm', 124 * 2000 * 2, envelope.pcm_size)
            if receiver is None:
                receiver = Receiver(envelope.binding, deadline=end)
            receiver.offer(envelope, video, pcm)
            pcm_hash.update(pcm)
            # Validation-only consumption; no real sink or transport ACK.
            receiver.ack(receiver.lease())
            frames += envelope.frame_stop - envelope.frame_start
            samples += envelope.sample_stop - envelope.sample_start
            video_bytes += len(video)
            pcm_bytes += len(pcm)
            windows.append(dict(index=index, frame_start=envelope.frame_start,
                                frame_stop=envelope.frame_stop,
                                sample_start=envelope.sample_start,
                                sample_stop=envelope.sample_stop,
                                duration_seconds=(envelope.sample_stop - envelope.sample_start) / 48000))
            del video, pcm
        if not receiver.snapshot()['complete']:
            raise ReplayPreflightError('incomplete validation sequence')
        return dict(schema='h3-transport-replay-preflight-v1', source_validation_complete=True,
                    transport_executed=False, video_decoded=False, playback_verified=False,
                    windows=windows, frame_count=frames, sample_count=samples,
                    media_seconds=samples / 48000, video_bytes=video_bytes,
                    pcm_bytes=pcm_bytes, capture_bytes=total,
                    source_manifest_sha256=manifest_hash.hexdigest(),
                    source_pcm_sha256=pcm_hash.hexdigest())
    except ReplayPreflightError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        # Never leak a source path, source identity, JSON contents or raw error.
        raise ReplayPreflightError('invalid capture; no transport executed') from None
    finally:
        if receiver is not None:
            receiver.disconnect()
