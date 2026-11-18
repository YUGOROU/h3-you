"""CPU/file-only diagnostics. No playback, network, or device adapters."""
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping
from fractions import Fraction

MAX_EVENTS = 144
# Every stage the relay assigns before raising a rejection (live_relay_server.py).
REJECT_STAGES = ('path', 'headers-duplicate', 'headers-framing', 'length', 'route-auth')


def timing_evidence(timing):
    """Strict allowlist; monotonic values belong to this receiver clock only."""
    if not isinstance(timing, Mapping):
        return {}
    clean = {'remote_verified': False, 'pipe_submission_only': True}
    for key in ('first_video_bytes_write_monotonic', 'first_audio_bytes_write_monotonic',
                'first_av_submission_monotonic', 'ack_count'):
        value = timing.get(key)
        if value is not None and type(value) in (int, float) and math.isfinite(value) and value >= 0:
            clean[key] = value
    values = timing.get('accepted_monotonic', ())
    clean['accepted_monotonic'] = [v for v in values[:MAX_EVENTS]
        if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    fields = ('sequence', 'frame_start', 'frame_stop', 'sample_start', 'sample_stop',
              'padding_samples', 'accept_started_monotonic', 'accepted_monotonic',
              'decode_seconds', 'pipe_seconds', 'total_seconds', 'media_seconds')
    clean['events'] = [{k: event[k] for k in fields if k in event and
        type(event[k]) in (int, float) and math.isfinite(event[k]) and event[k] >= 0}
        for event in timing.get('events', ())[:MAX_EVENTS] if isinstance(event, Mapping)]
    events = clean['events']
    clean['delivery_intervals_seconds'] = [b['accepted_monotonic'] - a['accepted_monotonic']
        for a, b in zip(events, events[1:])
        if 'accepted_monotonic' in a and 'accepted_monotonic' in b]
    clean['delivery_excess_over_previous_media_seconds'] = [
        b['accepted_monotonic'] - a['accepted_monotonic'] - a['media_seconds']
        for a, b in zip(events, events[1:])
        if 'accepted_monotonic' in a and 'accepted_monotonic' in b and 'media_seconds' in a]
    return clean


def failure_evidence(failure):
    if not isinstance(failure, Mapping):
        return {}
    # Exact enum membership, not character filtering of arbitrary secret strings.
    result = {}
    if failure.get('phase') in ('claim', 'claim-schema', 'download', 'ack', 'release', 'request', 'transfer',
        'response-headers', 'boot-identity', 'cache-policy', 'session-binding',
        'content-encoding', 'content-length', 'response-body'):
        result['phase'] = failure['phase']
    if type(failure.get('http_status')) is int and 100 <= failure['http_status'] <= 599:
        result['http_status'] = failure['http_status']
    if failure.get('error_type') in ('HTTPError', 'URLError', 'TimeoutError', 'ConnectionError',
                                     'OSError', 'RelayError', 'InvalidClaim', 'RemoteDisconnected',
                                     'IncompleteRead', 'ConnectionResetError'):
        result['error_type'] = failure['error_type']
    for key, allowed in (
        ('lane', ('claim', 'write', 'read', 'input', 'abort')),
        ('method', ('GET', 'POST')),
        ('framing', ('unknown', 'content-length', 'chunked', 'eof'))):
        if type(failure.get(key)) is str and failure[key] in allowed:
            result[key] = failure[key]
    for key in ('content_length', 'bytes_read'):
        value = failure.get(key)
        if type(value) is int and 0 <= value <= 524289:
            result[key] = value
    # Relay rejection explanation. Exact enum membership only: the server emits a
    # fixed stage name and a small integer bitmask, so no path/header/body text
    # can reach the evidence file through these fields.
    if failure.get('reject_stage') in REJECT_STAGES:
        result['reject_stage'] = failure['reject_stage']
    flags = failure.get('reject_flags')
    if type(flags) is str and flags.isascii() and flags.isdecimal() and len(flags) <= 4:
        result['reject_flags'] = int(flags)
    value=failure.get('stream_bytes')
    if type(value) is int and 0 <= value <= 2**32:
        result['stream_bytes']=value
    for key in ('elapsed_seconds', 'request_timeout_seconds', 'headers_elapsed_seconds',
                'last_progress_age_seconds'):
        value = failure.get(key)
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e9:
            result[key] = value
    return result


def persist(root, name, evidence):
    fd = os.open(os.path.join(root, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(evidence, out, allow_nan=False, sort_keys=True)
        out.write('\n')


def analyze_local_nut(path, original_pcm, *, sample_rate=48000, channels=1, timeout=30,
                      timing=None):
    """Decode a local NUT and compare exact S16LE PCM; never play it.

    Input is trusted local test media. Probe output is disk-backed with a 16MiB
    bound; execution is finite. Delivery gaps are not acoustic/remote quality.
    """
    if type(original_pcm) is not bytes or not 0 < timeout <= 300:
        raise ValueError('bytes and finite timeout required')
    if type(sample_rate) is not int or sample_rate <= 0 or type(channels) is not int or channels <= 0:
        raise ValueError('positive sample geometry required')
    started = time.monotonic()
    with tempfile.TemporaryFile() as probe, tempfile.TemporaryFile() as decoded:
        subprocess.run(['ffprobe', '-v', 'error', '-show_packets', '-show_streams',
            '-of', 'json', str(path)], stdout=probe, stderr=subprocess.DEVNULL,
            check=True, timeout=timeout)
        if probe.tell() > 16 * 1024 * 1024:
            raise ValueError('probe metadata bound exceeded')
        probe.seek(0)
        metadata = json.load(probe)
        probe_seconds = time.monotonic() - started
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
            '-map', '0:a:0', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'],
            stdout=decoded, stderr=subprocess.DEVNULL, check=True,
            timeout=max(.001, timeout - probe_seconds))
        size = decoded.tell()
        decoded.seek(0)
        exact = size == len(original_pcm)
        for offset in range(0, size, 65536):
            block = decoded.read(65536)
            exact = exact and block == original_pcm[offset:offset + len(block)]
    gaps = {}
    for stream in metadata['streams']:
        kind = stream['codec_type']
        if kind not in ('video', 'audio'):
            continue
        previous = None
        discontinuities = 0
        maximum = Fraction(0)
        base = Fraction(stream['time_base'])
        for packet in metadata['packets']:
            if packet['stream_index'] != stream['index']:
                continue
            pts = int(packet['pts']) * base
            if previous is not None:
                gap = pts - previous
                if abs(gap) > base:
                    discontinuities += 1
                maximum = max(maximum, abs(gap))
            previous = pts + int(packet.get('duration', 0)) * base
        gaps[kind] = dict(pts_discontinuities=discontinuities, max_abs_gap_seconds=float(maximum))
    audio = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
    return dict(original_pcm_exact=exact and int(audio['sample_rate']) == sample_rate
                and audio['channels'] == channels,
                decoded_pcm_bytes=size, pts=gaps, probe_seconds=probe_seconds,
                decode_seconds=time.monotonic() - started - probe_seconds,
                delivery=timing_evidence(timing), device_playback=False, remote_verified=False)
