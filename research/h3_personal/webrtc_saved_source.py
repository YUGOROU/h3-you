"""Bounded, same-host source replay; NOT a native producer/sink-credit adapter.

Source windows retire only after both source tracks consumed their original
bytes/frames. This two-window cache is not a network or playback ACK. No source
padding, frame dropping, per-window tracks or media-clock resets are allowed.
"""
import asyncio
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import time

from .paired_av import _bodies
from .paired_av_wire import parse_envelope
from .transport_replay_preflight import _read, _unique, inspect_capture, MAX_HEADER
from .webrtc_synthetic_probe import WebRTCProbeError, run_media_probe


def availability_schedule(path, count, *, envelopes=None):
    value = json.loads(_read(Path(path), 131072), object_pairs_hook=_unique)
    if type(value) is not dict or value.get('type') != 'h3_producer_timing' or value.get('clock') != 'local_monotonic':
        raise WebRTCProbeError('producer-local timing evidence required')
    rows = value.get('windows')
    if type(rows) is not list or len(rows) != count:
        raise WebRTCProbeError('matching producer timing required')
    times = []
    for index, row in enumerate(rows):
        if type(row) is not dict or type(row.get('index')) is not int or row['index'] != index:
            raise WebRTCProbeError('ordered timing index required')
        if envelopes is not None:
            for key, size in (('video_bytes', envelopes[index].video_size), ('pcm_bytes', envelopes[index].pcm_size)):
                if type(row.get(key)) is not int or row[key] != size:
                    raise WebRTCProbeError('timing payload sizes mismatch')
        t = row.get('packet_ready')
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
            raise WebRTCProbeError('finite producer time required')
        if times and t < times[-1]:
            raise WebRTCProbeError('nondecreasing producer time required')
        times.append(t)
    offsets = tuple(t - times[0] for t in times)
    if offsets[-1] > 75:
        raise WebRTCProbeError('recorded schedule outside finite replay scope')
    return offsets


class SavedSource:
    def __init__(self, directory, *, timing_path=None):
        self.root = Path(directory).resolve(strict=True)
        self.preflight = inspect_capture(self.root)
        self.envelopes = []
        digest = hashlib.sha256()
        for index in range(len(self.preflight['windows'])):
            header = _read(self.root / f'{index:03d}.json', MAX_HEADER)
            digest.update(len(header).to_bytes(4, 'big'))
            digest.update(header)
            self.envelopes.append(parse_envelope(dict(schema='h3-paired-av-wire-v1',
                envelope=json.loads(header, object_pairs_hook=_unique))))
        if digest.hexdigest() != self.preflight['source_manifest_sha256']:
            raise WebRTCProbeError('capture manifest changed')
        self.offsets = (availability_schedule(timing_path, len(self.envelopes), envelopes=self.envelopes)
                        if timing_path is not None else (0.0,) * len(self.envelopes))
        self.mode = 'recorded-packet-ready-open-loop' if timing_path is not None else 'burst-source'
        self.run_seconds = max(self.preflight['media_seconds'],
                               self.offsets[-1] + self.preflight['windows'][-1]['duration_seconds'])
        if self.run_seconds > 85:
            raise WebRTCProbeError('finite replay scope exceeded')
        self.cache = {}
        self.next = dict(audio=0, video=0)
        self.changed = asyncio.Event()
        self.high_windows = self.high_bytes = 0
        self.errors = set()
        self.late = {kind: dict(frames=0, max_seconds=0.0) for kind in ('audio', 'video')}

    async def acquire(self, kind, index, get_start):
        if index != self.next[kind]:
            raise WebRTCProbeError('source window reordered')
        await asyncio.sleep(max(0, get_start() + self.offsets[index] - time.monotonic()))
        e = self.envelopes[index]
        while index not in self.cache:
            size = sum(r['e'].video_size + r['e'].pcm_size for r in self.cache.values())
            if len(self.cache) < 2 and size + e.video_size + e.pcm_size <= e.binding.max_inflight_bytes:
                video = _read(self.root / f'{index:03d}.mp4', 16 * 1024 * 1024, e.video_size)
                pcm = _read(self.root / f'{index:03d}.pcm', 496000, e.pcm_size)
                _bodies(e, video, pcm)
                self.cache[index] = dict(e=e, video=video, pcm=pcm, done=set())
                self.high_windows = max(self.high_windows, len(self.cache))
                self.high_bytes = max(self.high_bytes, size + len(video) + len(pcm))
                break
            self.changed.clear()
            await self.changed.wait()
        return self.cache[index]

    def release(self, kind, index):
        row = self.cache.get(index)
        if row is None or kind in row['done']:
            return
        row['done'].add(kind)
        self.next[kind] += 1
        if len(row['done']) == 2:
            del self.cache[index]
            self.changed.set()

    async def pace(self, kind, offset, get_start):
        await asyncio.sleep(max(0, get_start() + offset - time.monotonic()))
        lag = max(0, time.monotonic() - get_start() - offset)
        if lag > (0.02 if kind == 'audio' else 1 / 24):
            self.late[kind]['frames'] += 1
        self.late[kind]['max_seconds'] = max(self.late[kind]['max_seconds'], lag)

    def make_tracks(self, gate, get_start):
        import av
        from aiortc import MediaStreamTrack
        from aiortc.mediastreams import MediaStreamError
        source = self

        class Video(MediaStreamTrack):
            kind = 'video'

            def __init__(self):
                super().__init__()
                self.generated = 0
                self.container = None
                self.active = None
                self.generator_closed = False
                self.frames = self.generate()

            async def generate(self):
                await gate.wait()
                for index, e in enumerate(source.envelopes):
                    row = await source.acquire('video', index, get_start)
                    self.active = index
                    try:
                        self.container = av.open(io.BytesIO(row['video']), mode='r', format='mp4')
                        streams = self.container.streams.video
                        if len(streams) != 1:
                            raise WebRTCProbeError('exact video stream required')
                        stream = streams[0]
                        if (stream.codec_context.width, stream.codec_context.height) != (252, 448):
                            raise WebRTCProbeError('source geometry changed')
                        stream.codec_context.thread_count = 1
                        count = 0
                        for frame in self.container.decode(stream):
                            if (frame.width, frame.height) != (252, 448) or count >= e.frame_stop - e.frame_start:
                                raise WebRTCProbeError('decoded source outside envelope')
                            number = e.frame_start + count
                            await source.pace('video', number / 24, get_start)
                            frame.pts, frame.time_base = number * 3750, Fraction(1, 90000)
                            count += 1
                            self.generated += 1
                            yield frame
                        if count != e.frame_stop - e.frame_start:
                            raise WebRTCProbeError('decoded source frame count mismatch')
                    finally:
                        if self.container is not None:
                            self.container.close()
                            self.container = None
                        source.release('video', index)
                        self.active = None
                        row = None

            async def recv(self):
                try:
                    return await anext(self.frames)
                except StopAsyncIteration:
                    self.stop()
                    raise MediaStreamError
                except Exception:
                    source.errors.add('video-source-failed')
                    self.stop()
                    raise

            def stop(self):
                if self.container is not None:
                    self.container.close()
                    self.container = None
                if self.active is not None:
                    source.release('video', self.active)
                    self.active = None
                super().stop()

            async def aclose(self):
                self.stop()
                await self.frames.aclose()
                self.generator_closed = True

        class Audio(MediaStreamTrack):
            kind = 'audio'

            def __init__(self):
                super().__init__()
                self.generated = self.samples_generated = self.index = self.offset = 0
                self.row = None
                self.digest = hashlib.sha256()

            async def recv(self):
                await gate.wait()
                if self.samples_generated == source.preflight['sample_count']:
                    self.stop()
                    raise MediaStreamError
                try:
                    parts = []
                    needed = min(960, source.preflight['sample_count'] - self.samples_generated) * 2
                    while needed:
                        if self.row is None:
                            self.row = await source.acquire('audio', self.index, get_start)
                            self.offset = 0
                        data = self.row['pcm']
                        take = min(needed, len(data) - self.offset)
                        parts.append(data[self.offset:self.offset + take])
                        self.offset += take
                        needed -= take
                        if self.offset == len(data):
                            source.release('audio', self.index)
                            self.row = None
                            self.index += 1
                        data = None
                    body = b''.join(parts)
                    samples = len(body) // 2
                    await source.pace('audio', self.samples_generated / 48000, get_start)
                    frame = av.AudioFrame(format='s16', layout='mono', samples=samples)
                    frame.planes[0].update(body)
                    frame.sample_rate = 48000
                    frame.pts, frame.time_base = self.samples_generated, Fraction(1, 48000)
                    self.digest.update(body)
                    self.samples_generated += samples
                    self.generated += 1
                    return frame
                except Exception:
                    source.errors.add('audio-source-failed')
                    self.stop()
                    raise

            def stop(self):
                if self.row is not None:
                    source.release('audio', self.index)
                    self.row = None
                super().stop()

        return [Audio(), Video()]

    def summary(self, tracks):
        audio, video = tracks
        if (self.errors or video.generated != self.preflight['frame_count']
                or audio.samples_generated != self.preflight['sample_count']
                or audio.digest.hexdigest() != self.preflight['source_pcm_sha256']
                or self.cache or self.next != dict(audio=len(self.envelopes), video=len(self.envelopes))):
            raise WebRTCProbeError('incomplete saved source; no replay')
        return dict(kind='saved-h3', mode=self.mode, media_seconds=self.preflight['media_seconds'],
                    windows=len(self.envelopes), generated_audio_samples=audio.samples_generated,
                    source_pcm_exact=True, window_cache_highwater=self.high_windows,
                    compressed_cache_bytes_highwater=self.high_bytes,
                    decoded_video_storage='one active decoder/frame stream; native codec buffers opaque',
                    audio_carry_bytes_max=1920,
                    final_audio_chunk_samples=self.preflight['sample_count'] % 960 or 960,
                    source_lateness=self.late,
                    schedule_offsets_seconds=list(self.offsets),
                    max_window_ready_minus_media_start_seconds=max(
                        0.0, max(t - e.sample_start / 48000 for t, e in zip(self.offsets, self.envelopes))),
                    schedule_capture_binding_verified=False,
                    schedule_attribution=('operator-selected; index/size matched, not cryptographically bound'
                                          if self.mode != 'burst-source' else 'all source available at start'),
                    native_producer_credits_exercised=False, playback_ack_exercised=False,
                    source_manifest_sha256=self.preflight['source_manifest_sha256'])


async def run_saved(directory, *, timing_path=None):
    source = SavedSource(directory, timing_path=timing_path)
    return await run_media_probe(source.make_tracks, run_seconds=source.run_seconds,
                                 source_summary=source.summary, private_media=True,
                                 schema='h3-webrtc-saved-probe-v1', operation_timeout=90)
