"""Growing original paired media. Consumption credits are NOT receiver/sink ACKs.

Only two admitted compressed windows; callers must await append serially (no
producer tasks/hidden queue). Decoder buffers are native/opaque, PCM carry <=1920.
"""
import asyncio
from fractions import Fraction
import io

from .paired_av import AVEnvelope, ProtocolError, _bodies


class LiveSource:
    def __init__(self, binding, *, timeout=30):
        binding.__post_init__()
        if not 0 < timeout <= 120:
            raise ValueError('finite timeout required')
        self.binding, self.timeout = binding, timeout
        self.cache = {}
        self.changed = asyncio.Event()
        self.index = self.frame_stop = self.sample_stop = 0
        self.finished = False
        self.failure = None
        self.appending = False
        self.next = dict(audio=0, video=0)
        self.high_windows = self.high_bytes = 0
        self.tracks = []

    def _check(self):
        if self.failure is not None:
            raise RuntimeError('live source aborted') from self.failure

    async def _wait(self):
        try:
            await asyncio.wait_for(self.changed.wait(), self.timeout)
        except (TimeoutError, asyncio.CancelledError) as exc:
            self.abort(exc)
            raise

    async def append(self, packet):
        if self.appending:
            raise ProtocolError('serial append required; no hidden producer queue')
        self.appending = True
        try:
            self._check()
            e = packet.envelope
            if type(e) is not AVEnvelope:
                raise ProtocolError('original envelope required')
            e.__post_init__()
            if (self.finished or e.binding != self.binding or
                    (e.index, e.frame_start, e.sample_start) !=
                    (self.index, self.frame_stop, self.sample_stop)):
                raise ProtocolError('contiguous global envelope required')
            _bodies(e, packet.video, packet.pcm)
            size = e.video_size + e.pcm_size
            while True:
                self.changed.clear()
                self._check()
                used = sum(r.envelope.video_size + r.envelope.pcm_size for r, _ in self.cache.values())
                if len(self.cache) < 2 and used + size <= self.binding.max_inflight_bytes:
                    break
                await self._wait()
            self.cache[e.index] = (packet, set())
            self.index += 1
            self.frame_stop, self.sample_stop = e.frame_stop, e.sample_stop
            self.finished = e.conversation_final
            self.high_windows = max(self.high_windows, len(self.cache))
            self.high_bytes = max(self.high_bytes, used + size)
            self.changed.set()
        except asyncio.CancelledError as exc:
            self.abort(exc)
            raise
        finally:
            self.appending = False

    async def acquire(self, kind, index):
        while True:
            self.changed.clear()
            self._check()
            if index != self.next[kind]:
                raise ProtocolError('source order changed')
            if index in self.cache:
                return self.cache[index][0]
            if self.finished and index == self.index:
                return None
            await self._wait()

    def release(self, kind, index):
        self._check()
        _, done = self.cache[index]
        if kind in done:
            raise ProtocolError('duplicate consumption')
        done.add(kind)
        self.next[kind] += 1
        if len(done) == 2:
            del self.cache[index]
        self.changed.set()

    async def wait_for_consumed(self):
        """Wait for final source retirement, never network delivery/playback."""
        while True:
            self.changed.clear()
            self._check()
            if self.finished and not self.cache:
                return
            await self._wait()

    def abort(self, reason=None):
        if self.failure is None:
            self.failure = reason or RuntimeError('closed')
        self.cache.clear()
        self.changed.set()

    async def aclose(self):
        self.abort()
        for track in self.tracks:
            await track.aclose()

    def make_tracks(self, gate, get_start):
        import av
        from aiortc import MediaStreamTrack
        from aiortc.mediastreams import MediaStreamError
        import time

        if self.tracks:
            raise ProtocolError('one audio/video track pair only')
        source = self

        async def pace(offset):
            source._check()
            delay = max(0, get_start() + offset - time.monotonic())
            if delay > source.timeout:
                raise ProtocolError('clock outside finite scope')
            await asyncio.sleep(delay)
            source._check()

        class Track(MediaStreamTrack):
            def __init__(self):
                super().__init__()
                self.frames = self.generate()
                self.exhausted = False

            async def recv(self):
                try:
                    return await anext(self.frames)
                except StopAsyncIteration:
                    self.exhausted = True
                    super().stop()
                    raise MediaStreamError
                except BaseException as exc:
                    source.abort(exc)
                    super().stop()
                    raise

            def stop(self):
                # aiortc stops each sender at its own normal EOF. Video may end
                # before the real final audio carry; don't cancel the sibling.
                if not self.exhausted:
                    source.abort()
                super().stop()

            async def aclose(self):
                self.stop()
                await self.frames.aclose()

        class Video(Track):
            kind = 'video'

            async def generate(self):
                await asyncio.wait_for(gate.wait(), source.timeout)
                index = 0
                while (packet := await source.acquire('video', index)) is not None:
                    e = packet.envelope
                    with av.open(io.BytesIO(packet.video), format='mp4') as container:
                        streams = container.streams.video
                        if len(streams) != 1:
                            raise ProtocolError('exactly one video stream required')
                        stream = streams[0]
                        if (stream.codec_context.width, stream.codec_context.height) != (252, 448):
                            raise ProtocolError('native geometry required')
                        stream.codec_context.thread_count = 1
                        count = 0
                        for frame in container.decode(stream):
                            if (frame.width, frame.height) != (252, 448) or count >= e.frame_stop - e.frame_start:
                                raise ProtocolError('decoded frame outside envelope')
                            number = e.frame_start + count
                            await pace(number / 24)
                            frame.pts, frame.time_base = number * 3750, Fraction(1, 90000)
                            count += 1
                            yield frame
                        if count != e.frame_stop - e.frame_start:
                            raise ProtocolError('decoded frame count mismatch')
                    source.release('video', index)
                    packet = None
                    index += 1

        class Audio(Track):
            kind = 'audio'

            async def generate(self):
                await asyncio.wait_for(gate.wait(), source.timeout)
                index = samples = 0
                carry = bytearray()
                while (packet := await source.acquire('audio', index)) is not None:
                    offset = 0
                    while offset < len(packet.pcm):
                        take = min(1920 - len(carry), len(packet.pcm) - offset)
                        carry.extend(packet.pcm[offset:offset + take])
                        offset += take
                        if len(carry) == 1920:
                            await pace(samples / 48000)
                            frame = av.AudioFrame(format='s16', layout='mono', samples=960)
                            frame.planes[0].update(bytes(carry))
                            frame.sample_rate = 48000
                            frame.pts, frame.time_base = samples, Fraction(1, 48000)
                            samples += 960
                            carry.clear()
                            yield frame
                    source.release('audio', index)
                    packet = None
                    index += 1
                if carry:
                    await pace(samples / 48000)
                    frame = av.AudioFrame(format='s16', layout='mono', samples=len(carry) // 2)
                    frame.planes[0].update(bytes(carry))
                    frame.sample_rate = 48000
                    frame.pts, frame.time_base = samples, Fraction(1, 48000)
                    yield frame

        self.tracks = [Audio(), Video()]
        return self.tracks
