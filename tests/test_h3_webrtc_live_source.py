import asyncio
import hashlib
import importlib.util
import io
import time
import unittest
from types import SimpleNamespace

from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode, ProtocolError
from research.h3_personal.webrtc_live_source import LiveSource

HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class LiveTests(unittest.IsolatedAsyncioTestCase):
    def packet(self, index, final=False):
        import av
        out = io.BytesIO()
        with av.open(out, 'w', format='mp4') as container:
            stream = container.add_stream('mpeg4', rate=24)
            stream.width, stream.height = 252, 448
            stream.pix_fmt = 'yuv420p'
            frame = av.VideoFrame(252, 448, 'yuv420p')
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        video, pcm = out.getvalue(), bytes(4000)
        e = encode(self.binding, index, index, index + 1, video, pcm,
                   (AudioRegion('silence', index * 2000, (index + 1) * 2000),), final)
        return SimpleNamespace(envelope=e, video=video, pcm=pcm)

    async def asyncSetUp(self):
        self.binding = LiveAVBinding('live-test', '0' * 64, 0)

    async def test_growing_native_decode_and_continuous_carry(self):
        from aiortc.mediastreams import MediaStreamError
        source = LiveSource(self.binding, timeout=2)
        gate = asyncio.Event()
        gate.set()
        start = time.monotonic()
        audio, video = source.make_tracks(gate, lambda: start)
        await source.append(self.packet(0))
        first_audio, first_video = await asyncio.gather(audio.recv(), video.recv())
        self.assertEqual((first_audio.pts, first_video.pts), (0, 0))
        self.assertFalse(source.finished)
        self.assertEqual(source.index, 1)  # consumed before second exists
        audio_frames, video_frames = [first_audio], [first_video]

        async def drain(track, result):
            while True:
                try:
                    result.append(await track.recv())
                except MediaStreamError:
                    return

        tasks = [asyncio.create_task(drain(audio, audio_frames)),
                 asyncio.create_task(drain(video, video_frames))]
        await asyncio.sleep(.06)  # deliberately late live producer
        await source.append(self.packet(1, True))
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        await source.wait_for_consumed()
        self.assertEqual([f.pts for f in video_frames], [0, 3750])
        self.assertTrue(all((f.width, f.height) == (252, 448) for f in video_frames))
        self.assertEqual([f.pts for f in audio_frames], [0, 960, 1920, 2880, 3840])
        self.assertEqual([f.samples for f in audio_frames], [960] * 4 + [160])
        self.assertEqual(b''.join(bytes(f.planes[0]) for f in audio_frames), bytes(8000))
        self.assertLessEqual(source.high_windows, 2)
        self.assertLessEqual(source.high_bytes, self.binding.max_inflight_bytes)
        await source.aclose()

    async def test_normal_video_sender_stop_preserves_real_audio_tail(self):
        from aiortc.mediastreams import MediaStreamError
        source=LiveSource(self.binding,timeout=2)
        gate=asyncio.Event();gate.set();start=time.monotonic()
        audio,video=source.make_tracks(gate,lambda:start)
        await source.append(self.packet(0,True))
        await video.recv()
        with self.assertRaises(MediaStreamError):await video.recv()
        video.stop()  # RTCRtpSender's normal EOF finally block
        result=[]
        while True:
            try:result.append(await audio.recv())
            except MediaStreamError:break
        self.assertEqual(sum(f.samples for f in result),2000)
        self.assertIsNone(source.failure)
        await source.aclose()

    async def test_validation_backpressure_abort(self):
        source = LiveSource(self.binding, timeout=1)
        first = self.packet(0)
        bad = SimpleNamespace(envelope=first.envelope, video=b'bad', pcm=first.pcm)
        with self.assertRaises(ProtocolError):
            await source.append(bad)
        await source.append(first)
        with self.assertRaises(ProtocolError):
            await source.append(first)
        await source.append(self.packet(1))
        pending = asyncio.create_task(source.append(self.packet(2, True)))
        await asyncio.sleep(.01)
        self.assertFalse(pending.done())
        source.abort()
        with self.assertRaises(RuntimeError):
            await pending
        with self.assertRaises(RuntimeError):
            await source.wait_for_consumed()
        await source.aclose()

    async def test_finite_wait_and_cancel(self):
        source = LiveSource(self.binding, timeout=.02)
        with self.assertRaises(TimeoutError):
            await source.acquire('audio', 0)
        await source.aclose()
        source = LiveSource(self.binding)
        task = asyncio.create_task(source.acquire('audio', 0))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(RuntimeError):
            await source.wait_for_consumed()
        await source.aclose()
