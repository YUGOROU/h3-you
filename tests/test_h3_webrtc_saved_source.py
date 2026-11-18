import asyncio
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest

from research.h3_personal.paired_av import AudioRegion, LiveAVBinding, encode
from research.h3_personal.webrtc_saved_source import SavedSource, availability_schedule, run_saved
from research.h3_personal.webrtc_synthetic_probe import WebRTCProbeError
from research.h3_personal.paired_av_wire import envelope_record

HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class SavedSourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import av
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'packets'
        self.root.mkdir()
        binding = LiveAVBinding('PRIVATE_SOURCE', '0' * 64, 0)
        self.original = bytearray()
        start = 0
        for index, count in enumerate((2, 3)):
            out = io.BytesIO()
            with av.open(out, mode='w', format='mp4') as container:
                stream = container.add_stream('libx264', rate=24)
                stream.width, stream.height = 252, 448
                stream.pix_fmt = 'yuv420p'
                stream.codec_context.thread_count = 1
                stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
                for i in range(count):
                    frame = av.VideoFrame(252, 448, 'yuv420p')
                    for plane in frame.planes:
                        plane.update(bytes([100 + index]) * plane.buffer_size)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode(None): container.mux(packet)
            video = out.getvalue()
            pcm = b''.join(struct.pack('<h', i % 1000 - 500) for i in range(start*2000, (start+count)*2000))
            self.original.extend(pcm)
            region = AudioRegion('speech', start*2000, (start+count)*2000,
                                 'PRIVATE_SOURCE', 0, start*2000, hashlib.sha256(pcm).hexdigest())
            e = encode(binding, index, start, start+count, video, pcm, (region,), index == 1)
            # Wire form omits the default presentation generation; asdict() would emit an explicit 0,
            # which the wire parser deliberately rejects.
            (self.root / f'{index:03d}.json').write_text(json.dumps(envelope_record(e)))
            (self.root / f'{index:03d}.mp4').write_bytes(video)
            (self.root / f'{index:03d}.pcm').write_bytes(pcm)
            start += count

    def timing(self, times, **fields):
        p = self.root.parent / 'timing.json'
        r = dict(type='h3_producer_timing', clock='local_monotonic',
                 windows=[dict(index=i, packet_ready=t) for i, t in enumerate(times)])
        for i, row in enumerate(r['windows']):
            header = self.root / f'{i:03d}.json'
            if header.exists():
                e = json.loads(header.read_text())
                row.update(video_bytes=e['video_size'], pcm_bytes=e['pcm_size'])
        r.update(fields)
        p.write_text(json.dumps(r))
        return p

    async def drain(self, source):
        gate = asyncio.Event()
        start = time.monotonic()
        tracks = source.make_tracks(gate, lambda: start)
        gate.set()
        pcm, video_pts = bytearray(), []

        async def consume(track):
            from aiortc.mediastreams import MediaStreamError
            try:
                while True:
                    frame = await track.recv()
                    if track.kind == 'audio':
                        pcm.extend(bytes(frame.planes[0])[:frame.samples*2])
                    else:
                        video_pts.append(float(frame.pts * frame.time_base))
            except MediaStreamError:
                pass

        try:
            await asyncio.wait_for(asyncio.gather(*(consume(t) for t in tracks)), 3)
            return source.summary(tracks), bytes(pcm), video_pts
        finally:
            for track in tracks: track.stop()
            await tracks[1].aclose()

    async def test_cross_window_pcm_and_continuous_video_exact(self):
        source = SavedSource(self.root)
        summary, pcm, pts = await self.drain(source)
        self.assertEqual(pcm, bytes(self.original))
        self.assertEqual(pts, [i/24 for i in range(5)])
        self.assertEqual(summary['generated_audio_samples'], 10000)
        self.assertEqual(summary['final_audio_chunk_samples'], 400)
        self.assertTrue(summary['source_pcm_exact'])
        self.assertLessEqual(summary['window_cache_highwater'], 2)
        self.assertFalse(summary['native_producer_credits_exercised'])
        self.assertNotIn('PRIVATE_SOURCE', json.dumps(summary))

    async def test_recorded_delay_is_exposed_not_padded_or_dropped(self):
        source = SavedSource(self.root, timing_path=self.timing([100, 100.3]))
        summary, pcm, pts = await self.drain(source)
        self.assertEqual(pcm, bytes(self.original))
        self.assertEqual(len(pts), 5)
        self.assertGreater(summary['source_lateness']['audio']['max_seconds'], 0.1)
        self.assertGreater(summary['source_lateness']['video']['max_seconds'], 0.1)
        self.assertEqual(summary['schedule_offsets_seconds'][0], 0)
        self.assertAlmostEqual(summary['schedule_offsets_seconds'][1], .3)
        self.assertAlmostEqual(summary['max_window_ready_minus_media_start_seconds'], .3 - 2/24)
        self.assertFalse(summary['schedule_capture_binding_verified'])
        self.assertIn('not cryptographically bound', summary['schedule_attribution'])

    async def test_real_saved_media_no_per_window_track_reset(self):
        r = await run_saved(self.root)
        self.assertTrue(r['cleanup_complete'])
        self.assertTrue(r['source']['source_pcm_exact'])
        self.assertEqual(r['source_generated_frames']['video'], 5)
        self.assertEqual(r['source_audio_samples'], 10000)
        self.assertGreater(r['receive']['video']['frames'], 0)
        self.assertGreater(r['receive']['audio']['samples'], 0)
        self.assertFalse(r['sample_exact'])
        self.assertFalse(r['playout_verified'])

    async def test_post_preflight_body_change_rejected(self):
        source = SavedSource(self.root)
        p = self.root / '000.mp4'
        b = bytearray(p.read_bytes()); b[-1] ^= 1; p.write_bytes(b)
        gate = asyncio.Event(); gate.set()
        tracks = source.make_tracks(gate, time.monotonic)
        try:
            with self.assertRaises(ValueError): await tracks[1].recv()
            self.assertIn('video-source-failed', source.errors)
        finally:
            for t in tracks: t.stop()

    async def test_cancellation_closes_decoder_and_releases_source(self):
        source = SavedSource(self.root)
        gate = asyncio.Event(); gate.set()
        start = time.monotonic()
        tracks = source.make_tracks(gate, lambda: start)
        tasks = []

        async def drain(track):
            while True: await track.recv()

        try:
            tasks = [asyncio.create_task(drain(t)) for t in tracks]
            await asyncio.sleep(0.03)
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for track in tracks: track.stop()
        self.assertEqual(source.cache, {})
        self.assertIsNone(tracks[1].container)
        await tracks[1].aclose()
        self.assertTrue(tracks[1].generator_closed)
        self.assertIsNone(tracks[1].frames.ag_frame)

    async def test_decoded_frame_count_mismatch_is_terminal(self):
        for suffix in ('json', 'mp4', 'pcm'):
            (self.root / f'001.{suffix}').unlink()
        p = self.root / '000.json'
        value = json.loads(p.read_text())
        pcm = (self.root / '000.pcm').read_bytes() + b'\\0' * 2000
        digest = hashlib.sha256(pcm).hexdigest()
        value.update(conversation_final=True, frame_stop=3, sample_stop=6000,
                     pcm_size=len(pcm), pcm_sha256=digest)
        value['regions'][0].update(sample_stop=6000, source_sha256=digest)
        p.write_text(json.dumps(value))
        (self.root / '000.pcm').write_bytes(pcm)
        source = SavedSource(self.root)
        gate = asyncio.Event(); gate.set()
        start = time.monotonic()
        tracks = source.make_tracks(gate, lambda: start)
        try:
            await tracks[1].recv()
            await tracks[1].recv()
            with self.assertRaises(WebRTCProbeError): await tracks[1].recv()
            self.assertIn('video-source-failed', source.errors)
        finally:
            for track in tracks: track.stop()

    def test_schedule_validation(self):
        for times in ([1], [2, 1], [0, float('nan')], [0, 76]):
            with self.subTest(times=times):
                with self.assertRaises(WebRTCProbeError): availability_schedule(self.timing(times), 2)
        with self.assertRaises(WebRTCProbeError):
            availability_schedule(self.timing([1, 2], clock='different_process'), 2)

    def test_unrelated_timing_sizes_rejected_and_attribution_explicit(self):
        p = self.timing([0, 1])
        value = json.loads(p.read_text())
        value['windows'][1]['video_bytes'] += 1
        p.write_text(json.dumps(value))
        with self.assertRaises(WebRTCProbeError): SavedSource(self.root, timing_path=p)

    async def test_cache_has_two_owners_and_retires_only_after_both(self):
        source = SavedSource(self.root)
        start = time.monotonic()
        a = await source.acquire('audio', 0, lambda: start)
        source.release('audio', 0)
        b = await source.acquire('audio', 1, lambda: start)
        self.assertEqual(len(source.cache), 2)
        self.assertEqual(source.high_windows, 2)
        source.release('video', 0)
        self.assertEqual(len(source.cache), 1)
        source.release('audio', 1)
        source.release('video', 1)
        self.assertEqual(len(source.cache), 0)
        del a, b


if __name__ == '__main__':
    unittest.main()
