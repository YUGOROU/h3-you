import asyncio
import contextlib
import importlib.util
import io
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.h3_personal.webrtc_synthetic_probe import (
    WebRTCProbeError, Reception, close_owned, run_synthetic, sanitized_stats, validate_duration,
)
from research.run_h3_webrtc_probe import main

HAVE_RTC = importlib.util.find_spec('aiortc') is not None and importlib.util.find_spec('av') is not None


class ContractTests(unittest.TestCase):
    def test_duration_bounds(self):
        self.assertEqual(validate_duration(1), 1.0)
        for value in (True, 0, 11, float('nan'), float('inf'), '4'):
            with self.subTest(value=value):
                with self.assertRaises(WebRTCProbeError): validate_duration(value)

    def test_stats_never_include_identifiers_or_addresses(self):
        value = SimpleNamespace(type='inbound-rtp', kind='audio', packetsReceived=3,
                                jitter=48, id='PRIVATE', ssrc=1234, address='PRIVATE',
                                timestamp='PRIVATE', roundTripTime=float('nan'))
        r = sanitized_stats({'PRIVATE': value})
        self.assertEqual(r, [dict(type='inbound-rtp', kind='audio', packetsReceived=3,
                                  jitter_rtp_timestamp_units=48)])
        self.assertNotIn('PRIVATE', json.dumps(r))

    def test_stats_unknown_string_rejected(self):
        r = sanitized_stats({'x': SimpleNamespace(type='transport', dtlsState='PRIVATE')})
        self.assertEqual(r, [dict(type='transport')])

    def test_stats_bound(self):
        with self.assertRaises(WebRTCProbeError):
            sanitized_stats({i: SimpleNamespace(type='transport') for i in range(65)})

    def test_media_timestamp_contract(self):
        r = Reception('audio')
        f = SimpleNamespace(pts=0, time_base=1/48000, samples=960,
                            sample_rate=48000, layout=SimpleNamespace(channels=[1, 2]))
        r.observe(f, 0.1)
        with self.assertRaises(WebRTCProbeError): r.observe(f, 0.2)
        f.pts = None
        with self.assertRaises(WebRTCProbeError): r.observe(f, 0.3)

    def test_default_cli_never_starts_transport(self):
        stream = io.StringIO()
        with patch('research.run_h3_webrtc_probe.subprocess.run') as child:
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main([]), 0)
        child.assert_not_called()
        self.assertFalse(json.loads(stream.getvalue())['transport_executed'])

    def test_subprocess_timeout_is_terminal_and_sanitized(self):
        stream = io.StringIO()
        with patch('research.run_h3_webrtc_probe.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('PRIVATE', 30)) as child:
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(['--run']), 1)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(child.call_args.kwargs['timeout'], 30)
        self.assertEqual(child.call_args.kwargs['stderr'], subprocess.DEVNULL)
        self.assertNotIn('PRIVATE', stream.getvalue())

    def test_child_failure_no_replay(self):
        with patch('research.run_h3_webrtc_probe.subprocess.run',
                   return_value=SimpleNamespace(returncode=1, stdout=b'PRIVATE')) as child:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--run']), 1)
        self.assertEqual(child.call_count, 1)

    def test_cleanup_receipt_required(self):
        bad = json.dumps(dict(schema='h3-webrtc-synthetic-probe-v1', cleanup_complete=False)).encode()
        with patch('research.run_h3_webrtc_probe.subprocess.run',
                   return_value=SimpleNamespace(returncode=0, stdout=bad)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--run']), 1)


class CleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_async_close_is_awaited(self):
        class Source:
            stopped = closed = False

            def stop(self): self.stopped = True

            async def aclose(self):
                await asyncio.sleep(0)
                self.closed = True

        source = Source()
        self.assertTrue(await close_owned([], [source], []))
        self.assertTrue(source.stopped and source.closed)

    async def test_source_async_close_failure_is_not_success(self):
        class Source:
            def stop(self): pass

            async def aclose(self): raise ValueError('private cleanup detail')

        self.assertFalse(await close_owned([], [Source()], []))

    async def test_stubborn_consumer_does_not_block_peer_close(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def stubborn():
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        class Peer:
            connectionState = 'connected'

            async def close(self):
                self.connectionState = 'closed'

        task = asyncio.create_task(stubborn())
        await entered.wait()
        peer = Peer()
        try:
            self.assertFalse(await close_owned([peer], [], [task], timeout=0.01))
            self.assertEqual(peer.connectionState, 'closed')
        finally:
            release.set()
            await asyncio.wait_for(task, 1)


@unittest.skipUnless(HAVE_RTC, 'optional uv WebRTC environment required')
class JitterTailTests(unittest.TestCase):
    def test_finite_input_retains_tail_without_packet_loss(self):
        # Pin-aware characterization, not a patch or a production flush policy.
        from aiortc.jitterbuffer import JitterBuffer
        for prefetch, expected in ((4, 4), (0, 7)):
            with self.subTest(prefetch=prefetch):
                buffer = JitterBuffer(capacity=16, prefetch=prefetch, is_video=prefetch == 0)
                output = []
                for i in range(8):
                    _, frame = buffer.add(SimpleNamespace(sequence_number=i, timestamp=i*960, _data=b'x'))
                    if frame is not None:
                        output.append(frame)
                self.assertEqual(len(output), expected)


@unittest.skipUnless(HAVE_RTC, 'optional uv WebRTC environment required')
class RealPeerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_synthetic_media_and_control(self):
        r = await run_synthetic(duration_seconds=1)
        self.assertTrue(r['cleanup_complete'])
        self.assertTrue(r['control_roundtrip'])
        self.assertEqual(r['external_ice_servers'], [])
        self.assertEqual(r['source_generated_frames'], dict(audio=50, video=24))
        self.assertGreater(r['receive']['audio']['frames'], 0)
        self.assertGreater(r['receive']['video']['frames'], 0)
        self.assertFalse(r['sample_exact'])
        self.assertFalse(r['playout_verified'])
        self.assertEqual(r['codec_preferences']['video']['mime_type'].lower(), 'video/h264')
        self.assertEqual(r['receive']['video']['formats'], [(252, 448)])
        self.assertLessEqual(r['receive']['audio']['samples'], r['source_audio_samples'])

    async def test_cancellation_closes_owned_peers(self):
        from aiortc import RTCPeerConnection
        peers = []

        def create(*args, **kwargs):
            peer = RTCPeerConnection(*args, **kwargs)
            peers.append(peer)
            return peer

        with patch('aiortc.RTCPeerConnection', side_effect=create):
            task = asyncio.create_task(run_synthetic(duration_seconds=5))
            # Cancel after peer construction, including a possible negotiation wait.
            for _ in range(100):
                if len(peers) == 2: break
                await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(len(peers), 2)
        self.assertTrue(all(p.connectionState == 'closed' for p in peers))

    async def test_second_constructor_failure_closes_first_peer(self):
        from aiortc import RTCPeerConnection
        peers = []

        def create(*args, **kwargs):
            if peers:
                raise RuntimeError('primary construction failure')
            peer = RTCPeerConnection(*args, **kwargs)
            peers.append(peer)
            return peer

        with patch('aiortc.RTCPeerConnection', side_effect=create):
            with self.assertRaisesRegex(RuntimeError, 'primary construction failure'):
                await run_synthetic(duration_seconds=1)
        self.assertEqual(peers[0].connectionState, 'closed')

    async def test_channel_setup_failure_closes_peers(self):
        from aiortc import RTCPeerConnection
        peers = []

        def create(*args, **kwargs):
            peer = RTCPeerConnection(*args, **kwargs)
            peers.append(peer)
            return peer

        with patch('aiortc.RTCPeerConnection', side_effect=create):
            with patch.object(RTCPeerConnection, 'createDataChannel', side_effect=RuntimeError('channel setup')):
                with self.assertRaisesRegex(RuntimeError, 'channel setup'):
                    await run_synthetic(duration_seconds=1)
        self.assertEqual(len(peers), 2)
        self.assertTrue(all(p.connectionState == 'closed' for p in peers))

    async def test_cleanup_error_does_not_mask_primary(self):
        async def failing_close(*args):
            await close_owned(*args)
            raise RuntimeError('secondary cleanup failure')

        with patch('research.h3_personal.webrtc_synthetic_probe.close_owned', side_effect=failing_close):
            with patch('aiortc.RTCRtpSender.getCapabilities', return_value=SimpleNamespace(codecs=[])):
                with self.assertRaisesRegex(WebRTCProbeError, 'explicit codec unavailable'):
                    await run_synthetic(duration_seconds=1)

    async def test_missing_codec_is_terminal_and_closes_peers(self):
        from aiortc import RTCPeerConnection
        peers = []

        def create(*args, **kwargs):
            peer = RTCPeerConnection(*args, **kwargs)
            peers.append(peer)
            return peer

        with patch('aiortc.RTCPeerConnection', side_effect=create):
            with patch('aiortc.RTCRtpSender.getCapabilities', return_value=SimpleNamespace(codecs=[])):
                with self.assertRaises(WebRTCProbeError): await run_synthetic(duration_seconds=1)
        self.assertTrue(all(p.connectionState == 'closed' for p in peers))


if __name__ == '__main__':
    unittest.main()
