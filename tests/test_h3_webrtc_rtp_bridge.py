import asyncio
import json
import socket
import time
import unittest

from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode, ProtocolError
from research.h3_personal import paired_av_wire as wire
from research.h3_personal.webrtc_rtp_bridge import RTPBridge


class Peer:
    def __init__(self):
        self.sent = asyncio.Queue(2)
        self.events = asyncio.Queue(2)

    async def send(self, value):
        await self.sent.put(json.loads(value))

    async def receive(self):
        return await self.events.get()


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, partial=False, bad=False):
        deadline = time.monotonic() + 5
        binding = LiveAVBinding('test', 'a' * 64, 1)
        bridge = RTPBridge(binding, deadline)
        peer = Peer()
        left, right = socket.socketpair()
        e = encode(binding, 0, 0, 1, b'original-video', bytes(4000),
                   (AudioRegion('silence', 0, 2000),), True)
        task = asyncio.create_task(bridge.run(peer, left))
        try:
            await asyncio.to_thread(wire.send_pair, right, e, b'original-video',
                                    bytes(4000), deadline=deadline)
            await asyncio.to_thread(wire.receive_ack, right, e, stage='verified', deadline=deadline)
            window = await peer.sent.get()
            self.assertEqual(window['kind'], 'window')
            self.assertEqual((await peer.sent.get())['kind'], 'source_eos')
            # Both codecs consuming verified source is explicitly not sink ACK.
            for kind in ('audio', 'video'):
                packet = await bridge.source.acquire(kind, 0)
                self.assertEqual(packet.pcm, bytes(4000))
                bridge.source.release(kind, 0)
            self.assertEqual(bridge.receiver.snapshot()['sink_acked'], 0)
            right.setblocking(False)
            with self.assertRaises(BlockingIOError):
                right.recv(1)
            right.setblocking(True)
            event = bridge._message('end-partial' if partial else 'sink-accepted',
                index=1 if bad else 0, video_frame_stop=0 if partial else 1,
                audio_sample_stop=1920 if partial else 2000)
            await peer.events.put(json.dumps(event))
            if bad:
                with self.assertRaises(ProtocolError):
                    await task
                return
            if partial:
                notice = await asyncio.to_thread(wire._json, right, deadline)
                self.assertEqual(notice['schema'], 'h3-rtc-closing-tail-v1')
                self.assertFalse(notice['media_complete'])
            else:
                await asyncio.to_thread(wire.receive_ack, right, e,
                                        stage='sink-accepted', deadline=deadline)
            result = await task
            self.assertTrue(result['source_complete'])
            self.assertEqual(result['media_complete'], not partial)
        finally:
            right.close()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_verified_and_consumed_do_not_release(self):
        await self.exercise()

    async def test_silent_partial_is_not_final_ack(self):
        await self.exercise(partial=True)

    async def test_wrong_fifo_fails_closed(self):
        await self.exercise(bad=True)

    async def test_cancel_joins_blocked_receive(self):
        bridge = RTPBridge(LiveAVBinding('test', 'a' * 64, 0), time.monotonic() + 3)
        left, right = socket.socketpair()
        task = asyncio.create_task(bridge.run(Peer(), left))
        await asyncio.sleep(.03)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertTrue(all(t.done() for t in bridge._threads))
        self.assertIsNotNone(bridge.source.failure)
        right.close()
