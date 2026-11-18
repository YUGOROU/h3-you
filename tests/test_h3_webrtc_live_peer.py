"""Run with research/webrtc_experiment/.venv/bin/python -m unittest discover
-s tests -p test_h3_webrtc_live_peer.py. Real local ICE; no devices/cloud.
"""
import importlib.util
import asyncio
import json
import time
import unittest

from aiortc import AudioStreamTrack, VideoStreamTrack
from research.h3_personal.webrtc_live_peer import Session, PeerError, description


HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class LivePeerTests(unittest.IsolatedAsyncioTestCase):
    async def pair(self, a, b):
        offer = json.loads(json.dumps(await a.create_offer()))
        answer = json.loads(json.dumps(await b.accept_offer(offer)))
        await a.set_answer(answer)
        await asyncio.gather(a.wait_connected(), b.wait_connected(),
                             a.wait_control(), b.wait_control())

    async def test_serialized_real_peers_media_and_control(self):
        audio, video = AudioStreamTrack(), VideoStreamTrack()
        deadline = time.monotonic() + 15
        async with Session(deadline=deadline, tracks=[audio, video], control=True) as a:
            async with Session(deadline=deadline, receive_audio=True, receive_video=True) as b:
                await self.pair(a, b)
                tracks = [await b.receive_track(), await b.receive_track()]
                frames = await asyncio.wait_for(asyncio.gather(*(t.recv() for t in tracks)), 4)
                self.assertEqual(len(frames), 2)
                for transceiver in a.pc.getTransceivers():
                    expected = "audio/opus" if transceiver.kind == "audio" else "video/H264"
                    self.assertTrue(all(c.mimeType == expected for c in transceiver._preferred_codecs))
                await a.send("ping")
                self.assertEqual(await b.receive(), "ping")
                await b.send(b"pong")
                self.assertEqual(await a.receive(), b"pong")
        self.assertEqual(audio.readyState, "ended")
        self.assertEqual(video.readyState, "ended")
        self.assertEqual(a.pc.connectionState, "closed")

    async def test_replay_terminal(self):
        async with Session(deadline=time.monotonic()+10, control=True) as a:
            await a.create_offer()
            with self.assertRaises(PeerError):
                await a.create_offer()
            self.assertEqual(a.pc.connectionState, "closed")

    async def test_invalid_answer_closes_owned_track(self):
        track = AudioStreamTrack()
        async with Session(deadline=time.monotonic()+5, tracks=[track]) as a:
            with self.assertRaises(PeerError):
                await a.set_answer({"type": "answer", "sdp": "bad"})
            self.assertEqual(track.readyState, "ended")

    async def test_deadline_closes_without_operation(self):
        a = Session(deadline=time.monotonic()+0.05, control=True)
        await asyncio.sleep(.15)
        await a.close()
        self.assertEqual(a.pc.connectionState, "closed")

    async def test_cancel_wait_closes(self):
        a = Session(deadline=time.monotonic()+10, control=True)
        task = asyncio.create_task(a.receive())
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(a.pc.connectionState, "closed")

    async def test_open_wait_deadline(self):
        a = Session(deadline=time.monotonic()+.05, control=True)
        with self.assertRaises(PeerError):
            await a.wait_control()
        self.assertEqual(a.pc.connectionState, "closed")

    async def test_message_bounds_and_overflow(self):
        deadline = time.monotonic()+15
        async with Session(deadline=deadline, control=True) as a:
            async with Session(deadline=deadline) as b:
                await self.pair(a, b)
                await a.send(b"x"*16384)
                self.assertEqual(len(await b.receive()), 16384)
                for _ in range(17):
                    if b._failed.is_set():
                        break
                    await a.send("x")
                    await asyncio.sleep(.01)
                await asyncio.wait_for(b._failed.wait(), 2)
                await b.close()
                self.assertLessEqual(b._messages.qsize(), 16)

    async def test_oversized_send_terminal(self):
        async with Session(deadline=time.monotonic()+5, control=True) as a:
            with self.assertRaises(PeerError):
                await a.send("あ"*6000)
            self.assertEqual(a.pc.connectionState, "closed")

    def test_description_validation(self):
        for value in (None, {}, {"type":"offer", "sdp": 2},
                      {"type":"answer", "sdp":"x"},
                      {"type":"offer", "sdp":"x"*65537},
                      {"type":"offer", "sdp":"x", "extra":True}):
            with self.assertRaises(PeerError):
                description(value, "offer")
        self.assertEqual(description({"type":"offer", "sdp":"x"*65536}, "offer").type, "offer")


if __name__ == "__main__":
    unittest.main()
