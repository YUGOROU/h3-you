"""Host-only synthetic native RTC/control tests; no credentials/network services."""
import importlib.util
import array
import asyncio
import http.client
import json
import threading
import time
import unittest
from unittest.mock import patch
from research.h3_personal.voxcpm_endpoint_service import Gate
from research.h3_personal.voxcpm_rtc_service import VoxRTCManager, Server
from research.h3_personal.webrtc_live_peer import Session
from research.h3_personal.webrtc_exact_channel import Binding, ExactChannel
from research.h3_personal.voxcpm_endpoint_protocol import metadata, frame

TURN = dict(urls='turns:turn.cloudflare.com:443?transport=tcp', username='test', credential='test')
MODULE = 'research.h3_personal.voxcpm_rtc_service.'

HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


class Candidate:
    sample_rate = 48000
    def stream(self, text):
        yield array.array('f', [.125, -.25, .5])
    def export_evidence(self):
        return {'synthetic': True}

@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gate = Gate(Candidate(), key='k'*40, session='session', boot='native-boot',
                         expires_at=time.time()+30)
        self.manager = VoxRTCManager(self.gate)
        self.addAsyncCleanup(self.manager.close)
        self.patch = patch(MODULE+'Session', side_effect=lambda **kw: Session(deadline=kw['deadline']))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def request(self, epoch=1):
        return dict(session=self.gate.session, boot=self.gate.boot, epoch=epoch, text='synthetic', turn_config=TURN)

    async def test_native_exact_success_retained_peer_and_replay(self):
        row = await self.manager.start(self.request())
        self.assertEqual(row['schema'], 'h3-voxcpm-rtc-turn-v1')
        self.assertEqual(row['binding']['boot'], self.gate.boot)
        self.assertEqual(row['signal_boot'], self.manager._signal.boot)
        self.assertNotEqual(row['signal_boot'], self.gate.boot)
        client = Session(deadline=time.monotonic()+8, control=True)
        self.addAsyncCleanup(client.close)
        await client.set_answer(await self.manager._peer.accept_offer(await client.create_offer()))
        await client.wait_control()
        channel = ExactChannel(client, Binding(**row['binding']))
        self.addAsyncCleanup(channel.close)
        await channel.start()
        received = bytearray()
        async def sink(data): received.extend(data)
        async def receive():
            while await channel.receive_into(sink): pass
        await asyncio.wait_for(asyncio.gather(channel.finish(), receive()), 5)
        await channel.barrier()
        await asyncio.wait_for(self.manager._task, 5)
        expected_pcm = array.array('f', [.125, -.25, .5]).tobytes()
        self.assertIn(frame(b'P', expected_pcm), received)
        self.assertTrue(received.startswith(metadata(b'R', self.gate.ready(1))))
        self.assertFalse(self.manager._failed)
        self.assertIsNotNone(self.manager._peer)
        with self.assertRaises(ValueError): await self.manager.start(self.request())

    async def test_binding_shape_bounds_and_credentials(self):
        for change in ({'boot':'wrong'}, {'session':'wrong'}, {'epoch':True}, {'epoch':3},
                       {'text':''}, {'extra':0}, {'turn_config':{}}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    await self.manager.start(dict(self.request(), **change))
        self.assertEqual(self.manager._epoch, 0)

    async def test_expiry_and_cancel_close_native(self):
        self.gate.expires_at = time.time()-1
        with self.assertRaises(ValueError): await self.manager.start(self.request())
        self.gate.expires_at = self.manager.expires_at
        await self.manager.start(self.request())
        await self.manager.close()
        self.assertTrue(self.gate.closed)
        with self.assertRaises(ValueError): await self.manager.start(self.request(2))

    async def test_gateway_auth_exact_ready_start_and_no_stream(self):
        server = Server(('127.0.0.1', 0), self.manager, asyncio.get_running_loop())
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.01})
        thread.start()
        def request(method, path, value=None, auth=True):
            conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=3)
            headers = {'Content-Type':'application/json'}
            if auth: headers.update({'X-H3-RTC-Key':self.gate.key, 'X-H3-TTS-Key':self.gate.key, 'X-H3-Session':self.gate.session})
            try:
                conn.request(method, path, json.dumps(value) if value is not None else None, headers)
                response = conn.getresponse()
                return response.status, json.loads(response.read())
            finally: conn.close()
        try:
            self.assertEqual((await asyncio.to_thread(request, 'GET', '/ready', None, False))[0], 401)
            self.assertEqual(await asyncio.to_thread(request, 'GET', '/ready'), (200, self.gate.ready(0)))
            self.assertEqual((await asyncio.to_thread(request, 'GET', '/evidence'))[1]['evidence'], {'synthetic':True})
            for method in ('GET', 'POST'):
                self.assertEqual((await asyncio.to_thread(request, method, '/stream'))[0], 404)
            self.assertEqual((await asyncio.to_thread(request, 'POST', '/rtc/start', self.request(), False))[0], 401)
            status, row = await asyncio.to_thread(request, 'POST', '/rtc/start', self.request())
            self.assertEqual(status, 200)
            self.assertNotIn('signal_port', row)
            self.assertEqual(row['binding']['boot'], self.gate.boot)
            self.assertEqual((await asyncio.to_thread(request, 'POST', '/rtc/start', self.request()))[0], 503)
            self.assertEqual((await asyncio.to_thread(request, 'POST', '/rtc/offer', {}, False))[0], 401)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(2)

if __name__ == '__main__': unittest.main()
