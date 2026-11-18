"""Loopback control/RTC/native Vox integration; no WAN, models or devices."""
import importlib.util
import array
import asyncio
import hashlib
import http.client
import json
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal import paired_av_wire as wire
from research.h3_personal.voxcpm_endpoint_protocol import decode, pcm_bytes
from research.h3_personal.voxcpm_endpoint_service import Gate
from research.h3_personal.voxcpm_rtc_client import serve
from research.h3_personal.voxcpm_rtc_service import VoxRTCManager, Server
from research.h3_personal.webrtc_live_peer import Session as RealSession

TURN = dict(urls='turns:turn.cloudflare.com:443?transport=tcp',
            username='test', credential='test')


HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class Integration(unittest.IsolatedAsyncioTestCase):
    async def test_local_control_rtc_native_exact_eof_without_media_fallback(self):
        chunks = [array.array('f', [.125, -.25, .5] * 1024),
                  array.array('f', [-.5, .25] * 1024)]
        texts = []

        class Candidate:
            sample_rate = 48000

            def stream(self, text):
                texts.append(text)
                yield from chunks

        gate = Gate(Candidate(), key='k'*40, session='session', boot='native-boot',
                    expires_at=time.time()+60)
        manager = VoxRTCManager(gate)
        server = Server(('127.0.0.1', 0), manager, asyncio.get_running_loop())
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        control, parent_control = socket.socketpair()
        media, parent_media = socket.socketpair()
        deadline = time.monotonic()+15
        task = reading = None
        peers = []
        calls = []
        received_wire = bytearray()

        def local_request(endpoint, path, token, key, session, deadline, value):
            calls.append((path, value))
            conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
            try:
                conn.request('POST', path, json.dumps(value), headers={
                    'Content-Type': 'application/json', 'X-H3-RTC-Key': key,
                    'X-H3-Session': session})
                response = conn.getresponse()
                row = json.loads(response.read())
                if response.status != 200:
                    raise RuntimeError('local control failed: '+str(row))
                return row
            finally:
                conn.close()

        def host_peer(**kwargs):
            peer = RealSession(deadline=kwargs['deadline'], control=kwargs.get('control', False))
            peers.append(peer)
            return peer

        def read_native():
            def read(n):
                parent_media.settimeout(max(.001, deadline-time.monotonic()))
                body = parent_media.recv(n)
                received_wire.extend(body)
                return body
            expected = {k: v for k, v in gate.ready(1).items()
                        if k in ('session', 'boot', 'expires_at', 'epoch')}
            return b''.join(decode(read, expected))

        try:
            with patch('research.h3_personal.voxcpm_rtc_service.Session', side_effect=host_peer), \
                 patch('research.h3_personal.webrtc_live_peer.Session', side_effect=host_peer), \
                 patch('research.h3_personal.voxcpm_rtc_client.request', side_effect=local_request), \
                 patch('research.h3_personal.webrtc_turn.fetch_turn_credentials', return_value=TURN) as turn, \
                 patch.dict(os.environ, {'H3_VOX_RTC_TOKEN': 'token', 'H3_VOX_RTC_KEY': gate.key}):
                task = asyncio.create_task(serve(dict(
                    endpoint='https://test.endpoints.huggingface.cloud', session=gate.session,
                    boot=gate.boot, epoch=1, text='synthetic', expires_at=gate.expires_at,
                    deadline=deadline), control, media))
                reading = asyncio.create_task(asyncio.to_thread(wire._json, parent_control, deadline))
                completed, _ = await asyncio.wait((reading, task), return_when=asyncio.FIRST_COMPLETED)
                if task in completed:
                    task.result()
                ready = await reading
                self.assertEqual(ready['schema'], 'h3-voxcpm-rtc-ipc-ready-v1')
                self.assertEqual(ready['binding'], dict(session=gate.session, boot=gate.boot,
                    epoch=1, purpose='voxcpm-native-wire-v1', max_bytes=8*1024*1024))
                self.assertNotEqual(manager._signal.boot, gate.boot)
                self.assertEqual(calls[1][1]['boot'], manager._signal.boot)
                parent_media.shutdown(socket.SHUT_WR)
                pcm = await asyncio.to_thread(read_native)
                receipt = await asyncio.to_thread(wire._json, parent_control, deadline)
                await task
                await asyncio.wait_for(manager._task, 5)
                turn.assert_called_once_with('token', 600)
            source = b''.join(pcm_bytes(chunk) for chunk in chunks)
            self.assertEqual(pcm, source)
            self.assertEqual(hashlib.sha256(pcm).digest(), hashlib.sha256(source).digest())
            self.assertEqual(texts, ['synthetic'])
            self.assertEqual([path for path, _ in calls], ['/rtc/start', '/rtc/offer'])
            self.assertEqual(receipt['schema'], 'h3-voxcpm-rtc-ipc-complete-v1')
            self.assertEqual(receipt['bytes'], 0)
            self.assertEqual(receipt['sha256'], hashlib.sha256(b'').hexdigest())
            self.assertGreater(receipt['received_bytes'], 0)
            self.assertEqual(receipt['received_bytes'], len(received_wire))
            self.assertEqual(receipt['received_sha256'], hashlib.sha256(received_wire).hexdigest())
            self.assertTrue(receipt['transport_complete'])
            self.assertFalse(receipt['native_eos_verified'])
            self.assertFalse(manager._failed)
            await manager.close()
            self.assertTrue(gate.closed)
            self.assertTrue(manager._closed)
            self.assertEqual(len(peers), 2)
            for peer in peers:
                self.assertEqual(peer.pc.connectionState, 'closed')
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await manager.close()
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(2)
            for sock in (control, parent_control, media, parent_media):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
            if reading is not None:
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)


if __name__ == '__main__':
    unittest.main()
