"""Loopback HTTP substitutes TLS dialing only; all tokens are test literals."""
import http.client
import json
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
from research.h3_personal.voxcpm_endpoint_service import create_server
from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL


class Tests(unittest.TestCase):
    def exercise(self, reject=False):
        entered = threading.Event(); release = threading.Event()
        class Candidate:
            sample_rate = 48000
            calls = 0
            active = False
            def stream(self, text):
                if self.active: raise RuntimeError('overlap')
                self.active = True
                self.calls += 1
                try:
                    yield [.25]
                    if self.calls == 1:
                        entered.set()
                        if not release.wait(3): raise TimeoutError()
                    yield [-.5]
                    yield [.125]
                finally:
                    self.active = False
        server = create_server(Candidate(), host='127.0.0.1', port=0, key='key',
            session='s', boot='b', expires_at=time.time()+60, cancel_protocol=CANCEL_PROTOCOL)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        def ready(method, path, body, headers, deadline):
            return 200, json.dumps(server.gate.ready(0)).encode()
        client = RemoteVoxCPM('https://test.endpoints.huggingface.cloud',
            token='test-token', key='key', session='s', boot='b',
            expires_at=server.gate.expires_at, cancel_protocol=CANCEL_PROTOCOL, transport=ready)
        client.read('tts_ready', time.monotonic()+5)
        errors = []; pcm = []
        def media():
            try: pcm.extend(client.stream('first', 1, time.monotonic()+8))
            except BaseException as error: errors.append(error)
        def dial(host, timeout):
            self.assertEqual(host, 'test.endpoints.huggingface.cloud')
            return http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=timeout)
        owner = threading.Thread(target=media)
        try:
            with patch('research.h3_personal.voxcpm_endpoint_client.http.client.HTTPSConnection', dial):
                owner.start()
                self.assertTrue(entered.wait(2))
                if reject:
                    with patch.object(server.gate, 'cancel', side_effect=ValueError('test rejection')):
                        with self.assertRaisesRegex(RuntimeError, 'terminal still required'):
                            client.request_cancel(1, time.monotonic()+2)
                    self.assertFalse(client._closed)
                else:
                    with patch('socket.getaddrinfo', side_effect=AssertionError('control must not resolve DNS')):
                        ack = client.request_cancel(1, time.monotonic()+2)
                    self.assertEqual(ack['state'], 'requested')
                    self.assertIsNone(client.last_completion)
                    self.assertTrue(owner.is_alive())
                    with self.assertRaisesRegex(RuntimeError, 'busy'):
                        list(client.stream('not allowed yet', 2, time.monotonic()+2))
                release.set(); owner.join(3)
                self.assertFalse(owner.is_alive()); self.assertEqual(errors, [])
                self.assertEqual(client.last_completion['finish_reason'],
                    'iterator_exhausted' if reject else 'cancelled')
                self.assertEqual(len(pcm), 3 if reject else 2)
                self.assertFalse(client._closed)
                self.assertEqual(len(list(client.stream('second', 2, time.monotonic()+3))), 3)
                self.assertEqual(client.last_completion['finish_reason'], 'iterator_exhausted')
                with self.assertRaises(RuntimeError): client.request_cancel(1, time.monotonic()+2)
        finally:
            release.set()
            if owner.ident is not None: owner.join(4)
            client.close(); server.shutdown(); server.server_close(); thread.join(3)
    def test_cancel_lane_does_not_wait_for_blocked_media(self):
        self.exercise()
    def test_control_rejection_keeps_bounded_media_drain_possible(self):
        self.exercise(reject=True)


if __name__ == '__main__': unittest.main()
