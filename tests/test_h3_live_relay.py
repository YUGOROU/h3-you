import http.client
import socket
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal.live_relay import Relay, RelayConfig, RelayError
from research.live_relay_server import build_server


def config(**kw):
    return RelayConfig('test-session', time.time() + 30, 'a' * 64, 'b' * 64,
                       wait_timeout=0.01, **kw)


class CoreTests(unittest.TestCase):
    def test_expired_configuration_cannot_be_restarted_into_a_new_lease(self):
        r=Relay(RelayConfig('old',time.time()-1,'a'*64,'b'*64))
        with self.assertRaises(RelayError):r.claim('producer')
        self.assertTrue(r._terminal)

    def test_reservations_credit_and_single_drain(self):
        relay = Relay(config(av_capacity=4))
        t = relay.reserve('av', 3)
        with self.assertRaises(RelayError) as cm:
            relay.reserve('av', 2)
        self.assertEqual(cm.exception.status, 429)
        relay.commit(t, b'abc')
        self.assertEqual(relay.take('av'), b'abc')
        self.assertEqual(relay.take('av'), b'')
        relay.commit(relay.reserve('av', 4), b'abcd')
        relay.abort()
        self.assertTrue(all(not q for q in relay._queues.values()))
        self.assertFalse(relay._tickets)

    def test_expiry_monotonic_and_wall(self):
        for clock in ('time', 'monotonic'):
            relay = Relay(config())
            now = getattr(time, clock)()
            with patch('research.h3_personal.live_relay.time.' + clock, return_value=now + 2000):
                with self.assertRaises(RelayError) as cm:
                    relay.status()
                self.assertEqual(cm.exception.status, 410)
            with self.assertRaises(RelayError):
                relay.claim('producer')

    def test_config_secrets_and_bounds(self):
        self.assertNotIn('a' * 64, repr(config()))
        with self.assertRaises(ValueError):
            config(av_capacity=40 * 1024 * 1024)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = build_server(('127.0.0.1', 0), config(body_timeout=0.2))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def headers(self, role):
        return {'X-H3-Session': 'test-session', 'X-H3-Role': role,
                'X-H3-Key': ('a' if role == 'producer' else 'b') * 64,
                'Content-Type': 'application/octet-stream'}

    def request(self, method, path, role='producer', body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
        conn.request(method, path, body, self.headers(role) if headers is None else headers)
        response = conn.getresponse()
        result = response.status, response.read()
        self.assertEqual(response.getheader('Cache-Control'), 'no-store')
        self.assertEqual(response.getheader('X-H3-Boot'), self.server.relay.boot_id)
        self.assertEqual(int(response.getheader('Content-Length')), len(result[1]))
        conn.close()
        return result

    def test_idle_service_housekeeping_expires_ram_without_more_requests(self):
        r=self.server.relay
        ticket=r.reserve('av',5);r.commit(ticket,b'bytes')
        r._deadline=time.monotonic()+.02
        time.sleep(.65)
        self.assertTrue(r._terminal)
        self.assertFalse(any(r._queues.values()))

    def test_duplicate_proxy_metadata_allowed_but_critical_headers_rejected(self):
        for header,wanted in (('X-Forwarded-For',200),('Content-Length',400)):
            c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=2)
            c.putrequest('GET','/health')
            c.putheader(header,'0');c.putheader(header,'0');c.endheaders()
            response=c.getresponse();self.assertEqual(response.status,wanted);response.read();c.close()

    def claim_both(self):
        for role in ('producer', 'consumer'):
            self.assertEqual(self.request('POST', '/v1/claim', role)[0], 200)

    def test_roles_claims_and_fifo(self):
        self.assertEqual(self.request('POST', '/v1/av', body=b'x')[0], 403)
        self.claim_both()
        self.assertEqual(self.request('POST', '/v1/claim')[0], 409)
        for channel, writer, reader in [('av', 'producer', 'consumer'),
                                        ('ack', 'consumer', 'producer'),
                                        ('input', 'consumer', 'producer')]:
            self.assertEqual(self.request('POST', '/v1/' + channel, reader, b'x')[0], 403)
            self.assertEqual(self.request('POST', '/v1/' + channel, writer, b'123')[0], 202)
            self.assertEqual(self.request('GET', '/v1/' + channel, reader), (200, b'123'))
            self.assertEqual(self.request('GET', '/v1/' + channel, reader), (204, b''))

    def raw(self, headers, suffix=b''):
        with socket.create_connection(('127.0.0.1', self.port), timeout=2) as sock:
            sock.sendall(headers + suffix)
            return sock.recv(4096)

    def test_auth_before_body_and_bad_auth_not_terminal(self):
        response = self.raw(b'POST /v1/av HTTP/1.1\r\nHost: local\r\nContent-Length: 999\r\n\r\n')
        self.assertIn(b' 401 ', response)
        self.claim_both()
        self.assertEqual(self.request('GET', '/v1/status')[0], 200)

    def test_truncated_body_terminal_without_persistence(self):
        self.claim_both()
        fields = ''.join(k + ': ' + v + '\r\n' for k, v in self.headers('producer').items())
        with patch('builtins.open', side_effect=AssertionError('durable IO forbidden')):
            with socket.create_connection(('127.0.0.1', self.port), timeout=2) as sock:
                sock.sendall(('POST /v1/av HTTP/1.1\r\nHost: local\r\n' + fields +
                              'Content-Length: 5\r\n\r\n').encode() + b'ab')
                sock.shutdown(socket.SHUT_WR)
                self.assertIn(b' 410 ', sock.recv(4096))
            self.assertEqual(self.request('GET', '/v1/status')[0], 410)
        self.assertFalse(any(self.server.relay._queues.values()))

    def test_framing_abort_and_health(self):
        self.assertEqual(self.request('GET', '/health', headers={}), (200, b'{"status":"ok"}'))
        self.assertEqual(self.request('GET', '/v1/status?key=x')[0], 400)
        self.assertIn(b' 400 ', self.raw(b'POST /v1/claim HTTP/1.1\r\nHost: local\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n'))
        self.assertEqual(self.request('POST', '/v1/abort')[0], 202)
        self.assertEqual(self.request('POST', '/v1/claim')[0], 410)
        self.assertEqual(self.request('GET', '/health', headers={})[0], 200)


if __name__ == '__main__':
    unittest.main()
