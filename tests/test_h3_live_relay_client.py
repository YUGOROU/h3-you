"""Local-only relay contract tests; no provider or device access."""
import collections
import http.server
import json
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal.live_relay_client import SocketAdapter, RelayError, MAX_BODY


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # Expected resets when a client rejects an oversized response.


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        s = self.server
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        s.calls.append((self.command, self.path, body, dict(self.headers)))
        boot = s.boot
        if self.path == '/v1/claim':
            status = s.claim_status
            data = json.dumps(dict(schema='h3-live-relay-v1', session='session',
                                   boot_id=boot, not_after=s.expiry)).encode()
        elif self.command == 'GET':
            status, data, boot = s.reads.popleft() if s.reads else (204, b'', boot)
        else:
            status, data = s.write_status, b''
        if getattr(s,'slow_headers',False):
            raw=(f'HTTP/1.1 {status} OK\r\nX-H3-Boot: {boot}\r\nCache-Control: no-store\r\n'
                 f'Content-Length: {len(data)}\r\nConnection: close\r\n\r\n').encode()
            self.close_connection=True
            try:
                for line in raw.splitlines(keepends=True):
                    self.wfile.write(line);self.wfile.flush();time.sleep(.12)
                self.wfile.write(data)
            except OSError:pass
            return
        self.send_response(status)
        self.send_header('X-H3-Boot', boot)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        if s.encoded:
            self.send_header('Content-Encoding', 'gzip')
        if getattr(s, 'connection_close', False):
            self.send_header('Connection', 'close')
            self.close_connection = True
        self.end_headers()
        self.wfile.flush()
        try:
            time.sleep(getattr(s, 'body_delay', 0))
            if getattr(s, 'body_trickle', False):
                for byte in data:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(.04)
            else:
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.server = Server(('127.0.0.1', 0), Handler)
        self.server.boot = 'a' * 32
        self.server.expiry = time.time() + 10
        self.server.claim_status = 200
        self.server.write_status = 202
        self.server.encoded = False
        self.server.calls = []
        self.server.reads = collections.deque()
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.clients = []

    def tearDown(self):
        for c in self.clients:
            c.close()
            if c._writer:
                self.assertFalse(c._writer.is_alive())
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def client(self, **kw):
        args = dict(session='session', not_after=self.server.expiry,
                    role='producer', role_key='b' * 64, hf_token='hf_secret',
                    allow_test_loopback=True, http_timeout=.2)
        args.update(kw)
        c = SocketAdapter('http://127.0.0.1:%s' % self.server.server_port, **args)
        self.clients.append(c)
        return c

    def test_header_trickle_cannot_renew_http_deadline(self):
        self.server.slow_headers=True
        c=self.client(http_timeout=.2)
        started=time.monotonic()
        with self.assertRaises(RelayError):c.open_claim()
        self.assertLess(time.monotonic()-started,.6)

    def test_connection_close_delayed_claim_and_read(self):
        self.server.connection_close = True
        self.server.body_delay = .05
        c = self.client(http_timeout=.5).open()
        self.assertIsNone(c._connections['claim'].sock)
        self.server.reads.append((200, b'delayed-body', self.server.boot))
        self.assertEqual(c.recv(64), b'delayed-body')
        self.assertIsNone(c._connections['read'].sock)

    def test_connection_close_delayed_body_deadline(self):
        self.server.connection_close = True
        self.server.body_delay = .4
        c = self.client(http_timeout=.15)
        started = time.monotonic()
        with self.assertRaises(RelayError):
            c.open()
        self.assertEqual(c.last_failure['phase'], 'response-body')
        self.assertEqual(c.last_failure['error_type'], 'TimeoutError')
        self.assertLess(time.monotonic() - started, .35)

    def test_connection_close_body_trickle_cannot_renew_deadline(self):
        self.server.connection_close = True
        self.server.body_trickle = True
        c = self.client(http_timeout=.2)
        started = time.monotonic()
        with self.assertRaises(RelayError):
            c.open()
        self.assertEqual(c.last_failure['phase'], 'response-body')
        self.assertEqual(c.last_failure['error_type'], 'TimeoutError')
        self.assertIsNone(c._connections['claim'].sock)
        self.assertLess(time.monotonic() - started, .45)

    def test_explicit_open_headers_and_redaction(self):
        c = self.client()
        self.assertEqual(self.server.calls, [])
        self.assertNotIn('hf_secret', repr(c))
        self.assertNotIn('b' * 64, repr(c))
        with self.assertRaises(RelayError):
            c.send(b'no')
        self.assertIs(c.open(), c)
        headers = self.server.calls[0][3]
        self.assertEqual(headers['Authorization'], 'Bearer hf_secret')
        self.assertEqual(headers['X-H3-Role'], 'producer')
        with self.assertRaises(RelayError):
            c.open()
        self.assertEqual(len(self.server.calls), 1)

    def test_url_and_types(self):
        for url in ['https://hf.space', 'https://x.hf.space.evil',
                    'https://user@x.hf.space', 'https://x.hf.space/',
                    'https://x.hf.space?', 'https://x.hf.space#',
                    'https://x.hf.space:444', 'http://x.hf.space',
                    'http://127.0.0.1:1234']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                SocketAdapter(url, session='s', not_after=time.time()+10,
                              role='producer', role_key='a'*64, hf_token='secret')
        for args in [dict(not_after=True), dict(deadline=float('inf')),
                     dict(hf_token='secret\r\nx'), dict(role_key='x'*64),
                     dict(http_timeout=0), dict(session='bad\n')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.client(**args)

    def test_no_redirect_or_claim_retry(self):
        for status in (409, 302, 500):
            self.server.claim_status = status
            c = self.client()
            before = len(self.server.calls)
            with self.assertRaises(RelayError):
                c.open()
            with self.assertRaises(RelayError):
                c.open()
            self.assertEqual(len(self.server.calls), before+1)

    def test_wait_cache_and_paths(self):
        c = self.client().open()
        self.server.reads.extend([(204, b'', self.server.boot),
                                 (200, b'abcdefghij', self.server.boot)])
        self.assertEqual(c.recv(6), b'abcdef')
        self.assertEqual(c.recv(4), b'ghij')
        gets = [x for x in self.server.calls if x[0] == 'GET']
        self.assertEqual([x[1] for x in gets], ['/v1/ack', '/v1/ack'])

    def test_coalesce_drain_without_abort(self):
        c = self.client().open()
        for _ in range(100):
            self.assertEqual(c.send(b'hello'), 5)
        c.shutdown()
        c.close()
        writes = [x for x in self.server.calls if x[1] == '/v1/av']
        self.assertEqual(b''.join(x[2] for x in writes), b'hello'*100)
        self.assertLess(len(writes), 100)
        self.assertFalse(any(x[1] == '/v1/abort' for x in self.server.calls))

    def test_boot_size_encoding_fail_closed(self):
        for mode in ('boot', 'size', 'encoding'):
            c = self.client().open()
            boot = 'c'*32 if mode == 'boot' else self.server.boot
            self.server.reads.append((200, b'x'*(MAX_BODY+1) if mode == 'size' else b'x', boot))
            self.server.encoded = mode == 'encoding'
            with self.assertRaises(RelayError):
                c.recv(1)
            with self.assertRaises(RelayError):
                c.send(b'x')
            c.close()
            self.server.encoded = False

    def test_deadline_and_drift(self):
        c = self.client(deadline=time.monotonic()+.1).open()
        old = c._deadline
        c.settimeout(100)
        self.assertEqual(c._deadline, old)
        with self.assertRaises((TimeoutError, RelayError)):
            c.recv(1)
        d = self.client().open()
        with patch('research.h3_personal.live_relay_client.time.time', return_value=time.time()+5):
            with self.assertRaises(RelayError):
                d.recv(1)

    def test_bounded_queue_and_writer_failure(self):
        c = self.client().open()
        self.server.write_status = 500
        # Holding the condition excludes the writer while testing the hard bound.
        with c._cv:
            self.assertEqual(c.send(b'x' * (2*MAX_BODY+1)), 2*MAX_BODY)
            self.assertEqual(c._queued, 2*MAX_BODY)
        end = time.monotonic()+1
        while c._error is None and time.monotonic() < end:
            time.sleep(.005)
        with self.assertRaises(RelayError):
            c.recv(1)
        c.close()
        self.assertEqual(sum(x[1] == '/v1/av' for x in self.server.calls), 1)

    def test_network_failure_has_no_retry(self):
        c = self.client()
        with patch('research.h3_personal.live_relay_client.http.client.HTTPConnection.request',
                   side_effect=OSError('private secret')) as request:
            with self.assertRaises(RelayError) as caught:
                c.open()
            self.assertNotIn('private secret', str(caught.exception))
            self.assertEqual(request.call_count, 1)

    def test_input_and_explicit_abort(self):
        c = self.client(role='consumer').open()
        self.assertEqual(c.send_input(b'input'), 5)
        c.send(b'ack')
        c.close()
        self.assertTrue(any(x[1] == '/v1/ack' and x[0] == 'POST' for x in self.server.calls))
        d = self.client().open()
        self.server.reads.append((200, b'input', self.server.boot))
        self.assertEqual(d.read_input(), b'input')
        d.abort()
        d.abort()
        self.assertEqual(sum(x[1] == '/v1/abort' for x in self.server.calls), 1)


if __name__ == '__main__':
    unittest.main()
