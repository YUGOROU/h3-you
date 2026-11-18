import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import socket
import threading
import http.client
from unittest.mock import patch
import time
import unittest
import wave

from research.h3_personal.irodori_endpoint_client import (
    RemoteIrodoriTTS, IrodoriEndpointError, MAX_RESPONSE,
    SOURCE, MODEL, CODEC, REFERENCE_SHA,
)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.ready = dict(schema='h3-irodori-endpoint-ready-v1', session='s',
                          device_name='NVIDIA L4', device_count=1, model_precision='fp32',
                          codec_precision='fp32', compile_model=False, watermark_ready=True,
                          source_revision=SOURCE, model_revision=MODEL, codec_revision=CODEC,
                          reference_sha256=REFERENCE_SHA)
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as wav:
            wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\0\0' * 240)
        self.audio = stream.getvalue()
        self.response = dict(schema='h3-irodori-endpoint-response-v1', session='s', epoch=1,
                             wav_base64=base64.b64encode(self.audio).decode(),
                             receipt=dict(source_revision=SOURCE, model_revision=MODEL,
                                          codec_revision=CODEC, watermark_applied=True,
                                          text_sha256=hashlib.sha256(b'hello').hexdigest(),
                                          sha256=hashlib.sha256(self.audio).hexdigest(),
                                          samples=240, sample_rate=24000, channels=1))
        self.status = 200
        self.raw = None
        self.client = self.make()
        self.deadline = time.monotonic() + 300

    def make(self, url='https://unit.endpoints.huggingface.cloud', **kwargs):
        return RemoteIrodoriTTS(url, token='secret', key='private', session='s',
                                expires_at=time.time() + 600, transport=self.transport, **kwargs)

    def transport(self, method, path, payload, headers, deadline):
        self.calls.append((method, path, payload, headers, deadline))
        return self.status, self.raw if self.raw is not None else json.dumps(
            self.ready if method == 'GET' else self.response).encode()

    def read(self):
        return self.client.read('tts_ready', self.deadline)

    def synth(self, path):
        return self.client.synthesize('hello', 'turn', path, self.deadline)

    def wire(self, response, *, hold=False):
        client_sock, server_sock = socket.socketpair()
        release = threading.Event()
        sent = threading.Event()
        def server():
            try:
                server_sock.recv(65536)
                server_sock.sendall(response)
                sent.set()
                if hold:
                    release.wait(3)
            finally:
                server_sock.close()
        thread = threading.Thread(target=server)
        thread.start()
        self.addCleanup(thread.join, 4)
        self.addCleanup(release.set)
        conn = http.client.HTTPConnection('unused')
        conn.connect = lambda: setattr(conn, 'sock', client_sock)
        self.client._transport = self.client._http
        return conn, sent

    def test_framed_complete_without_eof_read(self):
        conn, _ = self.wire(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}', hold=True)
        with patch('research.h3_personal.irodori_endpoint_client.http.client.HTTPSConnection', return_value=conn):
            self.assertEqual(self.client._http('GET', '/ready', None, {}, time.monotonic() + .5), (200, b'{}'))

    def test_wire_rejects_invalid_bodies(self):
        cases = [b'Content-Length: 3\r\n\r\n{}',
                 b'Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}',
                 b'Content-Length: 2\r\nContent-Length: 3\r\n\r\n{}',
                 b'Content-Length: +2\r\n\r\n{}',
                 b'Content-Length: 2, 2\r\n\r\n{}',
                 b'Content-Encoding: gzip\r\nContent-Length: 2\r\n\r\n{}',
                 ('Content-Length: %d\r\n\r\n' % (MAX_RESPONSE + 1)).encode()]
        for body in cases:
            with self.subTest(body=body):
                conn, _ = self.wire(b'HTTP/1.1 200 OK\r\n' + body)
                with patch('research.h3_personal.irodori_endpoint_client.http.client.HTTPSConnection', return_value=conn):
                    with self.assertRaises(IrodoriEndpointError):
                        self.client._http('GET', '/', None, {}, self.deadline)

    def test_detached_socket_deadline_and_close_interrupt(self):
        for cancel in (False, True):
            self.setUp()
            conn, sent = self.wire(b'HTTP/1.0 200 OK\r\nContent-Length: 20\r\n\r\n{', hold=True)
            errors = []
            def run():
                try:
                    self.client.read('tts_ready', time.monotonic() + (.2 if not cancel else 10))
                except IrodoriEndpointError as exc:
                    errors.append(exc)
            with patch('research.h3_personal.irodori_endpoint_client.http.client.HTTPSConnection', return_value=conn):
                worker = threading.Thread(target=run)
                worker.start()
                self.assertTrue(sent.wait(1))
                if cancel:
                    self.client.close()
                worker.join(1)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(errors), 1)

    def test_success_and_epoch(self):
        self.assertEqual(self.read(), self.ready)
        with tempfile.TemporaryDirectory() as tmp:
            for epoch in (1, 2):
                self.response['epoch'] = epoch
                result = self.synth(Path(tmp) / str(epoch))
                path = Path(result['wav'])
                self.assertEqual(path.read_bytes(), self.audio)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(result['receipt']['path'], str(path))
                self.assertEqual(result['turn_id'], 'turn')
                self.assertEqual(json.loads(self.calls[-1][2]), dict(session='s', epoch=epoch, text='hello'))
        self.assertEqual(self.calls[0][:3], ('GET', '/ready', None))
        self.assertEqual(self.calls[0][3]['Authorization'], 'Bearer secret')
        self.assertEqual(self.calls[0][3]['X-H3-TTS-Key'], 'private')
        self.assertLessEqual(self.calls[0][4], time.monotonic() + 120)

    def test_url_restrictions(self):
        for url in ['http://unit.endpoints.huggingface.cloud',
                    'https://unit.endpoints.huggingface.cloud/',
                    'https://unit.endpoints.huggingface.cloud?x=1',
                    'https://user@unit.endpoints.huggingface.cloud',
                    'https://unit.endpoints.huggingface.cloud:443',
                    'https://unit.endpoints.huggingface.cloud.evil',
                    'https://endpoints.huggingface.cloud']:
            with self.subTest(url=url), self.assertRaises(IrodoriEndpointError):
                self.make(url)

    def test_ready_proof(self):
        for field, value in [('session', 'other'), ('device_name', 'NVIDIA T4'),
                             ('device_count', True), ('model_precision', 'bf16'),
                             ('codec_precision', 'fp16'), ('compile_model', True),
                             ('watermark_ready', False), ('reference_sha256', 'bad'),
                             ('source_revision', 'bad'), ('schema', 'bad')]:
            with self.subTest(field=field):
                self.setUp()
                self.ready[field] = value
                with self.assertRaises(IrodoriEndpointError):
                    self.read()
                with self.assertRaises(IrodoriEndpointError):
                    self.read()
                self.assertEqual(len(self.calls), 1)

    def test_once_unknown_and_close(self):
        self.read()
        with self.assertRaises(IrodoriEndpointError):
            self.read()
        self.assertEqual(len(self.calls), 1)
        self.setUp()
        with self.assertRaises(IrodoriEndpointError):
            self.client.read('unknown', self.deadline)
        self.assertEqual(self.calls, [])
        self.client.close()
        self.assertEqual(self.calls, [])

    def test_bad_response_terminal(self):
        for raw, status in [(b'not json', 200), (b'{}', 302),
                            (b'x' * (MAX_RESPONSE + 1), 200),
                            (b'{"session":"s","session":"s"}', 200)]:
            self.setUp()
            self.raw, self.status = raw, status
            with self.assertRaises(IrodoriEndpointError):
                self.read()
            with self.assertRaises(IrodoriEndpointError):
                self.read()
            self.assertEqual(len(self.calls), 1)

    def test_synthesis_integrity(self):
        for field, value in [('session', 'other'), ('epoch', 0), ('epoch', True),
                             ('wav_base64', '!!!!'), ('schema', 'bad')]:
            self.setUp(); self.read(); self.response[field] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'new'
                with self.assertRaises(IrodoriEndpointError):
                    self.synth(path)
                self.assertFalse(path.exists())
        for field in ('sha256', 'samples', 'sample_rate', 'channels', 'text_sha256',
                      'watermark_applied', 'source_revision', 'model_revision', 'codec_revision'):
            self.setUp(); self.read(); self.response['receipt'][field] = None
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(IrodoriEndpointError):
                    self.synth(Path(tmp) / 'new')

    def test_no_overwrite(self):
        self.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reply.wav'
            path.write_bytes(b'original')
            with self.assertRaises(IrodoriEndpointError):
                self.synth(tmp)
            self.assertEqual(path.read_bytes(), b'original')

    def test_deadline_expiry_and_transport_exception(self):
        for deadline in (float('inf'), float('nan'), time.monotonic() - 1):
            self.setUp()
            with self.assertRaises(IrodoriEndpointError):
                self.client.read('tts_ready', deadline)
            self.assertFalse(self.calls)
        self.setUp()
        self.client.expires_at = time.time() - 1
        with self.assertRaises(IrodoriEndpointError):
            self.read()
        self.assertFalse(self.calls)
        self.setUp()
        def fail(*args):
            raise RuntimeError('secret private raw payload')
        self.client._transport = fail
        with self.assertRaises(IrodoriEndpointError) as caught:
            self.read()
        self.assertNotIn('secret', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)


if __name__ == '__main__':
    unittest.main()
