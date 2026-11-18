import array
import io
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal.voxcpm_endpoint_protocol import decode
from research.h3_personal.voxcpm_endpoint_service import Gate, create_server
from research.h3_personal.voxcpm_private_gate import PrivateGate


class Candidate:
    sample_rate = 48000

    def stream(self, text):
        yield array.array('f', [0, .25, -.5])
        yield array.array('f', [.125])

    def export_evidence(self):
        return {'completions': [], 'native': True}


class Tests(unittest.TestCase):
    def setUp(self):
        self.config = dict(key='private-key', session='session', boot='boot',
                           expires_at=time.time()+60)
        self.server = create_server(Candidate(), host='127.0.0.1', port=0, **self.config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.gate = PrivateGate(port=self.server.server_port, **self.config)
        self.request = dict(session='session', boot='boot', epoch=1, text='private text')

    def tearDown(self):
        self.gate.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_native_byte_identity_and_decode(self):
        self.assertTrue(self.gate.healthy())
        self.assertIs(self.gate.candidate, self.gate)
        self.assertEqual(self.gate.ready(0), self.server.gate.ready(0))
        direct = Gate(Candidate(), **self.config)
        expected, actual = [], []
        direct.stream(self.request, expected.append)
        with patch.dict('os.environ', {'http_proxy': 'http://192.0.2.1:1',
                                       'HTTP_PROXY': 'http://192.0.2.1:1', 'NO_PROXY': ''}):
            self.gate.stream(self.request, actual.append)
        self.assertEqual(b''.join(actual), b''.join(expected))
        binding = {k: v for k, v in direct.ready(1).items()
                   if k in ('session', 'boot', 'expires_at', 'epoch')}
        pcm = list(decode(io.BytesIO(b''.join(actual)).read, binding))
        self.assertEqual(sum(map(len, pcm)), 16)
        self.assertEqual(self.gate.export_evidence(), Candidate().export_evidence())

    def test_ready_required_and_local_close(self):
        with self.assertRaises(ValueError):
            self.gate.stream(self.request, lambda _: None)
        self.assertFalse(self.gate.healthy())
        self.assertEqual(self.server.gate.count, 0)
        self.assertTrue(self.server.gate.healthy())

    def test_wrong_key_and_binding_are_sanitized(self):
        for field in ('key', 'session', 'boot', 'expires_at'):
            config = dict(self.config)
            config[field] = time.time()+50 if field == 'expires_at' else 'sensitive-wrong'
            gate = PrivateGate(port=self.server.server_port, **config)
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                gate.ready(0)
            self.assertNotIn('sensitive-wrong', str(caught.exception))
            self.assertNotIn(self.config['key'], str(caught.exception))
            self.assertFalse(gate.healthy())

    def test_control_strictness(self):
        for mutation in (dict(channels=True), dict(epoch=1), dict(extra='secret')):
            with patch.object(self.server.gate, 'ready', return_value=dict(self.server.gate.ready(0), **mutation)):
                gate = PrivateGate(port=self.server.server_port, **self.config)
                with self.assertRaises(ValueError):
                    gate.ready(0)
                self.assertFalse(gate.healthy())

    def test_evidence_binding_checked(self):
        self.gate.ready(0)
        with patch.object(self.server.gate, 'ready', return_value={}):
            with self.assertRaises(ValueError):
                self.gate.export_evidence()
        self.assertFalse(self.gate.healthy())

    def test_close_interrupts_active_stream(self):
        emitted, release = threading.Event(), threading.Event()
        class Slow(Candidate):
            def stream(self, text):
                yield array.array('f', [.25])
                release.wait(3)
                yield array.array('f', [.5])
        self.server.gate.candidate = Slow()
        self.gate.ready(0)
        errors = []
        def run():
            try:
                self.gate.stream(self.request, lambda _: emitted.set())
            except ValueError:
                errors.append(True)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            self.assertTrue(emitted.wait(2))
            self.gate.close()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [True])
        finally:
            release.set()
            worker.join(2)

    def test_stream_rejects_wrong_content_type(self):
        self.gate.ready(0)
        handler = self.server.RequestHandlerClass
        original = handler.send_header
        def altered(instance, name, value):
            return original(instance, name, 'text/plain' if name == 'Content-Type' else value)
        with patch.object(handler, 'send_header', altered):
            out = []
            with self.assertRaises(ValueError):
                self.gate.stream(self.request, out.append)
            self.assertEqual(out, [])
        self.assertFalse(self.gate.healthy())

    def test_epoch_zero_only(self):
        with self.assertRaises(ValueError):
            self.gate.ready(True)

    def test_invalid_configuration(self):
        for field, value in [('key', 'secret\r\nx'), ('session', ''), ('boot', None),
                             ('expires_at', float('nan')), ('expires_at', time.time()+1300),
                             ('expires_at', time.time()-1), ('port', True), ('port', 0),
                             ('max_requests', 0), ('max_requests', True)]:
            config = dict(self.config, port=self.server.server_port)
            config[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                PrivateGate(**config)

    def test_write_error_closes_without_leaking_text(self):
        self.gate.ready(0)
        def fail(data):
            raise RuntimeError('private-key private text')
        with self.assertRaises(ValueError) as caught:
            self.gate.stream(self.request, fail)
        self.assertNotIn('private-key', str(caught.exception))
        self.assertNotIn('private text', str(caught.exception))
        self.assertFalse(self.gate.healthy())


if __name__ == '__main__':
    unittest.main()
