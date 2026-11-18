"""Offline mocked-transport contract tests; no device or network access."""
import io
import json
import socket
import threading
import unittest
from unittest.mock import patch

from research.h3_personal.endpoint_asr import transcribe, PARTIAL, FINAL


def server_frame(row, opcode=1, fin=True):
    payload = json.dumps(row).encode() if isinstance(row, dict) else row
    if len(payload) >= 126:
        return bytes([(128 if fin else 0) | opcode, 126]) + len(payload).to_bytes(2, 'big') + payload
    return bytes([(128 if fin else 0) | opcode, len(payload)]) + payload


class Connection:
    def __init__(self):
        self.sent = []
        self.closed = False
        self.shutdown_called = threading.Event()

    def sendall(self, data):
        self.sent.append(data)

    def settimeout(self, value):
        self.timeout = value

    def shutdown(self, how):
        self.shutdown_called.set()

    def close(self):
        self.closed = True


class AdapterTests(unittest.TestCase):
    def run_adapter(self, data, **kwargs):
        connection = Connection()
        source = io.BytesIO(data)
        def transport(host, token, owner, timeout):
            self.assertEqual(host, 'test.endpoints.huggingface.cloud')
            self.assertEqual((token, owner), ('hf_private', 'owner_private'))
            return connection, source
        result = transcribe('https://test.endpoints.huggingface.cloud',
                            token='hf_private', owner_key='owner_private',
                            pcm=[bytes(320)], _transport=transport, **kwargs)
        self.assertTrue(connection.closed)
        self.assertTrue(source.closed)
        self.assertTrue(result['cleanup_complete'])
        self.assertNotIn('private', repr(result))
        return result, connection

    def test_first_nonempty_native_final(self):
        data = b''.join(server_frame(row) for row in [
            {'type': PARTIAL, 'delta': 'hello'},
            {'type': FINAL, 'transcript': '  '},
            {'type': FINAL, 'transcript': 'hello world'},
            {'type': FINAL, 'transcript': 'must not consume'}])
        result, _ = self.run_adapter(data)
        self.assertEqual(result['text'], 'hello world')
        self.assertEqual(len(result['events']), 3)
        self.assertFalse(result['events'][0]['final'])
        self.assertFalse(result['events'][1]['final'])
        self.assertTrue(result['events'][2]['final'])
        self.assertFalse(result['capture_stopped'])

    def test_partial_never_promoted(self):
        result, _ = self.run_adapter(server_frame({'type': PARTIAL, 'delta': 'only partial'}))
        self.assertFalse(result['final'])
        self.assertIsNone(result['text'])

    def test_server_error_redacted(self):
        result, _ = self.run_adapter(server_frame({'type': 'error', 'message': 'owner_private'}))
        self.assertEqual(result['errors'], ['server_error'])

    def test_fragmentation_and_ping(self):
        payload = json.dumps({'type': FINAL, 'transcript': 'yes'}).encode()
        result, connection = self.run_adapter(
            server_frame(payload[:10], fin=False) + server_frame(b'p', opcode=9)
            + server_frame(payload[10:], opcode=0))
        self.assertEqual(result['text'], 'yes')
        self.assertTrue(any(frame[0] == 138 for frame in connection.sent))

    def test_invalid_fragment(self):
        result, _ = self.run_adapter(server_frame(b'{}', opcode=0))
        self.assertIn('transport_or_protocol_error', result['errors'])

    def test_event_bound(self):
        result, _ = self.run_adapter(server_frame({'type': 'progress'}) * 3, max_events=2)
        self.assertFalse(result['final'])
        self.assertIn('transport_or_protocol_error', result['errors'])

    def test_reject_endpoint_and_credentials_before_transport(self):
        for endpoint in ['http://a.huggingface.cloud', 'https://huggingface.cloud.evil',
                         'https://a.hf.space', 'https://user@a.huggingface.cloud',
                         'https://a.huggingface.cloud?x=1', 'https://a.huggingface.cloud:444']:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                transcribe(endpoint, token='secret', owner_key='owner', pcm=[])
        with self.assertRaises(ValueError):
            transcribe('https://a.huggingface.cloud', token='x\r\ny', owner_key='owner', pcm=[])

    def test_blocked_transport_has_bounded_return(self):
        release = threading.Event()
        closed = threading.Event()
        def transport(*args):
            release.wait(2)
            connection = Connection()
            original = connection.close
            def close():
                original()
                closed.set()
            connection.close = close
            return connection, io.BytesIO()
        try:
            result = transcribe('https://a.huggingface.cloud', token='secret',
                                owner_key='owner', pcm=[], timeout=.02,
                                join_timeout=.02, _transport=transport)
            self.assertIn('deadline', result['errors'])
            self.assertFalse(result['cleanup_complete'])
        finally:
            release.set()
        self.assertTrue(closed.wait(1))


if __name__ == '__main__':
    unittest.main()
