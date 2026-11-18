"""Deterministic HTTP framing/deadline tests; no external network."""
import http.client
import io
import unittest
from unittest.mock import patch

from research.h3_personal.live_relay_client import SocketAdapter, RelayError
from research.h3_personal.local_av_diagnostics import failure_evidence


class RawSocket:
    def __init__(self, raw):
        self.raw = raw
    def makefile(self, mode):
        return self.raw
    def settimeout(self, value):
        pass


class Partial(io.BytesIO):
    def read1(self, size=-1):
        if self.tell() == len(self.getvalue()):
            raise TimeoutError('SECRET payload URL token')
        return super().read1(size)


class Connection:
    def __init__(self, response):
        self.response = response
        self.sock = RawSocket(None)
    def request(self, *args, **kwargs):
        pass
    def getresponse(self):
        return self.response


class FailureEvidenceTests(unittest.TestCase):
    def adapter(self):
        return SocketAdapter('http://127.0.0.1:1234', session='private',
            not_after=1100, role='consumer', role_key='a'*64,
            hf_token='SECRET', allow_test_loopback=True)

    def response(self, body, length):
        raw = Partial(b'HTTP/1.1 200 OK\r\nX-H3-Boot: ' + b'a'*32 +
            b'\r\nCache-Control: no-store\r\nContent-Length: ' + str(length).encode() +
            b'\r\n\r\n' + body)
        response = http.client.HTTPResponse(RawSocket(raw))
        response.begin()
        return response

    @patch('research.h3_personal.live_relay_client.time.time', return_value=1000)
    @patch('research.h3_personal.live_relay_client.time.monotonic', return_value=100)
    def test_stalled_partial_body(self, mono, wall):
        adapter = self.adapter()
        adapter._connections['read'] = Connection(self.response(b'ab', 4))
        with self.assertRaises(RelayError):
            adapter._request('read', 'GET', '/secret?token=SECRET')
        evidence = adapter.last_failure
        self.assertEqual(evidence, failure_evidence(evidence))
        self.assertEqual(evidence['bytes_read'], 2)
        self.assertEqual(evidence['content_length'], 4)
        self.assertEqual(evidence['framing'], 'content-length')
        self.assertEqual(evidence['phase'], 'response-body')
        self.assertEqual(evidence['request_timeout_seconds'], 10)
        self.assertNotIn('SECRET', str(evidence))
        self.assertTrue(adapter._error)

    @patch('research.h3_personal.live_relay_client.time.time', return_value=1000)
    @patch('research.h3_personal.live_relay_client.time.monotonic', return_value=100)
    def test_exact_persistent_body_no_extra_deadline_read(self, mono, wall):
        adapter = self.adapter()
        response = self.response(b'ab', 2)
        adapter._connections['read'] = Connection(response)
        original = response.read1
        def read(size):
            data = original(size)
            # At the exact HTTP request deadline after the complete body arrives.
            mono.return_value = 110
            wall.return_value = 1010
            return data
        response.read1 = read
        self.assertEqual(adapter._request('read', 'GET', '/v1/av'), (200, b'ab'))
        self.assertFalse(response.will_close)

    def test_strict_sanitization_and_legacy(self):
        legacy = dict(phase='response-body', http_status=200, error_type='TimeoutError')
        self.assertEqual(failure_evidence(legacy), legacy)
        dirty = dict(legacy, lane='SECRET', method='GET SECRET', framing='SECRET',
            content_length=True, bytes_read=524290, elapsed_seconds=float('nan'),
            request_timeout_seconds=float('inf'), headers_elapsed_seconds=-1,
            last_progress_age_seconds='SECRET', url='SECRET', payload='SECRET')
        self.assertEqual(failure_evidence(dirty), legacy)
