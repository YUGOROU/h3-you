"""CPU-only real adapter event seam, injected WebSocket and event sink."""
import io
import threading
import unittest
from unittest.mock import patch

from test_h3_endpoint_asr_adapter import Connection, server_frame
from research.h3_personal.endpoint_asr import transcribe, PARTIAL, FINAL
from research.h3_personal.resident_endpoint import ResidentEndpoint


class EventsTests(unittest.TestCase):
    def run_stream(self, rows, sink, policy='input-finalized'):
        connection = Connection()
        committed = threading.Event()
        original = connection.sendall
        def send(data):
            original(data)
            # Session update then PCM then commit (masked wire payload).
            if len(connection.sent) == 3:
                committed.set()
        connection.sendall = send
        class Source(io.BytesIO):
            def read(self, size=-1):
                if not committed.wait(1):
                    raise AssertionError('commit missing')
                return super().read(size)
        source = Source(b''.join(server_frame(row) for row in rows))
        result = transcribe('https://test.endpoints.huggingface.cloud',
                            token='token', owner_key='key', pcm=[bytes(320)],
                            on_event=sink, completion_policy=policy,
                            _transport=lambda *args: (connection, source))
        self.assertTrue(connection.closed)
        self.assertTrue(source.closed)
        return result

    def test_order_isolation_and_terminal_coverage(self):
        seen = []
        def sink(event):
            seen.append(dict(event))
            event['text'] = 'mutated'
            event.get('errors', []).append('mutated')
        result = self.run_stream([
            {'type': PARTIAL, 'delta': 'hypothesis'},
            {'type': FINAL, 'transcript': 'speech', 'audio_processed': .01}], sink)
        self.assertEqual([e['event'] for e in seen], ['partial', 'native_completed', 'input_finalized'])
        self.assertTrue(seen[1]['provisional'])
        self.assertFalse(seen[1]['turn_final'])
        self.assertTrue(seen[2]['cleanup_complete'])
        self.assertEqual(seen[2]['audio_sent_seconds'], .01)
        self.assertEqual(result['text'], 'speech')
        self.assertEqual(result['events'][0]['text'], 'hypothesis')
        self.assertEqual(result['errors'], [])

    def test_no_terminal_on_eof_or_missing_coverage(self):
        for row in ({'type': PARTIAL, 'delta': 'p'},
                    {'type': FINAL, 'transcript': 'segment'}):
            seen = []
            result = self.run_stream([row], seen.append)
            self.assertEqual(len(seen), 1)
            self.assertFalse(result['final'])

    def test_bad_coverage_not_delivered(self):
        seen = []
        result = self.run_stream([{'type': FINAL, 'transcript': 'x',
                                   'audio_processed': True}], seen.append)
        self.assertEqual(seen, [])
        self.assertIn('transport_or_protocol_error', result['errors'])

    def test_callback_failure_cleanup(self):
        def fail(event):
            raise RuntimeError('private payload')
        result = self.run_stream([{'type': PARTIAL, 'delta': 'x'}], fail)
        self.assertEqual(result['errors'], ['event_callback_error'])
        self.assertTrue(result['cleanup_complete'])
        self.assertNotIn('private payload', repr(result))

    def test_invalid_callback_before_transport_or_capture(self):
        with patch('research.h3_personal.endpoint_asr._connect') as transport:
            with self.assertRaises(ValueError):
                transcribe('https://test.endpoints.huggingface.cloud', token='t',
                           owner_key='k', pcm=lambda: self.fail('capture'), on_event=3,
                           _transport=transport)
            transport.assert_not_called()
        with self.assertRaises(ValueError):
            ResidentEndpoint().capture(None, 0, on_asr_event=3)


if __name__ == '__main__':
    unittest.main()
