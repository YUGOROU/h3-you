"""Local socketpair only: multi-segment finalization, no GPU/device/network."""
import json
import socket
import struct
import threading
import unittest

from research.h3_personal.endpoint_asr import transcribe, FINAL, PARTIAL, native_completed_text, finalized_covers
from test_h3_endpoint_asr_adapter import server_frame


def exact(stream, n):
    out = b''
    while len(out) < n:
        part = stream.read(n - len(out))
        if not part:
            raise EOFError
        out += part
    return out


def client_frame(stream):
    a, b = exact(stream, 2)
    n = b & 127
    if n == 126:
        n = struct.unpack('!H', exact(stream, 2))[0]
    elif n == 127:
        n = struct.unpack('!Q', exact(stream, 8))[0]
    key = exact(stream, 4)
    data = exact(stream, n)
    return a & 15, bytes(v ^ key[i % 4] for i, v in enumerate(data))


class WholeInputTests(unittest.TestCase):
    def exercise(self, coverage=True, text=True, float32=False):
        client, server = socket.socketpair()
        server.settimeout(3)
        received = []
        failures = []

        def serve():
            try:
                with server, server.makefile('rb') as stream:
                    while True:
                        op, payload = client_frame(stream)
                        if op == 2:
                            received.append(payload)
                            if len(received) == 1:
                                server.sendall(server_frame(dict(type=FINAL, transcript='こんにちは', audio_processed=.01)))
                            elif len(received) == 2:
                                server.sendall(server_frame(dict(type=PARTIAL, delta='discard this partial')))
                                if text:
                                    server.sendall(server_frame(dict(type=FINAL, transcript='日本で最も多くの従業員を抱える企業は何ですか', audio_processed=.02)))
                        elif op == 1 and json.loads(payload).get('type') == 'input_audio_buffer.commit':
                            # Empty flush acknowledges full coverage, never adds text.
                            server.sendall(server_frame(dict(type=FINAL, transcript='', audio_processed=(.029999999329447746 if float32 else .03) if coverage else .02)))
                            return
            except Exception as exc:
                failures.append(type(exc).__name__)

        worker = threading.Thread(target=serve)
        worker.start()
        try:
            result = transcribe('https://test.endpoints.huggingface.cloud', token='test', owner_key='test',
                                pcm=[bytes(320)] * 3, timeout=2, completion_policy='input-finalized',
                                _transport=lambda *args: (client, client.makefile('rb')))
        finally:
            worker.join(4)
            client.close()
            server.close()
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures)
        self.assertEqual(len(received), 3, 'early final must not stop PCM input')
        return result

    def test_joins_segments_only_after_full_commit_coverage(self):
        r = self.exercise()
        self.assertTrue(r['final'])
        self.assertEqual(r['errors'], [])
        self.assertEqual(r['text'], 'こんにちは 日本で最も多くの従業員を抱える企業は何ですか')
        self.assertTrue(r['input_committed'])
        self.assertEqual(r['finalized_audio_seconds'], r['audio_sent_seconds'])
        self.assertTrue(r['cleanup_complete'])

    def test_incomplete_coverage_does_not_promote_earlier_final(self):
        r = self.exercise(coverage=False)
        self.assertFalse(r['final'])
        self.assertIsNone(r['text'])
        self.assertTrue(r['errors'])

    def test_float32_flush_completes_without_deadline(self):
        r = self.exercise(float32=True)
        self.assertTrue(r['final'])
        self.assertEqual(r['errors'], [])
        self.assertTrue(finalized_covers(r['finalized_audio_seconds'], r['audio_sent_seconds']))

    def test_partial_is_not_aggregated(self):
        r = self.exercise(text=False)
        self.assertEqual(r['text'], 'こんにちは')
        self.assertNotIn('discard', r['text'])

    def test_repeated_spoken_segments_preserved(self):
        e = dict(final=True, text='はい', endpoint_source='native_completed')
        self.assertEqual(native_completed_text([e, e]), 'はい はい')

    def test_float32_terminal_coverage_uses_pcm_sample_units(self):
        self.assertTrue(finalized_covers(44.98099899291992, 44.981))
        self.assertFalse(finalized_covers(44.9809, 44.981))
        self.assertFalse(finalized_covers(44.98, 44.981))
        for value in (float('nan'), float('inf'), -1, True, None):
            self.assertFalse(finalized_covers(value, 44.981))

    def test_unknown_policy_rejected_before_transport(self):
        with self.assertRaises(ValueError):
            transcribe('bad', token='bad', owner_key='bad', pcm=[], completion_policy='guess')


if __name__ == '__main__':
    unittest.main()
