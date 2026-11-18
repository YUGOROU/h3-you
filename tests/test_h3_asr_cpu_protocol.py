"""Offline RFC6455 helpers and plan: no network, audio/device/model execution."""
import contextlib
import io
import json
import struct
import sys
import unittest
from unittest.mock import patch
from research import probe_h3_asr_cpu_space as probe


class ProtocolTests(unittest.TestCase):
    def test_client_frames_masked(self):
        for size in (0, 20, 160, 5120, 65536):
            payload = bytes(i % 251 for i in range(size))
            encoded = probe.frame(payload)
            self.assertEqual(encoded[0], 0x82)
            self.assertTrue(encoded[1] & 0x80)
            offset = 2 + (2 if size >= 126 and size < 65536 else 8 if size >= 65536 else 0)
            mask = encoded[offset:offset+4]
            actual = bytes(v ^ mask[i % 4] for i, v in enumerate(encoded[offset+4:]))
            self.assertEqual(actual, payload)
        with self.assertRaises(ValueError):
            probe.frame(bytes(65537))

    def test_server_frames_and_limits(self):
        self.assertEqual(probe.recv_frame(io.BytesIO(b'\x81\x02{}')), (True, 1, b'{}'))
        payload = b'x' * 160
        self.assertEqual(probe.recv_frame(io.BytesIO(b'\x81\x7e'+struct.pack('!H',160)+payload))[2], payload)
        for bad in (b'\x81\x80', b'\xc1\x00', b'\x81\x7f'+struct.pack('!Q',65537)):
            with self.assertRaises(ValueError):
                probe.recv_frame(io.BytesIO(bad))
        with self.assertRaises(EOFError):
            probe.read_exact(io.BytesIO(b'a'), 2)

    def test_last_final_and_progress_not_confused_with_first_segment(self):
        rows = [dict(at_seconds=2, event=dict(type='conversation.item.input_audio_transcription.delta', delta='And', audio_processed=1)),
                dict(at_seconds=4, event=dict(type='conversation.item.input_audio_transcription.completed', transcript='And so', audio_processed=3)),
                dict(at_seconds=14, event=dict(type='conversation.item.input_audio_transcription.completed', transcript='final words', audio_processed=11))]
        result = probe.summarize(rows, 11, 12)
        self.assertEqual(result['first_final_seconds'], 4)
        self.assertEqual(result['last_final_seconds'], 14)
        self.assertEqual(result['final_after_input_end_seconds'], 2)
        self.assertAlmostEqual(result['observed_progress_wall_per_audio_second'], 1.2)
        self.assertEqual(result['final_segments'], ['And so', 'final words'])

    def test_offline_plan_no_network(self):
        out = io.StringIO()
        with patch.object(sys,'argv',['probe','--directory','/missing','--sample-directory','/missing']), contextlib.redirect_stdout(out):
            probe.main()
        self.assertFalse(json.loads(out.getvalue())['device_capture'])
        self.assertTrue(json.loads(out.getvalue())['pause_after_test'])


if __name__ == '__main__':
    unittest.main()
