import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal import paired_av_wire as wire
from research.h3_personal.transport_replay_preflight import (
    ReplayPreflightError, capabilities, inspect_capture,
)
from research.run_h3_transport_preflight import main


class ReplayPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binding = LiveAVBinding('PRIVATE_SESSION', '0' * 64, 1)
        for i in range(2):
            e = encode(self.binding, i, i, i+1, b'video', b'\0' * 4000,
                       (AudioRegion('silence', i*2000, (i+1)*2000),), i == 1)
            (self.root / f'{i:03d}.json').write_text(json.dumps(wire.envelope_record(e)))
            (self.root / f'{i:03d}.mp4').write_bytes(b'video')
            (self.root / f'{i:03d}.pcm').write_bytes(b'\0' * 4000)

    def change_header(self, index, mutate):
        p = self.root / f'{index:03d}.json'
        value = json.loads(p.read_text())
        mutate(value)
        p.write_text(json.dumps(value))

    def test_exact_sequence_without_transport_or_private_identity(self):
        r = inspect_capture(self.root)
        self.assertEqual((r['frame_count'], r['sample_count']), (2, 4000))
        self.assertEqual((r['video_bytes'], r['pcm_bytes']), (10, 8000))
        self.assertTrue(r['source_validation_complete'])
        for key in ('transport_executed', 'video_decoded', 'playback_verified'):
            self.assertIs(r[key], False)
        self.assertNotIn('PRIVATE_SESSION', json.dumps(r))
        self.assertNotIn(str(self.root), json.dumps(r))

    def test_corrupt_body_hash_rejected(self):
        (self.root / '000.mp4').write_bytes(b'wrong')
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_truncated_pcm_rejected(self):
        (self.root / '000.pcm').write_bytes(b'\0')
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_reordered_or_missing_triple_rejected(self):
        (self.root / '001.pcm').rename(self.root / '002.pcm')
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_final_missing_and_early_final_rejected(self):
        for index in (0, 1):
            with self.subTest(index=index):
                self.change_header(index, lambda v: v.update(conversation_final=index == 0))
                with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)
                self.change_header(index, lambda v: v.update(conversation_final=index == 1))

    def test_duplicate_json_field_rejected(self):
        p = self.root / '000.json'
        p.write_text(p.read_text()[:-1] + ', "index": 0}')
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_binding_change_rejected(self):
        self.change_header(1, lambda v: v['binding'].update(session='OTHER'))
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_header_size_bounded(self):
        (self.root / '000.json').write_bytes(b' ' * 16385)
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_total_byte_bound(self):
        with patch('research.h3_personal.transport_replay_preflight.MAX_CAPTURE_BYTES', 5000):
            with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_symlink_body_rejected(self):
        p = self.root / '000.mp4'
        p.unlink()
        p.symlink_to(self.root / '001.mp4')
        with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_nonfinite_timeout_and_expiry(self):
        for value in (0, True, float('inf'), float('nan'), 61):
            with self.subTest(value=value):
                with self.assertRaises(ReplayPreflightError): inspect_capture(self.root, timeout_seconds=value)
        with patch('research.h3_personal.transport_replay_preflight.time.monotonic', side_effect=[0, 31]):
            with self.assertRaises(ReplayPreflightError): inspect_capture(self.root)

    def test_dependency_probe_top_level_only(self):
        with patch('research.h3_personal.transport_replay_preflight.importlib.util.find_spec', return_value=None) as probe:
            self.assertTrue(all(value is False for value in capabilities().values()))
            self.assertTrue(all('.' not in call.args[0] for call in probe.call_args_list))

    def test_cli_failure_does_not_print_source_or_exception(self):
        stream = io.StringIO()
        with patch('research.run_h3_transport_preflight.capabilities', return_value={}):
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(['--capture-dir', str(self.root / 'PRIVATE_MISSING')]), 1)
        self.assertNotIn('PRIVATE', stream.getvalue())
        self.assertNotIn(str(self.root), stream.getvalue())


if __name__ == '__main__':
    unittest.main()
