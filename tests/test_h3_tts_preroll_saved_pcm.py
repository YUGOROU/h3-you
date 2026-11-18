"""Synthetic private-free fixtures; no devices, cloud, sleeps or source writes."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from research.h3_personal.tts_preroll import PCMPrerollPolicy
from research.h3_personal.tts_preroll_saved_pcm import verify_saved_case


def chunk(tag, body):
    return tag + struct.pack('<I', len(body)) + body + (b'\0' if len(body) & 1 else b'')


class SavedPCMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / 'synthetic.wav'
        self.data = struct.pack('<8f', 0, .25, -.25, 1, -1, 1.5, -.5, .125)
        self.fmt = struct.pack('<HHIIHH', 3, 1, 48000, 192000, 4, 32)
        self.save()
        self.row = dict(sample_rate=48000, total_samples=8, duration_seconds=8/48000,
                        pcm_sha256=hashlib.sha256(self.data).hexdigest(),
                        chunks=[dict(samples=0, arrival_seconds=.05),
                                dict(samples=4, arrival_seconds=.1),
                                dict(samples=4, arrival_seconds=.2)],
                        iterator_completion_seconds=.3)
        self.policy = PCMPrerollPolicy(reserve_samples=4)

    def save(self):
        body = b'WAVE' + chunk(b'JUNK', b'odd') + chunk(b'fmt ', self.fmt) + chunk(b'data', self.data)
        self.path.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)

    def test_threshold_before_end_preserves_original_and_separate_conversion(self):
        original = self.path.read_bytes()
        result = verify_saved_case(self.row, self.path, self.policy)
        self.assertEqual(result['first_published'], dict(time=.1, samples=4))
        receipt = result['receipt']
        self.assertEqual(receipt['native_sha256'], self.row['pcm_sha256'])
        expected = struct.pack('<8h', 0, 8192, -8192, 32767, -32768, 32767, -16384, 4096)
        self.assertEqual(receipt['converted_sha256'], hashlib.sha256(expected).hexdigest())
        self.assertEqual(receipt['converted_samples'], 8)
        self.assertEqual(receipt['saturation_count'], 2)
        self.assertTrue(result['virtual_replay'])
        self.assertFalse(result['actual_playback'])
        self.assertFalse(result['actual_model_run'])
        self.assertFalse(result['model_eos_verified'])
        self.assertEqual(self.path.read_bytes(), original)

    def test_short_completion(self):
        result = verify_saved_case(self.row, self.path, PCMPrerollPolicy(reserve_samples=16))
        self.assertEqual(result['first_published'], dict(time=.3, samples=8))
        self.assertTrue(result['receipt']['preroll']['short_sealed'])
        self.assertFalse(result['receipt']['model_eos_verified'])

    def test_bad_hash_count_duration_rate(self):
        for key, value in [('pcm_sha256', '0'*64), ('total_samples', 9),
                           ('duration_seconds', .1), ('sample_rate', 44100),
                           ('total_samples', True)]:
            with self.subTest(key=key):
                row = dict(self.row, **{key: value})
                with self.assertRaises(ValueError):
                    verify_saved_case(row, self.path, self.policy)

    def test_invalid_timing_including_empty_chunks(self):
        for arrival in [-1, float('nan'), 121, .25, True]:
            row = copy.deepcopy(self.row)
            row['chunks'][0]['arrival_seconds'] = arrival
            with self.subTest(arrival=arrival), self.assertRaises(ValueError):
                verify_saved_case(row, self.path, self.policy)
        for end in [.15, 121, float('inf')]:
            with self.subTest(end=end), self.assertRaises(ValueError):
                verify_saved_case(dict(self.row, iterator_completion_seconds=end), self.path, self.policy)
        row = dict(self.row, chunks=[dict(samples=0, arrival_seconds=0)]*1025)
        with self.assertRaises(ValueError):
            verify_saved_case(row, self.path, self.policy)

    def test_wrong_format_and_truncated_padding(self):
        self.fmt = struct.pack('<HHIIHH', 1, 1, 48000, 192000, 4, 32)
        self.save()
        with self.assertRaises(ValueError):
            verify_saved_case(self.row, self.path, self.policy)
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            verify_saved_case(self.row, self.path, self.policy)

    def test_nonfinite_rejected_by_bridge(self):
        self.data = struct.pack('<8f', *([float('nan')]*8))
        self.save()
        self.row['pcm_sha256'] = hashlib.sha256(self.data).hexdigest()
        with self.assertRaisesRegex(ValueError, 'finite float32'):
            verify_saved_case(self.row, self.path, self.policy)

    def test_symlinks_rejected(self):
        link = self.path.parent / 'link.wav'
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            verify_saved_case(self.row, link, self.policy)
        parent = self.path.parent / 'linked'
        parent.symlink_to(self.path.parent, target_is_directory=True)
        with self.assertRaises(OSError):
            verify_saved_case(self.row, parent / self.path.name, self.policy)


if __name__ == '__main__':
    unittest.main()
