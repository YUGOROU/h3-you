import hashlib
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from research.h3_personal.reference_candidate import audit_and_stage, _header, _rgb_png


def rgba_png(alpha):
    def chunk(k, p):
        return struct.pack('>I', len(p)) + k + p + struct.pack('>I', zlib.crc32(k + p))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', len(alpha), 1, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0' + b''.join(bytes([17, 42, 99, a]) for a in alpha)))
            + chunk(b'IEND', b''))


@unittest.skipUnless(shutil.which('ffmpeg'), 'existing ffmpeg required')
class CandidateTests(unittest.TestCase):
    def run_case(self, alpha):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source = root / 'input.png'
        original = rgba_png(alpha)
        source.write_bytes(original)
        before = source.stat().st_mtime_ns
        destination = root / 'candidate'
        result = audit_and_stage(source, destination)
        self.addCleanup(destination.chmod, 0o700)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(source.stat().st_mtime_ns, before)
        self.assertEqual((destination / 'source.png').read_bytes(), original)
        self.assertEqual(result['source_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(destination.stat().st_mode & 0o777, 0o500)
        with self.assertRaises(FileExistsError):
            audit_and_stage(source, destination)
        return result

    def test_opaque_lossless(self):
        result = self.run_case([255, 255])
        self.assertTrue(result['rgb_round_trip_verified'])
        self.assertEqual(result['alpha_counts']['opaque_255'], 2)

    def test_no_flatten(self):
        result = self.run_case([0, 128, 255])
        self.assertIsNone(result['candidate_path'])
        self.assertEqual(result['alpha_counts'], {'opaque_255': 1, 'transparent_0': 1, 'intermediate_1_254': 1})


class BoundsTests(unittest.TestCase):
    def test_pixel_bound_before_decoder(self):
        data = _rgb_png(1, 1, b'abc')
        payload = struct.pack('>IIBBBBB', 10001, 1000, 8, 6, 0, 0, 0)
        data = data[:16] + payload + struct.pack('>I', zlib.crc32(b'IHDR' + payload)) + data[33:]
        with self.assertRaises(ValueError):
            _header(data)

    def test_source_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'input.png'
            source.write_bytes(b'12345')
            with patch('research.h3_personal.reference_candidate.MAX_SOURCE', 4):
                with self.assertRaises(ValueError):
                    audit_and_stage(source, Path(temp) / 'out')
            self.assertFalse((Path(temp) / 'out').exists())

    def test_invalid_header(self):
        with self.assertRaises(ValueError):
            _header(b'not png')
