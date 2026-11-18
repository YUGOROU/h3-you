"""CPU-only spatial regression through the real live exporter and segment codec.

Synthetic grayscale coordinate tiles avoid chroma-subsampling ambiguity. Exact
pre-codec bytes establish the two-column crop; decoded tile interiors allow H264
loss, not rotations, flips, frame swaps, or flattened-buffer rearrangement.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'optional numpy is not installed in this interpreter')
@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'existing ffmpeg and ffprobe executables are required')
class LiveSpatialRoundtripTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # Seven rows by nine columns: each retained tile is exactly 64x28.
        # Native x=0,1 and 254,255 are deliberately different crop sentinels.
        frames = np.empty((4, 448, 256, 3), dtype=np.uint8)
        for t in range(4):
            frames[t, :, :2] = (255, 0, 255)
            frames[t, :, 254:] = (0, 255, 0)
            for y in range(7):
                for x in range(9):
                    value = 24 + ((y * 37 + x * 71 + t * 53) % 208)
                    frames[t, y*64:(y+1)*64, 2+x*28:2+(x+1)*28] = value
        return frames

    def assert_spatial_match(self, decoded, expected):
        self.assertEqual(decoded.shape, expected.shape)
        # Test every pixel away from codec tile edges, not just global averages.
        error = np.abs(decoded.astype(np.int16) - expected.astype(np.int16))
        for t in range(4):
            for y in range(7):
                for x in range(9):
                    interior = error[t, y*64+8:(y+1)*64-8,
                                     x*28+8:(x+1)*28-8]
                    self.assertLessEqual(int(interior.max()), 12,
                                         (t, y, x, 'interior maximum error'))
                    self.assertLessEqual(float(interior.mean()), 3.0,
                                         (t, y, x, 'interior mean error'))

    def test_actual_export_encode_decode_preserves_coordinates_and_frame_order(self):
        from research.h3_personal.live_native_backend import _export
        from research import run_h3_realtime_first as encoder

        native = self.fixture()
        expected = native[:, :, 2:254, :].copy(order='C')
        real_encode = encoder.encode_rgb_segment
        captured = []

        def inspect_then_encode(raw, path, timing, **kwargs):
            self.assertEqual(kwargs, {'width': 252, 'height': 448})
            self.assertEqual(raw, expected.tobytes(order='C'))
            self.assertEqual((timing.pts_start, timing.newly_emitted_frames), (119, 4))
            captured.append(True)
            return real_encode(raw, path, timing, **kwargs)

        with tempfile.TemporaryDirectory(prefix='h3-spatial-') as directory:
            path = Path(directory) / 'segment.mp4'
            # Spy delegates to the actual encoder: no replacement codec/runtime.
            with patch.object(encoder, 'encode_rgb_segment', inspect_then_encode):
                payload = _export(SimpleNamespace(frames=list(native)), path, 119, 123, 1)
            self.assertEqual(captured, [True])
            self.assertEqual(payload, path.read_bytes())
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            decoded = subprocess.run([
                'ffmpeg', '-v', 'error', '-xerror', '-threads', '1',
                '-i', str(path), '-map', '0:v:0', '-an', '-vsync', '0',
                '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1',
            ], check=True, capture_output=True, timeout=30).stdout
        self.assertEqual(len(decoded), 4*448*252*3)
        decoded = np.frombuffer(decoded, dtype=np.uint8).reshape(4, 448, 252, 3)
        self.assert_spatial_match(decoded, expected)

        # Demonstrate that the lossy tolerance cannot bless structural defects.
        mutations = {
            'horizontal flip': decoded[:, :, ::-1],
            'vertical flip': decoded[:, ::-1],
            'frame reversal': decoded[::-1],
            'axis reinterpretation': decoded.reshape(4, 252, 448, 3)
                                          .transpose(0, 2, 1, 3),
            'row roll': np.roll(decoded, 64, axis=1),
            'column roll': np.roll(decoded, 28, axis=2),
        }
        for name, wrong in mutations.items():
            with self.subTest(rejected=name):
                with self.assertRaises(AssertionError):
                    self.assert_spatial_match(wrong, expected)


if __name__ == '__main__':
    unittest.main()
