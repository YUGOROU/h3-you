"""CPU regression for review clocks from globally timestamped production fragments."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from research.h3_personal.conversation_scenario import segment_timing
from research.h3_personal.visual_ancestry_diagnostic import concat_manifest
from research.run_h3_realtime_first import encode_rgb_segment


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'requires CPU ffmpeg and ffprobe')
class VisualAncestryReviewClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        directory = cls.root / 'segments'
        directory.mkdir()
        cls.segments = []
        cls.rows = []
        for index, (start, stop) in enumerate(((0, 119), (119, 221), (221, 328))):
            path = directory / f'{index}.mp4'
            # Moving, nonuniform RGB content makes decoded-frame preservation meaningful.
            raw = b''.join(bytes(((x + frame * 3) % 256,
                                  (y * 5 + frame) % 256,
                                  (x + y + frame * 7) % 256))
                           for frame in range(start, stop)
                           for y in range(32) for x in range(32))
            encode_rgb_segment(raw, path, segment_timing(index, start, stop),
                               width=32, height=32)
            cls.segments.append(path)
            cls.rows.append({'emitted_frame': stop})

    def command(self, *args):
        return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=45).stdout

    def concatenate(self, manifest, name):
        listing = self.root / f'{name}.txt'
        listing.write_text(manifest)
        output = self.root / f'{name}.mov'
        # Stream copy deliberately has no -vf (filters cannot be used with copy).
        self.command('ffmpeg', '-v', 'error', '-nostdin', '-y',
                     '-f', 'concat', '-safe', '0', '-i', str(listing),
                     '-map', '0:v:0', '-an', '-c:v', 'copy', str(output))
        return output

    def timestamps(self, path):
        result = json.loads(self.command(
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_frames', '-show_entries', 'frame=pts_time',
            '-of', 'json', str(path)))
        return [float(frame['pts_time']) for frame in result['frames']]

    def assert_clock(self, path):
        timestamps = self.timestamps(path)
        self.assertEqual(len(timestamps), 328)
        for index, pts in enumerate(timestamps):
            self.assertAlmostEqual(pts, index / 24, delta=0.000002,
                                   msg=f'frame {index}: {pts}')

    def rgb_digest(self, path):
        raw = self.command('ffmpeg', '-v', 'error', '-nostdin',
                           '-i', str(path), '-map', '0:v:0', '-an',
                           '-vsync', '0', '-f', 'rawvideo',
                           '-pix_fmt', 'rgb24', 'pipe:1')
        self.assertEqual(len(raw), 328 * 32 * 32 * 3)
        return hashlib.sha256(raw).hexdigest()

    def test_duration_manifest_keeps_global_fragments_on_24hz_clock(self):
        manifest = concat_manifest(self.segments, self.rows)
        self.assertEqual(manifest.count('\nduration '), 3)
        self.assert_clock(self.concatenate(manifest, 'duration-review'))

    def test_offline_setts_repairs_old_concat_without_changing_decoded_rgb(self):
        old_manifest = ''.join(f"file 'segments/{path.name}'\n"
                               for path in self.segments)
        bad = self.concatenate(old_manifest, 'old-review')
        timestamps = self.timestamps(bad)
        self.assertEqual(len(timestamps), 328)
        self.assertGreater(max(abs(pts - i / 24)
                               for i, pts in enumerate(timestamps)), 0.000002)
        fixed = self.root / 'fixed-review.mov'
        self.command('ffmpeg', '-v', 'error', '-nostdin', '-y',
                     '-i', str(bad), '-map', '0:v:0', '-an', '-c:v', 'copy',
                     '-bsf:v', 'setts=ts=N:duration=1:time_base=1/24', str(fixed))
        self.assert_clock(fixed)
        self.assertEqual(self.rgb_digest(bad), self.rgb_digest(fixed))


if __name__ == '__main__':
    unittest.main()
