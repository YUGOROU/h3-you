import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from research.analyze_h3_idle_continuity import (
    COUNTS, analyze, load_manifest, parse_differences, summarize,
)


def manifest(root):
    (root / 'segments').mkdir()
    entries = []
    for i, n in enumerate(COUNTS):
        path = f'segments/native-{i:03d}.mkv'
        (root / path).touch()
        entries.append(dict(window=i, path=path, shape=[n, 448, 256, 3],
                            status='verified', rgb_bytes=n * 448 * 256 * 3,
                            rgb_sha256='0' * 64))
    (root / 'native-manifest.json').write_text(json.dumps(entries))
    return entries


class ContinuityTests(unittest.TestCase):
    def test_mapping(self):
        values = [0.] * 1449
        for i in (3, 6, 9, 12):
            values[119 + (i - 1) * 102 - 1] = 200.
        result = summarize(values)
        for row in result['boundaries']:
            self.assertEqual(row['global_frame'], sum(COUNTS[:row['window']]))
            self.assertEqual(row['boundary_luma_absolute'], 200 if row['refreshed'] else 0)
            self.assertEqual(row['first24_mean'], 200 / 24 if row['refreshed'] else 0)
            self.assertEqual(row['remaining_peak'], 0)

    def test_metadata_rejects_invalid(self):
        for text in ('', 'frame:0\nlavfi.signalstats.YAVG=nan',
                     'frame:1\nlavfi.signalstats.YAVG=1'):
            with self.assertRaises(ValueError):
                parse_differences(text)

    def test_manifest_rejects_shape_path_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = manifest(root)
            load_manifest(root)
            for key, value in [('path', '../native-000.mkv'), ('shape', [119, 448, 252, 3]),
                               ('path', 'https://example.org/video.mkv')]:
                bad = [dict(e) for e in entries]
                bad[0][key] = value
                (root / 'native-manifest.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_manifest(root)
            (root / 'native-manifest.json').write_text(json.dumps(entries))
            target = root / entries[0]['path']
            target.unlink()
            target.symlink_to(root / entries[1]['path'])
            with self.assertRaises(ValueError):
                load_manifest(root)

    def test_cli_rejects_unknown_argument(self):
        p = subprocess.run([sys.executable, '-m', 'research.analyze_h3_idle_continuity',
                            '--network'], capture_output=True, timeout=10)
        self.assertEqual(p.returncode, 2)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_synthetic_concat_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = manifest(root)
            for i, n in enumerate(COUNTS):
                # One jump at global frame 323; subsequent windows remain white.
                color = 'black' if i < 3 else 'white'
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'color={color}:s=256x448:r=24', '-frames:v', str(n),
                    '-an', '-c:v', 'ffv1', '-threads', '1', str(root / entries[i]['path'])],
                    check=True, capture_output=True, timeout=30)
            result = analyze(root)
            self.assertEqual(result['differences'], 1449)
            for row in result['boundaries']:
                if row['window'] == 3:
                    self.assertGreater(row['boundary_luma_absolute'], 200)
                    self.assertGreater(row['first24_peak'], 200)
                else:
                    self.assertEqual(row['boundary_luma_absolute'], 0)
                self.assertEqual(row['remaining_peak'], 0)


if __name__ == '__main__':
    unittest.main()
