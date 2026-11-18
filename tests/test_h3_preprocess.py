import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from research.h3_personal.preprocess import AUDIO_SAMPLES, DURATION, command, starts, validate, inventory


class PlanningTests(unittest.TestCase):
    def test_empty_short_source(self):
        self.assertEqual(starts(1.965), [])

    def test_exact_multiple(self):
        self.assertEqual(starts(DURATION * 3), [0, DURATION, DURATION * 2])

    def test_tail_is_covered_not_discarded(self):
        for seconds in (DURATION + .01, 588.696667, 1932.4):
            cuts = starts(seconds)
            self.assertAlmostEqual(cuts[0], 0)
            self.assertAlmostEqual(cuts[-1] + DURATION, seconds)
            self.assertTrue(all(b - a <= DURATION + 1e-6 for a, b in zip(cuts, cuts[1:])))

    def test_real_capture_clip_count(self):
        self.assertEqual(len(starts(588.696667)) + len(starts(1932.4)), 489)

    def test_command_has_one_shared_seek_and_no_crop(self):
        cmd = command('source.mov', 2, Path('v.mp4'), Path('a.wav'), 768, 1344)
        self.assertEqual(cmd.count('-i'), 1)
        self.assertEqual(cmd.count('-ss'), 1)
        self.assertNotIn('crop', ' '.join(cmd))
        self.assertNotIn('asetpts', ' '.join(cmd))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class MediaTests(unittest.TestCase):
    def test_sync_grid_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'capture.mov'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                            '-f', 'lavfi', '-i', 'color=c=red:size=96x160:rate=30:duration=9',
                            '-f', 'lavfi', '-i', "aevalsrc=if(between(t\\,3\\,3.01)\\,0.9\\,0):s=48000:d=9",
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le',
                            str(source)], check=True, capture_output=True)
            source.with_suffix('.json').write_text(json.dumps({'status': 'completed'}))
            video, audio = root / 'clip.mp4', root / 'clip.wav'
            subprocess.run(command(str(source), 2, video, audio, 96, 160), check=True, capture_output=True)
            validate(video, audio, 96, 160)
            with wave.open(str(audio)) as wav:
                import array
                data = array.array('h', wav.readframes(AUDIO_SAMPLES))
                first_peak = next(i // 2 for i, val in enumerate(data) if abs(val) > 15000)
                self.assertLess(abs(first_peak / 32000 - 1), .002)
            records = inventory(root)
            self.assertEqual(records[0]['exclusion_reasons'], [])
            # File with rejected_ prefix is never automatically reintroduced.
            rejected = root / 'rejected_capture.mov'
            shutil.copyfile(source, rejected)
            rec = next(r for r in inventory(root) if r['path'] == str(rejected.resolve()))
            self.assertIn('explicitly_rejected_source', rec['exclusion_reasons'])


if __name__ == '__main__':
    unittest.main()
