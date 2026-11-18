import shutil
import tempfile
import unittest
from pathlib import Path

from research import analyze_h3_obs_delivery_quality as a
from research.h3_personal.obs_av_sink import output_arguments


class QualityTests(unittest.TestCase):
    def metadata(self):
        return {'streams': [dict(codec_type='video', width=252, height=448,
                                 avg_frame_rate='24/1', nb_read_frames='2368'),
                            dict(codec_type='audio')],
                'format': {'duration': str(a.DURATION)}}

    def test_geometry(self):
        m = self.metadata()
        self.assertEqual(a.validate_source(m), (252, 448))
        for key, value in [('width', 253), ('height', 0),
                           ('nb_read_frames', '2367'), ('avg_frame_rate', '25/1')]:
            bad = self.metadata()
            bad['streams'][0][key] = value
            with self.assertRaises(ValueError):
                a.validate_source(bad)

    def test_parser(self):
        text = '\n'.join(f'n:{i+1} Y:0.1 All:0.99 (20)' for i in range(a.FRAMES))
        values = a.parse_ssim(text)
        self.assertEqual(len(values), 2368)
        self.assertEqual(a.summarize(values)['frame_index_range'], [0, 2367])
        for bad in [text + '\nn:2369 All:1', text.replace('n:2 ', 'n:1 ', 1),
                    text.replace('All:0.99', 'All:nan', 1), '']:
            with self.assertRaises(ValueError):
                a.parse_ssim(bad)

    def test_command(self):
        command = a.encode_command(Path('/tmp/source.mp4'), Path('/tmp/stage1.ts'))
        production = output_arguments(48124)
        self.assertEqual(command[-len(production):-1], production[:-1])
        self.assertNotIn('-re', command)
        self.assertFalse(any('udp:' in item for item in command))
        self.assertIn('0:v:0', command)
        self.assertIn('0:a:0', command)
        self.assertEqual(a.TIMEOUT, 120)

    def test_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'large'
            with p.open('wb') as f:
                f.truncate(a.LIMIT + 1)
            with self.assertRaises(ValueError):
                a.check_bound(Path(tmp))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_actual_local_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, stage, stats = root/'source.mkv', root/'stage1.ts', root/'ssim.txt'
            a.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                   'testsrc2=size=64x96:rate=24:duration=1', '-f', 'lavfi', '-i',
                   'sine=frequency=440:duration=1', '-c:v', 'ffv1', '-c:a', 'pcm_s16le',
                   str(source)])
            a.run(a.encode_command(source, stage, duration=1))
            a.run(a.ssim_command(source, stage, stats))
            values = a.parse_ssim(stats.read_text(), expected=24)
            self.assertGreater(min(values), 0.5)
            metadata = a.probe(stage)
            self.assertEqual(int(metadata['streams'][0]['nb_read_frames']), 24)
            self.assertTrue(metadata['packet_sizes'])


if __name__ == '__main__':
    unittest.main()
