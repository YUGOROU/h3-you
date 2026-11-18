import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from research.h3_personal.local_av_diagnostics import (
    analyze_local_nut, failure_evidence, timing_evidence)
from research.run_h3_relay_conversation import _persist_output_timing


class DiagnosticsTests(unittest.TestCase):
    def test_allowlist_and_bound(self):
        clean = timing_evidence(dict(accepted_monotonic=list(range(200)),
            events=[dict(sequence=i, token='SECRET') for i in range(200)],
            token='SECRET', first_av_submission_monotonic=2.0, ack_count=200))
        self.assertEqual(len(clean['events']), 144)
        self.assertEqual(len(clean['accepted_monotonic']), 144)
        self.assertNotIn('SECRET', json.dumps(clean))
        self.assertEqual(clean['ack_count'], 200)

    def test_failure(self):
        clean = failure_evidence(dict(phase='response-body', http_status=200,
            error_type='IncompleteRead', message='SECRET', url='SECRET'))
        self.assertEqual(clean, dict(phase='response-body', http_status=200,
                                    error_type='IncompleteRead'))
        self.assertEqual(failure_evidence(dict(phase='SECRET', error_type='SECRET')), {})

    def test_unified_persistence(self):
        class Sink:
            timing = dict(first_av_submission_monotonic=10., accepted_monotonic=(11.,), ack_count=1)
        with tempfile.TemporaryDirectory() as root:
            _persist_output_timing(root, Sink())
            data = json.loads((Path(root) / 'unified-av-timing.json').read_text())
            self.assertEqual(data['accepted_monotonic'], [11.])
            self.assertFalse(data['remote_verified'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_synthetic_nut_exact_pcm_and_pts(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pcm = bytes(range(256)) * 750  # exactly two seconds mono 48k S16LE
            (root / 'input.pcm').write_bytes(pcm)
            output = root / 'fixture.nut'
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi',
                '-i', 'color=size=16x16:rate=24:duration=2', '-f', 's16le',
                '-ar', '48000', '-ac', '1', '-i', str(root / 'input.pcm'),
                '-c:v', 'ffv1', '-threads', '1', '-c:a', 'pcm_s16le', str(output)], check=True)
            result = analyze_local_nut(output, pcm)
            self.assertTrue(result['original_pcm_exact'])
            self.assertEqual(result['pts']['audio']['pts_discontinuities'], 0)
            self.assertEqual(result['pts']['video']['pts_discontinuities'], 0)
            self.assertFalse(analyze_local_nut(output, b'bad')['original_pcm_exact'])


if __name__ == '__main__':
    unittest.main()
