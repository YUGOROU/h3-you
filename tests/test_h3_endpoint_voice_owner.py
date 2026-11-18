import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from research import run_h3_endpoint_voice as owner


class OwnerTests(unittest.TestCase):
    def test_offline_plan_never_captures_or_allocates(self):
        with patch('sys.argv',['owner','--directory','/tmp/unused']), patch.object(owner,'run') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            owner.main()
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['mode'],'offline')

    def test_schedule_validation(self):
        self.assertIsNone(owner.validate_capture_time(None, 100))
        for value in (100, 700):
            self.assertEqual(owner.validate_capture_time(value, 100), value)
        for value in (99, 701, float('nan'), float('inf')):
            with self.assertRaises(ValueError):owner.validate_capture_time(value, 100)

    def test_wait_and_receipt(self):
        clock = [100.0]
        def sleep(seconds):clock[0] += seconds
        with patch.object(owner.time, 'time', side_effect=lambda:clock[0]), patch.object(owner.time, 'monotonic', side_effect=lambda:clock[0]), patch.object(owner.time, 'sleep', side_effect=sleep):
            receipt = owner.wait_for_capture(101, 2000)
        self.assertEqual(receipt['actual_time'], 101)
        self.assertEqual(receipt['wait_status'], 'waited')
        self.assertTrue(receipt['actual_utc'].endswith('+00:00'))

    def test_late_or_expiring_capture_rejected(self):
        with patch.object(owner.time, 'time', return_value=104):
            with self.assertRaises(RuntimeError):owner.wait_for_capture(100, 2000)
            with self.assertRaises(RuntimeError):owner.wait_for_capture(None, 164)
        with patch.object(owner.time, 'time', return_value=103):
            self.assertEqual(owner.wait_for_capture(100, 2000)['wait_status'], 'not_needed')

    def test_parent_requires_paused_matching_branch_and_original_pins(self):
        original = owner.deployment('owned/runtime', 'original', 'endpoint', 1000)
        parent = owner.deployment('owned/runtime', 'fresh', 'endpoint', 2000)
        info = SimpleNamespace(private=True, sha='fresh')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            def save(state='paused'):
                (path/'intent.json').write_text(json.dumps(parent))
                (path/'result.json').write_text(json.dumps(dict(terminal_state=state)))
            save()
            self.assertEqual(owner.parent_intent(original, path, info), parent)
            with self.assertRaises(ValueError):owner.parent_intent(original, None, info)
            save('unconfirmed')
            with self.assertRaises(ValueError):owner.parent_intent(original, path, info)
            save()
            info.sha = 'drift'
            with self.assertRaises(ValueError):owner.parent_intent(original, path, info)
            info.sha = 'fresh'
            parent['model']['command'] = ['unreviewed']
            save()
            with self.assertRaises(ValueError):owner.parent_intent(original, path, info)

    def test_capture_termination_is_graceful_and_bounded(self):
        process=MagicMock()
        process.poll.return_value=None
        process.returncode=143
        owner.stop_capture(process)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=4)

    def test_killed_capture_is_not_clean_capture(self):
        process=MagicMock()
        process.poll.return_value=None
        process.wait.side_effect=[subprocess.TimeoutExpired('capture',4),None]
        with self.assertRaises(RuntimeError):owner.stop_capture(process)
        process.kill.assert_called_once()

    def test_half_duplex_and_truthful_origin(self):
        source=Path(owner.__file__).read_text()
        self.assertLess(source.index('try:stop_capture(process)'),source.index("'research.run_h3_director'"))
        self.assertIn("origin='comet_process_audio_endpoint_asr'",source)
        self.assertIn('discord_reception_verified=False',source)
        self.assertIn('h3_generated=False',source)
        self.assertIn('pause_owned(token,path)',source)
        self.assertIn("'capture-metadata.jsonl'",source)
        self.assertNotIn('ffmpeg',source)
        self.assertNotIn("'--text',prompt",source)
        self.assertIn('signal.SIGTERM',source)
        self.assertIn('pcm=start_capture',source)


if __name__=='__main__':unittest.main()
