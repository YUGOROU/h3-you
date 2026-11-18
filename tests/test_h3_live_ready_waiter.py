import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from research.wait_h3_live_ready import main, wait


class Clock:
    def __init__(self):
        self.now = 0

    def sleep(self, seconds):
        self.now += seconds

    def wall(self):
        return 1000 + self.now


class API:
    def __init__(self, clock, scheduled=0, terminal=None):
        self.clock = clock
        self.scheduled = scheduled
        self.terminal = terminal
        self.canceled = []
        self.revisions = []
        self.private = True

    def whoami(self):
        return {'name': 'YUGOROU'}

    def inspect_job(self, *, job_id, namespace):
        assert (job_id, namespace) == ('owned', 'YUGOROU')
        stage = self.terminal or ('PENDING' if self.clock.now < self.scheduled else 'RUNNING')
        return NS(status=NS(stage=stage))

    def cancel_job(self, **kwargs):
        self.canceled.append(kwargs)
        self.terminal = 'CANCELED'

    def repo_info(self, repo, **kwargs):
        assert repo == 'YUGOROU/session'
        assert kwargs['repo_type'] == 'dataset'
        return NS(private=self.private, sha='pinned')

    def list_repo_files(self, repo, **kwargs):
        self.revisions.append(kwargs['revision'])
        return ['live/state.json']


class WaiterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        (self.d / 'deployment').mkdir()
        (self.d / 'submission.json').write_text(json.dumps({'job_id': 'owned'}))
        self.spec = {'publication_repo': 'YUGOROU/session', 'config': {'timeout_seconds': 660},
                     'live_voice_session': {'id': 'session'}}
        self.save_spec()
        self.clock = Clock()
        self.api = API(self.clock)
        self.fetches = []
        self.ready_at = 0
        self.state = {'status': 'ready', 'session_id': 'session', 'previous_audio_reused': False}

    def save_spec(self):
        (self.d / 'deployment/spec.json').write_text(json.dumps(self.spec))

    def fetch(self, repo, name, revision):
        self.fetches.append((repo, name, revision))
        return dict(self.state, status=self.state['status'] if self.clock.now >= self.ready_at else 'preparing',
                    expires_at=self.clock.wall() + 300)

    def run_wait(self, **kwargs):
        return wait(self.d, self.api, self.fetch, monotonic=lambda: self.clock.now,
                    wall=self.clock.wall, sleep=self.clock.sleep, **kwargs)

    def receipt(self):
        return json.loads((self.d / 'ready-cancellation.json').read_text())

    def test_scheduling_does_not_consume_running_preparation(self):
        self.api.scheduled = 230
        self.ready_at = 700  # 470s running; old combined deadline would fail.
        result = self.run_wait()
        self.assertEqual(self.clock.now, 700)
        self.assertEqual(result['remaining_seconds'], 190)
        self.assertEqual(self.api.canceled, [])
        self.assertTrue((self.d / 'ready-observed.json').exists())
        self.assertEqual(set(self.api.revisions), {'pinned'})
        self.assertTrue(all(row[2] == 'pinned' for row in self.fetches))

    def test_resident_readiness_failure_keeps_warm_job(self):
        self.api.scheduled=300
        with self.assertRaises(TimeoutError):self.run_wait(cancel_on_failure=False)
        self.assertEqual(self.api.canceled,[])
        self.assertFalse((self.d/'ready-cancellation.json').exists())
        receipt=json.loads((self.d/'ready-error.json').read_text())
        self.assertEqual(receipt['skipped'],'resident_operator_policy_keep_warm')
        self.assertEqual(receipt['cancel_attempts'],0)

    def test_scheduling_deadline(self):
        self.api.scheduled = 300
        with self.assertRaises(TimeoutError):
            self.run_wait()
        self.assertLessEqual(self.clock.now, 243)
        self.assertEqual(len(self.api.canceled), 1)
        self.assertTrue(self.receipt()['confirmed_terminal'])

    def test_running_deadline(self):
        self.ready_at = 9999
        with self.assertRaises(TimeoutError):
            self.run_wait()
        self.assertLessEqual(self.clock.now, 663)
        self.assertTrue(self.receipt()['confirmed_terminal'])

    def test_small_window(self):
        self.ready_at = 500
        with self.assertRaises(TimeoutError):
            self.run_wait()
        self.assertEqual(len(self.api.canceled), 1)

    def test_stale_session_and_reuse(self):
        for key, value in [('session_id', 'other'), ('previous_audio_reused', True),
                           ('previous_audio_reused', None)]:
            with self.subTest(key=key, value=value):
                self.state[key] = value
                with self.assertRaises(ValueError):
                    self.run_wait()
                self.assertTrue(self.receipt()['confirmed_terminal'])
                (self.d / 'ready-cancellation.json').unlink()
                self.api.terminal = None
                self.state = {'status': 'ready', 'session_id': 'session', 'previous_audio_reused': False}

    def test_changed_submission_never_canceled(self):
        original = self.fetch

        def changed(*args):
            (self.d / 'submission.json').write_text('{"job_id":"unrelated"}')
            return original(*args)
        self.fetch = changed
        with self.assertRaises(ValueError):
            self.run_wait()
        self.assertEqual(self.api.canceled, [])
        self.assertEqual(self.receipt()['skipped'], 'submission_or_spec_changed')

    def test_terminal_needs_no_cancel(self):
        self.api.terminal = 'ERROR'
        with self.assertRaises(RuntimeError):
            self.run_wait()
        self.assertEqual(self.api.canceled, [])
        self.assertTrue(self.receipt()['confirmed_terminal'])

    def test_private_required(self):
        self.api.private = False
        with self.assertRaises(ValueError):
            self.run_wait()
        self.assertEqual(self.fetches, [])

    def test_cancel_confirmation_is_bounded_even_on_errors(self):
        def broken(**kwargs):
            self.api.canceled.append(kwargs)
            raise RuntimeError('secret must not be recorded')
        self.api.cancel_job = broken
        self.state['status'] = 'error'
        with self.assertRaises(RuntimeError):
            self.run_wait()
        self.assertEqual(len(self.api.canceled), 3)
        self.assertFalse(self.receipt()['confirmed_terminal'])
        self.assertNotIn('secret', (self.d / 'ready-cancellation.json').read_text())

    def test_foreign_repo_and_provider_rejected_without_calls(self):
        self.spec['publication_repo'] = 'OTHER/session'
        self.save_spec()
        with self.assertRaises(ValueError):
            self.run_wait()
        self.assertEqual(self.api.canceled, [])

    def test_offline_default(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--directory', str(self.d)]), 0)
        self.assertFalse(json.loads(output.getvalue())['network'])
        self.assertFalse((self.d / 'owner.lock').exists())

    def test_bounds_reject_infinite_or_oversized(self):
        for kwargs in ({'running_seconds': 661}, {'scheduling_seconds': 241},
                       {'poll_seconds': float('nan')}):
            with self.assertRaises(ValueError):
                self.run_wait(**kwargs)
        self.assertEqual(self.api.canceled, [])


if __name__ == '__main__':
    unittest.main()
