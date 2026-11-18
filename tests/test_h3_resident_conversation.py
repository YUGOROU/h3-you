import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
from research.run_h3_resident_conversation import full_budget_gate, write, command
from research.h3_personal.resident_recovery import ResidentRecovery, plan_turn_recovery
from research.h3_personal.ref2va_budget import HF_CAP

class ConversationTests(unittest.TestCase):
    def test_full_session_includes_all_pi_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json'
            path.write_text(json.dumps(dict(spent_usd=str(HF_CAP-6),reservations={})))
            self.assertEqual(full_budget_gate(path),'4.883340')
            path.write_text(json.dumps(dict(spent_usd=str(HF_CAP-4),reservations={})))
            with self.assertRaises(ValueError):full_budget_gate(path)

    def test_existing_reservations_cannot_be_reused_as_new_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json'
            path.write_text(json.dumps(dict(spent_usd=str(HF_CAP-5),reservations={'other':{'upper_bound_usd':'.5'}})))
            with self.assertRaises(ValueError):full_budget_gate(path)

    def test_receipts_are_private_and_replaceable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'status.json'
            write(path,{'status':'preparing'});write(path,{'status':'ready'})
            self.assertEqual(json.loads(path.read_text()),{'status':'ready'})
            self.assertEqual(path.stat().st_mode & 0o777,0o600)

class WorkerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.pool.shutdown)
        self.state = dict(status='ready', turns={'1': {'video_sequence': 1}})
        self.saved = Mock()
        self.owner = ResidentRecovery(self.state, threading.RLock(), self.saved)
        self.delivery = Mock()
        self.delivery.turns = {}
        self.pending = dict(sequence=1, end=time.monotonic()+60)
        self.delivery.recover_pending.return_value = self.pending
        self.endpoint = SimpleNamespace(closed=False)
        self.render = Mock(side_effect=lambda *args: self.owner.complete(1, 'played'))
        self.playback = Mock(side_effect=lambda *args: self.owner.complete(1, 'played'))

    def prepare(self, fault, action='retry'):
        return plan_turn_recovery(1, fault, action, state=self.state,
            endpoint=self.endpoint, delivery=self.delivery, complete=self.owner.complete,
            render=self.render, playback=self.playback)

    def recover(self, action='retry', **kwargs):
        return self.owner.recover(1, lambda fault: self.prepare(fault, action),
            self.pool, now=time.time(), deadline=time.time()+60, **kwargs)

    def test_worker_failure_then_same_pending_recovery_returns_ready(self):
        def worker():
            try:raise ConnectionError('injected lost acknowledgement')
            except Exception as exc:self.owner.fail(1, 'generation_submit', exc)
        self.owner.submit(self.pool, worker).result(1)
        self.assertEqual(self.state['status'], 'degraded')
        self.recover().result(1)
        self.render.assert_called_once_with(None, None, 1, 1, self.pending)
        self.delivery.submit.assert_not_called()
        self.assertEqual(self.state['status'], 'ready')
        self.assertEqual(self.state['failures'], {})
        self.assertEqual(self.state['recovery_attempts'], 1)

    def test_finished_asset_recovery_does_not_regenerate(self):
        self.owner.fail(1, 'playback', ConnectionError())
        self.delivery.recover_asset.return_value = Path('/private/existing.mp4')
        self.recover().result(1)
        self.playback.assert_called_once_with(Path('/private/existing.mp4'), 1, 1)
        self.render.assert_not_called()
        self.assertEqual(self.state['status'], 'ready')

    def test_started_playback_rejected_but_explicit_skip_allows_next_turn(self):
        self.owner.fail(1, 'playback', RuntimeError())
        self.delivery.recover_asset.side_effect = ValueError('started')
        self.delivery.turns[1] = dict(play_start=1, play_end=2, play_success=False)
        with self.assertRaises(ValueError):self.recover()
        self.recover('skip').result(1)
        self.assertEqual(self.state['turns']['1']['stage'], 'skipped_failure')
        self.assertEqual(self.state['status'], 'ready')
        self.playback.assert_not_called()

    def test_recovery_cannot_race_any_queued_or_running_worker(self):
        release = threading.Event()
        self.addCleanup(release.set)
        worker = self.owner.submit(self.pool, release.wait, 2)
        self.owner.fail(1, 'generation_finish', ConnectionError())
        with self.assertRaisesRegex(ValueError, 'workers still active'):self.recover()
        self.delivery.recover_pending.assert_not_called()
        release.set();worker.result(1)
        self.recover().result(1)

    def test_repeated_failure_is_bounded_without_resubmission(self):
        self.owner.fail(1, 'generation_finish', ConnectionError())
        self.render.side_effect = ConnectionError('still offline')
        for _ in range(2):
            self.recover().result(1)
            self.assertEqual(self.state['status'], 'degraded')
        with self.assertRaisesRegex(ValueError, 'recovery limit'):self.recover()
        self.delivery.submit.assert_not_called()

    def test_uncertain_pi_or_startup_never_replayed(self):
        for phase in ('pi_request', 'prepare_h3'):
            self.owner.fail(1, phase, ConnectionError())
            for action in ('retry', 'skip'):
                with self.assertRaises(ValueError):self.recover(action)
        self.assertEqual(self.state['recovery_attempts'], 0)
        self.render.assert_not_called();self.playback.assert_not_called()

    def test_capture_skip_requires_still_open_asr(self):
        self.owner.fail(1, 'capture', ConnectionError())
        self.endpoint.closed = True
        with self.assertRaises(ValueError):self.recover('skip')
        self.endpoint.closed = False
        self.recover('skip').result(1)
        self.assertEqual(self.state['status'], 'ready')
        self.render.assert_not_called()

    def test_expired_or_stopping_session_never_recovers(self):
        self.owner.fail(1, 'generation_finish', ConnectionError())
        with self.assertRaises(ValueError):self.recover(stopping=True)
        with self.assertRaises(ValueError):
            self.owner.recover(1, self.prepare, self.pool, now=100, deadline=100)
        self.pending['end'] = time.monotonic()-1
        with self.assertRaisesRegex(ValueError, 'original generation wait expired'):self.recover()
        self.render.assert_not_called()

    def test_success_does_not_clear_another_turns_failure(self):
        self.state['turns']['2'] = {}
        self.owner.fail(1, 'generation_finish', ConnectionError())
        self.owner.fail(2, 'capture', ConnectionError())
        self.recover().result(1)
        self.assertEqual(self.state['status'], 'degraded')
        self.assertEqual(set(self.state['failures']), {'2'})

    def test_hold_rejects_recovery_before_socket_or_device_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'operator-hold.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'operator hold'):
                command(root, dict(op='recover', sequence=1))


if __name__=='__main__':unittest.main()
