"""Offline socket/worker integration: recover an uncertain submit, never resend it."""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import run_h3_resident_conversation as owner
from research.h3_personal.resident_recovery import ResidentRecovery


class QueuePi:
    """Line-oriented Pi readiness and request-ID-correlated replies, including EOF."""

    def __init__(self):
        self.rows = queue.Queue()
        self.rows.put(json.dumps(dict(type='ready', model='Qwen/Qwen3.8-27B:cerebras')) + '\n')
        self.requests = []
        self.closed = False
        self.stdin = SimpleNamespace(write=Mock(side_effect=self.write),
                                     flush=Mock(), close=Mock(side_effect=self.close))
        self.stdout = SimpleNamespace(readline=Mock(side_effect=lambda _limit: self.rows.get(timeout=1)))

    def write(self, line):
        packet = json.loads(line)
        self.requests.append(packet)
        self.rows.put(json.dumps(dict(type='turn', turn_id=packet['turn_id'],
                                      reply=f'reply {len(self.requests)}')) + '\n')

    def close(self):
        self.closed = True
        self.rows.put('')

    def poll(self):
        return 0 if self.closed else None


class EOFSelector:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def register(self, *_args):
        pass

    def select(self, _timeout):
        return [(None, 1)]


def fake_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class ResidentRecoveryIntegrationTests(unittest.TestCase):
    def test_lost_submit_ack_recovers_existing_attempt_over_socket(self):
        pi = QueuePi()
        api = Mock()
        api.list_jobs.return_value = []
        api.inspect_job.side_effect = lambda **_kw: SimpleNamespace(
            status=SimpleNamespace(stage='CANCELED' if api.cancel_job.called else 'RUNNING'))
        hf_api = Mock(return_value=api)
        download = Mock(side_effect=AssertionError('unexpected Hub download'))
        token = Mock(return_value='offline-placeholder')
        endpoint = Mock()
        endpoint.deadline = time.time() + 1800
        endpoint.lease_id = 'offline-asr-lease'
        endpoint.closed = False
        endpoint.receipt = dict(terminal_state='running')

        def capture(turn, _planned):
            turn.mkdir(parents=True)
            if endpoint.capture.call_count <= 3:
                raise NoSpeech('injected silent capture')
            return dict(text=f'fresh speech {endpoint.capture.call_count}')

        def close_endpoint():
            endpoint.closed = True
            endpoint.receipt['terminal_state'] = 'paused'

        endpoint.capture.side_effect = capture
        endpoint.close.side_effect = close_endpoint
        endpoint_type = SimpleNamespace(open=Mock(return_value=endpoint))
        delivery = Mock()
        delivery.turns = {}
        pending = dict(sequence=1, end=time.monotonic() + 600,
                       deadline=time.time() + 700, request_id='existing-request')
        original_pending = dict(pending)
        retained = {}
        asset = Path('/offline/existing-asset.mp4')

        def submit(_result, _turn, video_sequence):
            retained[video_sequence] = pending
            raise ConnectionError('ACK lost after request accepted')

        def recover_pending(video_sequence):
            return retained[video_sequence]

        def finish(existing):
            self.assertIs(existing, pending)
            return asset

        delivery.submit.side_effect = submit
        delivery.recover_pending.side_effect = recover_pending
        delivery.finish.side_effect = finish
        delivery_type = Mock(return_value=delivery)
        h3_expiry = time.time() + 1200
        ready_wait = Mock(return_value=dict(expires_at=h3_expiry, observed_at=time.time()))
        recoveries = []

        def recovery_factory(*args):
            recovery = ResidentRecovery(*args)
            recoveries.append(recovery)
            return recovery

        class NoSpeech(Exception):
            pass

        class TurnRejected(Exception):
            pass

        modules = {
            'huggingface_hub': fake_module('huggingface_hub', HfApi=hf_api, hf_hub_download=download),
            'research.provision_modal_identity_secret': fake_module(
                'research.provision_modal_identity_secret', cached_token=token),
            'research.h3_personal.resident_endpoint': fake_module(
                'research.h3_personal.resident_endpoint', ResidentEndpoint=endpoint_type,
                NoSpeech=NoSpeech, TurnRejected=TurnRejected),
            'research.h3_personal.resident_delivery': fake_module(
                'research.h3_personal.resident_delivery', ResidentDelivery=delivery_type),
            'research.wait_h3_live_ready': fake_module('research.wait_h3_live_ready', wait=ready_wait),
        }
        errors = []
        # Short /tmp path keeps the real AF_UNIX address below macOS's limit.
        with tempfile.TemporaryDirectory(prefix='h3-recovery-', dir='/tmp') as folder:
            root = Path(folder)
            owner.write(root / 'owner.json', dict(socket=str(root / 'owner.sock')))

            def launch():
                self.assertEqual(endpoint.capture.call_count, 4)
                self.assertEqual(len(pi.requests), 1)
                (root / 'h3').mkdir()
                owner.write(root / 'h3' / 'submission.json', dict(job_id='offline-job'))

            launcher = Mock(side_effect=launch)
            modules['research.launch_h3_live_voice'] = fake_module(
                'research.launch_h3_live_voice', main=launcher)

            def run():
                try:
                    owner.serve(root, root / 'parent', root / 'obs.json')
                except BaseException as exc:
                    errors.append(exc)

            def status():
                try:
                    return json.loads((root / 'status.json').read_text())
                except FileNotFoundError:
                    return {}

            def await_condition(predicate):
                end = time.monotonic() + 2
                while time.monotonic() < end:
                    if errors:
                        raise AssertionError(f'serve failed: {errors!r}')
                    if predicate():
                        return
                    threading.Event().wait(.005)
                self.fail(f'timed out; state={status()!r}, errors={errors!r}')

            def quiescent():
                if not recoveries:
                    return False
                recovery = recoveries[0]
                with recovery.lock:
                    return all(future.done() for future in recovery.workers)

            # All heavy imports and external boundaries are replaced before serve.
            with patch.dict('sys.modules', modules), \
                    patch.object(owner, 'full_budget_gate', return_value='offline') as budget, \
                    patch.object(owner.signal, 'signal') as signal_handler, \
                    patch.object(owner.subprocess, 'Popen', return_value=pi) as popen, \
                    patch.object(owner.selectors, 'DefaultSelector', EOFSelector), \
                    patch.object(owner, 'time', SimpleNamespace(
                        time=time.time, monotonic=time.monotonic, sleep=Mock())), \
                    patch('research.h3_personal.resident_recovery.ResidentRecovery',
                          side_effect=recovery_factory):
                thread = threading.Thread(target=run, daemon=True)
                thread.start()
                try:
                    await_condition(lambda: status().get('status') == 'asr_ready_preflight')
                    self.assertTrue((root / 'owner.sock').exists())
                    for silent_sequence in range(1, 4):
                        silent = owner.command(root, dict(op='turn', start=time.time() + 4))
                        self.assertEqual(silent.get('sequence'), silent_sequence, silent)
                        await_condition(lambda: status().get('turns', {}).get(str(silent_sequence), {}).get('stage') == 'no_speech' and quiescent())
                        self.assertEqual(status()['pi_requests'], 0)
                        launcher.assert_not_called()
                    first = owner.command(root, dict(op='turn', start=time.time() + 4))
                    self.assertEqual(first.get('sequence'), 4, first)
                    await_condition(lambda: status().get('status') == 'ready' and quiescent())
                    initial = status()
                    self.assertEqual(initial['turns']['4']['stage'], 'preflight_only_not_AV_input')
                    delivery.submit.assert_not_called()
                    second = owner.command(root, dict(op='turn', start=time.time() + 4))
                    self.assertEqual(second.get('sequence'), 5, second)
                    await_condition(lambda: status().get('status') == 'degraded' and quiescent())
                    degraded = status()
                    self.assertEqual(degraded['failures']['5'], dict(
                        phase='generation_submit', error_type='ConnectionError'))
                    self.assertIs(retained[1], pending)
                    delivery.finish.assert_not_called()
                    delivery.play.assert_not_called()
                    api.cancel_job.assert_not_called()
                    endpoint.close.assert_not_called()
                    input_counts = (endpoint.capture.call_count, len(pi.requests),
                                    pi.stdin.write.call_count, pi.stdout.readline.call_count)
                    submission = json.loads((root / 'h3' / 'submission.json').read_text())
                    reply = owner.command(root, dict(op='recover', sequence=5, action='retry'))
                    self.assertEqual(reply, dict(recovering=True, sequence=5, action='retry'))
                    await_condition(lambda: status().get('status') == 'ready' and quiescent())
                    recovered = status()
                    self.assertEqual(recovered['turns']['5']['stage'], 'played')
                    self.assertEqual(recovered['failures'], {})
                    self.assertEqual(recovered['recovery_attempts'], 1)
                    self.assertFalse(recovered['requires_operator_review'])
                    self.assertEqual(input_counts, (endpoint.capture.call_count, len(pi.requests),
                                                   pi.stdin.write.call_count, pi.stdout.readline.call_count))
                    self.assertEqual(endpoint.capture.call_count, 5)
                    self.assertEqual(recovered['pi_requests'], 2)
                    self.assertEqual(recovered['maximum_capture_attempts'], 12)
                    self.assertEqual(len(pi.requests), 2)
                    self.assertNotEqual(pi.requests[0]['turn_id'], pi.requests[1]['turn_id'])
                    self.assertNotEqual(pi.requests[0]['text'], pi.requests[1]['text'])
                    launcher.assert_called_once_with()
                    delivery.submit.assert_called_once()
                    delivery.recover_pending.assert_called_once_with(1)
                    delivery.finish.assert_called_once_with(pending)
                    delivery.play.assert_called_once_with(asset, 1)
                    for key in ('deadline', 'asr_deadline'):
                        self.assertEqual(initial[key], degraded[key])
                        self.assertEqual(initial[key], recovered[key])
                    self.assertEqual(recovered['deadline'], h3_expiry - 10)
                    self.assertEqual(endpoint.deadline, initial['asr_deadline'])
                    self.assertEqual(pending, original_pending)
                    self.assertEqual(json.loads((root / 'h3' / 'submission.json').read_text()), submission)
                    api.cancel_job.assert_not_called()
                    endpoint.close.assert_not_called()
                finally:
                    try:
                        if thread.is_alive() and (root / 'owner.sock').exists():
                            self.assertEqual(owner.command(root, dict(op='stop')), dict(stopping=True))
                    finally:
                        thread.join(timeout=3)
                self.assertFalse(thread.is_alive(), 'serve did not terminate after explicit stop')
                self.assertEqual(errors, [])
                self.assertEqual(status()['status'], 'stopped')
                self.assertTrue(status()['pi_closed'])
                self.assertFalse((root / 'owner.sock').exists())
                api.cancel_job.assert_called_once_with(job_id='offline-job', namespace='YUGOROU')
                endpoint.close.assert_called_once_with()
                delivery.close.assert_called_once_with()
                pi.stdin.close.assert_called_once_with()
                hf_api.assert_called_once_with(token='offline-placeholder')
                api.list_jobs.assert_called_once_with(namespace='YUGOROU')
                ready_wait.assert_called_once()
                budget.assert_called_once()
                signal_handler.assert_called_once()
                popen.assert_called_once()
                download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
