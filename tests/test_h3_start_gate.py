"""Operator start gate and pump-timing receipt contracts (offline only)."""
import contextlib
import io
import json
from pathlib import Path
import importlib.util
import tempfile
import time
import unittest
from unittest.mock import patch

from research import launch_h3_direct_conversation as direct
from research.run_h3_relay_conversation import run_conversation
import test_h3_relay_conversation as fixtures


class StartGateTests(unittest.TestCase):
    def test_gate_requires_run_and_path(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            direct.main(['--directory', '/tmp/not-created-gate-a', '--start-gate', '/tmp/g'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            direct.main(['--directory', '/tmp/not-created-gate-a', '--run', '--start-gate'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            direct.main(['--directory', '/tmp/not-created-gate-b', '--run', '--start-gate', ' '])

    def test_gate_type_checked_before_any_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (1, True, object()):
                with self.assertRaises(TypeError):
                    direct.run(Path(tmp)/'never', ['hello'], start_gate=value)

    def test_gate_blocks_until_file_appears(self):
        # Reuses only the local wait loop shape through run() validation: a gate
        # path that never appears must not be treated as released.
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / 'START'
            target = Path(tmp) / 'fresh-session'
            # run() imports httpx/huggingface_hub at call time, before the gate is
            # consulted; without them the refusal would be an ImportError, not the gate.
            for module in ('httpx', 'huggingface_hub'):
                if importlib.util.find_spec(module) is None:
                    self.skipTest(f'{module} not installed')
            # The gate is consulted after the director starts but before the Job is
            # submitted; an absent gate must time out without reaching the provider.
            from research import provision_modal_identity_secret as secret
            from research import run_h3_relay_conversation as relay
            with patch.object(direct, 'prepare') as prepare, \
                    patch.object(secret, 'cached_token', return_value='t'), \
                    patch.object(relay, 'PiDirector') as director, \
                    patch.object(direct, 'submit') as submit:
                prepare.side_effect = TimeoutError('stop before provider work')
                with self.assertRaises(TimeoutError):
                    direct.run(target, ['hello'], start_gate=gate)
                submit.assert_not_called()


class PumpTimingReceiptTests(unittest.TestCase):
    def test_completed_session_writes_pump_timing(self):
        transport = fixtures.Transport()
        director = fixtures.Director()
        from research.h3_personal.paired_av import LiveAVBinding
        binding = LiveAVBinding('fresh', 'a'*64, 1, max_total_frames=248)
        ready = dict(status='ready', session='fresh', profile_sha256='a'*64,
                     epoch=1, expires_at=time.time()+60)

        class Pump:
            def __init__(self, transport, binding, root, sink, *, deadline, **kwargs):
                self.transport = transport
                self.timing = lambda: dict(receive=[dict(index=0, received=0.0)],
                                           prefetch=True, clock='local_monotonic')
            def run(self):
                self.transport.done.wait(1)
                return {'complete': True, 'frames': 1}

        with tempfile.TemporaryDirectory() as tmp:
            run_conversation(transport, binding, Path(tmp), ['a→', 'b→'], director,
                             ready=ready, max_seconds=2, sink_factory=fixtures.Sink,
                             pump_factory=Pump)
            receipt = json.loads((Path(tmp)/'pump-timing.json').read_text())
            self.assertEqual(receipt['clock'], 'local_monotonic')
            self.assertTrue(receipt['prefetch'])
            self.assertEqual(receipt['receive'][0]['index'], 0)

    def test_missing_timing_callable_is_not_an_error(self):
        transport = fixtures.Transport()
        director = fixtures.Director()
        from research.h3_personal.paired_av import LiveAVBinding
        binding = LiveAVBinding('fresh', 'a'*64, 1, max_total_frames=248)
        ready = dict(status='ready', session='fresh', profile_sha256='a'*64,
                     epoch=1, expires_at=time.time()+60)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_conversation(transport, binding, Path(tmp), ['a→'], director,
                                      ready=ready, max_seconds=2, sink_factory=fixtures.Sink,
                                      pump_factory=fixtures.Pump)
            self.assertTrue(result['complete'])
            self.assertFalse((Path(tmp)/'pump-timing.json').exists())


if __name__ == '__main__':
    unittest.main()
