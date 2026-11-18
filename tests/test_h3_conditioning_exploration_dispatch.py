"""Offline admission and dispatch regression tests; no provider/device access."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import launch_h3_continuation_diagnostic as launcher
from research import hf_h3_continuation_diagnostic as worker
from research import hf_h3_conditioning_exploration as exploration
from test_h3_continuation_diagnostic_worker import spec_fixture


class ConditioningDispatchTests(unittest.TestCase):
    def setUp(self):
        # Dispatch contract doubles do not assert worker execution/quality acceptance.
        self.contract = dict(schema='h3-conditioning-exploration-v1', enabled=True)
        self.binding = patch.object(exploration, 'CONTRACT', self.contract, create=True)
        self.binding.start()
        self.addCleanup(self.binding.stop)
        self.spec = spec_fixture()
        self.spec['conditioning_exploration'] = copy.deepcopy(self.contract)

    def test_strict_contract_before_runtime(self):
        worker.validate_spec(self.spec)
        for value in (None, False, {}, dict(self.contract, extra=True),
                      dict(self.contract, enabled=1)):
            with self.subTest(value=value):
                spec = dict(self.spec, conditioning_exploration=value)
                with patch.object(worker, 'LiveVoiceRuntime') as runtime, \
                     patch.object(exploration, 'run') as run:
                    with self.assertRaises(ValueError):
                        worker.run(None, None, spec, None, 0)
                    runtime.assert_not_called()
                    run.assert_not_called()

    def test_all_old_mode_keys_exclusive_even_null(self):
        for key in worker.CONDITIONING_EXCLUSIVE:
            with self.subTest(key=key):
                spec = dict(self.spec, **{key: None})
                with self.assertRaises(ValueError):
                    worker.validate_spec(spec)

    def test_integrated_dispatch(self):
        with patch.object(exploration, 'run', return_value='isolated') as run, \
             patch.object(worker, 'LiveVoiceRuntime') as runtime:
            args = ('root', {'base': 'model'}, self.spec, 'private', 42)
            self.assertEqual(worker.run(*args), 'isolated')
            run.assert_called_once_with(*args)
            runtime.assert_not_called()

    def test_packaged_and_inventory_sealed(self):
        for name in ('research/hf_h3_conditioning_exploration.py',
                     'research/h3_personal/conditioning_experiment.py'):
            self.assertIn(name, launcher.EXTRA)
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            file = stage/'helper.py'
            file.write_text('original')
            before = launcher.inventory(stage)
            file.write_text('tampered')
            self.assertNotEqual(before, launcher.inventory(stage))

    def test_prepare_default_and_conflicts_before_staging(self):
        self.assertIs(inspect.signature(launcher.prepare).parameters[
            'conditioning_exploration'].default, False)
        for key in ('comparison', 'decode_parity', 'efficiency_comparison',
                    'utilization_comparison', 'window_audio_context', 'defer_progress_receipts'):
            with self.subTest(key=key), patch.object(launcher.staging, 'prepare') as prepare:
                with self.assertRaises(ValueError):
                    launcher.prepare(None, None, None, None,
                        conditioning_exploration=True, **{key: True})
                prepare.assert_not_called()
        with self.assertRaises(ValueError):
            launcher.prepare(None, None, None, None, conditioning_exploration=1)

    def test_cli_exclusive_and_prepare_only(self):
        common = ['--directory', '/unused', '--ledger', '/unused', '--conditioning-exploration']
        for extra in ([], ['--prepare', '--comparison'],
                      ['--prepare', '--window-audio-context']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                launcher.main(common + extra)


if __name__ == '__main__':
    unittest.main()
