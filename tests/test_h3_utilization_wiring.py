"""CPU-only request policy and lazy worker admission regression."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research.h3_personal.utilization_policy import TRACE, DENOISE, policies
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
from research.h3_personal.fast_ref2va import warm_forward


class UtilizationTests(unittest.TestCase):
    def test_strict_copy(self):
        self.assertEqual(policies(), (None, None))
        value = dict(TRACE)
        normalized, _ = policies(value)
        value['max_events'] = 1
        self.assertEqual(normalized, TRACE)
        for key in TRACE:
            bad = dict(TRACE)
            bad.pop(key)
            with self.assertRaises(ValueError):
                policies(bad)
        for bad in ({**TRACE, 'max_events': True}, {**TRACE, 'extra': 1},
                    {**TRACE, 'selected_window': 2}):
            with self.assertRaises(ValueError):
                policies(bad)
        with self.assertRaises(ValueError):
            policies(None, {**DENOISE, 'max_compile_seconds': 181})

    def test_configure_preserves_generator_and_rejects_active(self):
        runtime = LiveVoiceRuntime.__new__(LiveVoiceRuntime)
        runtime._lock = threading.Lock()
        runtime._state = 'ready'
        runtime._continuation_enabled = True
        runtime._decode_parity = runtime._comparison_diagnostics = False
        runtime._parallel_acceptance_enabled = runtime._profile_stages = False
        runtime.generator = object()
        generator = runtime.generator
        runtime.configure_utilization(TRACE, DENOISE)
        self.assertIs(runtime.generator, generator)
        self.assertEqual(runtime._trace_policy, TRACE)
        runtime._state = 'degraded'
        with self.assertRaises(RuntimeError):
            runtime.configure_utilization()

    def test_default_worker_no_optional_modules(self):
        with patch.dict(os.environ, {}, clear=True), patch(
                'research.h3_personal.fast_ref2va._warm_forward', return_value='ok'):
            self.assertEqual(warm_forward(None, None, None, None), 'ok')

    def test_controller_lifecycle_and_cleanup_failure(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(dict(mode='latent-tail-v1', session_id='turn',
                                            denoise_policy=DENOISE)))
            pipeline, batch = SimpleNamespace(), SimpleNamespace(extra={})
            controller = Mock()
            apply = Mock(return_value=controller)
            module = SimpleNamespace(apply_request=apply)
            with patch.dict(os.environ, {'H3_SMD_CONTINUATION_WINDOW_FILE': str(path)}), patch(
                    'research.h3_personal.fast_continuation_stages._manifest'), patch(
                    'research.h3_personal.fast_ref2va.static_merge_receipt'), patch.dict(
                    'sys.modules', {'research.h3_personal.denoise_execution': module}), patch(
                    'research.h3_personal.fast_ref2va._warm_forward', return_value=batch):
                self.assertIs(warm_forward(pipeline, batch, 'args', None), batch)
                apply.assert_called_once_with(pipeline, batch, 'args', DENOISE)
                controller.close.assert_called_once_with()
                self.assertEqual(batch.extra, {})
                controller.close.side_effect = RuntimeError('cleanup rejected')
                with self.assertRaises(RuntimeError):
                    warm_forward(pipeline, batch, 'args', None)
                self.assertTrue(pipeline._h3_utilization_faulted)

    def test_policy_drift_faults_before_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(dict(mode='latent-tail-v1', session_id='turn')))
            pipeline = SimpleNamespace(_h3_utilization_identity=('turn', TRACE, None))
            with patch.dict(os.environ, {'H3_SMD_CONTINUATION_WINDOW_FILE': str(path)}), patch(
                    'research.h3_personal.fast_ref2va._warm_forward') as forward:
                with self.assertRaises(ValueError):
                    warm_forward(pipeline, SimpleNamespace(extra={}), None, None)
                self.assertTrue(pipeline._h3_utilization_faulted)
                forward.assert_not_called()

    def test_missing_controller_faults_and_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(dict(mode='latent-tail-v1', denoise_policy=DENOISE)))
            pipeline = SimpleNamespace()
            batch = SimpleNamespace(extra={})
            with patch.dict(os.environ, {'H3_SMD_CONTINUATION_WINDOW_FILE': str(path)}), patch(
                    'research.h3_personal.fast_continuation_stages._manifest'), patch(
                    'research.h3_personal.fast_ref2va.static_merge_receipt'), patch.dict(
                    'sys.modules', {'research.h3_personal.denoise_execution': None}):
                with self.assertRaises(ImportError):
                    warm_forward(pipeline, batch, None, None)
                self.assertTrue(pipeline._h3_utilization_faulted)
                with self.assertRaises(RuntimeError):
                    warm_forward(pipeline, batch, None, None)
            self.assertEqual(batch.extra, {})


if __name__ == '__main__':
    unittest.main()
