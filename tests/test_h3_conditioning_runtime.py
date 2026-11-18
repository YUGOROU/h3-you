"""CPU-only runtime isolation; no model/device initialization."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
from research.h3_personal.conditioning_experiment import EXPERIMENT_ENV, SCHEMA

POLICY = dict(schema=SCHEMA, short_edges=[1024, 512], active_short_edge=1024)


class RuntimeTests(unittest.TestCase):
    def construct(self, **kwargs):
        return LiveVoiceRuntime(**dict.fromkeys(('model', 'model_manifest', 'identity',
            'identity_config', 'smd', 'reference', 'archive', 'output'), '/unused'), **kwargs)

    def test_constructor_is_explicit_and_offline(self):
        runtime = self.construct()
        self.assertIsNone(runtime._conditioning_experiment)
        runtime = self.construct(continuation=True, conditioning_experiment=POLICY)
        self.assertEqual(runtime._conditioning_experiment, POLICY)
        for flags in ({}, {'continuation': True, 'profile_stages': True},
                      {'continuation': True, 'comparison_diagnostics': True},
                      {'continuation': True, 'decode_parity': True}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.construct(conditioning_experiment=POLICY, **flags)

    def test_switch_requires_completed_turn_and_keeps_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.construct(continuation=True, conditioning_experiment=POLICY)
            runtime._state = 'ready'
            runtime.conditioning_file = Path(tmp).resolve()/'conditioning.json'
            runtime.continuation_file = Path(tmp).resolve()/'window.json'
            runtime.continuation_file.write_text(json.dumps({'mode': 'latent-tail-v1'}))
            generator = runtime.generator = object()
            with self.assertRaises(RuntimeError):
                runtime.configure_conditioning(512)
            runtime.continuation_file.write_text(json.dumps({'mode': 'independent'}))
            runtime.configure_conditioning(512)
            self.assertIs(runtime.generator, generator)
            self.assertEqual(json.loads(runtime.conditioning_file.read_text())['active_short_edge'], 512)
            self.assertEqual(runtime.conditioning_file.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                runtime.configure_conditioning(True)
            with self.assertRaises(RuntimeError):
                runtime.configure_utilization(trace_policy={'unsupported': True})

    def test_close_removes_only_owned_experiment_environment_even_on_failure(self):
        for failure in (False, True):
            runtime = self.construct(continuation=True, conditioning_experiment=POLICY)
            runtime.conditioning_file = Path('/private/owned.json')
            runtime.generator = Mock()
            if failure:
                runtime.generator.shutdown.side_effect = RuntimeError('shutdown')
            with patch.dict(os.environ, {EXPERIMENT_ENV: str(runtime.conditioning_file)}):
                if failure:
                    with self.assertRaises(RuntimeError): runtime.close()
                else:
                    runtime.close()
                self.assertNotIn(EXPERIMENT_ENV, os.environ)
                self.assertEqual(runtime._state, 'closed')
                self.assertIsNone(runtime.generator)


if __name__ == '__main__':
    unittest.main()
