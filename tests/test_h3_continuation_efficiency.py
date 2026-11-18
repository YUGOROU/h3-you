"""Strict opt-in/admission tests, no models or GPU operations."""
import builtins
import unittest
from unittest.mock import patch

from research.h3_personal.continuation_efficiency import make_options, normalize
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime


class EfficiencyTests(unittest.TestCase):
    def paths(self):
        return {k: '.' for k in ('model', 'model_manifest', 'identity', 'identity_config',
                                 'smd', 'reference', 'archive', 'output')}

    def test_options_and_strict_types(self):
        self.assertIsNone(make_options())
        for device in (False, True):
            for waves in (False, True):
                options = make_options(device_audio_validation=device, parallel_decode_waves=waves)
                self.assertEqual(normalize(options), options)
                if options is not None:
                    self.assertIsNot(normalize(options), options)
        for value in (1, None, 'yes'):
            with self.assertRaises(ValueError):
                make_options(device_audio_validation=value)
            with self.assertRaises(ValueError):
                make_options(parallel_decode_waves=value)
        options = make_options(device_audio_validation=True)
        for bad in ({}, dict(options, extra=True), dict(options, audio_validation=True),
                    dict(options, decode_schedule='unknown')):
            with self.assertRaises(ValueError):
                normalize(bad)

    def test_runtime_requires_v6_and_isolation(self):
        for device, waves in ((True, False), (False, True), (True, True)):
            flags = dict(device_audio_validation=device, parallel_decode_waves=waves)
            with self.assertRaisesRegex(ValueError, 'v6'):
                LiveVoiceRuntime(**self.paths(), continuation=True, **flags)
            runtime = LiveVoiceRuntime(**self.paths(), continuation=True,
                                       window_audio_context=True, **flags)
            self.assertEqual(runtime._efficiency, make_options(**flags))
            self.assertIsNone(runtime.generator)
            for blocked in ({'decode_parity': True}, {'comparison_diagnostics': True}):
                with self.assertRaises(ValueError):
                    LiveVoiceRuntime(**self.paths(), continuation=True,
                                     window_audio_context=True, **flags, **blocked)

    def test_default_owner_needs_no_new_optional_policy_import(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.endswith(('continuation_candidate', 'continuation_efficiency')):
                raise AssertionError('default requested an optional policy module')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guard):
            runtime = LiveVoiceRuntime(**self.paths())
        self.assertIsNone(runtime._candidate)
        self.assertIsNone(runtime._efficiency)


if __name__ == '__main__':
    unittest.main()
