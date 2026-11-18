"""Offline integration of real admission, parity and resident-owner boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_h3_continuation_diagnostic_worker import spec_fixture
from research import hf_h3_continuation_diagnostic as base
from research import hf_h3_efficiency_comparison as worker
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
from research.h3_personal.continuation_candidate import make_options
from research.h3_personal.continuation_efficiency import comparison_contract, make_options as efficiency


def parity_fixture():
    row = dict(schema='h3-parallel-decode-parity-v1', session_id='session', worker_rank=0,
        total_frames=328, arrivals=[37, 30, 30], emitted_counts=[119, 102, 107],
        normalized_shape=[1, 24, 97, 28, 16], decoded_shape=[1, 3, 328, 448, 256],
        inputs_unchanged=True, frames_exactly_once=True, closed=True,
        native_parallel_verified=False, diagnostic_only=True, quality_verified=False,
        realtime_verified=False, serial=False, atol=0.0,
        source_sha256='e3cec57d51623c8343c3bc25160e863d253997fb4158aa283bba874154f70fa1',
        source_revision='ba371e33ed080e51f1884ae166baf7c4447a66fe',
        cuda_fp16_autocast=True, denormalized_dtype='torch.float32',
        reference_schedule='alternating-v1', candidate_schedule='two-clip-waves-v1',
        reference_emitted_counts=[119, 102, 107], input_sha256='a'*64, parity=True)
    for name in ('raw_float', 'denormalized_pixels', 'uint8'):
        row[name] = dict(max=0., mean=0., per_frame_max=[0.]*328,
                         per_frame_mean=[0.]*328, exact=True, within_atol=True)
    return row


class EfficiencyIntegrationTests(unittest.TestCase):
    def test_actual_spec_admission(self):
        spec = spec_fixture()
        spec['efficiency_comparison'] = comparison_contract()
        self.assertNotIn('continuation_candidate', spec)
        base.validate_spec(spec)
        worker.validate_spec(spec)
        bad = copy.deepcopy(spec)
        bad['efficiency_comparison']['schema'] = 'wrong'
        for validate in (base.validate_spec, worker.validate_spec):
            with self.assertRaises(ValueError):
                validate(bad)
        for key, value in (('continuation_candidate', make_options(window_audio_context=True)),
                           ('continuation_comparison', base.CONTINUATION_COMPARISON),
                           ('decode_parity_diagnostic', base.DECODE_PARITY_DIAGNOSTIC)):
            bad = copy.deepcopy(spec)
            bad[key] = value
            with self.subTest(mode=key), self.assertRaises(ValueError):
                worker.validate_spec(bad)
        with self.assertRaises(ValueError):
            worker.validate_spec(spec_fixture())

    def test_actual_parallel_parity_and_coherent_negative_finding(self):
        row = parity_fixture()
        self.assertTrue(base.validate_parity(row, 0, 'session', parallel=True)['parity'])
        negative = copy.deepcopy(row)
        negative['parity'] = False
        negative['raw_float'].update(max=1., mean=.5, exact=False, within_atol=False,
            per_frame_max=[1.]*328, per_frame_mean=[.5]*328)
        self.assertFalse(base.validate_parity(negative, 0, 'session', parallel=True)['parity'])
        for key, value in dict(serial=True, source_revision='wrong', cuda_fp16_autocast=False,
                denormalized_dtype='torch.float16', reference_schedule='wrong',
                candidate_schedule='wrong', emitted_counts=[119, 102, 106],
                reference_emitted_counts=[328], worker_rank=True, session_id='other',
                input_sha256='x'*64, parity=False).items():
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                base.validate_parity(bad, 0, 'session', parallel=True)
        bad = copy.deepcopy(row)
        bad['uint8']['per_frame_max'].pop()
        with self.assertRaises(ValueError):
            base.validate_parity(bad, 0, 'session', parallel=True)

    def owner(self, **kwargs):
        paths = {k: '/tmp/h3-efficiency-unused' for k in
                 ('model', 'model_manifest', 'identity', 'identity_config', 'smd',
                  'reference', 'archive', 'output')}
        return LiveVoiceRuntime(**paths, **kwargs)

    def test_constructor_defaults_and_boolean_opt_in(self):
        runtime = self.owner()
        self.assertFalse(runtime._parallel_acceptance_enabled)
        self.assertFalse(runtime._parallel_acceptance_pending)
        self.assertFalse(runtime._measure_memory)
        for flag in ('parallel_decode_acceptance', 'measure_memory'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.owner(**{flag: 1})

    def test_successful_turn_only_consumes_acceptance_and_switch_preserves_owner(self):
        runtime = self.owner(continuation=True, window_audio_context=True,
                             parallel_decode_acceptance=True, measure_memory=True)
        runtime._state = 'ready'
        runtime.generator = Mock()
        generator = runtime.generator
        runtime._requests = 7
        switch = dict(device_audio_validation=True, parallel_decode_waves=True,
                      defer_progress_receipts=True)
        with self.assertRaises((ValueError, RuntimeError)):
            runtime.configure_efficiency(**switch)
        flags = []
        def render(owner, *args, **kwargs):
            flags.append(owner._parallel_acceptance_pending)
            return 'rendered'
        with tempfile.TemporaryDirectory() as directory, \
                patch('research.h3_personal.fast_continuation_render.plan'), \
                patch('research.h3_personal.fast_continuation_render.render', side_effect=render):
            for turn in ('first', 'second'):
                self.assertEqual(runtime.render_continuation('/tmp/unused.wav', 'a'*64,
                    Path(directory)/turn), 'rendered')
        self.assertEqual(flags, [True, False])
        runtime.configure_efficiency(**switch)
        self.assertIs(runtime.generator, generator)
        self.assertEqual(runtime._requests, 7)
        generator.assert_not_called()
        self.assertEqual(generator.mock_calls, [])
        self.assertEqual(runtime._candidate, make_options(window_audio_context=True,
                                                        defer_progress_receipts=True))
        self.assertEqual(runtime._efficiency, efficiency(device_audio_validation=True,
                                                       parallel_decode_waves=True))
        self.assertTrue(runtime._measure_memory)
        self.assertTrue(runtime._parallel_acceptance_enabled)
        runtime._parallel_acceptance_pending = True
        with tempfile.TemporaryDirectory() as directory, \
                patch('research.h3_personal.fast_continuation_render.plan'), \
                patch('research.h3_personal.fast_continuation_render.render',
                      side_effect=RuntimeError('fault')):
            with self.assertRaises(RuntimeError):
                runtime.render_continuation('/tmp/unused.wav', 'a'*64, Path(directory)/'fault')
        self.assertTrue(runtime._parallel_acceptance_pending)
        self.assertEqual(runtime._state, 'degraded')
        runtime._parallel_acceptance_pending = False
        with self.assertRaises((ValueError, RuntimeError)):
            runtime.configure_efficiency(**switch)
        self.assertIs(runtime.generator, generator)
        self.assertEqual(runtime._requests, 7)


if __name__ == '__main__':
    unittest.main()
