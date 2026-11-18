"""CPU-only warm execution contracts; cold loading and GPU quality are not covered."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research.h3_personal import fast_ref2va
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime


class Tensor:
    """Only the CPU-cache transfer/clone operations used by warm_forward."""

    def __init__(self, values, device='cpu'):
        self.values = list(values)
        self.device = device

    def to(self, device):
        # Deliberately return self like a no-op torch transfer: clone is mandatory.
        return self

    @property
    def shape(self):
        return (len(self.values),)

    def clone(self):
        return Tensor(self.values, self.device)


class WarmResidencyReuseTests(unittest.TestCase):
    def test_four_warm_requests_reuse_resident_pipeline_and_copy_cache(self):
        self.exercise_warm_memory(False)

    def test_four_warm_requests_measure_allocator_only(self):
        self.exercise_warm_memory(True)

    def exercise_warm_memory(self, measure):
        forbidden = Mock(side_effect=AssertionError('cold/model/device operation'))
        calls = []
        batches = []
        conditioning = SimpleNamespace()
        cache = ('already-validated-key', [Tensor([11, 22])], Tensor([3, 4]))
        args = object()

        def stage(name):
            def run(batch, actual_args):
                self.assertIs(actual_args, args)
                calls.append((batch.request, name))
                return batch
            return run

        names = ('input_preparation_stage', 'latent_preparation_stage',
                 'denoising_stage', 'video_decoding_stage')
        mapping = {name: stage(name) for name in names}
        mapping.update(conditioning_stage=conditioning, audio_decoding_stage=forbidden,
                       text_encoding_stage=forbidden)
        pipeline = SimpleNamespace(_stage_name_mapping=mapping, get_module=forbidden,
                                   _release_all_lazy_modules=forbidden,
                                   release_module=forbidden, _release_module=forbidden,
                                   _load_modules=forbidden, compose=forbidden)
        torch = ModuleType('torch')
        torch.distributed = SimpleNamespace(get_rank=lambda: 0)
        reset, allocated, reserved = Mock(), Mock(return_value=10), Mock(return_value=20)
        torch.cuda = SimpleNamespace(Event=forbidden, synchronize=forbidden,
            reset_peak_memory_stats=reset, max_memory_allocated=allocated,
            max_memory_reserved=reserved)
        base = ModuleType('fastvideo.pipelines.composed_pipeline_base')
        base.get_local_torch_device = Mock(return_value='cpu')
        modules = {'torch': torch, 'fastvideo': ModuleType('fastvideo'),
                   'fastvideo.pipelines': ModuleType('fastvideo.pipelines'),
                   'fastvideo.pipelines.composed_pipeline_base': base}

        def skip_audio(batch, audio_stage, actual_torch):
            self.assertIs(audio_stage, forbidden)
            self.assertIs(actual_torch, torch)
            calls.append((batch.request, 'audio_skip'))
            return batch

        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {
                    # Fixture avoids the unrelated packed cold-probe path only.
                    'H3_REF2VA_ATTENTION_MODE': 'fa2',
                    'H3_REF2VA_RUNTIME_POLICY': 'resident-visual-cached',
                    'H3_REF2VA_COVERAGE_DIR': str(Path(root)/'coverage'),
                    'H3_REF2VA_STAGE_PROFILE_DIR': '',
                    'H3_REF2VA_COMPARISON_PROFILE': '0',
                    'H3_REF2VA_REGIONAL_COMPILE': '0',
                    'H3_SMD_MEASURE_MEMORY': '1' if measure else '0',
                    'H3_SMD_CONTINUATION_WINDOW_FILE': str(Path(root)/'coverage'/'window.json'),
                }), \
                patch.object(fast_ref2va, 'build_adapter', forbidden), \
                patch.object(fast_ref2va, 'load_deferred_adapter', forbidden), \
                patch.object(fast_ref2va, 'store_condition', forbidden), \
                patch('research.h3_personal.ref2va_runtime_options.finish_visual_only',
                      side_effect=skip_audio) as finish, \
                patch('research.h3_personal.ref2va_runtime_options.residency_receipt',
                      return_value={'resident': True}) as residency:
            folder = str(Path(root)/'coverage')
            Path(folder).mkdir()
            for hit, request in enumerate(('turn1-window0', 'turn1-window1',
                                           'turn1-window2', 'turn2-window0'), 1):
                Path(folder, 'window.json').write_text(json.dumps(dict(
                    mode='latent-tail-v1', session_id='warm-baseline', index=hit-1,
                    measure_memory=measure, parallel_acceptance=hit <= 3)))
                batch = SimpleNamespace(request=request, extra={})
                batches.append(batch)
                self.assertIs(fast_ref2va.warm_forward(pipeline, batch, args, cache), batch)
                self.assertEqual(conditioning._smd_condition_hits, hit)
                self.assertEqual(json.loads(Path(folder, 'cache-rank-0.json').read_text()),
                                 {'hits': hit})
                self.assertEqual(calls[-5:], [(request, name) for name in (*names, 'audio_skip')])
                self.assertIsNot(batch.prompt_embeds, cache[1])
                self.assertIsNot(batch.prompt_embeds[0], cache[1][0])
                self.assertIsNot(batch.extra['minimax_h3_text_token_tags'], cache[2])
                self.assertEqual(batch.prompt_embeds[0].values, [11, 22])
                self.assertEqual(batch.extra['minimax_h3_text_token_tags'].values, [3, 4])
                batch.prompt_embeds[0].values[0] = hit
                batch.extra['minimax_h3_text_token_tags'].values[0] = hit
            self.assertEqual(finish.call_count, 4)
            self.assertEqual(residency.call_count, 4)
            self.assertEqual(base.get_local_torch_device.call_count, 4)
            self.assertEqual(json.loads(Path(folder, 'residency-rank-0.json').read_text()),
                             {'resident': True})
            receipts = list(Path(folder).glob('utilization-*.json'))
            self.assertEqual(len(receipts), 4 if measure else 0)
            for receipt in receipts:
                row = json.loads(receipt.read_text())
                self.assertEqual(row['session_id'], 'warm-baseline')
                self.assertIsNone(row['trace_policy'])
                self.assertIsNone(row['denoise_policy'])
                self.assertFalse(row['faulted'])
                receipt.unlink()
            self.assertEqual(len(list(Path(folder).iterdir())), 3)
            for counter in (reset, allocated, reserved):
                self.assertEqual(counter.call_count, 4 if measure else 0)
                if measure:
                    self.assertEqual(counter.call_args_list, [unittest.mock.call('cpu')]*4)
            if measure:
                for hit in range(1, 5):
                    path = Path(folder).parent/'memory'/f'rank-0-request-hit{hit}.json'
                    self.assertEqual(json.loads(path.read_text()), dict(
                        schema='h3-worker-memory-v1', worker_rank=0, cache_hit=hit,
                        peak_allocated_bytes=10, peak_reserved_bytes=20,
                        contains_parallel_acceptance=hit <= 3))
                    path.unlink()
                path.parent.rmdir()
        self.assertEqual(len({id(b.prompt_embeds[0]) for b in batches}), 4)
        self.assertEqual(len({id(b.extra['minimax_h3_text_token_tags']) for b in batches}), 4)
        self.assertEqual(cache[1][0].values, [11, 22])
        self.assertEqual(cache[2].values, [3, 4])
        forbidden.assert_not_called()

    def test_generate_reuses_generator_and_accounts_for_warmup_and_continuation(self):
        runtime = object.__new__(LiveVoiceRuntime)
        runtime.api = object()
        runtime.reference_type = object()
        runtime.paths = {'reference': 'unused-reference'}
        runtime.config = object()
        runtime._continuation_enabled = True
        runtime._requests = 0
        seen = []
        generator = SimpleNamespace(generate=lambda req: seen.append(req) or req)
        runtime.generator = generator
        sync = Mock()
        runtime.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=sync))

        def request(*args):
            return SimpleNamespace(sampling=SimpleNamespace(seed=314159))

        with patch('research.h3_personal.ref2va_runtime.request', side_effect=request):
            for continuation in (False, False, False, True, True):
                result, times = runtime._generate(continuation=continuation)
                self.assertIs(runtime.generator, generator)
                self.assertIs(result, seen[-1])
                self.assertLessEqual(*times)
                self.assertEqual(runtime._last_seed, result.sampling.seed)
        self.assertEqual([r.sampling.seed for r in seen],
                         [314159, 314159, 314160, 314159, 314159])
        self.assertEqual(len({id(r) for r in seen}), 5)
        self.assertEqual(runtime._requests, 5)
        # _generate explicitly synchronizes; only the default profile is sync-free.
        self.assertEqual(sync.call_count, 5)


if __name__ == '__main__':
    unittest.main()
