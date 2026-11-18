"""CPU protocol tests using a minimal tensor double (no torch dependency)."""
import random
import sys
import types
import unittest
from unittest.mock import patch

from research.h3_personal.continuation_candidate import (
    encode_window_context, make_options, normalize,
)
from research.h3_personal.continuation_plan import plan


class Tensor:
    def __init__(self, shape, values=None, *, offset=0, finite=True, floating=True, device='cpu'):
        self.shape = shape
        self.values = list(values) if values is not None else None
        self.offset = offset
        self.finite = finite
        self.floating = floating
        self.device = types.SimpleNamespace(type=device)

    def is_floating_point(self):
        return self.floating

    def clone(self):
        return Tensor(self.shape, self.values, offset=self.offset, finite=self.finite,
                      floating=self.floating, device=self.device.type)

    def __getitem__(self, key):
        s = key[1]
        start, stop, step = s.indices(self.shape[1])
        assert step == 1
        return Tensor((2, stop - start) + self.shape[2:],
                      None if self.values is None else self.values[start:stop],
                      offset=self.offset + start, finite=self.finite)


def cat(parts, dim):
    assert dim == 1
    return Tensor((2, sum(p.shape[1] for p in parts), 32),
                  [v for p in parts for v in p.values])


TORCH = types.SimpleNamespace(Tensor=Tensor, cat=cat,
                              isfinite=lambda t: types.SimpleNamespace(all=lambda: t.finite))


class CandidateTests(unittest.TestCase):
    def test_default_owner_does_not_require_candidate_package(self):
        import builtins
        from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.endswith('continuation_candidate'):
                raise ImportError('candidate module intentionally absent')
            return original(name, *args, **kwargs)
        paths = dict(model='.', model_manifest='.', identity='.', identity_config='.',
                     smd='.', reference='.', archive='.', output='.')
        with patch('builtins.__import__', side_effect=blocked):
            self.assertIsNone(LiveVoiceRuntime(**paths)._candidate)
            with self.assertRaises(ImportError):
                LiveVoiceRuntime(**paths, continuation=True, window_audio_context=True)
            with self.assertRaises(ValueError):
                LiveVoiceRuntime(**paths, window_audio_context=1)

    def test_options(self):
        self.assertIsNone(normalize(None))
        self.assertIsNone(make_options())
        for a, b in ((True, False), (False, True), (True, True)):
            o = make_options(window_audio_context=a, defer_progress_receipts=b)
            self.assertEqual(normalize(o), o)
            self.assertIsNot(normalize(o), o)
        base = make_options(window_audio_context=True)
        bad = [{}, [], dict(base, extra=1), dict(base, schema=1),
               dict(base, schema='wrong'), dict(base, audio_context=True),
               dict(base, progress_receipts='unknown')]
        for o in bad:
            with self.subTest(options=o), self.assertRaises(ValueError):
                normalize(o)
        for value in (0, 1, None, 'false'):
            with self.assertRaises(ValueError):
                make_options(window_audio_context=value)
            with self.assertRaises(ValueError):
                make_options(defer_progress_receipts=value)

    def setUp(self):
        self.windows = plan(328, window_frames=124)
        self.wave = Tensor((2, 437600))
        self.torch = patch.dict(sys.modules, torch=TORCH)
        self.torch.start()
        self.addCleanup(self.torch.stop)

    def test_exact_offsets_overlap_ownership_and_rng(self):
        calls = []
        reused = Tensor((2, 207, 32), range(207))

        def encode(wave):
            calls.append((wave.offset, wave.offset + wave.shape[1]))
            wave.offset = -1  # callback cannot mutate source/slices
            reused.values[:] = [len(calls) * 1000 + t for t in range(207)]
            return reused

        random.seed(27)
        before = random.getstate()
        result, info = encode_window_context(self.wave, self.windows, encode)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(calls, [(0, 165600), (136000, 301600), (272000, 437600)])
        self.assertEqual(result.shape, (2, 547, 32))
        self.assertEqual(result.values, list(range(1000, 1207)) +
                         list(range(2037, 2207)) + list(range(3037, 3207)))
        self.assertEqual(result.values[170], 1170)  # not second encoder's 2000
        self.assertEqual(result.values[340], 2170)  # not third encoder's 3000
        self.assertEqual(self.wave.offset, 0)
        self.assertEqual(info, {'source_window_encodes': 3,
                               'audio_overlap_policy': 'prior-canonical-wins'})
        calls.clear()
        again, _ = encode_window_context(self.wave, self.windows, encode)
        self.assertEqual(again.values, result.values)
        reused.values[0] = -999
        self.assertEqual(result.values[0], 1000)

    def test_invalid_inputs_before_callback(self):
        for wave, windows in ((Tensor((2, 437599)), self.windows),
                              (Tensor((2, 437601)), self.windows),
                              (Tensor((1, 437600)), self.windows),
                              (Tensor((2, 437600), finite=False), self.windows),
                              (Tensor((2, 437600), floating=False), self.windows),
                              (Tensor((2, 437600), device='cuda'), self.windows),
                              (self.wave, ()), (self.wave, self.windows[1:]),
                              (self.wave, (self.windows[0], self.windows[2]))):
            with self.subTest(shape=wave.shape, windows=windows):
                with self.assertRaises(ValueError):
                    encode_window_context(wave, windows, lambda _: self.fail('called'))

    def test_invalid_encoder_results(self):
        for result in (None, Tensor((2, 206, 32)), Tensor((2, 207, 31)),
                       Tensor((2, 207, 32), finite=False),
                       Tensor((2, 207, 32), floating=False),
                       Tensor((2, 207, 32), device='cuda')):
            with self.subTest(result=result), self.assertRaises(ValueError):
                encode_window_context(self.wave, self.windows, lambda _: result)


if __name__ == '__main__':
    unittest.main()
