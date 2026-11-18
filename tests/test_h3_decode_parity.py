"""Offline protocol tests. Toy Torch decoder is NOT pinned GPU acceptance."""
import importlib.util
import itertools
import math
import types
from contextlib import nullcontext
import unittest
from unittest.mock import patch

from research.h3_personal import decode_parity as parity


class ValidationTests(unittest.TestCase):
    def test_arrival_grids_and_limits(self):
        parity._validate((1, 24, 97, 2, 2), (37, 30, 30), 0)
        for arrivals in ((30, 37, 30), (37, 60), (37, 30), (True, 96), ()):
            with self.assertRaises(ValueError):
                parity._validate((1, 24, 97, 2, 2), arrivals, 0)
        for atol in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                parity._validate((1, 24, 97, 2, 2), (37, 30, 30), atol)

    def test_geometry(self):
        for shape in ((1, 24, 132, 2, 2), (1, 24, 7, 3, 2), (2, 24, 7, 2, 2)):
            with self.assertRaises(ValueError):
                parity._validate(shape, (7,), 0)

    def test_source_guard_and_instance_override(self):
        class Toy:
            def _decode(self): pass
            def _decode_chunks(self): pass
            def _temporal_decode_plan(self): pass
        toy = Toy()
        with self.assertRaises(ValueError):
            parity._verify_native(toy)
        with patch.object(parity, 'verify_fast_source'):
            parity._verify_native(toy)
            toy._decode = lambda: None
            with self.assertRaises(ValueError):
                parity._verify_native(toy)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional local Torch unavailable')
class ToyTorchTests(unittest.TestCase):
    def run_toy(self, delta=0, mutate=False):
        import torch
        class VAE:
            spatial_compression_ratio = 1
            def denormalize_latents(self, z): return z * 2
            def denormalize_pixels(self, z): return z * .5 + .5
            def _decode(self, z):
                if mutate: z.add_(1)
                return torch.zeros((1, 3, 22, 2, 2))
        class Stream:
            def __init__(self, vae, parallel=False):
                self.closed = False
                self.received = self.emitted_frames = 0
                self.buffer = self.overlap = None
            def push(self, z, final=False):
                self.received += z.shape[2]
                self.emitted_frames = 22
                self.closed = final
                yield torch.full((1, 3, 22, 2, 2), float(delta))
        with patch.object(parity, '_verify_native'), patch.object(parity, 'FastVideoDecodeStream', Stream):
            return parity.compare_decode(VAE(), torch.zeros((1, 24, 7, 2, 2)), arrivals=(7,))

    def test_exact_and_failed_metrics(self):
        result = self.run_toy()
        self.assertTrue(result['parity'])
        self.assertEqual(result['emitted_counts'], [22])
        failed = self.run_toy(.1)
        self.assertFalse(failed['parity'])
        self.assertFalse(failed['uint8']['exact'])
        self.assertEqual(len(failed['raw_float']['per_frame_max']), 22)

    def test_native_input_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mutated'):
            self.run_toy(mutate=True)





class NumericTensor:
    """Small dense CPU double: values, indexing and reductions are real arithmetic."""
    device = types.SimpleNamespace(type='cpu')

    def __init__(self, values, shape, dtype='torch.float32'):
        self.values = list(values)
        self.shape = tuple(shape)
        self.dtype = dtype
        assert len(self.values) == math.prod(self.shape)

    @property
    def ndim(self): return len(self.shape)
    def coords(self): return itertools.product(*(range(n) for n in self.shape))
    def at(self, coord):
        index = 0
        for n, i in zip(self.shape, coord): index = index*n+i
        return self.values[index]
    def clone(self): return NumericTensor(self.values, self.shape, self.dtype)
    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key += (slice(None),)*(self.ndim-len(key))
        axes = [list(range(n))[k] if isinstance(k, slice) else [k]
                for n, k in zip(self.shape, key)]
        shape = tuple(len(a) for a, k in zip(axes, key) if isinstance(k, slice))
        coords = list(itertools.product(*axes))
        result = NumericTensor((self.at(c) for c in coords), shape, self.dtype)
        result.parent = self
        result.parent_indices = []
        for coord in coords:
            index = 0
            for n, i in zip(self.shape, coord): index = index*n+i
            result.parent_indices.append(index)
        return result
    def op(self, other, fn):
        values = other.values if isinstance(other, NumericTensor) else [other]*len(self.values)
        return NumericTensor((fn(a, b) for a, b in zip(self.values, values)), self.shape, self.dtype)
    def __add__(self, x): return self.op(x, lambda a, b: a+b)
    def __sub__(self, x): return self.op(x, lambda a, b: a-b)
    def __mul__(self, x): return self.op(x, lambda a, b: a*b)
    def mul(self, x): return self*x
    def add_(self, x):
        self.values[:] = (self+x).values
        if hasattr(self, 'parent'):
            for index, value in zip(self.parent_indices, self.values):
                self.parent.values[index] = value
        return self
    def abs(self): return NumericTensor(map(abs, self.values), self.shape, self.dtype)
    def clamp(self, lo, hi):
        return NumericTensor((min(hi, max(lo, v)) for v in self.values), self.shape, self.dtype)
    def to(self, dtype):
        return NumericTensor((int(v) if dtype == 'torch.uint8' else float(v)
                              for v in self.values), self.shape, dtype)
    def float(self): return self.to('torch.float32')
    def item(self):
        assert len(self.values) == 1
        return self.values[0]
    def all(self): return NumericTensor([all(self.values)], ())
    def tolist(self): return self.values[:]
    def permute(self, *axes):
        shape = tuple(self.shape[a] for a in axes)
        def value(c):
            original = [0]*self.ndim
            for a, v in zip(axes, c): original[a] = v
            return self.at(original)
        return NumericTensor((value(c) for c in itertools.product(*(range(n) for n in shape))), shape, self.dtype)
    def reshape(self, *shape):
        if -1 in shape:
            shape = tuple(len(self.values)//math.prod(n for n in shape if n != -1)
                          if n == -1 else n for n in shape)
        return NumericTensor(self.values, shape, self.dtype)
    def reduce(self, fn, dim):
        if dim is None: return NumericTensor([fn(self.values)], ())
        assert self.ndim == 2 and dim == 1
        width = self.shape[1]
        return NumericTensor((fn(self.values[i:i+width])
                              for i in range(0, len(self.values), width)), (self.shape[0],))
    def max(self, dim=None):
        result = self.reduce(max, dim)
        return result if dim is None else types.SimpleNamespace(values=result)
    def mean(self, dim=None): return self.reduce(lambda x: sum(x)/len(x), dim)


class NumericTorch(types.ModuleType):
    float32 = 'torch.float32'
    float16 = 'torch.float16'
    uint8 = 'torch.uint8'
    inference_mode = staticmethod(nullcontext)
    @staticmethod
    def autocast(**kwargs):
        assert kwargs == dict(device_type='cpu', dtype='torch.float16', enabled=False)
        return nullcontext()
    @staticmethod
    def equal(a, b): return a.shape == b.shape and a.values == b.values
    @staticmethod
    def isfinite(x): return NumericTensor(map(math.isfinite, x.values), x.shape)
    @staticmethod
    def cat(tensors, dim):
        shape = list(tensors[0].shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        def value(c):
            c = list(c)
            for t in tensors:
                if c[dim] < t.shape[dim]: return t.at(c)
                c[dim] -= t.shape[dim]
            raise AssertionError('out of bounds')
        return NumericTensor((value(c) for c in itertools.product(*(range(n) for n in shape))), shape)


class StdlibDecodeTests(unittest.TestCase):
    """Protocol/numerical diagnostic coverage, NOT learned decoder parity or GPU evidence."""
    def run_double(self, delta=0, mutate=None, fail_clip=False):
        from research.h3_personal.incremental_decode import _DecodeStream
        from research.h3_personal.fast_continuation_decode import FastVideoDecodeStream
        owner = self
        self.streams = []
        class VAE:
            tokens_chunk_size = 5
            temporal_compression_ratio = 4
            token_overlap = 2
            frame_pre_padding = 3
            frame_overlap = 5
            spatial_compression_ratio = 1
            config = types.SimpleNamespace(token_drop=3)
            def denormalize_latents(self, z): return z*2
            def denormalize_pixels(self, z): return z*.5+.5
            def _decode_clip(self, z):
                if fail_clip and owner.streams: raise RuntimeError('toy clip failure')
                # Context-dependent toy output, three channels and actual spatial values.
                shape = (1, 3, 28, 2, 2)
                return NumericTensor((z.at((0, c, t//4, h, w))*.1
                                      for _, c, t, h, w in itertools.product(*(range(n) for n in shape))), shape)
            def _blend(self, previous, chunk, overlap, dim):
                # Toy blend with meaningful overlap arithmetic; no learned weights.
                head = previous[:, :, :overlap]*.5+chunk[:, :, :overlap]*.5
                return NumericTorch.cat((head, chunk[:, :, overlap:]), dim=2)
            def _decode(self, z):
                if mutate == 'native': z.add_(1)
                state = _DecodeStream(self)
                return NumericTorch.cat(list(state.push(z, final=True)), dim=2)
        class ObservedStream(FastVideoDecodeStream):
            def __init__(self, vae, parallel=False):
                super().__init__(vae, parallel=parallel)
                owner.streams.append(self)
            def push(self, z, final=False):
                if mutate == 'candidate': z.add_(1)
                for chunk in super().push(z, final=final): yield chunk+delta
        z = NumericTensor(((i % 31-15)/32 for i in range(24*97*4)), (1, 24, 97, 2, 2))
        original = z.clone()
        # Explicit offline source-verifier bypass only; real stream algorithm runs.
        with patch.dict('sys.modules', {'torch': NumericTorch('torch')}), \
                patch.object(parity, '_verify_native'), \
                patch('research.h3_personal.fast_continuation_decode.verify_fast_source'), \
                patch.object(parity, 'FastVideoDecodeStream', ObservedStream):
            try:
                return parity.compare_decode(VAE(), z)
            finally:
                self.assertTrue(NumericTorch.equal(z, original))

    def test_exact_raw_and_uint8_and_arrival_mapping(self):
        result = self.run_double()
        self.assertTrue(result['parity'])
        self.assertEqual(result['emitted_counts'], [119, 102, 107])
        self.assertEqual(result['decoded_shape'], [1, 3, 328, 2, 2])
        for name in ('raw_float', 'uint8'):
            self.assertEqual(result[name], dict(max=0., mean=0., exact=True,
                within_atol=True, per_frame_max=[0.]*328, per_frame_mean=[0.]*328))
        self.assertTrue(result['inputs_unchanged'])
        self.assertFalse(result['quality_verified'])
        state, = self.streams
        self.assertTrue(state.closed)
        self.assertFalse(state.busy)
        self.assertIsNone(state.buffer)
        self.assertIsNone(state.overlap)

    def test_nonzero_error_is_not_parity(self):
        result = self.run_double(delta=.125)
        self.assertFalse(result['parity'])
        for name in ('raw_float', 'uint8'):
            self.assertFalse(result[name]['exact'])
            self.assertFalse(result[name]['within_atol'])
            self.assertGreater(result[name]['mean'], 0)
            self.assertTrue(all(v > 0 for v in result[name]['per_frame_max']))
        self.assertAlmostEqual(result['raw_float']['max'], .125)
        self.assertAlmostEqual(result['raw_float']['mean'], .125)

    def test_failure_closes_stream_but_does_not_claim_buffer_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, 'toy clip failure'):
            self.run_double(fail_clip=True)
        state, = self.streams
        self.assertTrue(state.closed)
        self.assertFalse(state.busy)
        # Current implementation retains failed input; it does not promise release.
        self.assertIsNotNone(state.buffer)
        self.assertIsNone(state.overlap)
        with patch.dict('sys.modules', {'torch': NumericTorch('torch')}), \
                self.assertRaisesRegex(ValueError, 'already finalized'):
            next(state.push(state.buffer))

    def test_input_mutation_rejected(self):
        for side in ('native', 'candidate'):
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, 'mutated'):
                self.run_double(mutate=side)


if __name__ == '__main__':
    unittest.main()
