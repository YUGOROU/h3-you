"""CPU numerical/protocol doubles only; no learned/GPU acceptance."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_h3_decode_parity import NumericTensor, NumericTorch
from test_h3_decode_parity_capture import Torch, manifest, suffix
from research.h3_personal import parallel_decode_parity as parity
from research.h3_personal import decode_parity_capture as capture
from research.h3_personal.incremental_decode import _DecodeStream


class NumericalTests(unittest.TestCase):
    def run_compare(self, delta=0, fail=False, peer_failure=False):
        owner = self
        self.votes = []
        self.streams = []
        self.denormalizations = 0
        class T(NumericTorch):
            int32 = 'int32'
            @staticmethod
            def tensor(value, **kwargs): return NumericTensor([value], ())
        t = T('torch')
        def reduce(value, op):
            self.votes.append(value.item())
            if peer_failure and len(self.votes) == 1:
                value.values[:] = [0]
        t.distributed = types.SimpleNamespace(is_available=lambda: True,
            is_initialized=lambda: True, get_world_size=lambda: 2,
            all_reduce=reduce, ReduceOp=types.SimpleNamespace(MIN='min'))
        class VAE:
            tokens_chunk_size = 5
            temporal_compression_ratio = 4
            token_overlap = 2
            frame_pre_padding = 3
            frame_overlap = 5
            spatial_compression_ratio = 1
            config = types.SimpleNamespace(token_drop=3)
            latents_mean = types.SimpleNamespace(device=NumericTensor.device)
            def denormalize_latents(self, z):
                owner.denormalizations += 1
                return z*2
            def denormalize_pixels(self, z): return z*.5+.5
            def _decode_clip(self, z):
                if fail: raise ValueError('local numerical failure')
                shape = (1, 3, 28, 2, 2)
                import itertools
                return NumericTensor((z.at((0, c, f//4, h, w))*.1
                    for _, c, f, h, w in itertools.product(*(range(n) for n in shape))), shape)
            def _blend(self, a, b, overlap, dim):
                return t.cat((a[:, :, :overlap]*.5+b[:, :, :overlap]*.5,
                              b[:, :, overlap:]), dim=2)
        class Stream(_DecodeStream):
            # Numerical delegate: actual temporal math, not distributed transport.
            def __init__(self, vae, *, parallel, waves):
                assert parallel
                self.waves = waves
                super().__init__(vae)
                owner.streams.append(self)
            def push(self, z, final=False):
                for chunk in super().push(z, final=final):
                    yield chunk+(delta if self.waves else 0)
        z = NumericTensor(((i % 31-15)/32 for i in range(24*97*4)), (1, 24, 97, 2, 2))
        with patch.dict(sys.modules, {'torch': t}), patch.object(parity, 'verify_fast_source'), \
                patch.object(parity, 'FastVideoDecodeStream', Stream):
            return parity.compare_decode(VAE(), z)

    def test_full_function_exact_and_metrics(self):
        result = self.run_compare()
        self.assertEqual(self.votes, [1, 1, 1, 1])
        self.assertEqual(self.denormalizations, 1)
        self.assertEqual([s.waves for s in self.streams], [False, True])
        self.assertTrue(all(s.closed for s in self.streams))
        self.assertEqual(result['schema'], 'h3-parallel-decode-parity-v1')
        self.assertEqual(result['emitted_counts'], [119, 102, 107])
        self.assertEqual(result['reference_emitted_counts'], [119, 102, 107])
        self.assertFalse(result['serial'])
        self.assertFalse(result['native_parallel_verified'])
        self.assertFalse(result['cuda_fp16_autocast'])
        self.assertTrue(result['parity'])
        for key in ('raw_float', 'denormalized_pixels', 'uint8'):
            self.assertTrue(result[key]['exact'])
            self.assertEqual(result[key]['per_frame_max'], [0.]*328)
        json.dumps(result, allow_nan=False)

    def test_numeric_difference(self):
        result = self.run_compare(delta=.125)
        self.assertFalse(result['parity'])
        self.assertAlmostEqual(result['raw_float']['max'], .125)
        self.assertAlmostEqual(result['raw_float']['mean'], .125)
        self.assertFalse(result['uint8']['exact'])

    def test_phase_failure_does_not_enter_next_run(self):
        with self.assertRaisesRegex(RuntimeError, 'alternating completion'):
            self.run_compare(fail=True)
        self.assertEqual(self.votes, [1, 0])
        self.assertEqual(self.streams[1].received, 0)
        with self.assertRaisesRegex(RuntimeError, 'input preparation'):
            self.run_compare(peer_failure=True)
        self.assertEqual(self.votes, [1])
        self.assertTrue(all(s.received == 0 for s in self.streams))


class CaptureProtocolTests(unittest.TestCase):
    def test_decode_default_and_false_do_not_record(self):
        from test_h3_fast_continuation_decode import (
            DecodeProtocolTests, VAE, session, committed, modules,
        )
        from research.h3_personal import fast_continuation_decode as decode
        helper = DecodeProtocolTests()
        for flag in (None, False):
            helper.unpack_offset = 0
            helper.unpack_counts = []
            owner = session()
            stage = types.SimpleNamespace(vae=VAE())
            with patch.dict(os.environ, {'H3_SMD_DECODE_PARITY': '0', 'H3_SMD_PARALLEL_DECODE_WAVES': '0'}), \
                    patch.dict(sys.modules, modules()), patch.object(decode, 'verify_fast_source'), \
                    patch.object(decode, '_unpack', side_effect=helper.unpack), \
                    patch.object(capture, 'record_suffix') as record:
                for i in range(3):
                    batch = committed(owner, i)
                    if flag is not None:
                        batch.extra[decode.KEY]['manifest']['parallel_acceptance'] = flag
                    decode.decode_suffix(stage, batch, helper.args())
                record.assert_not_called()

    def test_dispatch_and_filenames_default_is_serial(self):
        for flag in (None, False, True):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as folder:
                t = Torch()
                vae = types.SimpleNamespace(latents_mean=types.SimpleNamespace(device='cpu'))
                m = manifest(**({} if flag is None else {'parallel_acceptance': flag}))
                c = capture.Capture(m)
                stage = types.SimpleNamespace(vae=vae, _h3_decode_parity_capture=c)
                with patch.dict(sys.modules, {'torch': t}), \
                        patch.object(capture, '_digest', return_value='digest'), \
                        patch.object(capture, 'compare_decode', return_value={'serial': True}) as serial, \
                        patch.object(parity, 'compare_decode', return_value={'serial': False}) as parallel, \
                        patch.dict(os.environ, {'H3_REF2VA_COVERAGE_DIR': folder}):
                    for i in range(3): c.append(manifest(i), suffix(i))
                    result = capture.finish_parity(stage, c)
                self.assertEqual(serial.call_count, int(flag is not True))
                self.assertEqual(parallel.call_count, int(flag is True))
                prefix = 'parallel-parity' if flag is True else 'decode-parity'
                self.assertEqual([p.name for p in Path(folder).iterdir()], [f'{prefix}-rank-0.json'])
                self.assertEqual(result['serial'], flag is not True)
                self.assertTrue(c.closed)
                self.assertEqual(c.parts, [])


if __name__ == '__main__':
    unittest.main()
