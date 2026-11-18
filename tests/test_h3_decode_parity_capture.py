"""Capture lifecycle/collective contracts using stdlib-only Torch doubles."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from research.h3_personal import decode_parity_capture as capture


class Tensor:
    def __init__(self, values, shape=None):
        self.values = list(values)
        self.shape = shape or (1, 24, len(self.values), 2, 2)
        self.device = 'cpu'

    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def clone(self):
        return Tensor(self.values, self.shape)

    def contiguous(self):
        return self


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def all(self):
        return self


class Torch(types.ModuleType):
    float32 = 'float32'
    int32 = 'int32'

    def __init__(self):
        super().__init__('torch')
        self.votes = []
        self.peer = 1
        self.transport_error = None
        self.initialized = True
        self.world_size = 2
        self.rank = 0
        self.distributed = types.SimpleNamespace(
            is_initialized=lambda: self.initialized,
            get_world_size=lambda: self.world_size,
            get_rank=lambda: self.rank,
            ReduceOp=types.SimpleNamespace(MIN=object()),
            all_reduce=self.all_reduce)

    def all_reduce(self, value, op):
        self.votes.append((value.item(), op))
        if self.transport_error:
            raise self.transport_error
        value.value = min(value.value, self.peer)

    @staticmethod
    def tensor(value, **kwargs):
        return Scalar(value)

    @staticmethod
    def isfinite(value):
        return Scalar(all(math.isfinite(v) for v in value.values))

    @staticmethod
    def cat(parts, dim):
        assert dim == 2
        return Tensor([v for p in parts for v in p.values],
                      (1, 24, sum(p.shape[2] for p in parts), *parts[0].shape[3:]))


def manifest(index=0, **overrides):
    result = dict(session_id='private-session', total_frames=328,
                  index=index, final=index == 2)
    result.update(overrides)
    return result


def suffix(index):
    start, count = ((0, 37), (37, 30), (67, 30))[index]
    return Tensor(range(start, start + count))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.torch = Torch()
        self.enterContext(patch.dict(sys.modules, {'torch': self.torch}))
        self.compare = self.enterContext(patch.object(
            capture, 'compare_decode', return_value={'equal': True, 'max_error': 0.0}))
        self.digest = self.enterContext(patch.object(
            capture, '_digest', side_effect=lambda t: hashlib.sha256(
                repr((t.shape, t.values)).encode()).hexdigest()))
        self.vae = types.SimpleNamespace(latents_mean=types.SimpleNamespace(device='cpu'))
        self.stage = types.SimpleNamespace(vae=self.vae)

    def complete(self):
        owner = capture.Capture(manifest())
        for i in range(3):
            owner.append(manifest(i), suffix(i))
        return owner

    def assert_faulted(self, owner):
        self.assertTrue(owner.failed)
        self.assertTrue(owner.closed)
        self.assertEqual(owner.parts, [])
        with self.assertRaises(ValueError):
            owner.append(manifest(), suffix(0))
        with self.assertRaises(ValueError):
            owner.finish(self.vae)

    def test_exact_owned_copies_37_30_30_finish_closes_and_releases(self):
        owner = capture.Capture(manifest())
        sources = [suffix(i) for i in range(3)]
        for i, source in enumerate(sources):
            owner.append(manifest(i), source)
            self.assertIsNot(owner.parts[i], source)
            self.assertIsNot(owner.parts[i].values, source.values)
            self.assertEqual(owner.parts[i].values, source.values)
            self.assertEqual(owner.parts[i].device, 'cpu')
            source.values[:] = [-999] * len(source.values)
        self.assertEqual(owner.arrivals, [37, 30, 30])
        result = owner.finish(self.vae)
        args, kwargs = self.compare.call_args
        self.assertIs(args[0], self.vae)
        self.assertEqual(args[1].values, list(range(97)))
        self.assertEqual(args[1].shape, (1, 24, 97, 2, 2))
        self.assertEqual(kwargs, dict(arrivals=(37, 30, 30), atol=0.0))
        self.assertEqual(result['session_id'], 'private-session')
        self.assertEqual(len(result['input_sha256']), 64)
        self.assertTrue(owner.closed)
        self.assertFalse(owner.failed)
        self.assertEqual(owner.parts, [])
        with self.assertRaises(ValueError):
            owner.append(manifest(2), suffix(2))

    def test_manifest_shape_and_canvas_rejections_do_not_append(self):
        cases = [
            (manifest(session_id='other'), suffix(0)),
            (manifest(total_frames=226), suffix(0)),
            (manifest(index=1), suffix(0)),
            (manifest(index=False), suffix(0)),
            (manifest(final=True), suffix(0)),
            (manifest(final=0), suffix(0)),
        ]
        for shape in [(1, 24, 30, 2, 2), (1, 23, 37, 2, 2),
                      (2, 24, 37, 2, 2), (1, 24, 37, 2),
                      (1, 24, 37, 3, 2), (1, 24, 37, 0, 2),
                      (1, 24, 37, 56, 34)]:
            cases.append((manifest(), Tensor(range(37), shape)))
        cases.append((manifest(), Tensor([float('nan')] * 37)))
        for metadata, tensor in cases:
            with self.subTest(metadata=metadata, shape=tensor.shape):
                owner = capture.Capture(manifest())
                with self.assertRaises(ValueError):
                    owner.append(metadata, tensor)
                self.assertEqual((owner.parts, owner.arrivals, owner.index), ([], [], 0))
        owner = capture.Capture(manifest())
        owner.append(manifest(), suffix(0))
        with self.assertRaisesRegex(ValueError, 'canvas drift'):
            owner.append(manifest(1), Tensor(range(30), (1, 24, 30, 4, 2)))
        self.assertEqual(owner.index, 1)
        self.compare.assert_not_called()

    def test_incomplete_finish_faults_and_refuses_reuse(self):
        owner = capture.Capture(manifest())
        owner.append(manifest(), suffix(0))
        with self.assertRaisesRegex(ValueError, 'complete committed'):
            owner.finish(self.vae)
        self.assert_faulted(owner)
        self.compare.assert_not_called()

    def test_compare_error_releases_parts_and_faults(self):
        owner = self.complete()
        self.compare.side_effect = RuntimeError('compare failed')
        with self.assertRaisesRegex(RuntimeError, 'compare failed'):
            owner.finish(self.vae)
        self.assert_faulted(owner)

    def test_two_rank_min_local_recording_peer_rejection_and_transport_error(self):
        for mode, vote in [('success', 1), ('local', 0), ('peer', 1), ('transport', 1)]:
            with self.subTest(mode=mode):
                self.stage = types.SimpleNamespace(vae=self.vae)
                self.torch.peer = 0 if mode == 'peer' else 1
                self.torch.transport_error = OSError('transport') if mode == 'transport' else None
                context = {'manifest': manifest(final=True) if mode == 'local' else manifest()}
                if mode == 'success':
                    owner = capture.record_suffix(self.stage, context, suffix(0))
                    self.assertIs(owner, self.stage._h3_decode_parity_capture)
                    self.assertEqual(owner.arrivals, [37])
                else:
                    with self.assertRaises(OSError if mode == 'transport' else RuntimeError):
                        capture.record_suffix(self.stage, context, suffix(0))
                    self.assert_faulted(self.stage._h3_decode_parity_capture)
                self.assertEqual(self.torch.votes[-1],
                                 (vote, self.torch.distributed.ReduceOp.MIN))

    def test_finish_private_scalar_receipt_no_raw_files_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                os.environ, {'H3_REF2VA_COVERAGE_DIR': folder}):
            owner = self.complete()
            self.stage._h3_decode_parity_capture = owner
            result = capture.finish_parity(self.stage, owner)
            path = Path(folder) / 'decode-parity-rank-0.json'
            original = path.read_bytes()
            self.assertEqual(json.loads(original), result)
            self.assertEqual(set(result), {'equal', 'max_error', 'session_id',
                                          'total_frames', 'input_sha256', 'worker_rank'})
            self.assertTrue(all(type(v) in (str, int, float, bool) for v in result.values()))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            replacement = self.complete()
            self.stage._h3_decode_parity_capture = replacement
            with self.assertRaisesRegex(RuntimeError, 'rejected') as caught:
                capture.finish_parity(self.stage, replacement)
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            self.assertIn('never overwrite', str(caught.exception.__cause__))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            self.assert_faulted(replacement)
            self.assertEqual(self.torch.votes[-1][0], 0)


if __name__ == '__main__':
    unittest.main()
