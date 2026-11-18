"""Actual _DecodeStream temporal math with shape tensors, not GPU acceptance."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import test_h3_conversation_continuation as fixtures
from research.h3_personal.incremental_decode import _DecodeStream
from research.h3_personal.conversation_decode import decode, abandon


class Shape:
    device = 'cpu'
    dtype = 'float32'

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def __getitem__(self, key):
        key = key + (slice(None),)*(self.ndim-len(key))
        return Shape([len(range(*k.indices(n))) for k, n in zip(key, self.shape)])

    def clone(self):
        return Shape(self.shape)


def cat(parts, dim):
    shape = list(parts[0].shape)
    shape[dim] = sum(p.shape[dim] for p in parts)
    return Shape(shape)


class VAE:
    tokens_chunk_size = 5
    temporal_compression_ratio = 4
    token_overlap = 2
    frame_pre_padding = 3
    frame_overlap = 5
    config = SimpleNamespace(token_drop=3)

    def _decode_clip(self, z):
        return Shape((1, 3, 28, 2, 2))

    def _blend(self, old, new, *args, **kw):
        return new


class Ops:
    rank = 0
    fail = None

    def __init__(self, vae):
        self.gates = 0

    def common(self, error):
        self.gates += 1
        if error:
            raise RuntimeError('common failure') from error

    def prepare(self, vae, rows, count, geometry):
        if self.fail == 'prepare':
            raise ValueError('copy failed')
        return Shape((1, 24, count, 2, 2))

    def inference(self):
        return nullcontext()

    autocast = inference

    def pixels(self, vae, chunk):
        return chunk

    def output(self, chunks):
        if self.fail == 'output':
            raise MemoryError('allocation failed')
        return cat(chunks, 2) if self.rank == 0 else Shape((0, 3, 0, 0, 0))


def stream(vae, *, parallel, waves):
    s = _DecodeStream(vae)
    s.native, s.parallel, s.waves = vae, parallel, waves
    return s


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.ConversationTests()
        self.owner = self.helper.owner()
        self.stage = SimpleNamespace(vae=VAE(), _h3_conversation_tensor_ops=Ops,
                                     _h3_conversation_stream_factory=stream)
        self.args = SimpleNamespace(vae_parallel_decode=True)
        self.fake_torch = patch.dict(sys.modules, torch=SimpleNamespace(cat=cat))
        self.fake_torch.start()
        self.addCleanup(self.fake_torch.stop)

    def committed(self, source=None, total=124, final=False):
        if source:
            self.helper.source(self.owner, source, total)
        t = self.owner.begin_window()
        r = self.helper.commit(self.owner, t, final)
        b = SimpleNamespace(raw_latent_shape=(1, 24, 37, 2, 2), extra={})
        ctx = dict(conversation_owner=self.owner, global_token=t, global_suffix=r,
                   manifest=dict(source_index=t.local.index, final=True))
        return b, ctx

    def run_decode(self, pair):
        return decode(self.stage, pair[0], self.args, pair[1])

    def test_real_stream_across_sources_actual_global_math(self):
        counts = []
        pairs = [('listen', 124, False), ('speech', 328, False),
                 (None, 124, False), (None, 124, False), ('return', 124, True)]
        identity = None
        received = []
        for sid, total, final in pairs:
            b = self.run_decode(self.committed(sid, total, final))
            counts.append(b.output.shape[2])
            state = self.stage._h3_conversation_decoder
            identity = identity or state
            self.assertIs(state, identity)
            self.assertIs(state.owner, self.owner)
            received.append(state.stream.received)
            self.assertEqual(state.stream.closed, final)
            self.assertEqual(self.owner.emittedF, state.stream.emitted_frames)
        self.assertEqual(counts, [119, 102, 102, 102, 107])
        self.assertEqual(received, [37, 67, 97, 127, 157])
        self.assertEqual(sum(counts), 532)
        with self.assertRaises(RuntimeError):
            self.run_decode((b, dict(conversation_owner=self.owner)))

    def test_source_terminal_does_not_flush_and_stale_offset_poisoned(self):
        self.run_decode(self.committed('a'))
        self.assertFalse(self.stage._h3_conversation_decoder.stream.closed)
        pair = self.committed('b')
        r = pair[1]['global_suffix']
        pair[1]['global_suffix'] = replace(r, suffix=replace(r.suffix, video_row_offset=0))
        with self.assertRaises(RuntimeError): self.run_decode(pair)
        self.assertEqual(self.owner.emittedF, 119)
        self.assertTrue(self.stage._h3_conversation_decoder.failed)

    def test_profile_drift_and_explicit_abandon_no_model_close(self):
        self.run_decode(self.committed('a'))
        pair = self.committed('b')
        pair[1]['global_token'] = replace(pair[1]['global_token'], profile='other')
        with self.assertRaises(RuntimeError): self.run_decode(pair)
        abandon(self.stage)
        self.assertIsNone(self.stage._h3_conversation_decoder.stream.buffer)
        self.assertFalse(self.owner._terminal)  # no inferred model health
        with self.assertRaises(RuntimeError): self.run_decode(pair)

    def test_copy_and_output_failures_never_ack(self):
        for failure in ('prepare', 'output'):
            with self.subTest(failure=failure):
                self.setUp()
                with patch.object(Ops, 'fail', failure):
                    with self.assertRaises(RuntimeError):
                        self.run_decode(self.committed('a'))
                self.assertEqual(self.owner.emittedF, 0)
                self.assertTrue(self.stage._h3_conversation_decoder.failed)

    def test_rank_one_participates_but_returns_empty(self):
        with patch.object(Ops, 'rank', 1):
            result = self.run_decode(self.committed('a'))
        self.assertEqual(result.output.shape, (0, 3, 0, 0, 0))
        self.assertEqual(self.owner.emittedF, 119)

    def test_shared_post_decode_receipt_is_written(self):
        from research.h3_personal.fast_continuation_decode import RECEIPT_KEY
        result = self.run_decode(self.committed('a'))
        receipt = result.extra[RECEIPT_KEY]
        self.assertTrue(receipt['conversation_owner'])
        self.assertEqual(receipt['emitted_frames'], 119)
        self.assertEqual(receipt['committed_frames'], 124)
        self.assertEqual(receipt['frame_count'], 119)
        self.assertEqual(receipt['retained_latents'], 2)
        self.assertIs(receipt['final'], False)
        self.assertTrue(receipt['bounded_buffers'])


if __name__ == '__main__':
    unittest.main()
