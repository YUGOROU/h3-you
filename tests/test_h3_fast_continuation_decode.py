"""Stdlib protocol/source-AST checks; optional real CPU Torch numerical check.

No model import, dependency install, GPU, audio decode, device or cloud activity.
"""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from research.h3_personal import fast_continuation_decode as decode
from research.h3_personal.incremental_decode import NativeDecodeStream, _DecodeStream
from research.h3_personal.fast_continuation import Session, SamplerContract
from research.h3_personal.continuation_plan import video_t, audio_boundary
from test_h3_fast_continuation import Tensor, Ops

SOURCE = Path(os.environ.get('H3_PINNED_FASTVIDEO_VAE',
    '/tmp/fv-src/FastVideo-ba371e33ed080e51f1884ae166baf7c4447a66fe/fastvideo/models/vaes/minimax_h3_video.py'))
if 'H3_PINNED_FASTVIDEO_VAE' in os.environ and not SOURCE.is_file():
    raise ValueError('explicit pinned VAE fixture missing; refusing silent skip')


class Temporal:
    """Temporal-only tensor fake: not a numerical Torch/packing substitute."""
    def __init__(self, values, channels=3):
        self.values = list(values)
        self.shape = (1, channels, len(self.values), 1, 1)
        self.ndim = 5
        self.device = types.SimpleNamespace(type='cpu')
        self.dtype = 'float32'

    def __getitem__(self, key):
        return Temporal(self.values[key[2]], self.shape[1])

    def contiguous(self):
        return self

    def clone(self):
        return Temporal(self.values, self.shape[1])

    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def clamp_(self, low, high):
        self.values = [min(high, max(low, v)) for v in self.values]
        return self

    def mul_(self, value):
        self.values = [v*value for v in self.values]
        return self


class FakeTorch(types.ModuleType):
    Tensor = Temporal
    float32 = 'float32'
    float16 = 'float16'
    uint8 = 'uint8'

    def __init__(self):
        super().__init__('torch')

    @staticmethod
    def cat(parts, dim):
        assert dim == 2
        return Temporal([v for p in parts for v in p.values], parts[0].shape[1])

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def autocast(**kwargs):
        assert not kwargs['enabled']
        return nullcontext()

    @staticmethod
    def empty(shape, **kwargs):
        return types.SimpleNamespace(shape=shape, **kwargs)


class VAE:
    tokens_chunk_size = 5
    temporal_compression_ratio = 4
    token_overlap = 2
    frame_pre_padding = 3
    frame_overlap = 5
    config = types.SimpleNamespace(token_drop=3, clip_length=17)
    latents_mean = types.SimpleNamespace(device=types.SimpleNamespace(type='cpu'))

    def __init__(self):
        self.calls = []

    def _decode_clip(self, z):
        self.calls.append(tuple(z.values))
        return Temporal([v/100 for v in z.values for _ in range(4)])

    @staticmethod
    def _blend(a, b, extent, dim):
        assert extent == 5 and dim == -3
        return Temporal([a.values[-5+i]*(1-i/5)+b.values[i]*i/5 for i in range(5)] + b.values[5:])

    def denormalize_latents(self, z):
        return z

    def denormalize_pixels(self, z):
        return z.clone()


def modules(output_rank=True):
    distributed = types.ModuleType('fastvideo.distributed')
    distributed.model_parallel_is_initialized = lambda: True
    encoder = types.ModuleType('fastvideo.pipelines.basic.minimax_h3.encoder_split')
    encoder.h3_is_output_worker = lambda args: output_rank
    return {'torch': FakeTorch(), 'fastvideo.distributed': distributed,
            'fastvideo.pipelines.basic.minimax_h3.encoder_split': encoder}


def session(name='turn', total=328):
    return Session(name, source_audio=Tensor((2, audio_boundary(total), 32)),
                   total_frames=total, height=2, width=2, window_frames=124, ops=Ops())


def committed(owner, index):
    token = owner.begin_window(index)
    suffix = owner.commit_window(token,
        clean_video=Tensor((video_t(token.window.frames), 96)),
        clean_audio=token.locked_audio, observed_sampler=SamplerContract())
    context = dict(manifest=dict(session_id=owner.session_id, index=index,
                                final=index == len(owner.windows)-1, total_frames=owner._total),
                   owner=owner, token=token, suffix=suffix)
    return types.SimpleNamespace(extra={decode.KEY: context}, raw_latent_shape=(1, 24, 37, 2, 2))


class DecodeProtocolTests(unittest.TestCase):
    def args(self, **kwargs):
        return types.SimpleNamespace(vae_parallel_decode=False, **kwargs)

    def test_shared_base_and_strict_native_pin_unchanged(self):
        self.assertIs(NativeDecodeStream.__bases__[0], _DecodeStream)
        self.assertIs(decode.FastVideoDecodeStream.__bases__[0], _DecodeStream)
        with self.assertRaises(ValueError):
            NativeDecodeStream(VAE())
        with self.assertRaises(ValueError):
            decode.FastVideoDecodeStream(VAE())
        bad = VAE()
        bad.frame_overlap = 4
        with patch.object(decode, 'verify_fast_source'), self.assertRaises(ValueError):
            decode.FastVideoDecodeStream(bad)

    def test_opt_in_and_unsupported_policies_before_tensor_work(self):
        self.assertIsNone(decode.decode_suffix(None, types.SimpleNamespace(extra={}), None))
        batch = types.SimpleNamespace(extra={decode.KEY: {}})
        with self.assertRaisesRegex(ValueError, 'boolean'):
            decode.decode_suffix(None, batch, types.SimpleNamespace(vae_parallel_decode='yes'))
        for args in (self.args(vae_cpu_offload=True), self.args(video_decode_backend='taeh3'),
                     self.args(output_type='latent')):
            with self.assertRaises(ValueError):
                decode.decode_suffix(None, batch, args)

    def unpack(self, rows, count, height, width, channels):
        self.assertEqual((rows.shape, height, width, channels), ((count, 96), 2, 2, 24))
        start = self.unpack_offset
        self.unpack_offset += count
        self.unpack_counts.append(count)
        return Temporal(range(start, start+count), 24)

    def test_actual_commits_suffix_counts_bounded_clip_participation_and_new_turn(self):
        owner = session()
        stage = types.SimpleNamespace(vae=VAE())
        self.unpack_offset = 0
        self.unpack_counts = []
        with patch.dict(sys.modules, modules()), patch.object(decode, 'verify_fast_source'), \
                patch.object(decode, '_unpack', side_effect=self.unpack):
            for index, frames, offset in ((0, 119, 0), (1, 102, 119), (2, 107, 221)):
                batch = committed(owner, index)
                audio = batch.extra[decode.KEY]['suffix'].audio
                self.assertIs(decode.decode_suffix(stage, batch, self.args()), batch)
                receipt = batch.extra[decode.RECEIPT_KEY]
                self.assertEqual(batch.output.shape[2], frames)
                self.assertEqual((receipt['frame_offset'], receipt['frame_count']), (offset, frames))
                self.assertEqual(receipt['retained_latents'], 0 if index == 2 else 2)
                self.assertEqual(receipt['retained_overlap_frames'], 0 if index == 2 else 5)
                self.assertEqual(receipt['two_latent_lookahead'], 2)
                self.assertFalse(receipt['causal'])
                self.assertFalse(receipt['parallel_preserved'])
                self.assertIs(batch.extra[decode.KEY]['suffix'].audio, audio)
                with self.assertRaises(ValueError):
                    decode.decode_suffix(stage, batch, self.args())
            self.assertEqual(self.unpack_counts, [37, 30, 30])
            self.assertEqual(stage.vae.calls, [tuple(range(i, i+7)) for i in range(0, 95, 5)])
            self.assertEqual(stage._h3_continuation_decoder.stream.emitted_frames, 328)
            fresh = committed(session('next', total=124), 0)
            self.unpack_offset = 0
            decode.decode_suffix(stage, fresh, self.args())
            self.assertEqual(fresh.output.shape[2], 124)

    def test_parity_opt_in_records_normalized_suffixes_and_finishes_once(self):
        from research.h3_personal import decode_parity_capture as parity
        self.unpack_offset = 0
        self.unpack_counts = []
        owner, stage = session(), types.SimpleNamespace(vae=VAE())
        marker = object()
        observed = []
        def record(stage_arg, context, z):
            self.assertIs(stage_arg, stage)
            observed.append((context['manifest']['index'], tuple(z.values)))
            return marker
        with patch.dict(os.environ, {parity.ENV: '1'}), patch.dict(sys.modules, modules()), \
                patch.object(decode, 'verify_fast_source'), patch.object(decode, '_unpack', side_effect=self.unpack), \
                patch.object(parity, 'record_suffix', side_effect=record), \
                patch.object(parity, 'finish_parity') as finish:
            for i in range(3):
                decode.decode_suffix(stage, committed(owner, i), self.args())
                self.assertEqual(finish.call_count, int(i == 2))
            finish.assert_called_once_with(stage, marker)
        self.assertEqual([len(z) for _, z in observed], [37, 30, 30])
        self.assertEqual([v for _, z in observed for v in z], list(range(97)))

    def test_parity_failure_poisoned_before_final_output(self):
        from research.h3_personal import decode_parity_capture as parity
        self.unpack_offset = 0
        self.unpack_counts = []
        owner, stage = session(total=124), types.SimpleNamespace(vae=VAE())
        capture = types.SimpleNamespace(fault=lambda: None)
        stage._h3_decode_parity_capture = capture
        batch = committed(owner, 0)
        with patch.dict(os.environ, {parity.ENV: '1'}), patch.dict(sys.modules, modules()), \
                patch.object(decode, 'verify_fast_source'), patch.object(decode, '_unpack', side_effect=self.unpack), \
                patch.object(parity, 'record_suffix', return_value=capture), \
                patch.object(parity, 'finish_parity', side_effect=RuntimeError('parity failed')):
            with self.assertRaisesRegex(RuntimeError, 'parity failed'):
                decode.decode_suffix(stage, batch, self.args())
        self.assertTrue(stage._h3_continuation_decoder.failed)
        self.assertNotIn(decode.RECEIPT_KEY, batch.extra)
        self.assertFalse(hasattr(batch, 'output'))

    def test_nonoutput_ranks_placeholder_at_each_index_no_vae_work(self):
        owner = session()
        stage = types.SimpleNamespace(vae=None)
        with patch.dict(sys.modules, modules(False)), \
                patch.object(decode, 'FastVideoDecodeStream', side_effect=AssertionError('VAE touched')):
            for i in range(3):
                batch = committed(owner, i)
                decode.decode_suffix(stage, batch, self.args())
                self.assertEqual(batch.output.shape, (0, 3, 0, 0, 0))
                self.assertEqual(batch.output.device, 'cpu')

    def test_manifest_suffix_and_final_grid_validation(self):
        owner = session()
        batch = committed(owner, 0)
        context = batch.extra[decode.KEY]
        for key, bad in (('index', 1), ('final', True), ('total_frames', 327), ('session_id', 'foreign')):
            prior = context['manifest'][key]
            context['manifest'][key] = bad
            with self.assertRaises(ValueError):
                decode._validate(context, batch)
            context['manifest'][key] = prior
        batch.raw_latent_shape = (1, 24, 37, 3, 2)
        with self.assertRaises(ValueError):
            decode._validate(context, batch)

    def test_cannot_replace_unfinished_turn_or_decode_skipped_index(self):
        stage = types.SimpleNamespace(vae=VAE())
        owner = session()
        self.unpack_offset = 0
        self.unpack_counts = []
        with patch.dict(sys.modules, modules()), patch.object(decode, 'verify_fast_source'), \
                patch.object(decode, '_unpack', side_effect=self.unpack):
            decode.decode_suffix(stage, committed(owner, 0), self.args())
            with self.assertRaisesRegex(ValueError, 'previous complete'):
                decode.decode_suffix(stage, committed(session('other'), 0), self.args())
            committed(owner, 1)
            with self.assertRaisesRegex(ValueError, 'index'):
                decode.decode_suffix(stage, committed(owner, 2), self.args())

    def test_failed_turn_is_poisoned_no_redecode(self):
        stage = types.SimpleNamespace(vae=VAE())
        batch = committed(session(), 0)
        with patch.dict(sys.modules, modules()), patch.object(decode, 'verify_fast_source'), \
                patch.object(decode, '_unpack', side_effect=RuntimeError('input failure')):
            with self.assertRaises(RuntimeError):
                decode.decode_suffix(stage, batch, self.args())
            self.assertTrue(stage._h3_continuation_decoder.failed)
            with self.assertRaises(ValueError):
                decode.decode_suffix(stage, batch, self.args())
            self.assertEqual(stage.vae.calls, [])


class Wire:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class TensorGroup:
    """Two real host threads, fixed tensor collectives only; no Torch/network."""
    def __init__(self):
        self.local = threading.local()
        self.barrier = threading.Barrier(2, timeout=3)
        self.slots = {}
        self.payloads = []

    def is_available(self): return True
    def is_initialized(self): return True
    def get_world_size(self): return 2
    def get_rank(self): return self.local.rank
    def get_backend(self): return 'gloo'

    def exchange(self, tensor, src=None):
        rank = self.get_rank()
        self.slots[rank] = copy.deepcopy(tensor)
        self.barrier.wait()
        result = copy.deepcopy(self.slots if src is None else self.slots[src])
        self.barrier.wait()
        return result

    def all_gather(self, outputs, tensor):
        peers = self.exchange(tensor)
        for rank, out in enumerate(outputs):
            out.values = peers[rank].values

    def broadcast(self, tensor, src):
        received = self.exchange(tensor, src)
        tensor.__dict__.update(received.__dict__)
        if self.get_rank() == 0 and isinstance(tensor, Temporal):
            self.payloads.append((src, tensor.shape, tensor.dtype))


class ParallelTorch(FakeTorch):
    bfloat16 = 'bfloat16'
    int64 = 'int64'

    def __init__(self, group):
        super().__init__()
        self.distributed = group

    @staticmethod
    def tensor(values, **kwargs): return Wire(values)

    @staticmethod
    def empty_like(tensor): return Wire([0]*len(tensor.values))

    @staticmethod
    def empty(shape, **kwargs):
        if shape[0] == 0:
            return FakeTorch.empty(shape, **kwargs)
        out = Temporal([0]*shape[2], shape[1])
        out.shape = tuple(shape)
        out.dtype = kwargs['dtype']
        out.device = kwargs['device']
        return out


class TwoRankTests(unittest.TestCase):
    def run_pair(self, fn, group):
        def run(rank):
            group.local.rank = rank
            return fn(rank)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, rank) for rank in range(2)]
            return [f.result(timeout=10) for f in futures]

    def test_full_suffix_entry_both_ranks_alternating_and_rank1_no_pixels(self):
        self.full_suffix_pair(False)

    def test_two_clip_waves_match_alternating_outputs_and_owners(self):
        old = self.full_suffix_pair(False)
        new = self.full_suffix_pair(True)
        self.assertEqual(old, new)

    def full_suffix_pair(self, waves):
        group = TensorGroup()
        stages = [types.SimpleNamespace(vae=VAE()) for _ in range(2)]
        for stage in stages:
            stage.vae.spatial_compression_ratio = 1
        # Any CPU RGB conversion on rank1 is a test failure.
        stages[1].vae.denormalize_pixels = lambda z: self.fail('rank1 converted pixels')
        def run(rank):
            owner = session()
            offset = 0
            receipts = []
            outputs = []
            for i in range(3):
                batch = committed(owner, i)
                count = video_t(owner.windows[i].frames)-owner.windows[i].guide_video_t
                z = Temporal(range(offset, offset+count), 24)
                offset += count
                # Thread-local unpack without changing the shared mocked function.
                group.local.z = z
                decode.decode_suffix(stages[rank], batch,
                                     types.SimpleNamespace(vae_parallel_decode=True))
                receipts.append(batch.extra[decode.RECEIPT_KEY])
                outputs.append(batch.output)
            return receipts, outputs
        mods = modules(False)  # Even encoder nonoutput ranks must decode.
        mods['torch'] = ParallelTorch(group)
        with patch.dict(os.environ, {'H3_SMD_PARALLEL_DECODE_WAVES': '1' if waves else '0'}), \
                patch.dict(sys.modules, mods), patch.object(decode, 'verify_fast_source'), \
                patch.object(decode, '_unpack', side_effect=lambda *a: group.local.z):
            results = self.run_pair(run, group)
        for rank in range(2):
            self.assertTrue(stages[rank]._h3_continuation_decoder.stream.closed)
            self.assertEqual(stages[rank].vae.calls,
                             [tuple(range(i*5, i*5+7)) for i in range(19) if i % 2 == rank])
            for receipt in results[rank][0]:
                self.assertEqual(receipt['execution_policy'],
                                 'two-rank-two-clip-waves' if waves else 'two-rank-alternating-native-clips')
                if waves:
                    self.assertEqual(receipt['decode_schedule'], 'two-clip-waves-v1')
                    self.assertEqual(receipt['max_buffer_latents'], 12)
                    self.assertEqual(receipt['max_queued_clips'], 1)
                self.assertEqual(receipt['max_clip_latents'], 7)
                self.assertTrue(receipt['parallel_preserved'])
                self.assertTrue(receipt['participation'])
                self.assertFalse(receipt['parity_verified'])
                self.assertFalse(receipt['speed_verified'])
        self.assertEqual([o.shape[2] for o in results[0][1]], [119, 102, 107])
        self.assertTrue(all(o.shape == (0, 3, 0, 0, 0) for o in results[1][1]))
        self.assertEqual([p[0] for p in group.payloads], [i % 2 for i in range(19)])
        return [o.values for o in results[0][1]]

    def test_wave_forwards_concurrent_before_rendezvous_and_singleton(self):
        group = TensorGroup()
        forward_barrier = threading.Barrier(2, timeout=5)
        natives = [VAE(), VAE()]
        def run(rank):
            native = natives[rank]
            native.spatial_compression_ratio = 1
            original = native._decode_clip
            def forward(z):
                if z.values[0] < 10:
                    forward_barrier.wait()
                return original(z)
            native._decode_clip = forward
            adapter = decode._TwoRankWaveVAE(native)
            adapter.prepare_decode_wave(Temporal(range(12), 24))
            self.assertEqual(len(adapter.queued), 1)
            self.assertIsNotNone(adapter.current)
            first = adapter._decode_clip(Temporal(range(7), 24))
            self.assertIsNone(adapter.current)
            self.assertEqual(len(adapter.queued), 1)
            adapter.prepare_decode_wave(Temporal(range(5, 12), 24))
            second = adapter._decode_clip(Temporal(range(5, 12), 24))
            adapter.prepare_decode_wave(Temporal(range(10, 17), 24))
            third = adapter._decode_clip(Temporal(range(10, 17), 24))
            self.assertEqual(adapter.queued, [])
            return [x.values for x in (first, second, third)]
        with patch.dict(sys.modules, {'torch': ParallelTorch(group)}):
            result = self.run_pair(run, group)
        self.assertEqual(result[0], result[1])
        self.assertEqual(natives[0].calls, [tuple(range(7)), tuple(range(10, 17))])
        self.assertEqual(natives[1].calls, [tuple(range(5, 12))])

    def test_wave_iterator_cancel_clears_queued_clip_and_buffer(self):
        group = TensorGroup()
        def run(rank):
            native = VAE()
            native.spatial_compression_ratio = 1
            stream = _DecodeStream(native)
            stream.vae = decode._TwoRankWaveVAE(native)
            iterator = stream.push(Temporal(range(12), 24))
            next(iterator)
            self.assertEqual(len(stream.vae.queued), 1)
            iterator.close()
            self.assertTrue(stream.closed)
            self.assertIsNone(stream.buffer)
            self.assertIsNone(stream.overlap)
            self.assertEqual(stream.vae.queued, [])
            self.assertIsNone(stream.vae.current)
        with patch.dict(sys.modules, {'torch': ParallelTorch(group)}):
            self.run_pair(run, group)

    def test_cached_wave_state_drift_fails_both_before_payload(self):
        for fault in ('queued', 'index', 'count', 'current', 'geometry'):
            with self.subTest(fault=fault):
                group = TensorGroup()
                def run(rank):
                    native = VAE()
                    native.spatial_compression_ratio = 1
                    adapter = decode._TwoRankWaveVAE(native)
                    adapter.prepare_decode_wave(Temporal(range(12), 24))
                    adapter._decode_clip(Temporal(range(7), 24))
                    z = Temporal(range(5, 12), 24)
                    if rank == 1:
                        if fault == 'queued': adapter.queued.clear()
                        if fault == 'index': adapter.index += 1
                        if fault == 'count': z = Temporal(range(12), 24)
                        if fault == 'current': adapter.current = adapter.queued[0]
                        if fault == 'geometry': z.shape = (1, 24, 7, 2, 1)
                    try:
                        adapter.prepare_decode_wave(z)
                    except ValueError as exc:
                        self.assertEqual(adapter.queued, [])
                        self.assertIsNone(adapter.current)
                        self.assertEqual(len(native.calls), 1)
                        return str(exc)
                    self.fail('cached drift hidden')
                with patch.dict(sys.modules, {'torch': ParallelTorch(group)}):
                    results = self.run_pair(run, group)
                self.assertEqual(results, ['two-rank decode request/order/geometry drift']*2)
                self.assertEqual(len(group.payloads), 2)

    def test_wave_current_cleanup_and_native_output_ownership(self):
        group = TensorGroup()
        def run(rank):
            native = VAE()
            native.spatial_compression_ratio = 1
            original = native._decode_clip
            outputs = []
            def forward(z):
                out = original(z)
                outputs.append(out)
                return out
            native._decode_clip = forward
            adapter = decode._TwoRankWaveVAE(native)
            adapter.prepare_decode_wave(Temporal(range(12), 24))
            self.assertEqual(len(adapter.queued), 1)
            self.assertIsNotNone(adapter.current)
            owned = adapter.current if rank == 0 else adapter.queued[0]
            self.assertIsNot(owned, outputs[0])
            before = list(owned.values)
            outputs[0].values[:] = [-1]*28
            self.assertEqual(owned.values, before)
            adapter.clear_decode_wave()
            self.assertIsNone(adapter.current)
            self.assertEqual(adapter.queued, [])
        with patch.dict(sys.modules, {'torch': ParallelTorch(group)}):
            self.run_pair(run, group)

    def test_wave_constructor_requires_parallel(self):
        with self.assertRaisesRegex(ValueError, 'requires parallel'):
            decode.FastVideoDecodeStream(VAE(), waves=True)

    def test_wave_faults_stop_both_ranks_and_clear_state(self):
        for fault in ('descriptor', 'local', 'allocation', 'drift', 'header'):
            with self.subTest(fault=fault):
                group = TensorGroup()
                torch = ParallelTorch(group)
                empty = torch.empty
                def allocate(shape, **kwargs):
                    if fault == 'allocation' and group.get_rank() == 1:
                        raise RuntimeError('allocation')
                    return empty(shape, **kwargs)
                torch.empty = allocate
                broadcast = group.broadcast
                def corrupt(tensor, src):
                    broadcast(tensor, src)
                    if fault == 'header' and isinstance(tensor, Wire):
                        tensor.values[1] = -1
                group.broadcast = corrupt
                streams = []
                def run(rank):
                    native = VAE()
                    native.spatial_compression_ratio = 1
                    if fault == 'local' and rank == 1:
                        native._decode_clip = lambda z: (_ for _ in ()).throw(RuntimeError('local'))
                    stream = _DecodeStream(native)
                    stream.vae = decode._TwoRankWaveVAE(native)
                    streams.append(stream)
                    if fault == 'drift' and rank == 1:stream.vae.index = 2
                    z = Temporal(range(12), 24)
                    if fault == 'descriptor' and rank == 1:z.dtype = 'int64'
                    try:
                        list(stream.push(z))
                    except (ValueError, RuntimeError) as exc:
                        return str(exc)
                    self.fail('wave fault hidden')
                with patch.dict(sys.modules, {'torch': torch}):
                    result = self.run_pair(run, group)
                self.assertEqual(result[0], result[1])
                self.assertEqual(group.payloads, [])
                for stream in streams:
                    self.assertTrue(stream.closed)
                    self.assertIsNone(stream.buffer)
                    self.assertIsNone(stream.overlap)
                    self.assertEqual(stream.vae.queued, [])
                    self.assertIsNone(stream.vae.current)

    def adapter_pair(self, *, dtype='float32', bad=None, drift=False):
        group = TensorGroup()
        natives = [VAE(), VAE()]
        delegates = []
        def run(rank):
            native = natives[rank]
            native.spatial_compression_ratio = 1
            original = native._decode_clip
            def clip(z):
                if bad == 'failure': raise RuntimeError('owner error')
                out = original(z)
                out.dtype = 'int64' if bad == 'dtype' else dtype
                if bad == 'shape': out.shape = (1, 3, 40, 1, 1)
                return out
            native._decode_clip = clip
            adapter = decode._TwoRankVAE(native)
            delegates.append(adapter)
            if drift and rank == 1: adapter.index = 1
            try:
                out = adapter._decode_clip(Temporal(range(7), 24))
                return out.dtype, out.shape
            except (RuntimeError, ValueError) as exc:
                return str(exc)
        with patch.dict(sys.modules, {'torch': ParallelTorch(group)}):
            result = self.run_pair(run, group)
        return result, group, natives

    def test_exact_payload_dtype_and_bounded_shape(self):
        for dtype in ('float16', 'bfloat16', 'float32'):
            result, group, natives = self.adapter_pair(dtype=dtype)
            self.assertEqual(result, [(dtype, (1, 3, 28, 1, 1))]*2)
            self.assertEqual([len(n.calls) for n in natives], [1, 0])
            self.assertEqual(group.payloads, [(0, (1, 3, 28, 1, 1), dtype)])

    def test_owner_failure_invalid_payload_and_request_drift_synchronize(self):
        for bad in ('failure', 'dtype', 'shape'):
            result, group, _ = self.adapter_pair(bad=bad)
            self.assertEqual(result, ['native clip 0 owner 0 failed']*2)
            self.assertEqual(group.payloads, [])
        result, group, natives = self.adapter_pair(drift=True)
        self.assertTrue(all('request/order/geometry drift' in r for r in result))
        self.assertEqual([n.calls for n in natives], [[], []])
        self.assertEqual(group.payloads, [])

    def test_rank1_owner_failure_poisons_both_turns_without_fallback(self):
        group = TensorGroup()
        stages = [types.SimpleNamespace(vae=VAE()) for _ in range(2)]
        for stage in stages:
            stage.vae.spatial_compression_ratio = 1
        stages[1].vae._decode_clip = lambda z: (_ for _ in ()).throw(RuntimeError('rank1'))
        def run(rank):
            batch = committed(session(), 0)
            try:
                decode.decode_suffix(stages[rank], batch,
                                     types.SimpleNamespace(vae_parallel_decode=True))
            except RuntimeError as exc:
                return str(exc)
            self.fail('owner error was hidden')
        mods = modules()
        mods['torch'] = ParallelTorch(group)
        with patch.dict(sys.modules, mods), patch.object(decode, 'verify_fast_source'), \
                patch.object(decode, '_unpack', side_effect=lambda rows, count, *a: Temporal(range(count), 24)):
            self.assertEqual(self.run_pair(run, group), ['native clip 1 owner 1 failed']*2)
        for stage in stages:
            turn = stage._h3_continuation_decoder
            self.assertTrue(turn.failed)
            self.assertTrue(turn.stream.closed)
            self.assertIsNone(turn.stream.buffer)
            self.assertIsNone(turn.stream.overlap)
        self.assertEqual(len(stages[0].vae.calls), 1)
        self.assertEqual(len(group.payloads), 1)

    def test_payload_receiver_allocation_failure_is_collective(self):
        group = TensorGroup()
        torch = ParallelTorch(group)
        empty = torch.empty
        def allocate(shape, **kwargs):
            if group.get_rank() == 1:
                raise RuntimeError('allocation failed')
            return empty(shape, **kwargs)
        torch.empty = allocate
        def run(rank):
            native = VAE()
            native.spatial_compression_ratio = 1
            try:
                decode._TwoRankVAE(native)._decode_clip(Temporal(range(7), 24))
            except RuntimeError as exc:
                return str(exc)
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(self.run_pair(run, group),
                             ['two-rank native clip payload allocation failed']*2)
        self.assertEqual(group.payloads, [])

    def test_group_and_local_shape_rejected_before_collectives(self):
        group = TensorGroup()
        group.local.rank = 0
        native = VAE()
        native.spatial_compression_ratio = 1
        with patch.dict(sys.modules, {'torch': ParallelTorch(group)}):
            with patch.object(group, 'get_world_size', return_value=1):
                with self.assertRaisesRegex(ValueError, 'two-rank'):
                    decode._TwoRankVAE(native)
            adapter = decode._TwoRankVAE(native)
            with patch.object(group, 'all_gather', side_effect=AssertionError('collective')):
                with self.assertRaisesRegex(ValueError, 'seven-latent'):
                    adapter._decode_clip(Temporal(range(8), 24))
        self.assertEqual(native.calls, [])


@unittest.skipUnless(SOURCE.is_file(), 'local pinned FastVideo source required for source AST audit')
class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != decode.FAST_VAE_SHA:
            raise AssertionError('wrong FastVideo source SHA')
        cls.methods = {node.name: node for node in ast.walk(ast.parse(SOURCE.read_text()))
                       if isinstance(node, ast.FunctionDef)}

    def extract(self, names, torch):
        import copy
        namespace = {'torch': torch, 'nvtx_range': lambda *a: nullcontext(), 'Iterator': list}
        for name in names:
            node = copy.deepcopy(self.methods[name])
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         str(SOURCE), 'exec'), namespace)
        return namespace

    def test_sha_pin_and_method_ownership_are_enforced(self):
        methods = self.extract(('_decode_clip', '_blend', 'denormalize_latents', 'denormalize_pixels'), FakeTorch())
        name = '_h3_pinned_test_shell'
        module = types.ModuleType(name)
        module.__file__ = str(SOURCE)
        cls = type('PinnedShell', (VAE,), dict(
            __module__=name, _decode_clip=methods['_decode_clip'],
            _blend=staticmethod(methods['_blend']),
            denormalize_latents=methods['denormalize_latents'],
            denormalize_pixels=methods['denormalize_pixels']))
        with patch.dict(sys.modules, {name: module}):
            vae = cls()
            decode.verify_fast_source(vae)
            with patch.object(Path, 'read_bytes', return_value=b'tampered'):
                with self.assertRaisesRegex(ValueError, 'pinned'):
                    decode.verify_fast_source(vae)
            vae._decode_clip = lambda z: z
            with self.assertRaisesRegex(ValueError, 'overridden'):
                decode.verify_fast_source(vae)

    def test_native_decoder_seven_latents_produce_28_not_40_frames(self):
        tree = ast.parse(SOURCE.read_text())
        decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and
                       n.name == 'MiniMaxH3VideoViTDecoder3d')
        forward = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and
                       n.name == 'forward')
        returned = next(n for n in ast.walk(forward) if isinstance(n, ast.Return))
        self.assertEqual(ast.unparse(returned.value.func), 'hidden_states.reshape')
        self.assertEqual([ast.unparse(a) for a in returned.value.args],
                         ['batch_size', 'self.out_channels', 'num_frames * patch_size_t',
                          'height * patch_size', 'width * patch_size'])
        # Stream geometry strictly requires temporal_compression_ratio == 4;
        # native constructor passes that ratio as decoder patch_size_t.
        constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                        isinstance(n.func, ast.Name) and n.func.id == 'MiniMaxH3VideoViTDecoder3d']
        self.assertEqual(len(constructors), 1)
        keywords = {k.arg: ast.unparse(k.value) for k in constructors[0].keywords}
        self.assertEqual(keywords['patch_size_t'], 'self.temporal_compression_ratio')

    def test_native_source_ast_decode_boundaries_match_common_stream(self):
        methods = self.extract(('_temporal_decode_plan', '_decode_chunks', '_decode'), FakeTorch())
        native_type = type('NativeAST', (VAE,), {k: methods[k] for k in
                           ('_temporal_decode_plan', '_decode_chunks', '_decode')})
        for count in (7, 37, 97, 202):
            native = native_type()
            expected = native._decode(Temporal(range(count), 24))
            for width in (1, 5, 13, 64):
                incremental = VAE()
                with patch.dict(sys.modules, modules()), patch.object(decode, 'verify_fast_source'):
                    stream = decode.FastVideoDecodeStream(incremental)
                    parts = []
                    for start in range(0, count, width):
                        parts.extend(stream.push(Temporal(range(start, min(count, start+width)), 24),
                                                 final=start+width >= count))
                        if stream.buffer is not None:
                            self.assertLess(stream.buffer.shape[2], 7)
                    actual = FakeTorch.cat(parts, dim=2)
                self.assertEqual(actual.values, expected.values)
                self.assertEqual(incremental.calls, native.calls)
                self.assertEqual(stream.emitted_frames, 17*((count-2)//5)+5)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch unavailable: no numerical parity claim')
    def test_optional_torch_native_ast_blend_and_channel_major_unpack(self):
        import torch
        methods = self.extract(('_temporal_decode_plan', '_decode_chunks', '_decode', '_blend'), torch)
        class Native:
            tokens_chunk_size = 5
            temporal_compression_ratio = 4
            token_overlap = 2
            frame_pre_padding = 3
            frame_overlap = 5
            config = types.SimpleNamespace(token_drop=3, clip_length=17)
            _blend = staticmethod(methods['_blend'])
            _temporal_decode_plan = methods['_temporal_decode_plan']
            _decode_chunks = methods['_decode_chunks']
            _decode = methods['_decode']
            def _decode_clip(self, z):
                return z.repeat_interleave(4, dim=2)+z.mean(dim=2, keepdim=True)
        vae = Native()
        z = torch.arange(97, dtype=torch.float32).reshape(1, 1, 97, 1, 1)
        with patch.object(decode, 'verify_fast_source'):
            stream = decode.FastVideoDecodeStream(vae)
        parts = []
        for start, end in ((0, 37), (37, 67), (67, 97)):
            parts.extend(stream.push(z[:, :, start:end], final=end == 97))
        self.assertTrue(torch.equal(torch.cat(parts, dim=2), vae._decode(z)))
        packing_path = SOURCE.parents[2]/'pipelines/basic/minimax_h3/packing.py'
        self.assertEqual(hashlib.sha256(packing_path.read_bytes()).hexdigest(), decode.PACKING_SHA)
        tree = ast.parse(packing_path.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                     n.name in ('patchify_video_latents', 'unpatchify_video_tokens')]
        ns = {'torch': torch}
        for node in functions:
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         str(packing_path), 'exec'), ns)
        raw = torch.arange(1*24*30*2*4).reshape(1, 24, 30, 2, 4)
        rows = ns['patchify_video_latents'](raw, (1, 2, 2))
        self.assertTrue(torch.equal(ns['unpatchify_video_tokens'](rows, 30, 2, 4, 24, (1, 2, 2)), raw))


if __name__ == '__main__':
    unittest.main()
