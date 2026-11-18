"""Offline worker-hook tests. Fake tensors exercise operations, NOT GPU parity."""
import ast
import contextlib
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import types
import unittest
from unittest.mock import patch

from research.h3_personal import fast_continuation_stages as stages
from research.h3_personal.fast_continuation import Session, WindowCancelled
from research.h3_personal.continuation_plan import audio_boundary, plan
from research.h3_personal import continuation_candidate as candidate
from research.h3_personal.private_root import root as private_root


@contextlib.contextmanager
def pinned_source():
    """Offline archive fixture; only absent automatic discovery may skip.

    H3_FASTVIDEO_ARCHIVE selects an existing archive explicitly. A missing,
    corrupt or incompatible selected archive is an error, never a skip.
    No archive paths/links are extracted: copy only exact regular-file names.
    """
    from research.run_h3_ref2va import ARCHIVE_SHA, FASTVIDEO_REV
    selected = os.environ.get('H3_FASTVIDEO_ARCHIVE')
    if selected is not None:
        if not selected.strip():
            raise ValueError('H3_FASTVIDEO_ARCHIVE must not be empty')
        archive = Path(selected).expanduser()
    else:
        folder = private_root()
        archive = next(iter(sorted(folder.glob('**/deployment/runtime/FastVideo.tar.gz'))), None)
        if archive is None:
            raise unittest.SkipTest('pinned archive unavailable; set H3_FASTVIDEO_ARCHIVE')
    base = 'fastvideo/pipelines/basic/minimax_h3/'
    files = ['fastvideo/pipelines/lora_pipeline.py', base + 'minimax_h3_pipeline.py',
             base + 'packing.py', base + 'reference.py']
    from research.h3_personal.fast_continuation import SP_TRANSPORT_SHAS
    files += ['fastvideo/' + name for name in SP_TRANSPORT_SHAS]
    files += [base + 'stages/minimax_h3_' + name + '.py' for name in
              ('latent_preparation', 'denoising', 'decoding', 'conditioning', 'input_preparation')]
    # Hash and read from the same open descriptor. Never modify the original.
    with archive.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        if digest.hexdigest() != ARCHIVE_SHA:
            raise ValueError(f'pinned FastVideo archive SHA256 differs: {archive}')
        stream.seek(0)
        with tempfile.TemporaryDirectory(prefix='h3-pinned-source-') as folder:
            root = Path(folder)
            with tarfile.open(fileobj=stream, mode='r:gz') as tar:
                members = tar.getmembers()
                for relative in files:
                    name = f'FastVideo-{FASTVIDEO_REV}/{relative}'
                    matches = [m for m in members if m.name == name]
                    if len(matches) != 1 or not matches[0].isreg():
                        raise ValueError(f'pinned source member missing/unsafe/duplicate: {name}')
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(matches[0]) as source:
                        target.write_bytes(source.read())
                    target.chmod(0o600)
            yield root


class Tensor:
    """Small row-major CPU tensor, including real channel-major slice semantics."""
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        self.values = list(values) if values is not None else [0.] * math.prod(shape)
        assert len(self.values) == math.prod(shape)
        self.dtype = 'float32'
        self.device = types.SimpleNamespace(type='cpu')

    @property
    def ndim(self):
        return len(self.shape)

    def clone(self):
        return Tensor(self.shape, self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def reshape(self, *shape):
        if -1 in shape:
            known = math.prod(n for n in shape if n != -1)
            shape = tuple(len(self.values) // known if n == -1 else n for n in shape)
        return Tensor(shape, self.values)

    def __getitem__(self, selectors):
        selectors = selectors if isinstance(selectors, tuple) else (selectors,)
        selectors += (slice(None),) * (self.ndim - sum(s is not None for s in selectors))
        axes, shape, original = [], [], 0
        for s in selectors:
            if s is None:
                axes.append([None])
                shape.append(1)
            elif isinstance(s, int):
                axes.append([s])
                original += 1
            else:
                coords = list(range(*s.indices(self.shape[original])))
                axes.append(coords)
                shape.append(len(coords))
                original += 1
        out = []
        for coords in itertools.product(*axes):
            index = 0
            for coord, size in zip((c for c in coords if c is not None), self.shape):
                index = index * size + coord
            out.append(self.values[index])
        return Tensor(shape, out)

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        out = []
        for coords in itertools.product(*(range(n) for n in shape)):
            original = list(coords)
            original[a], original[b] = original[b], original[a]
            index = 0
            for c, n in zip(original, self.shape):
                index = index * n + c
            out.append(self.values[index])
        return Tensor(shape, out)


class Ops:
    def shape(self, t):
        return t.shape

    def copy(self, t):
        return t.clone()

    def slice(self, t, axis, start, end):
        selectors = [slice(None)] * t.ndim
        selectors[axis] = slice(start, end)
        return t[tuple(selectors)]

    def digest(self, t):
        return hashlib.sha256(repr((t.shape, t.values)).encode()).hexdigest()

    def equal(self, a, b):
        return a.shape == b.shape and a.values == b.values

    def finite_cpu(self, t):
        return all(math.isfinite(v) for v in t.values)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def fill_(self, value):
        self.value = value


class DeviceBool(Scalar):
    """Fake device scalar: host reads are observable, never implicit."""
    def __init__(self, value, device, dtype='bool', shape=()):
        super().__init__(value)
        self.device, self.dtype, self.shape = device, dtype, shape
        self.item_calls = 0

    def item(self):
        self.item_calls += 1
        return self.value

    def all(self):
        return self

    def to(self, *, dtype):
        self.dtype = dtype
        return self


class Distributed:
    def __init__(self):
        self.initialized = True
        self.peer_ok = True
        self.calls = []
        self.ReduceOp = types.SimpleNamespace(MIN='MIN')

    def is_initialized(self):
        return self.initialized

    def get_world_size(self):
        return 2

    def get_rank(self):
        return 0

    def all_reduce(self, value, op):
        # The fake transport must not introduce host reads into worker counts.
        self.calls.append((value.value, op))
        value.fill_(min(value.value, int(self.peer_ok)))


def fake_torch():
    module = types.ModuleType('torch')
    module.distributed = Distributed()
    module.int32 = 'int32'
    module.bool = 'bool'
    module.eq = lambda a, b: DeviceBool(Ops().equal(a, b), a.device)
    module.logical_and = lambda a, b: DeviceBool(a.value and b.value, a.device)
    module.tensor = lambda value, **kw: Scalar(value)
    module.equal = Ops().equal
    module.isfinite = lambda t: types.SimpleNamespace(all=lambda: Ops().finite_cpu(t))
    module.zeros = lambda shape, **kw: Tensor(shape)
    module.no_grad = contextlib.nullcontext
    def cat(tensors, dim=0):
        first = tensors[0]
        dim %= first.ndim
        shape = list(first.shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        out = []
        for coords in itertools.product(*(range(n) for n in shape)):
            local = list(coords)
            for t in tensors:
                if local[dim] < t.shape[dim]:
                    index = 0
                    for c, n in zip(local, t.shape):
                        index = index * n + c
                    out.append(t.values[index])
                    break
                local[dim] -= t.shape[dim]
        return Tensor(shape, out)
    module.cat = cat
    return module


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wav = self.root / 'source.wav'
        self.wav.write_bytes(b'private offline source fixture')
        self.manifest_path = self.root / 'window.json'
        self.torch = fake_torch()
        self.enterContext(patch.dict(os.environ, {
            stages.ENV: str(self.manifest_path), 'H3_REF2VA_COVERAGE_DIR': str(self.root / 'coverage')}))
        self.enterContext(patch.dict(sys.modules, {'torch': self.torch}))
        self.enterContext(patch.object(stages, 'TorchOps', Ops))
        self.enterContext(patch.object(stages, 'Session', lambda *a, **kw: Session(*a, **kw, ops=Ops())))
        self.ref_module = types.ModuleType('fastvideo.pipelines.basic.minimax_h3.reference')
        self.ref_module.MiniMaxH3PreparedReference = lambda **kw: types.SimpleNamespace(**kw)
        self.enterContext(patch.dict(sys.modules, {self.ref_module.__name__: self.ref_module}))
        self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
        self.denoise = types.SimpleNamespace(interrupt=False, scheduler=types.SimpleNamespace(timesteps=[0]*4),
                                             audio_scheduler=types.SimpleNamespace(timesteps=[0]*4))
        self.args = types.SimpleNamespace()
        self.encodes = 0
        def encode(stage, batch, args, m):
            self.encodes += 1
            t = audio_boundary(m['total_frames'])
            # Distinct left/right and per-time values catch flattened audio slicing.
            info = dict(source_samples=10, source_sample_rate=32000,
                        padded_samples=t*800-10, global_audio_t=t, full_source_encodes=1)
            if m.get('candidate') is not None:
                options = m['candidate']
                windowed = options['audio_context'] == 'window-context-canonical-v1'
                info.update(candidate_audio_context=options['audio_context'],
                            candidate_progress_receipts=options['progress_receipts'],
                            full_source_encodes=0 if windowed else 1,
                            source_window_encodes=m['expected_windows'] if windowed else 0)
                if windowed:
                    info['audio_overlap_policy'] = 'prior-canonical-wins'
            return Tensor((2, t, 32), range(2*t*32)), info
        self.enterContext(patch.object(stages, '_encode_source', encode))
        self.write_manifest()

    def write_manifest(self, index=0, windows=3, **updates):
        m = dict(mode='latent-tail-v1', session_id='turn-one', index=index, wav=str(self.wav),
                 sha256=hashlib.sha256(self.wav.read_bytes()).hexdigest(), total_frames=124+(windows-1)*102,
                 window_frames=124, overlap_frames=22, seed=314159, expected_windows=windows,
                 final=index == windows-1)
        m.update(updates)
        temp = self.manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(m))
        temp.replace(self.manifest_path)
        return m

    def prepare(self, fixed_value=9.):
        ref = types.SimpleNamespace(media_type='image', has_audio=False, num_audio_latents=0,
                                    num_latent_frames=1, latent_height=2, latent_width=2, image=object())
        batch = types.SimpleNamespace(extra={}, references=[ref], num_frames=124,
                    generator=types.SimpleNamespace(initial_seed=lambda:314159),
                    raw_latent_shape=(1,24,37,2,2))
        local = list(batch.references)
        original_rows = [Tensor((1,96), [fixed_value]*96)]
        vr, ar = stages.append_conditions(self.stage, batch, self.args, local, original_rows, [])
        t = batch.extra[stages.KEY]['token']
        n = 2*t.window.guide_audio_t
        batch.extra[stages.LAYOUT_KEY] = types.SimpleNamespace(
            num_condition_audio_rows=n, num_audio_latents=t.window.audio_end-t.window.audio_start,
            num_condition_video_rows=1+t.window.guide_video_t)
        batch.audio_latents = Tensor((n+2*(t.window.audio_end-t.window.audio_start),32))
        batch.latents = Tensor((1+t.window.guide_video_t+37,96),
                               [float(t.index+1)]*((1+t.window.guide_video_t+37)*96))
        stages.bind_audio(self.stage, batch, self.args)
        self.assertEqual(len(original_rows), 1)
        self.assertIs(batch.references[0], ref)
        self.assertIsNotNone(ref.image)
        self.assertEqual(len(local), len(batch.references))
        return batch, vr, ar

    def finish(self, batch):
        for i in range(4):
            stages.before_forward(self.denoise, batch, i)
            stages.hold_audio(self.denoise, batch, object())
        return stages.commit(self.denoise, batch, self.args)

    def efficiency_manifest(self, index=0, **updates):
        return self.write_manifest(index, candidate=candidate.make_options(
            window_audio_context=True, defer_progress_receipts=True), efficiency={
                'schema': 'h3-continuation-efficiency-v1',
                'audio_validation': 'device-steps-canonical-boundaries-v1',
                'decode_schedule': 'alternating-v1'}, **updates)

    def test_efficiency_device_scalar_has_no_local_host_reads(self):
        self.efficiency_manifest()
        batch, _, _ = self.prepare()
        c = batch.extra[stages.KEY]
        with patch.object(Tensor, 'cpu', side_effect=AssertionError('D2H')), \
                patch.object(self.torch, 'equal', side_effect=AssertionError('host equal')), \
                patch.object(DeviceBool, 'item', side_effect=AssertionError('local item')):
            result = stages._assert_audio(batch, c, canonical=False)
        self.assertIsInstance(result, DeviceBool)
        self.assertEqual((result.shape, result.dtype, result.device),
                         ((), self.torch.bool, batch.latents.device))
        self.assertTrue(result.value)
        stages._collective_check(self.stage, batch, lambda: result)
        self.assertEqual(result.item_calls, 1)
        self.assertEqual(self.torch.distributed.calls[-1], (1, 'MIN'))

    def test_efficiency_exact_legacy_outputs_and_two_canonical_boundaries(self):
        outputs = []
        for efficient in (False, True):
            self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
            run = []
            for index in range(3):
                (self.efficiency_manifest if efficient else self.write_manifest)(index)
                start = len(self.torch.distributed.calls)
                with patch.object(Tensor, 'cpu', autospec=True, side_effect=Tensor.cpu) as cpu:
                    batch, _, _ = self.prepare()
                    self.finish(batch)
                # Each canonical assertion checks target plus guide (when present).
                self.assertEqual(cpu.call_count, (2 if efficient else 10) *
                                 (2 if index else 1))
                self.assertEqual(len(self.torch.distributed.calls) - start, 10)
                self.assertEqual(self.torch.distributed.calls[start:], [(1, 'MIN')] * 10)
                suffix = batch.extra[stages.KEY]['suffix']
                run.append((suffix.video.shape, suffix.video.values,
                            suffix.audio.shape, suffix.audio.values,
                            suffix.video_row_offset, suffix.audio_time_offset))
            outputs.append(run)
        self.assertEqual(*outputs)

    def test_efficiency_step_corruption_and_peer_rejection(self):
        for boundary in ('forward', 'hold'):
            for failure in ('audio', 'lock', 'dtype', 'device', 'shape', 'peer'):
                with self.subTest(boundary=boundary, failure=failure):
                    self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
                    self.torch.distributed.peer_ok = True
                    self.efficiency_manifest()
                    batch, _, _ = self.prepare()
                    c = batch.extra[stages.KEY]
                    if boundary == 'hold':
                        stages.before_forward(self.denoise, batch, 0)
                    if failure == 'audio':
                        batch.audio_latents.values[-1] += 1
                    elif failure == 'lock':
                        c['target_locked'].values[-1] += 1
                    elif failure == 'dtype':
                        batch.audio_latents.dtype = 'float16'
                    elif failure == 'device':
                        batch.audio_latents.device = types.SimpleNamespace(type='other')
                    elif failure == 'shape':
                        batch.audio_latents.shape = (1, 32)
                    else:
                        self.torch.distributed.peer_ok = False
                    with self.assertRaises(WindowCancelled):
                        if boundary == 'hold':
                            stages.hold_audio(self.denoise, batch, object())
                        else:
                            stages.before_forward(self.denoise, batch, 0)
                    self.assertEqual(self.torch.distributed.calls[-1],
                                     (1 if failure == 'peer' else 0, 'MIN'))
                    self.assertTrue(c['owner'].faulted)
                    self.assertNotIn('suffix', c)
        self.torch.distributed.peer_ok = True

    def test_efficiency_invalid_scalar_participates_in_zero_min(self):
        for failure in ('type', 'dtype', 'device', 'shape'):
            with self.subTest(failure=failure):
                self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
                self.efficiency_manifest()
                batch, _, _ = self.prepare()
                value = DeviceBool(True, batch.latents.device)
                if failure == 'type':
                    value = True
                else:
                    setattr(value, failure, {'dtype': 'int32', 'device': 'other',
                                             'shape': (1,)}[failure])
                with self.assertRaises(WindowCancelled):
                    stages._collective_check(self.stage, batch, lambda: value)
                self.assertEqual(self.torch.distributed.calls[-1], (0, 'MIN'))
                self.assertTrue(batch.extra[stages.KEY]['owner'].faulted)

    def test_efficiency_matching_device_corruption_still_fails_canonical_commit(self):
        self.efficiency_manifest()
        batch, _, _ = self.prepare()
        c = batch.extra[stages.KEY]
        batch.audio_latents.values[-1] += 1
        c['target_locked'].values[-1] += 1
        for i in range(4):
            stages.before_forward(self.denoise, batch, i)
            stages.hold_audio(self.denoise, batch, object())
        with self.assertRaises(WindowCancelled) as rejected:
            stages.commit(self.denoise, batch, self.args)
        self.assertIn('source clone', str(rejected.exception.__cause__))
        self.assertEqual(self.torch.distributed.calls[-1], (0, 'MIN'))
        self.assertNotIn('suffix', c)

    def test_efficiency_policy_fail_closed_and_drift(self):
        valid = self.efficiency_manifest()['efficiency']
        for policy in (None, True, {}, dict(valid, extra=True),
                       dict(valid, schema='wrong'), dict(valid, audio_validation=False),
                       dict(valid, decode_schedule='wrong')):
            self.efficiency_manifest()
            m = json.loads(self.manifest_path.read_text())
            m['efficiency'] = policy
            self.manifest_path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                stages.active()
        for options in (None, candidate.make_options(defer_progress_receipts=True)):
            self.write_manifest(efficiency=valid, **({'candidate': options} if options else {}))
            with self.assertRaises(ValueError):
                stages.active()
        self.efficiency_manifest()
        batch, _, _ = self.prepare()
        self.write_manifest(candidate=candidate.make_options(
            window_audio_context=True, defer_progress_receipts=True))
        with self.assertRaises(WindowCancelled):
            stages.before_forward(self.denoise, batch, 0)
        self.assertEqual(self.torch.distributed.calls[-1], (0, 'MIN'))
        self.assertTrue(batch.extra[stages.KEY]['owner'].faulted)

    def test_global_once_guides_exact_source_suffix_and_receipts(self):
        for index in range(3):
            self.write_manifest(index)
            batch, vr, ar = self.prepare()
            c = batch.extra[stages.KEY]
            token = c['token']
            if index:
                self.assertEqual(len(vr), 2)
                self.assertEqual(len(ar), 1)
                self.assertEqual(vr[-1].shape, (7,96))
                self.assertEqual(ar[-1].shape, (74,32))
                self.assertEqual(vr[-1].values, [float(index)]*(7*96))
                vr[-1].values[0] = -999  # native noise path cannot mutate clean token
                ar[-1].values[0] = -999
                self.assertNotEqual(token.guide_video.values[0], -999)
                self.assertNotEqual(token.guide_audio.values[0], -999)
                self.assertEqual(batch.references[-1].media_type, 'video')
            self.finish(batch)
            suffix = c['suffix']
            self.assertEqual(suffix.video.shape, (37 if index == 0 else 30,96))
            self.assertEqual(suffix.audio.shape, (2,207 if index == 0 else 170,32))
            self.assertEqual(suffix.audio_time_offset, 0 if index == 0 else 207+(index-1)*170)
            path = self.root / 'coverage' / f'h3-continuation-rank-0-window-{index}.json'
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt['schema'], stages.SCHEMA)
            self.assertEqual(receipt['steps_held'], 4)
            self.assertEqual(receipt['forwards'], 4)
            self.assertTrue(receipt['committed'])
            self.assertFalse(receipt['gpu_validated'])
            self.assertEqual(receipt['audio_timestep_policy'], 'fixed_clean_1.0')
            self.assertNotIn(str(self.wav), path.read_text())
            self.assertTrue(all(type(v) in (str,int,bool,float) for v in receipt.values()))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.encodes, 1)
        self.assertEqual(c['owner'].committed_shape, (97,547))
        self.assertTrue(c['state']['completed'])
        with self.assertRaises(WindowCancelled):
            stages.commit(self.denoise, batch, self.args)

    def test_independent_and_absent_are_identity_passthrough(self):
        self.manifest_path.write_text('{"mode":"independent"}')
        self.assertFalse(stages.active())
        rows, audio = [object()], []
        got = stages.append_conditions(None, None, None, None, rows, audio)
        self.assertIs(got[0], rows)
        self.assertIs(got[1], audio)
        with patch.dict(os.environ, {stages.ENV:''}):
            self.assertFalse(stages.active())
        self.assertEqual(self.encodes, 0)

    def test_manifest_fail_closed_no_273_bool_index_or_bad_final(self):
        for updates in (dict(mode='typo'), dict(total_frames=273), dict(index=True),
                        dict(expected_windows=5), dict(seed=42), dict(final=True),
                        dict(overlap_frames=0), dict(source_samples=0), dict(cancelled=1)):
            self.write_manifest(**updates)
            with self.assertRaises(ValueError):
                stages.active()
        self.manifest_path.write_text('{')
        with self.assertRaises(ValueError):
            stages.active()

    def test_cancel_and_manifest_replacement_collective_no_close(self):
        for change in ('manifest', 'source', 'interrupt', 'peer'):
            with self.subTest(change=change):
                self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
                self.write_manifest()
                batch, _, _ = self.prepare()
                self.torch.distributed.initialized = True
                self.torch.distributed.peer_ok = True
                if change == 'manifest':
                    self.write_manifest(mode='independent')
                elif change == 'source':
                    self.wav.write_bytes(b'changed')
                elif change == 'interrupt':
                    self.denoise.interrupt = True
                else:
                    self.torch.distributed.peer_ok = False
                with self.assertRaises(WindowCancelled):
                    stages.before_forward(self.denoise, batch, 0)
                c = batch.extra[stages.KEY]
                self.assertTrue(c['owner'].faulted)
                self.assertFalse(c['owner']._closed)
                self.assertEqual(c['forwards'], 0)
                self.assertEqual(self.torch.distributed.calls[-1][1], 'MIN')
                self.denoise.interrupt = False

    def test_audio_mutations_and_early_or_extra_forward_fail(self):
        batch, _, _ = self.prepare()
        with self.assertRaises(WindowCancelled):
            stages.commit(self.denoise, batch, self.args)
        self.assertNotIn('suffix', batch.extra[stages.KEY])
        self.stage = types.SimpleNamespace(ref2va=True)
        batch, _, _ = self.prepare()
        stages.before_forward(self.denoise, batch, 0)
        batch.audio_latents.values[-1] += 1
        with self.assertRaises(WindowCancelled) as rejected:
            stages.hold_audio(self.denoise, batch, object())
        self.assertIn('locked audio', str(rejected.exception.__cause__))
        self.stage = types.SimpleNamespace(ref2va=True)
        batch, _, _ = self.prepare()
        for i in range(4):
            stages.before_forward(self.denoise, batch, i)
            stages.hold_audio(self.denoise, batch, object())
        with self.assertRaises(WindowCancelled):
            stages.before_forward(self.denoise, batch, 4)

    def test_guide_audio_corruption_and_borrowed_video_rejected(self):
        self.finish(self.prepare()[0])
        self.write_manifest(1)
        batch, _, _ = self.prepare()
        batch.extra[stages.KEY]['guide_locked'].values[0] += 1
        with self.assertRaises(WindowCancelled):
            stages.before_forward(self.denoise, batch, 0)
        self.stage._h3_continuation_owner['owner'].abort_window(batch.extra[stages.KEY]['token'])
        batch, _, _ = self.prepare()
        batch.extra[stages.KEY]['token'].guide_video.values[0] += 1
        for i in range(4):
            stages.before_forward(self.denoise, batch, i)
            stages.hold_audio(self.denoise, batch, object())
        with self.assertRaises(WindowCancelled):
            stages.commit(self.denoise, batch, self.args)
        self.assertNotIn('suffix', batch.extra[stages.KEY])

    def test_session_change_and_global_plan_drift_rejected_without_reencode(self):
        self.finish(self.prepare()[0])
        self.write_manifest(1, windows=4)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.prepare()
        self.write_manifest(session_id='new-turn')
        with self.assertRaisesRegex(ValueError, 'completed previous'):
            self.prepare()
        self.assertEqual(self.encodes, 1)

    def test_fixed_identity_rows_immutable_across_windows(self):
        self.finish(self.prepare()[0])
        self.write_manifest(1)
        with self.assertRaisesRegex(ValueError, 'fixed image rows'):
            self.prepare(fixed_value=10.)
        self.assertEqual(self.encodes, 1)
        self.assertIsNone(self.stage._h3_continuation_owner['owner'].pending_window)

    def test_commit_cancellation_and_incomplete_schedule(self):
        self.torch.distributed.initialized = True
        batch, _, _ = self.prepare()
        for i in range(4):
            stages.before_forward(self.denoise, batch, i)
            stages.hold_audio(self.denoise, batch, object())
        self.write_manifest(cancelled=True)
        with self.assertRaises(WindowCancelled):
            stages.commit(self.denoise, batch, self.args)
        self.assertEqual(self.torch.distributed.calls[-1], (0, 'MIN'))
        self.assertNotIn('suffix', batch.extra[stages.KEY])
        self.stage = types.SimpleNamespace(ref2va=True)
        self.write_manifest()
        batch, _, _ = self.prepare()
        self.denoise.scheduler.timesteps = [0]*8
        with self.assertRaises(WindowCancelled):
            stages.before_forward(self.denoise, batch, 0)

    def test_new_turn_only_after_final_four_window_bound(self):
        for index in range(4):
            self.write_manifest(index, windows=4)
            batch, _, _ = self.prepare()
            self.finish(batch)
        self.assertEqual(self.encodes, 1)
        self.assertEqual(batch.extra[stages.KEY]['owner'].committed_shape, (127,717))
        self.write_manifest(windows=1, session_id='second')
        batch, _, _ = self.prepare()
        self.finish(batch)
        self.assertEqual(self.encodes, 2)

    def test_no_serial_silent_fallback_at_forward(self):
        batch, _, _ = self.prepare()
        self.torch.distributed.initialized = False
        with self.assertRaises(WindowCancelled):
            stages.before_forward(self.denoise, batch, 0)
        self.assertTrue(batch.extra[stages.KEY]['owner'].faulted)
        self.assertEqual(self.torch.distributed.calls, [(1, 'MIN')])  # prior bind

    def test_bind_and_hold_local_and_peer_failures_fault_without_suffix(self):
        for boundary in ('bind', 'hold'):
            for failure in ('validation', 'receipt', 'peer', 'transport'):
                with self.subTest(boundary=boundary, failure=failure):
                    self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
                    self.torch.distributed.peer_ok = True
                    batch, _, _ = self.prepare()
                    c = batch.extra[stages.KEY]
                    if boundary == 'hold':
                        stages.before_forward(self.denoise, batch, 0)
                    operation = (lambda: stages.bind_audio(self.stage, batch, self.args)
                                 if boundary == 'bind' else
                                 stages.hold_audio(self.denoise, batch, object()))
                    if failure == 'validation':
                        if boundary == 'bind':
                            batch.extra[stages.LAYOUT_KEY].num_audio_latents += 1
                        else:
                            batch.audio_latents.values[0] += 1
                    if failure == 'peer':
                        self.torch.distributed.peer_ok = False
                    with contextlib.ExitStack() as stack:
                        if failure == 'receipt':
                            stack.enter_context(patch.object(stages, '_receipt', side_effect=OSError('disk')))
                        if failure == 'transport':
                            stack.enter_context(patch.object(self.torch.distributed, 'all_reduce',
                                                            side_effect=TimeoutError('finite timeout')))
                        with self.assertRaises(TimeoutError if failure == 'transport' else WindowCancelled):
                            operation()
                    self.assertTrue(c['owner'].faulted)
                    self.assertFalse(c['owner']._closed)
                    self.assertNotIn('suffix', c)
                    self.torch.distributed.peer_ok = True
                    with self.assertRaises(WindowCancelled):
                        stages.before_forward(self.denoise, batch, 0)

    def test_missing_velocity_or_duplicate_hold_participates_in_rejection(self):
        for duplicate in (False, True):
            self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
            batch, _, _ = self.prepare()
            stages.before_forward(self.denoise, batch, 0)
            if duplicate:
                stages.hold_audio(self.denoise, batch, object())
            with self.assertRaises(WindowCancelled):
                stages.hold_audio(self.denoise, batch, object() if duplicate else None)
            self.assertEqual(self.torch.distributed.calls[-1], (0, 'MIN'))
            self.assertTrue(batch.extra[stages.KEY]['owner'].faulted)

    def test_candidate_exact_guides_source_and_receipt_frequency(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
                options = candidate.make_options(window_audio_context=True,
                                                 defer_progress_receipts=deferred)
                self.torch.distributed.calls.clear()
                with patch.object(stages, '_receipt', wraps=stages._receipt) as receipt:
                    for index in range(3):
                        self.write_manifest(index, candidate=options)
                        batch, vr, ar = self.prepare()
                        c = batch.extra[stages.KEY]
                        token = c['token']
                        if index:
                            self.assertTrue(Ops().equal(vr[-1], token.guide_video))
                            expected = Tensor((2, audio_boundary(328), 32),
                                              range(2*audio_boundary(328)*32))
                            start = token.window.audio_start
                            guide = expected[:, start:start+token.window.guide_audio_t]
                            self.assertTrue(Ops().equal(ar[-1], guide.reshape(-1, 32)))
                        self.finish(batch)
                        self.assertEqual(c['forwards'], 4)
                        self.assertEqual(c['steps_held'], 4)
                        self.assertIn('suffix', c)
                    self.assertEqual(receipt.call_count, 3*(2 if deferred else 6))
                self.assertEqual(self.torch.distributed.calls, [(1, 'MIN')]*30)

    def test_candidate_deferred_hold_failure_still_faults_without_suffix(self):
        for failure in ('audio', 'peer'):
            with self.subTest(failure=failure):
                self.stage = types.SimpleNamespace(ref2va=True, interrupt=False)
                self.torch.distributed.peer_ok = True
                self.write_manifest(candidate=candidate.make_options(
                    window_audio_context=True, defer_progress_receipts=True))
                batch, _, _ = self.prepare()
                stages.before_forward(self.denoise, batch, 0)
                if failure == 'audio':
                    batch.audio_latents.values[-1] += 1
                else:
                    self.torch.distributed.peer_ok = False
                with patch.object(stages, '_receipt', wraps=stages._receipt) as receipt:
                    with self.assertRaises(WindowCancelled):
                        stages.hold_audio(self.denoise, batch, object())
                    receipt.assert_not_called()
                c = batch.extra[stages.KEY]
                self.assertTrue(c['owner'].faulted)
                self.assertNotIn('suffix', c)

    def test_candidate_manifest_invalid_values_and_mutable_policy_drift(self):
        options = candidate.make_options(window_audio_context=True)
        invalid = [None, {}, dict(options, unknown=True)]
        for key in options:
            invalid.extend(dict(options, **{key: value}) for value in (None, 'unknown'))
        for value in invalid:
            with self.subTest(value=value):
                self.write_manifest(candidate=value)
                with self.assertRaises(ValueError):
                    stages.active()
        self.write_manifest(candidate=options)
        batch, _, _ = self.prepare()
        self.write_manifest(candidate=dict(options, progress_receipts='bind-and-commit-v1'))
        with self.assertRaises(WindowCancelled):
            stages.before_forward(self.denoise, batch, 0)
        c = batch.extra[stages.KEY]
        self.assertTrue(c['owner'].faulted)
        self.assertNotIn('suffix', c)

    def test_collective_at_every_forward_and_commit(self):
        self.torch.distributed.initialized = True
        batch, _, _ = self.prepare()
        self.finish(batch)
        self.assertEqual(self.torch.distributed.calls, [(1,'MIN')]*10)  # bind +4forward +4hold +commit


class EncodeTests(unittest.TestCase):
    def test_candidate_wrapper_collective_failures(self):
        torch = fake_torch()
        dist = types.ModuleType('fastvideo.distributed')
        dist.get_local_torch_device = lambda: 'cpu'
        m = dict(candidate=candidate.make_options(window_audio_context=True))
        with patch.dict(sys.modules, {'torch': torch, dist.__name__: dist}):
            for failure in ('local', 'peer', 'uninitialized'):
                with self.subTest(failure=failure):
                    torch.distributed.calls.clear()
                    torch.distributed.initialized = failure != 'uninitialized'
                    torch.distributed.peer_ok = failure != 'peer'
                    error = ValueError('local encode') if failure == 'local' else None
                    with patch.object(stages, '_encode_source_local', side_effect=error,
                                      return_value=('latent', {})) as local:
                        with self.assertRaises(ValueError if failure == 'uninitialized'
                                               else WindowCancelled) as rejected:
                            stages._encode_source(None, None, None, m)
                        self.assertEqual(local.call_count, int(failure != 'uninitialized'))
                        self.assertEqual(torch.distributed.calls,
                                         [] if failure == 'uninitialized' else
                                         [(0 if failure == 'local' else 1, 'MIN')])
                        if error:
                            self.assertIs(rejected.exception.__cause__, error)


    def test_native_full_waveform_encode_padding_shape_and_no_oversize(self):
        # Execute real _encode_source, replacing only third-party native APIs.
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'x.wav'
            source.write_bytes(b'fixture')
            torch = fake_torch()
            encoded = []
            unloaded = []
            device = types.SimpleNamespace(type='cpu')
            vae = types.SimpleNamespace(hop_length=800, sampling_rate=32000, latent_channels=32,
                                       parameters=lambda: [], normalize_latents=lambda x:x)
            def encode(x):
                encoded.append(x)
                t = x.shape[-1] // 800
                return types.SimpleNamespace(latent_dist=types.SimpleNamespace(
                    mode=lambda: Tensor((2,32,t), range(2*32*t))))
            vae.encode = encode
            ref = types.ModuleType('fastvideo.pipelines.basic.minimax_h3.reference')
            ref.decode_reference_audio = lambda path:(Tensor((2,17)),32000)
            ref.prepare_reference_waveform = lambda waveform, rate, target, **kw:waveform
            dist = types.ModuleType('fastvideo.distributed')
            dist.get_local_torch_device = lambda:device
            models = types.ModuleType('fastvideo.models')
            models.pinned_offload = types.SimpleNamespace(load=lambda *a,**kw:None,
                                                         unload=lambda *a:unloaded.append(True))
            modules = {'torch':torch, ref.__name__:ref, dist.__name__:dist, models.__name__:models}
            m = dict(wav=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     total_frames=226, source_samples=17, source_sample_rate=32000)
            with patch.dict(sys.modules, modules), patch.object(stages, 'TorchOps', Ops):
                latent, info = stages._encode_source(types.SimpleNamespace(audio_vae=vae), None,
                                                     types.SimpleNamespace(), m)
                self.assertEqual(len(encoded), 1)
                self.assertEqual(encoded[0].shape, (2,1,377*800))
                self.assertEqual(info['padded_samples'], 377*800-17)
                self.assertEqual(latent.shape, (2,377,32))
                self.assertEqual(latent.values[:3], [0,377,754])
                self.assertEqual(info['full_source_encodes'], 1)
                self.assertEqual(torch.distributed.calls, [])
                torch.Tensor = Tensor
                with patch.object(Tensor, 'is_floating_point', lambda self: True, create=True), \
                        patch.object(candidate, 'encode_window_context',
                                     wraps=candidate.encode_window_context) as helper:
                    m['candidate'] = candidate.make_options(window_audio_context=True)
                    encoded.clear()
                    canonical, info = stages._encode_source(
                        types.SimpleNamespace(audio_vae=vae), None, types.SimpleNamespace(), m)
                    self.assertEqual(helper.call_count, 1)
                    speech, windows, callback = helper.call_args.args
                    self.assertEqual(speech.shape, (2, 377*800))
                    self.assertEqual(windows, plan(226, 124, 22))
                    self.assertTrue(callable(callback))
                    self.assertEqual([x.shape for x in encoded],
                                     [(2, 1, (w.audio_end-w.audio_start)*800) for w in windows])
                    self.assertEqual(canonical.shape, (2, 377, 32))
                    self.assertEqual(info['full_source_encodes'], 0)
                    self.assertEqual(info['source_window_encodes'], 2)
                    self.assertEqual(info['audio_overlap_policy'], 'prior-canonical-wins')
                    self.assertEqual(torch.distributed.calls, [(1, 'MIN')])
                m.pop('candidate')
                encoded.clear()
                # Full-source normalized transpose is cloned, not a window encode.
                ref.decode_reference_audio = lambda path:(Tensor((2,377*800+1)),32000)
                m.pop('source_samples')
                with self.assertRaisesRegex(ValueError, 'never truncate'):
                    stages._encode_source(types.SimpleNamespace(audio_vae=vae), None,
                                          types.SimpleNamespace(), m)
                self.assertEqual(len(encoded), 0)
                self.assertFalse(unloaded)


class SourceBindingTests(unittest.TestCase):
    def test_lazy_import_without_torch(self):
        code = ('import sys; import research.h3_personal.fast_continuation_stages; '
                'assert "torch" not in sys.modules; assert "fastvideo" not in sys.modules')
        subprocess.run([sys.executable, '-S', '-c', code], check=True)

    def test_pinned_native_hook_order_noise_and_timestep_contract(self):
        root = self.enterContext(pinned_source())
        base = root / 'fastvideo/pipelines/basic/minimax_h3/stages'
        latent = (base / 'minimax_h3_latent_preparation.py').read_text()
        tree = ast.parse(latent)
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name.endswith('Stage'))
        encode = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '_encode_ref2va_conditions')
        source = ast.get_source_segment(latent, encode)
        self.assertLess(source.index('audio_rows = self._encode_audio_rows'), source.index('shapes = tuple'))
        self.assertLess(source.index('shapes = tuple'), source.index('self.scheduler.scale_noise'))
        self.assertIn('MINIMAX_H3_KEYFRAME_NOISE_AUG', source)
        denoise = (base / 'minimax_h3_denoising.py').read_text()
        self.assertIn('condition_audio_timestep=1.0', denoise)
        external = Path('research/h3_personal/external_audio.py').read_text()
        self.assertIn('def locked_audio_timestep(', external)
        self.assertIn('return 1.0', external)
        ours = Path(stages.__file__).read_text()
        code_tree = ast.parse(ours)
        calls = [ast.unparse(n.func) for n in ast.walk(code_tree) if isinstance(n,ast.Call)]
        self.assertNotIn('stage.audio_scheduler.step', calls)
        self.assertNotIn('stage.vae.encode_keyframe', calls)
        self.assertNotIn('stage.transformer', calls)
        self.assertNotIn('owner.close', calls)
        self.assertIn('vae.encode', calls)
        self.assertIn('vae.normalize_latents', calls)
        self.assertIn('torch.distributed.all_reduce', calls)

    def test_layout_calls_existing_global_clock_builder_with_original_images(self):
        torch = fake_torch()
        batch = types.SimpleNamespace(num_frames=124, raw_latent_shape=(1,24,37,2,2),
                    generator=types.SimpleNamespace(initial_seed=lambda:314159),
                    references=[types.SimpleNamespace(media_type='image'), types.SimpleNamespace(media_type='video')],
                    extra={stages.KEY:dict(token=types.SimpleNamespace(window=object())),
                           'minimax_h3_text_token_tags':object()})
        package = types.ModuleType('fastvideo.pipelines.basic.minimax_h3')
        package.packing = object()
        package.reference = object()
        with patch.dict(sys.modules, {'torch':torch, package.__name__:package}), patch.object(stages, 'build_layout') as build:
            self.assertIs(stages.make_layout(None,batch,None), build.return_value)
            self.assertEqual(build.call_args.kwargs['images'], batch.references[:1])
            self.assertIs(build.call_args.kwargs['window'], batch.extra[stages.KEY]['token'].window)
            self.assertEqual(build.call_args.kwargs['ranks'], 2)


class ProfileRangeTests(unittest.TestCase):
    def test_selected_cpu_ranges_and_default_no_torch(self):
        from unittest.mock import patch
        from contextlib import contextmanager
        from research.h3_personal import fast_continuation_stages as stages
        batch = types.SimpleNamespace(extra={})
        with patch.dict('sys.modules', {'torch': None}):
            with stages.forward_profile(batch, 0):
                pass
        events = []
        @contextmanager
        def record(label):
            events.append(('enter', label))
            try:
                yield
            finally:
                events.append(('exit', label))
        fake = types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=record))
        batch.extra[stages.KEY] = {'manifest': {'index': 1, 'trace_policy': {'selected_window': 1}}}
        with patch.dict('sys.modules', {'torch': fake}):
            with self.assertRaisesRegex(RuntimeError, 'original'):
                with stages.forward_profile(batch, 3):
                    raise RuntimeError('original')
            for label in ('h3.before_forward', 'h3.hold', 'h3.commit'):
                @stages._profile_stage(label)
                def hook(stage, batch):
                    return 42
                self.assertEqual(hook(None, batch), 42)
        self.assertEqual(events[:2], [('enter', 'h3.forward3'), ('exit', 'h3.forward3')])
        self.assertEqual(len(events), 8)
        batch.extra[stages.KEY]['manifest']['index'] = 0
        with patch.dict('sys.modules', {'torch': None}):
            with stages.forward_profile(batch, 1):
                pass


if __name__ == '__main__':
    unittest.main()
