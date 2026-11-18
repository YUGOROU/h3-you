import unittest
from unittest.mock import patch
from types import SimpleNamespace
from research.h3_personal.denoise_execution import DenoiseController, CompileBudget, CandidateRejected
from research.h3_personal.private_root import base as private_base


class Status:
    def __init__(self, fail=None, peer_code=0):
        self.labels = []
        self.fail = fail
        self.peer_code = peer_code
        self.votes = []
    def rejection_vote(self, code, label):
        self.votes.append((code, label))
        return max(code, self.peer_code)
    def min(self, ok, label):
        self.labels.append(label)
        return ok and label != self.fail


class Model:
    def named_parameters(self):
        return []
    def state_dict(self, keep_vars=False):
        assert keep_vars
        return {}


class Tests(unittest.TestCase):
    def controller(self, status=None):
        blocks = [SimpleNamespace(forward=lambda: None) for _ in range(50)]
        return DenoiseController(Model(), blocks, status=status or Status(),
            compiler=lambda *a, **k: None, backend=lambda: None,
            max_compile_count=4, max_compile_seconds=10)

    def install(self, c, fail_prepare=None, fail_commit=None, source='', real_extract=False,
                gates=None):
        seen = []
        def prepare(block, **kw):
            i = kw['main_block_index']
            self.assertTrue(all(not hasattr(b, '_h3_region_b_dispatch') for b in c.blocks))
            seen.append(i)
            if i == fail_prepare:
                raise ValueError('bad')
            def commit():
                if i == fail_commit:
                    raise RuntimeError('install failed')
                block._h3_local_probe = lambda: None
                block._h3_region_b_dispatch = lambda *args: args
                block.forward = lambda: 'installed'
            return commit
        from contextlib import nullcontext
        extraction = nullcontext() if real_extract else patch(
            'research.h3_personal.denoise_execution.local.extract', return_value=compile(
                'def _h3_region_b(): pass\ndef forward(): pass', '<test>', 'exec'))
        with extraction, patch(
                'research.h3_personal.denoise_execution.local.install', side_effect=prepare):
            c.install(source=source, namespace={}, admission={}, stage_shapes={}, gates=gates if gates is not None else {
                'source_pinned': True, 'real_bf16_weights': True,
                'static_adapter_receipt_verified': True, 'main50_verified': True})
        return seen

    def test_asymmetric_prepare_typed_proof_and_unexpected_dominance(self):
        import threading
        from research.h3_personal import local_denoise_compile as local
        from research.h3_personal.denoise_execution import CandidateUnavailable
        for peer_failure in (None, ValueError('native source SHA mismatch'),
                             RuntimeError('native unexpected')):
            barrier = threading.Barrier(2, timeout=3)
            values, sequences = {}, [[], []]
            class PairStatus:
                def __init__(self, rank): self.rank = rank
                def reduce(self, value, label, op):
                    sequences[self.rank].append((op, label))
                    values[self.rank] = value
                    barrier.wait()
                    result = min(values.values()) if op == 'MIN' else max(values.values())
                    barrier.wait()
                    return result
                def min(self, value, label): return self.reduce(value, label, 'MIN')
                def rejection_vote(self, value, label): return self.reduce(value, label, 'MAX')
            owners = [self.controller(PairStatus(r)) for r in range(2)]
            original = local.AdmissionUnsupported('native-provenance-unsupported',
                                                 'native class/forward globals provenance required')
            errors = [None, None]
            def prepare(block, **kw):
                if block is owners[0].blocks[0]: raise original
                if peer_failure is not None and block is owners[1].blocks[0]:
                    raise peer_failure
                return lambda: None
            def run(rank):
                try:
                    owners[rank].install(source='', namespace={}, admission={}, stage_shapes={},
                        gates=dict(source_pinned=True, real_bf16_weights=True,
                                   static_adapter_receipt_verified=True, main50_verified=True))
                except Exception as exc: errors[rank] = exc
            with patch.object(local, 'extract', return_value=compile(
                    'def _h3_region_b(): pass\ndef forward(): pass', '<test>', 'exec')), patch.object(
                    local, 'install', side_effect=prepare):
                threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
                for t in threads: t.start()
                for t in threads: t.join(5)
            self.assertFalse(any(t.is_alive() for t in threads))
            self.assertIs(errors[0].__cause__, original)
            self.assertEqual(sequences[0], [('MIN', 'prepare-main50'), ('MAX', 'prepare-main50')])
            self.assertEqual(sequences[0], sequences[1])
            for c, exc in zip(owners, errors):
                self.assertEqual(isinstance(exc, CandidateUnavailable), peer_failure is None)
                self.assertEqual(c.state, 'faulted')
                self.assertTrue(c.receipt['identity_unchanged'])
                self.assertFalse(any(hasattr(b, '_h3_region_b_dispatch') for b in c.blocks))
                if peer_failure is None:
                    self.assertEqual(c.receipt['prepare_error_code'], 'native-provenance-unsupported')
                    self.assertEqual(c.receipt['rejection_kind'], 'controlled-admission-rejection')
                    self.assertTrue(c.receipt['source_pinned_verified'])
                    self.assertTrue(c.receipt['main50_verified'])

    def test_all50_prepared_before_mutation(self):
        c = self.controller()
        self.assertEqual(self.install(c), list(range(50)))
        self.assertTrue(c.receipt['identity_unchanged'])
        self.assertEqual(c.state, 'ready')

    def test_prepare_failure_never_mutates(self):
        c = self.controller()
        with self.assertRaises(RuntimeError): self.install(c, fail_prepare=49)
        self.assertEqual(c.state, 'faulted')
        self.assertFalse(any(hasattr(b, '_h3_region_b_dispatch') for b in c.blocks))

    def test_commit_failure_rolls_back_and_refuses(self):
        c = self.controller()
        old = [b.forward for b in c.blocks]
        with self.assertRaises(RuntimeError): self.install(c, fail_commit=25)
        self.assertEqual([b.forward for b in c.blocks], old)
        self.assertFalse(any(hasattr(b, '_h3_local_probe') for b in c.blocks))
        self.assertFalse(any(hasattr(b, '_h3_region_b_dispatch') for b in c.blocks))
        with self.assertRaises(RuntimeError): c.run('first', 0, {}, lambda: None)

    def test_remote_install_failure_rolls_back(self):
        c = self.controller(Status('install-main50'))
        with self.assertRaises(RuntimeError): self.install(c)
        self.assertFalse(any(hasattr(b, '_h3_region_b_dispatch') for b in c.blocks))

    def test_post_failure_common_min_fault(self):
        c = self.controller(Status('local-post'))
        self.install(c)
        with self.assertRaises(RuntimeError):
            c.run('first', 0, {'R': 7, 'T': 9, 'local_shape': (1, 11, 5376)},
                  lambda: c.blocks[0]._h3_region_b_dispatch(1))
        self.assertEqual(c.state, 'faulted')
        self.assertEqual(c.status.labels[-1], 'local-post')

    def test_peer_candidate_rejection_has_same_exception(self):
        from research.h3_personal.denoise_execution import CandidateRejected
        c = self.controller(Status('local-post', peer_code=1))
        self.install(c)
        with self.assertRaises(CandidateRejected):
            c.run('first', 0, {'R': 7, 'T': 9, 'local_shape': (1, 11, 5376)},
                  lambda: c.blocks[0]._h3_region_b_dispatch(1))
        self.assertEqual(c.state, 'faulted')

    def test_failure_votes(self):
        unsupported = type('Unsupported', (RuntimeError,), {'__module__': 'torch._dynamo.exc'})
        for error, peer, controlled in [(None, 1, True), (None, 2, False),
                (CandidateRejected('budget'), 0, True), (unsupported('x'), 0, True),
                (RuntimeError('Unsupported OOM'), 0, False),
                (CandidateRejected('x'), 2, False)]:
            with self.subTest(error=error, peer=peer):
                c = self.controller(Status(peer_code=peer))
                self.install(c)
                # Inject at the underlying dispatch, before guarded wrapping.
                if error is not None:
                    with patch.object(c, '_post_agreement', wraps=c._post_agreement):
                        c.context = ('first', 0, {})
                        with self.assertRaises(CandidateRejected if controlled else RuntimeError) as caught:
                            c._post_agreement(error, 'local-post')
                else:
                    c.status.fail = 'local-post'
                    with self.assertRaises(CandidateRejected if controlled else RuntimeError) as caught:
                        c.run('first', 0, {'R': 7, 'T': 9, 'local_shape': (1, 11, 5376)},
                              lambda: c.blocks[0]._h3_region_b_dispatch(1))
                self.assertEqual(isinstance(caught.exception, CandidateRejected), controlled)
                self.assertEqual(c.receipt.get('candidate_rejected', False), controlled)
                self.assertEqual(c.state, 'faulted')

    def test_vote_api_required(self):
        c = self.controller(SimpleNamespace(min=lambda ok, label: False))
        with self.assertRaisesRegex(RuntimeError, 'rejection_vote'):
            c._post_agreement(None, 'local-post')

    def test_healthy_post_has_no_vote(self):
        c = self.controller()
        self.install(c)
        c.run('first', 0, {'R': 7, 'T': 9, 'local_shape': (1, 11, 5376)},
              lambda: c.blocks[0]._h3_region_b_dispatch(1))
        self.assertEqual(c.status.votes, [])

    def test_strict_seconds(self):
        for value in (True, False, float('nan'), float('inf'), 0, -1, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CompileBudget(lambda: None, 1, value)

    def test_source_class_fp32_admission(self):
        class NativeModel(Model):
            _keep_in_fp32_modules = ['final_norm']
            def named_parameters(self):
                return [('final_norm.weight', self.parameter)]
        for source, admitted in [
                ('class NativeModel:\n _keep_in_fp32_modules = ["final_norm"]', True),
                ('class Other:\n _keep_in_fp32_modules = ["final_norm"]\nclass NativeModel:\n pass', False)]:
            with self.subTest(admitted=admitted):
                c = self.controller()
                c.model = NativeModel()
                c.model.parameter = SimpleNamespace(dtype='torch.float32', shape=(8,))
                if admitted:
                    self.install(c, source=source)
                else:
                    with self.assertRaises(RuntimeError):
                        self.install(c, source=source)

    def test_strict_gate_booleans(self):
        for key in ('source_pinned', 'real_bf16_weights',
                    'static_adapter_receipt_verified', 'main50_verified'):
            gates = dict.fromkeys(('source_pinned', 'real_bf16_weights',
                                   'static_adapter_receipt_verified', 'main50_verified'), True)
            gates[key] = 1
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.install(self.controller(), gates=gates)

    def test_hash_before_source_admission(self):
        c = self.controller()
        with patch('research.h3_personal.denoise_execution._native_fp32_roots') as roots:
            with self.assertRaises(RuntimeError):
                self.install(c, source='not even valid Python!', real_extract=True)
            roots.assert_not_called()

    def test_pinned_archive_native_projection_baseline(self):
        import ast
        import pathlib
        import tarfile
        from research.h3_personal.denoise_execution import _native_fp32_roots
        paths = list(private_base().glob(
            '**/gpu-hf-fast-ref2va-smd4-v14-live-transfer/deployment/runtime/FastVideo.tar.gz'))
        if not paths:
            self.skipTest('pinned local native archive unavailable')
        with tarfile.open(paths[0]) as archive:
            member = next(m for m in archive.getmembers()
                          if m.name.endswith('/fastvideo/models/dits/minimax_h3.py'))
            source = archive.extractfile(member).read().decode()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and any(isinstance(a, ast.Assign) and any(
                       isinstance(t, ast.Name) and t.id == '_keep_in_fp32_modules'
                       for t in a.targets) for a in n.body))
        roots = frozenset({'proj_in', 'audio_proj_in', 'time_embedder',
                           'proj_out', 'audio_proj_out', 'rope'})
        Native = type(cls.name, (Model,), {'_keep_in_fp32_modules': roots,
            'named_parameters': lambda self: self.params,
            'state_dict': lambda self, keep_vars=False: dict(self.params)})
        for name, accepted in [('proj_in.weight', True),
                               ('transformer_blocks.0.attn.weight', False)]:
            c = self.controller()
            c.model = Native()
            p = SimpleNamespace(dtype='torch.float32', shape=(8, 8))
            c.model.params = [(name, p)]
            self.assertEqual(_native_fp32_roots(source, c.model), roots)
            before = c._identity()
            if accepted:
                self.install(c, source=source, real_extract=True)
            else:
                with self.assertRaises(RuntimeError):
                    self.install(c, source=source, real_extract=True)
            self.assertEqual(c._identity(), before)
            self.assertEqual(p.dtype, 'torch.float32')
        self.assertIn('_keep_in_fp32_modules = frozenset', source)

    def test_backend_budget_counts_recompiles(self):
        b = CompileBudget(lambda: 42, 1, 10)
        self.assertEqual(b(), 42)
        with self.assertRaises(CandidateRejected): b()
        self.assertEqual(b.count, 1)

    def test_over_time_rejected_after_return(self):
        ticks = iter([0, 3])
        b = CompileBudget(lambda: 42, 4, 2, lambda: next(ticks))
        with self.assertRaises(CandidateRejected): b()
        self.assertEqual(b.seconds, 3)


class ProbeProtocolTests(Tests):
    def test_asymmetric_probe_failure_matches_through_outer_guard(self):
        import threading
        from research.h3_personal.denoise_execution import RequestAdapter
        unsupported = type('Unsupported', (RuntimeError,), {'__module__': 'torch._dynamo.exc'})
        for failure in (unsupported('compiler'), RuntimeError('GPU'), CandidateRejected('benefit')):
            barrier = threading.Barrier(2, timeout=3)
            values, sequences, errors = {}, [[], []], [None, None]

            class ThreadStatus:
                def __init__(self, rank): self.rank = rank
                def reduce(self, value, label, op):
                    sequences[self.rank].append((op, label))
                    values[self.rank] = value
                    barrier.wait()
                    result = min(values.values()) if op == 'MIN' else max(values.values())
                    barrier.wait()
                    return result
                def min(self, ok, label): return self.reduce(ok, label, 'MIN')
                def rejection_vote(self, code, label): return self.reduce(code, label, 'MAX')

            owners = []
            for rank in range(2):
                c = self.controller()
                adapter = RequestAdapter(None, None, None, {})
                adapter.controller = c
                def compute(*args, rank=rank):
                    if rank == 0: raise failure
                    return dict(exact=True, max=0., mean=0., eager_ms=2., compiled_ms=1., host_seconds=1.)
                adapter._probe_compute = compute
                def prepare(block, _adapter=adapter, **kw):
                    def commit():
                        block._h3_region_b_dispatch = lambda *args: _adapter._probe(None, None, ())
                    return commit
                with patch('research.h3_personal.denoise_execution.local.extract', return_value=compile(
                        'def _h3_region_b(): pass\ndef forward(): pass', '<test>', 'exec')), patch(
                        'research.h3_personal.denoise_execution.local.install', side_effect=prepare):
                    c.install(source='', namespace={}, admission={}, stage_shapes={}, gates=dict(
                        source_pinned=True, real_bf16_weights=True,
                        static_adapter_receipt_verified=True, main50_verified=True))
                owners.append((c, adapter, c.blocks[0]._h3_region_b_dispatch))
            for rank, (c, adapter, dispatch) in enumerate(owners):
                c.status = ThreadStatus(rank)

            def run(rank):
                c, adapter, dispatch = owners[rank]
                c.context = ('first', 0, {})
                try:
                    dispatch()
                except Exception as exc: errors[rank] = exc
            threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join(5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(sequences[0], sequences[1])
            self.assertEqual(sequences[0], [('MIN', 'representative-probe'),
                ('MAX', 'representative-probe'), ('MIN', 'local-post'), ('MAX', 'local-post')])
            for rank, error in enumerate(errors):
                self.assertEqual(isinstance(error, CandidateRejected), not isinstance(failure, RuntimeError)
                                 or isinstance(failure, (unsupported, CandidateRejected)))
                self.assertEqual(owners[rank][0].state, 'faulted')
                self.assertIsNone(owners[rank][0].receipt['representative_metrics'])


class AdapterTests(unittest.TestCase):
    def test_delayed_binding_before_batch_preparation(self):
        from research.h3_personal.denoise_execution import apply_request
        from research.h3_personal.utilization_policy import DENOISE
        pipeline, batch = SimpleNamespace(), SimpleNamespace(extra={})
        adapter = apply_request(pipeline, batch, None, DENOISE)
        self.assertIsNone(adapter.controller)
        adapter.close()
        self.assertTrue(adapter.closed)

    def test_binding_fault_common_boundary_rolls_back(self):
        import sys
        from research.h3_personal.denoise_execution import RequestAdapter
        for local_failure in (False, True):
            c = Tests().controller(Status('bind-forward', peer_code=2))
            c.install = lambda **kw: setattr(c, 'state', 'ready')
            model = c.model
            original = lambda: None
            model.forward = original
            adapter = RequestAdapter(None, None, None, {})
            def preadmit(*args):
                adapter.controller, adapter.model = c, model
                adapter._install_context = '', {}, {}
            adapter._preadmit = preadmit
            bind = adapter._bind_forward
            def binding(*args):
                bind(*args)
                if local_failure: raise RuntimeError('binding setter')
            adapter._bind_forward = binding
            with patch.dict(sys.modules, {'torch': SimpleNamespace()}), patch(
                    'research.h3_personal.denoise_execution.TorchStatus', return_value=Status()):
                with self.assertRaises(RuntimeError):
                    adapter._initialize(None, SimpleNamespace(latents=SimpleNamespace(device='cpu')), {}, None)
            self.assertEqual(c.status.labels, ['bind-forward'])
            self.assertEqual(c.state, 'faulted')
            self.assertIs(model.forward, original)
            self.assertFalse(hasattr(model, '_h3_denoise_owner'))
            self.assertTrue(all(not hasattr(b, '_h3_local_probe') for b in c.blocks))

    def test_real_layout_shape_plan(self):
        from research.h3_personal.denoise_execution import shape_plan
        layout = SimpleNamespace(num_condition_video_rows=7296,
            num_condition_audio_rows=0, text_indices=SimpleNamespace(numel=lambda: 93))
        batch = SimpleNamespace(latents=SimpleNamespace(shape=(11440, 96)),
            audio_latents=SimpleNamespace(shape=(414, 32)))
        context = dict(token=SimpleNamespace(window=SimpleNamespace(guide_video_t=0, guide_audio_t=0)))
        r, t, plans = shape_plan(layout, batch, context)
        self.assertEqual((r, t), (7296, 93))
        self.assertEqual(plans['first'], (1, 5974, 5376))
        self.assertEqual(plans['guide'], (1, 6403, 5376))
        layout.num_condition_video_rows += 784
        layout.num_condition_audio_rows = 74
        batch.latents.shape = (12224, 96)
        batch.audio_latents.shape = (488, 32)
        context['token'].window.guide_video_t = 7
        context['token'].window.guide_audio_t = 37
        self.assertEqual(shape_plan(layout, batch, context), (r, t, plans))

    def test_custom_backend_options_forwarded(self):
        import sys
        from research.h3_personal.denoise_execution import inductor_backend
        calls = []
        fake = SimpleNamespace(compile_fx=lambda *a, **kw: calls.append((a, kw)))
        with patch.dict(sys.modules, {'torch._inductor.compile_fx': fake}):
            inductor_backend('graph', ['input'], options={'triton.cudagraphs': False})
        self.assertEqual(calls[0][1], dict(config_patches={'triton.cudagraphs': False}))


class RequestInitProtocolTests(unittest.TestCase):
    def test_asymmetric_preadmission_never_enters_prepare(self):
        import sys
        import threading
        from research.h3_personal.denoise_execution import RequestAdapter
        for failing_rank in (0, 1):
            for reason in ('env', 'static', 'model', 'shape', 'ownership', 'source'):
                barrier = threading.Barrier(2, timeout=2)
                values, sequences, errors = {}, [[], []], [None, None]

                class ThreadStatus:
                    def __init__(self, torch, device):
                        self.rank = device
                    def min(self, ok, label):
                        sequences[self.rank].append(label)
                        values[self.rank] = ok
                        barrier.wait()
                        result = all(values.values())
                        barrier.wait()
                        return result

                class Adapter(RequestAdapter):
                    def _preadmit(self, stage, batch, context, layout, status):
                        if status.rank == failing_rank:
                            raise ValueError(reason)
                        self.model = SimpleNamespace()
                        self.controller = SimpleNamespace(state='new', install=lambda **kw:
                            status.min(True, 'prepare-main50'))
                        self._install_context = '', {}, {}

                def run(rank):
                    status = ThreadStatus(None, rank)
                    adapter = Adapter(None, None, None, {})
                    try:
                        adapter._initialize(None, SimpleNamespace(
                            latents=SimpleNamespace(device=rank)), {}, None)
                    except Exception as exc:
                        errors[rank] = type(exc)
                    status.min(errors[rank] is None, 'outer-status')

                with patch.dict(sys.modules, {'torch': SimpleNamespace()}), patch(
                        'research.h3_personal.denoise_execution.TorchStatus', ThreadStatus):
                    threads = [threading.Thread(target=run, args=(r,), daemon=True)
                               for r in (0, 1)]
                    for thread in threads: thread.start()
                    for thread in threads: thread.join(3)
                self.assertFalse(any(t.is_alive() for t in threads))
                self.assertEqual(sequences, [['request-init', 'outer-status']] * 2)
                self.assertIs(errors[0], errors[1])
                self.assertTrue(issubclass(errors[0], RuntimeError))


if __name__ == '__main__': unittest.main()
