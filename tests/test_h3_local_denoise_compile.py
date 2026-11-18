"""Source-only native archive tests; no torch, GPU, extraction, or network."""
import ast
import pathlib
import tarfile
import types
import unittest
from contextlib import nullcontext

from research.h3_personal.local_denoise_compile import install, extract
from research.h3_personal.private_root import base as private_base


class Tensor:
    dtype = 'torch.bfloat16'
    device = 'cpu'
    shape = (1, 4, 5376)

    def __init__(self, value):
        self.value = value

    def stride(self):
        return (21504, 5376, 1)

    def __add__(self, other):
        return Tensor(self.value + (other.value if isinstance(other, Tensor) else other))

    __radd__ = __add__

    def __mul__(self, other):
        return Tensor(self.value * (other.value if isinstance(other, Tensor) else other))

    __rmul__ = __mul__

    def to(self, dtype):
        return self

    def index_select(self, *args):
        return self

    def is_contiguous(self):
        return True


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = private_base()
        paths = list(root.glob('**/gpu-hf-fast-ref2va-smd4-v14-live-transfer/deployment/runtime/FastVideo.tar.gz'))
        if not paths:
            raise unittest.SkipTest('pinned local native archive unavailable')
        with tarfile.open(paths[0]) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith('/fastvideo/models/dits/minimax_h3.py'))
            cls.source = archive.extractfile(member).read().decode()
            packing = next(m for m in archive.getmembers() if m.name.endswith('/minimax_h3/packing.py'))
            cls.packing_source = archive.extractfile(packing).read().decode()
        extract(cls.source)

    def make(self, native_future=False):
        events = []
        compiling = [False]
        ns = dict(nvtx_range=lambda *a: nullcontext(),
                  STAGES=types.SimpleNamespace(span=lambda *a: nullcontext()),
                  torch=types.SimpleNamespace(compiler=types.SimpleNamespace(is_compiling=lambda: compiling[0])),
                  _can_run_minimax_h3_fusion=lambda x: False)
        tree = ast.parse(self.source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniMaxH3TransformerBlock')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        fn.returns = None
        for a in fn.args.args:
            a.annotation = None
        import __future__
        flags = __future__.annotations.compiler_flag if native_future else 0
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     '<native>', 'exec', flags=flags, dont_inherit=True), ns)
        native_class = type('MiniMaxH3TransformerBlock', (), {'forward': ns['forward']})
        ns['MiniMaxH3TransformerBlock'] = native_class
        block = native_class()
        block.fuse_modulate = False
        block.exact_modulate = False
        block._h3_local_forward_index = 0
        block.norm1 = lambda x: events.append('norm1') or x
        block.norm2 = lambda x: events.append('norm2') or x
        block.attn = lambda x, *a: events.append('attention') or x
        block.ff = types.SimpleNamespace(chunk_tokens=0, use_mxfp8=False)
        class FF:
            chunk_tokens = 0
            use_mxfp8 = False
            def __call__(self, x):
                events.append('ff')
                return x
        block.ff = FF()
        block.adaln_proj = lambda x: [Tensor(i) for i in range(6)]
        block.forward = types.MethodType(ns['forward'], block)
        return block, ns, events, compiling

    def config(self):
        return dict(source=self.source, main_block_index=0,
                    admission=dict(bf16=True, static_merged=True, fa4=True,
                                   portrait=True, resident=True, old_regional=False),
                    stage_shapes={'first': (1, 4, 5376), 'guide': (1, 5, 5376)})

    def test_eager_default_and_source_pin(self):
        block = object()
        self.assertFalse(install(block, source='', namespace={}))
        with self.assertRaises(ValueError):
            extract(self.source + '\n')

    def test_real_native_future_annotation_flags(self):
        block, ns, _, _ = self.make(native_future=True)
        install(block, namespace=ns, enabled=True,
                compile_factory=lambda fn, **kw: fn, **self.config())
        block._h3_local_stage = 'first'
        self.assertIsNotNone(block.forward(Tensor(2), Tensor(1), Tensor(0), (), 4))

    def test_native_branch_parity_order_and_identity(self):
        block, ns, events, compiling = self.make()
        args = (Tensor(2), Tensor(1), Tensor(0), (), 4)
        expected = block.forward(*args).value
        order = events[:]
        identities = tuple(id(getattr(block, k)) for k in ('ff', 'norm1', 'norm2', 'attn', 'adaln_proj'))
        calls = []
        def factory(fn, **options):
            calls.append(options)
            def run(*a):
                compiling[0] = True
                try:
                    return fn(*a)
                finally:
                    compiling[0] = False
            return run
        install(block, namespace=ns, enabled=True, compile_factory=factory, **self.config())
        self.assertEqual(calls, [])
        block._h3_local_stage = 'first'
        for _ in range(2):
            events.clear()
            self.assertEqual(block.forward(*args).value, expected)
            self.assertEqual(events, order)
        self.assertEqual(calls, [dict(fullgraph=True, dynamic=False, options={'emulate_precision_casts': True, 'triton.cudagraphs': False})])
        self.assertEqual(identities, tuple(id(getattr(block, k)) for k in ('ff', 'norm1', 'norm2', 'attn', 'adaln_proj')))
        block._h3_local_post_attention = False
        self.assertEqual(block.forward(*args).value, expected)
        block._h3_local_post_attention = True
        block._h3_local_stage = 'guide'
        with self.assertRaises(ValueError):
            block.forward(*args)

    def test_rejection_transaction_and_no_fallback(self):
        block, ns, _, _ = self.make()
        original = block.forward
        block.ff.chunk_tokens = 4
        with self.assertRaises(ValueError):
            install(block, namespace=ns, enabled=True, **self.config())
        self.assertIs(block.forward, original)
        block.ff.chunk_tokens = 0
        def fail(*a, **kw):
            raise RuntimeError('compiler failure /private/secret')
        install(block, namespace=ns, enabled=True, compile_factory=fail, **self.config())
        block._h3_local_stage = 'first'
        with self.assertRaisesRegex(RuntimeError, 'compiler failure'):
            block.forward(Tensor(2), Tensor(1), Tensor(0), (), 4)
        self.assertEqual(block._h3_local_compile_receipt['state'], 'faulted')
        self.assertEqual(block._h3_local_compile_receipt['fault'], 'RuntimeError')
        with self.assertRaisesRegex(RuntimeError, 'permanently faulted'):
            block.forward(Tensor(2), Tensor(1), Tensor(0), (), 4)

    def test_phase_signatures_and_late_compiler_fault(self):
        block, ns, _, _ = self.make()
        calls = []
        def factory(fn, **kw):
            def run(*args):
                calls.append(1)
                if len(calls) == 3:
                    raise RuntimeError('backend execution failed')
                return fn(*args)
            return run
        install(block, namespace=ns, enabled=True, compile_factory=factory, **self.config())
        block._h3_local_stage = 'first'
        args = (Tensor(2), Tensor(1), Tensor(0), (), 4)
        block.forward(*args)
        block._h3_local_forward_index = 1
        block.adaln_proj = lambda x: [self.table(i, 6) for i in range(6)]
        block.forward(*args)
        self.assertEqual(len(block._h3_local_compile_receipt['signatures']), 2)
        block._h3_local_forward_index = 2
        with self.assertRaisesRegex(RuntimeError, 'backend execution failed'):
            block.forward(*args)
        with self.assertRaisesRegex(RuntimeError, 'permanently faulted'):
            block.forward(*args)
        self.assertEqual(len(calls), 3)

    @staticmethod
    def table(value, rows):
        t = Tensor(value)
        t.shape = (rows, 5376)
        return t

    def test_fused_and_exact_native_branches(self):
        for mode in ('fused', 'exact'):
            with self.subTest(mode=mode):
                block, ns, events, compiling = self.make()
                def modulate(x, scale, shift, indices):
                    events.append('modulate')
                    return x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
                def residual(x, gate, output, indices):
                    events.append('residual')
                    return x + gate.index_select(0, indices) * output
                def gated(x, gate, indices, output):
                    events.append('outer_gated')
                    return x + gate.index_select(0, indices) * output
                def fused_norm(x, weight, scale, shift, indices, eps):
                    return modulate(block.norm1(x), scale, shift, indices)
                def fused_residual(x, output, gate, weight, scale, shift, indices, eps):
                    x = residual(x, gate, output, indices)
                    return x, modulate(block.norm2(x), scale, shift, indices)
                class Norm:
                    weight = Tensor(1)
                    eps = 1e-6
                    def __call__(self, x):
                        return x
                block.norm1 = Norm()
                block.norm2 = Norm()
                block.fuse_modulate = mode == 'fused'
                block.exact_modulate = mode == 'exact'
                ns.update(_can_run_minimax_h3_fusion=lambda x: True,
                          fused_rmsnorm_modulate=fused_norm,
                          fused_residual_gate_rmsnorm_modulate=fused_residual,
                          _gated_residual=gated,
                          exact_kernels=types.SimpleNamespace(supports_rowwise=lambda *a: True,
                                                              modulate=modulate, gate_residual=residual))
                args = (Tensor(2), Tensor(1), Tensor(0), (), 4)
                expected = block.forward(*args).value
                self.assertEqual(events.count('modulate'), 2)
                self.assertEqual(events.count('outer_gated'), int(mode == 'fused'))
                def factory(fn, **kw):
                    def run(*a):
                        compiling[0] = True
                        try:
                            return fn(*a)
                        finally:
                            compiling[0] = False
                    return run
                install(block, namespace=ns, enabled=True, compile_factory=factory, **self.config())
                block._h3_local_stage = 'first'
                events.clear()
                self.assertEqual(block.forward(*args).value, expected)
                self.assertEqual(events.count('modulate'), 2)
                self.assertNotIn('outer_gated', events)
                self.assertEqual(events.count('residual'), 1 if mode == 'fused' else 2)

    def test_drift_permanently_faults(self):
        for drift in ('shape', 'index', 'ff', 'signature'):
            with self.subTest(drift=drift):
                block, ns, _, _ = self.make()
                install(block, namespace=ns, enabled=True, compile_factory=lambda fn, **kw: fn, **self.config())
                block._h3_local_stage = 'first'
                args = (Tensor(2), Tensor(1), Tensor(0), (), 4)
                block.forward(*args)
                if drift == 'shape':
                    block._h3_local_stage = 'guide'
                elif drift == 'index':
                    block._h3_local_forward_index = True
                elif drift == 'ff':
                    block.ff.chunk_tokens = 1
                else:
                    block.adaln_proj = lambda x: [self.table(i, 7) for i in range(6)]
                with self.assertRaises(ValueError):
                    block.forward(*args)
                self.assertEqual(block._h3_local_compile_receipt['state'], 'faulted')
                with self.assertRaisesRegex(RuntimeError, 'permanently faulted'):
                    block.forward(*args)

    def test_source_row_timestep_shape_plan_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        tree = ast.parse(self.packing_source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_row_timesteps')
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        ns = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<pinned-packing>', 'exec'), ns)
        layout = types.SimpleNamespace(sequence_length=8, video_indices=torch.tensor([0, 1, 2, 3]),
                                       audio_indices=torch.tensor([4, 5, 6, 7]),
                                       num_condition_video_rows=1, num_condition_audio_rows=1)
        for values, count in (((1., 1., 1., 1.), 1), ((1., 2., 0., 0.), 3), ((1., 2., 3., 4.), 4)):
            unique, inverse = ns['build_row_timesteps'](layout, *values)
            self.assertEqual(tuple(unique.shape), (count,))
            self.assertEqual(tuple(inverse.shape), (8,))
            rows = unique[inverse]
            self.assertEqual(rows.tolist(), [values[2], values[0], values[0], values[0],
                                             values[3], values[1], values[1], values[1]])

    def test_strict_admission(self):
        for change in ('bool', 'extra', 'shape', 'index'):
            block, ns, _, _ = self.make()
            config = self.config()
            if change == 'bool':
                config['admission']['bf16'] = 1
            elif change == 'extra':
                config['admission']['extra'] = True
            elif change == 'shape':
                config['stage_shapes']['first'] = (True, 4, 5376)
            else:
                config['main_block_index'] = True
            with self.assertRaises(ValueError):
                install(block, namespace=ns, enabled=True, **config)
            self.assertFalse(hasattr(block, '_h3_region_b_dispatch'))


if __name__ == '__main__':
    unittest.main()
