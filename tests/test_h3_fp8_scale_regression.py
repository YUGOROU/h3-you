"""Pinned TorchAO source math on CPU; NOT H200/CP kernel compatibility proof."""
import ast,hashlib,os,typing,unittest,zipfile
from research.h3_personal.acceleration_experiment import WHEEL_SHA,fp8_recipe,probe_fp8_linear

class RecipeWiring(unittest.TestCase):
    def test_explicit_floor_and_native_kernel_are_wired(self):
        import inspect
        tree=ast.parse(inspect.getsource(fp8_recipe))
        recipe=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Float8DynamicActivationFloat8WeightConfig')
        kw={k.arg:k.value for k in recipe.keywords}
        self.assertEqual(ast.literal_eval(kw['activation_value_lb']),1e-12)
        self.assertEqual(ast.unparse(kw['kernel_preference']),'KernelPreference.TORCH')
    def test_finite_and_scaled_mm_checks_not_weakened(self):
        import inspect
        source=inspect.getsource(probe_fp8_linear)
        self.assertIn("'zero'",source);self.assertIn("'nonzero'",source)
        self.assertIn("'_scaled_mm'",source);self.assertIn('not all(probes.values())',source)


@unittest.skipUnless(os.environ.get('H3_TORCHAO_WHEEL'),'explicit pinned TorchAO wheel required')
class TorchAOScaleRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        path=os.environ['H3_TORCHAO_WHEEL']
        with open(path,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==WHEEL_SHA
        with zipfile.ZipFile(path) as z:tree=ast.parse(z.read('torchao/quantization/quant_primitives.py').decode())
        names={'_choose_scale_float8','_quantize_affine_float8','_expand_scale_to_tensor_shape'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(nodes)==3
        # Remove registry decorators only; execute exact pinned function bodies.
        for node in nodes:node.decorator_list=[]
        env=dict(torch=torch,List=typing.List,Optional=typing.Optional)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),env)
        cls.choose=staticmethod(env['_choose_scale_float8']);cls.quantize=staticmethod(env['_quantize_affine_float8'])
    def test_old_zero_probe_reproduces_nan(self):
        t=self.torch;x=t.zeros((32,16),dtype=t.bfloat16)
        scale=self.choose(x,[]);self.assertEqual(scale.item(),0)
        self.assertFalse(t.isfinite(self.quantize(x,scale).float()).all())
    def test_floor_keeps_zero_probe_finite(self):
        t=self.torch;x=t.zeros((32,16),dtype=t.bfloat16)
        scale=self.choose(x,[],hp_value_lb=1e-12)
        self.assertGreater(scale.item(),0)
        self.assertTrue(t.isfinite(self.quantize(x,scale).float()).all())
        self.assertEqual(self.quantize(x,scale).float().abs().max().item(),0)
    def test_floor_does_not_change_normal_scale_or_quantized_values(self):
        t=self.torch;x=t.linspace(-1,1,512,dtype=t.bfloat16).reshape(32,16)
        before=self.choose(x,[]);after=self.choose(x,[],hp_value_lb=1e-12)
        self.assertTrue(t.equal(before,after))
        self.assertTrue(t.equal(self.quantize(x,before).float(),self.quantize(x,after).float()))
