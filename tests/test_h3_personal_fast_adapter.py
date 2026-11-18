import importlib.util
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.h3_personal.personal_fast_adapter import validate_identity_config

class PersonalFastConfigTests(unittest.TestCase):
    def test_loaded_coverage_fails_closed(self):
        from research.h3_personal.personal_fast_adapter import assert_loaded_coverage
        state={f'x{i}.lora_A':SimpleNamespace(shape=(80 if i<208 else 64,3)) for i in range(362)}
        state.update({f'x{i}.lora_B':SimpleNamespace(shape=(2,80 if i<208 else 64)) for i in range(362)})
        with patch.dict('os.environ',{},clear=True):
            assert_loaded_coverage(state,set(state),362)
            for consumed,count in ((set(),362),(set(state),361)):
                with self.assertRaises(RuntimeError):assert_loaded_coverage(state,consumed,count)

    def test_guard_overlay_is_explicit_and_nonrepeatable(self):
        from research.h3_personal.personal_fast_adapter import install_coverage_guard
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'loader.py';p.write_text('        # The loop above reports model layers the adapter has nothing for. This is the\n')
            with self.assertRaises(ValueError):install_coverage_guard(p)
            report=install_coverage_guard(p,expected_sha=__import__('hashlib').sha256(p.read_bytes()).hexdigest())
            self.assertNotEqual(report['original_sha256'],report['patched_sha256'])
            self.assertIn('assert_loaded_coverage(self.lora_adapters',p.read_text())
            with self.assertRaises(ValueError):install_coverage_guard(p)

    def test_scale_contract(self):
        validate_identity_config({'r':16,'lora_alpha':16})
        for extra in ({'lora_alpha':8},{'use_rslora':True},{'use_dora':True},{'alpha_pattern':{'x':8}},{'bias':'all'}):
            with self.assertRaises(ValueError):validate_identity_config(dict(r=16,lora_alpha=16,**{})|extra)

@unittest.skipUnless(importlib.util.find_spec('torch'),'explicit CPU Torch environment required')
class PersonalFastCompositionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;g=torch.Generator().manual_seed(1)
        self.fast={'x.lora_A.weight':torch.randn(64,3,generator=g,dtype=torch.float64),
                   'x.lora_B.weight':torch.randn(4,64,generator=g,dtype=torch.float64),
                   'y.diff':torch.ones(4,3,dtype=torch.float64)}
        self.identity={'x.lora_A.weight':torch.randn(16,3,generator=g,dtype=torch.float64),
                       'x.lora_B.weight':torch.randn(4,16,generator=g,dtype=torch.float64)}
        self.config={'r':16,'lora_alpha':16}
    def test_rank80_sum_dense_and_original_invariance(self):
        from research.h3_personal.personal_fast_adapter import compose
        torch=self.torch;before={k:v.clone() for k,v in self.fast.items()};personal={k:v.clone() for k,v in self.identity.items()}
        result,report=compose(self.fast,self.identity,self.config)
        expected=self.fast['x.lora_B.weight']@self.fast['x.lora_A.weight']+self.identity['x.lora_B.weight']@self.identity['x.lora_A.weight']
        actual=result['x.lora_B.weight']@result['x.lora_A.weight']
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        self.assertEqual(result['x.lora_A.weight'].shape,(80,3));self.assertIs(result['y.diff'],self.fast['y.diff'])
        for k,v in self.fast.items():self.assertTrue(torch.equal(v,before[k]))
        for k,v in self.identity.items():self.assertTrue(torch.equal(v,personal[k]))
        self.assertFalse(report['gpu_parity_verified'])
    def test_gate_replacement_is_preserved(self):
        from research.h3_personal.personal_fast_adapter import compose
        self.fast['transformer_blocks.0.attn.to_gate_compress.set_weight']=self.torch.ones(1)
        result,report=compose(self.fast,self.identity,self.config)
        self.assertIs(result['transformer_blocks.0.attn.to_gate_compress.set_weight'],self.fast['transformer_blocks.0.attn.to_gate_compress.set_weight'])
        self.assertEqual(report['dense_tensors_preserved'],2)

    def test_missing_target_incomplete_or_alpha_rejected(self):
        from research.h3_personal.personal_fast_adapter import compose
        for identity in ({'z.lora_A.weight':self.identity['x.lora_A.weight'],'z.lora_B.weight':self.identity['x.lora_B.weight']},
                         {'x.lora_A.weight':self.identity['x.lora_A.weight']}):
            with self.assertRaises(ValueError):compose(self.fast,identity,self.config)
        with self.assertRaises(ValueError):compose(self.fast|{'x.alpha':self.torch.ones(1)},self.identity,self.config)
