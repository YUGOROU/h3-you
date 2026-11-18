import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),'CPU Torch+safetensors required')
class PersonalBuilderTests(unittest.TestCase):
    def test_full_count_derived_serialization_and_immutability(self):
        import torch
        from safetensors.torch import save_file,load_file
        from research import build_h3_personal_fast_adapter as builder
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fast={};identity={}
            for i in range(362):
                fast[f'x{i}.lora_A.weight']=torch.full((64,3),.01,dtype=torch.bfloat16)
                fast[f'x{i}.lora_B.weight']=torch.full((2,64),.02,dtype=torch.bfloat16)
                if i<208:
                    identity[f'x{i}.lora_A.weight']=torch.full((16,3),.03,dtype=torch.bfloat16)
                    identity[f'x{i}.lora_B.weight']=torch.full((2,16),.04,dtype=torch.bfloat16)
            for i in range(82):fast[f'dense{i}.diff']=torch.ones(2,3,dtype=torch.bfloat16)
            for i in range(50):fast[f'transformer_blocks.{i}.attn.to_gate_compress.set_weight']=torch.ones(2,3,dtype=torch.bfloat16)
            f=root/'fast.safetensors';p=root/'identity.safetensors';c=root/'config.json';d=root/'derived.safetensors'
            save_file(fast,str(f),metadata={'format':'fastvideo-lora-v2','rank':'64'})
            save_file(identity,str(p));c.write_text(json.dumps({'r':16,'lora_alpha':16}))
            fs,ps=builder.sha(f),builder.sha(p)
            with patch.object(builder,'FAST_SHA',fs),patch.object(builder,'IDENTITY_SHA',ps):
                report=builder.build(f,p,c,d)
                with self.assertRaises(ValueError):builder.build(f,p,c,d)
            self.assertEqual((builder.sha(f),builder.sha(p)),(fs,ps))
            self.assertEqual(report['identity_modules'],208);self.assertEqual(report['dense_tensors_preserved'],132)
            actual=load_file(str(d));self.assertEqual(actual['x0.lora_A.weight'].shape,(80,3))
            self.assertEqual(actual['x361.lora_A.weight'].shape,(64,3))
            for name in fast:
                if name.endswith(('.diff','.set_weight')):self.assertTrue(torch.equal(actual[name],fast[name]))
            self.assertTrue(report['strict_cpu_reload'])
