import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research.h3_personal.incremental_decode import NativeDecodeStream,VAE_SHA


@unittest.skipUnless(importlib.util.find_spec('torch') and os.environ.get('H3_PINNED_VIDEO_VAE'),
                     'explicit pinned native VAE source and CPU Torch required')
class IncrementalDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        path=Path(os.environ['H3_PINNED_VIDEO_VAE'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=VAE_SHA:raise AssertionError('wrong VAE source')
        tree=ast.parse(path.read_text());methods={}
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name in ('_decode','_blend'):
                node.decorator_list=[];namespace={'torch':torch}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
                methods[node.name]=namespace[node.name]
        class VAE:
            tokens_chunk_size=5;temporal_compression_ratio=4;token_overlap=2
            frame_pre_padding=3;frame_overlap=5;config=SimpleNamespace(token_drop=3,clip_length=17)
            _decode=methods['_decode'];_blend=methods['_blend']
            def _decode_clip(self,z):return z.repeat_interleave(4,dim=2)+z.mean(dim=2,keepdim=True)
        cls.VAE=VAE

    def state(self):
        with patch('research.h3_personal.incremental_decode.verify_source'):
            return NativeDecodeStream(self.VAE())

    def test_exact_pinned_decode_agreement_varied_chunk_arrivals(self):
        torch=self.torch
        for total in (7,37,52,202):
            z=torch.arange(total,dtype=torch.float32).reshape(1,1,total,1,1)
            expected=self.VAE()._decode(z)
            for width in (1,5,13,64):
                state=self.state();parts=[]
                for offset in range(0,total,width):
                    parts.extend(state.push(z[:,:,offset:offset+width],final=offset+width>=total))
                    if state.buffer is not None:self.assertLess(state.buffer.shape[2],7)
                actual=torch.cat(parts,dim=2)
                self.assertTrue(torch.equal(expected,actual),(total,width))
                self.assertTrue(state.closed)
                self.assertEqual(state.emitted_frames,expected.shape[2])

    def test_visual_audit_provenance_matches_pinned_decode_blends(self):
        from research.audit_h3_native_video import frame_context
        torch=self.torch
        class TaggedVAE(self.VAE):
            def _decode_clip(self,z):
                tag=int(z[0,0,0,0,0].item())//5
                return torch.full((1,1,z.shape[2]*4,1,1),float(tag))
        z=torch.arange(202,dtype=torch.float32).reshape(1,1,202,1,1)
        decoded=TaggedVAE()._decode(z).flatten();expected=[]
        for frame in range(685):
            context=frame_context(frame);clip=context['decode_clip']
            if context['temporal_overlap_slot']:
                weight=context['body_phase']/5
                expected.append((clip-1)*(1-weight)+clip*weight)
            else:expected.append(float(clip))
        self.assertTrue(torch.allclose(decoded,torch.tensor(expected),atol=1e-5,rtol=0))
        self.assertTrue(torch.equal(decoded[:17],torch.zeros(17)))

    def test_emits_before_full_video_available_and_refuses_reuse(self):
        torch=self.torch;state=self.state()
        chunks=list(state.push(torch.zeros(1,1,7,1,1)))
        self.assertEqual(sum(c.shape[2] for c in chunks),17)
        self.assertFalse(state.closed);self.assertEqual(state.buffer.shape[2],2)
        tail=list(state.push(torch.zeros(1,1,5,1,1),final=True))
        self.assertEqual(sum(c.shape[2] for c in tail),22)
        with self.assertRaises(ValueError):list(state.push(torch.zeros(1,1,5,1,1)))

    def test_no_speculative_grid_padding_or_concurrent_push(self):
        torch=self.torch;state=self.state()
        with self.assertRaises(ValueError):list(state.push(torch.zeros(1,1,8,1,1),final=True))
        state=self.state();iterator=state.push(torch.zeros(1,1,12,1,1));next(iterator)
        with self.assertRaises(ValueError):list(state.push(torch.zeros(1,1,5,1,1)))
        iterator.close();self.assertTrue(state.closed)
