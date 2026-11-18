import ast
from collections.abc import Mapping,Sequence
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave
from research.h3_personal.continuation_plan import Window,plan,request,covering_frames,video_t,audio_boundary
from research.plan_h3_continuation import main


class ContinuationPlanTests(unittest.TestCase):
    def test_recording_coverage_and_exact_clock(self):
        r=request(1364312,48000)
        self.assertEqual(r['total_frames'],685)
        windows=plan(685)
        self.assertEqual([(w.start,w.end) for w in windows],[(0,277),(255,532),(510,685)])
        self.assertEqual([w.guide_audio_t for w in windows],[0,37,37])
        self.assertEqual(sum(video_t(w.frames)-w.guide_video_t for w in windows),202)
        self.assertEqual(sum(w.audio_end-w.audio_start-w.guide_audio_t for w in windows),1142)
        self.assertEqual(str(windows[1].temporal_offset),'425')

    def test_all_finite_grid_totals_accept_or_explicitly_reject_short_tail(self):
        for frames in range(124,7197,17):
            try:windows=plan(frames)
            except ValueError as e:
                self.assertIn('short final target',str(e));continue
            self.assertEqual(sum(video_t(w.frames)-w.guide_video_t for w in windows),video_t(frames))
            self.assertEqual(sum(w.audio_end-w.audio_start-w.guide_audio_t for w in windows),audio_boundary(frames))
            self.assertTrue(all(124<=w.frames<=345 for w in windows))

    def test_no_silent_short_tail_guard_bypass_or_recipe_change(self):
        with self.assertRaises(ValueError):plan(685,345,22)  # final39frames is unsupported here
        with self.assertRaises(ValueError):request(1364312,48000,video_shift=6)
        for args in [(True,277,22),(685,277,True),(685,107,22),(685,277,277),(7213,277,22),(684,277,22)]:
            with self.assertRaises(ValueError):plan(*args)
        for args in [(1,278,22),(0,277,22),(255,532,0)]:
            with self.assertRaises(ValueError):Window(*args)
        with self.assertRaises(ValueError):covering_frames(0,48000)

    def test_cli_local_wav_only_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'source.wav';output=root/'plan.json'
            with wave.open(str(path),'wb') as w:
                w.setnchannels(1);w.setsampwidth(3);w.setframerate(48000);w.writeframes(b'\0'*48000*3)
            before=path.read_bytes();self.assertEqual(main(['--audio',str(path),'--output',str(output)]),0)
            r=json.loads(output.read_text());self.assertTrue(r['plan_only']);self.assertFalse(r['gpu_launched'])
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(FileExistsError):main(['--audio',str(path),'--output',str(output)])


@unittest.skipUnless(os.environ.get('H3_PINNED_PACKING_DIR') and importlib.util.find_spec('torch'),
                     'explicit offline pinned Diffusers fixture and CPU torch required')
class ContinuationPinnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        # No installed source edits: temporarily load exact external fixture files
        # under their package names, then restore module bindings in tearDownClass.
        import importlib
        importlib.import_module('diffusers.modular_pipelines.minimax_h3')
        cls.old={};cls.loaded={}
        for short in ('packing','packing_ref2va'):
            name='diffusers.modular_pipelines.minimax_h3.'+short
            cls.old[name]=sys.modules.get(name)
            spec=importlib.util.spec_from_file_location(name,Path(os.environ['H3_PINNED_PACKING_DIR'])/(short+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
            cls.loaded[short]=module
        cls.p=cls.loaded['packing'];cls.r=cls.loaded['packing_ref2va']
        from research.h3_personal.continuation_layout import verify_modules
        verify_modules(cls.p,cls.r)

    @classmethod
    def tearDownClass(cls):
        for name,old in cls.old.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old

    def image(self):
        return self.r.MiniMaxH3PreparedReference(kind='image',num_latent_frames=1,latent_height=4,latent_width=4)

    def layout(self,w,tags=None,ranks=2):
        from research.h3_personal.continuation_layout import build_layout
        return build_layout(self.p,self.r,text_tags=self.torch.tensor([1,0]) if tags is None else tags,
                            image=self.image(),window=w,height=4,width=4,ranks=ranks)

    def test_guide_shares_target_global_clock_static_image_fixed(self):
        t=self.torch;w=plan(685)[1];layout=self.layout(w)
        base=self.layout(plan(685)[0]);self.assertEqual(layout.position_ids.dtype,t.float64)
        self.assertEqual(float(layout.position_ids[layout.video_indices[0],0]),2)
        guide_first=layout.video_indices[4];target_first=layout.video_indices[layout.num_condition_video_rows]
        self.assertEqual(float(layout.position_ids[guide_first,0]),428)
        self.assertEqual(float(layout.position_ids[target_first,0]),428)
        self.assertEqual(float(layout.position_ids[layout.audio_indices[0],0]),428)
        self.assertEqual(float(layout.position_ids[layout.audio_indices[layout.num_condition_audio_rows],0]),428)
        self.assertTrue(t.equal(layout.position_ids[:6],base.position_ids[:6]))
        times,indices=self.p.build_row_timesteps(layout,.5,1.,.999,1.)
        rows=times[indices]
        self.assertTrue(t.allclose(rows[layout.video_indices[:layout.num_condition_video_rows]],t.full((32,),.999)))
        self.assertTrue(t.equal(rows[layout.audio_indices],t.ones_like(rows[layout.audio_indices])))

    def test_cp_rejects_odd_without_padding(self):
        with self.assertRaises(ValueError):self.layout(plan(685)[0],self.torch.ones(3,dtype=self.torch.long))
        self.layout(plan(685)[0],self.torch.ones(3,dtype=self.torch.long),ranks=1)

    def test_run_reuses_model_preserves_full_audio_and_no_qwen(self):
        from research.h3_personal.continuation_runtime import run
        t=self.torch;calls=[];events=[];source=t.arange(2284*32,dtype=t.float32).reshape(2284,32)/100000
        original=source.clone();fixed=t.zeros(4,96)
        def model(**kw):
            calls.append(kw['hidden_states'].shape[1]);return t.zeros_like(kw['hidden_states']),None
        class Scheduler:
            config=SimpleNamespace(shift=12)
            def set_timesteps(self,n,device):
                assert n==9;self.timesteps=t.linspace(.99,.01,8)
            def step(self,v,timestep,x,return_dict):return (x-v*.1,)
        v,a,r=run(model,Scheduler,self.p,self.r,text=t.zeros(2,8),text_tags=t.tensor([1,0]),
                  image=self.image(),image_rows=fixed,source_audio=source,total_frames=685,
                  height=4,width=4,seed=42,device='cpu',on_window=events.append)
        self.assertEqual(v.shape,(808,96));self.assertTrue(t.equal(a,original))
        self.assertTrue(t.equal(source,original));self.assertTrue(t.equal(fixed,t.zeros_like(fixed)))
        self.assertEqual(len(calls),24);self.assertEqual(len(events),3)
        self.assertFalse(r['decode_performed']);self.assertEqual([e['cumulative_audio_t'] for e in events],[462,887,1142])

    def test_interrupt_does_not_emit_stale_window(self):
        from research.h3_personal.continuation_runtime import run,ContinuationInterrupted
        t=self.torch;events=[]
        with self.assertRaises(ContinuationInterrupted):
            run(None,None,self.p,self.r,text=t.zeros(2,8),text_tags=t.tensor([1,0]),image=self.image(),
                image_rows=t.zeros(4,96),source_audio=t.zeros(2284,32),total_frames=685,
                height=4,width=4,seed=42,device='cpu',is_current=lambda:False,on_window=events.append)
        self.assertEqual(events,[])

    def test_mid_forward_cancellation_emits_no_partial_window(self):
        from research.h3_personal.continuation_runtime import run,ContinuationInterrupted
        t=self.torch;calls=[];events=[]
        def model(**kw):calls.append(1);return t.zeros_like(kw['hidden_states']),None
        class Scheduler:
            config=SimpleNamespace(shift=12)
            def set_timesteps(self,n,device):self.timesteps=t.linspace(.99,.01,8)
            def step(self,v,ts,x,return_dict):return (x,)
        with self.assertRaises(ContinuationInterrupted):
            run(model,Scheduler,self.p,self.r,text=t.zeros(2,8),text_tags=t.tensor([1,0]),
                image=self.image(),image_rows=t.zeros(4,96),source_audio=t.zeros(2284,32),
                total_frames=685,height=4,width=4,seed=42,device='cpu',
                is_current=lambda:len(calls)<2,on_window=events.append)
        self.assertEqual(len(calls),2);self.assertEqual(events,[])

    def test_cp_collective_cancellation_gate(self):
        from unittest.mock import patch
        from research.h3_personal.continuation_runtime import run,ContinuationInterrupted
        t=self.torch
        with patch.object(t.distributed,'is_initialized',return_value=True), \
             patch.object(t.distributed,'get_world_size',return_value=2), \
             patch.object(t.distributed,'all_reduce',side_effect=lambda status,op:status.zero_()) as collective:
            with self.assertRaises(ContinuationInterrupted):
                run(None,None,self.p,self.r,text=t.zeros(2,8),text_tags=t.tensor([1,0]),image=self.image(),
                    image_rows=t.zeros(4,96),source_audio=t.zeros(2284,32),total_frames=685,
                    height=4,width=4,seed=42,device='cpu')
            collective.assert_called_once()

    def test_corrected_reference_text_binding_and_no_legacy_mixing(self):
        from research.h3_personal import continuation_conditioning as c
        t=self.torch
        class Backend:
            def image(self,unit,geometry_policy):
                if geometry_policy!='correct-hw-v2':raise AssertionError('legacy path')
                return SimpleNamespace(width=64,height=96)
            def reference(self,unit,image):return dict(latents=t.zeros(6,96),geometry=t.tensor([[0,1,6,4,0]]))
            def presentation(self,image,geometry):return [geometry]
            def text(self,prompt,refs):return dict(prompt_embeds=t.zeros(2,8),text_token_tags=t.tensor([1,0]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'private';path.write_bytes(b'synthetic reference');sha=hashlib.sha256(path.read_bytes()).hexdigest()
            base={'fixture':'not actual model'};backend=Backend()
            refs,binding=c.reference(backend,path,sha,base_identity=base)
            text,report=c.text(backend,path,sha,refs,binding,'hello',base_identity=base)
            self.assertEqual(report['reference_key'],binding['key'])
            with self.assertRaises(ValueError):c.text(backend,path,sha,refs,{**binding,'geometry_policy':'legacy-wh-v1'},'hello',base_identity=base)
            refs['latents'][0,0]=1
            with self.assertRaises(ValueError):c.text(backend,path,sha,refs,binding,'hello',base_identity=base)

    def upstream(self,file,expected,names):
        root=os.environ.get('H3_CONTINUATION_UPSTREAM_DIR')
        if not root:self.skipTest('explicit pinned upstream source fixture required')
        path=Path(root)/file;data=path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),expected)
        import numpy as np
        ns={'torch':self.torch,'np':np,'math':math,'Sequence':Sequence,'Mapping':Mapping,'OmniClientError':ValueError}
        tree=ast.parse(data)
        for node in tree.body:
            if isinstance(node,ast.Assign):
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                for name in node.targets:
                    if isinstance(name,ast.Name):ns[name.id]=value
        ns.update(_FRAME_RESCALE=5/3,MINIMAX_H3_MAX_PAD_SEQ_LEN=1<<20)
        body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual(len(body),len(names))
        exec(compile(ast.Module(body=body,type_ignores=[]),'reviewed-pinned-upstream-pure-functions','exec'),ns)
        return ns

    def test_layout_matches_upstream_native_guide_positions_and_indices(self):
        ns=self.upstream('packed_sequence.py','a206d2d2fb72a5b485f224e12a71ee867cef650e7b6fe52dabf575a9cd10acb9',
            {'_resolve_padded_seq_len','_positive_int','_axis_from_sqrt_area','_video_t_grid','_video_t_span',
             '_range_for_slice','_cat_ranges','minimax_h3_packed_sequence_ref2va_blocks'})
        t=self.torch
        for w in plan(685):
            layout=self.layout(w);blocks=[dict(kind='image',latent_h=4,latent_w=4)]
            if w.overlap:blocks.append(dict(kind='latent_guide',latent_t=w.guide_video_t,latent_h=4,latent_w=4,ref_audio_t=w.guide_audio_t))
            g=ns['minimax_h3_packed_sequence_ref2va_blocks'](text_len=2,latent_t=video_t(w.frames),
                latent_h=4,latent_w=4,audio_t=w.audio_end-w.audio_start,ref_blocks=blocks,
                seq_len=layout.sequence_length,temporal_offset=float(w.temporal_offset))
            self.assertTrue(t.allclose(layout.position_ids,g['img_position_ids'],atol=1e-10,rtol=0))
            self.assertTrue(t.equal(layout.video_indices,g['img_pos']))
            self.assertTrue(t.equal(layout.audio_indices,g['audio_pos']))
            g['token_tags'][:2]=t.tensor([1,0]);self.assertTrue(t.equal(layout.token_tags,g['token_tags']))

    def test_visual_noise_matches_upstream_per_block_seed_draw(self):
        ns=self.upstream('condition_noise.py','4b4d2d05e55124770fd288201cd1ebe612c9e6c647f0593a4d1381b5e175b108',
            {'minimax_h3_imgvid_cond_noise_aug_rows'})
        ns['minimax_h3_patchify_video_latent']=lambda x,patch_size:self.p.patchify_video_latents(x,patch_size)
        from research.h3_personal.continuation_layout import condition_video
        t=self.torch;clean=t.arange(32*96,dtype=t.float32).reshape(32,96)/1000;before=clean.clone()
        args=dict(condition_shapes=[(1,4,4),(7,4,4)],target_latent_t=82,imgvid_cond_num_frames=2,seed=42,noise_aug=.999)
        expected=ns['minimax_h3_imgvid_cond_noise_aug_rows'](clean,**args)
        actual=condition_video(self.p,clean,[(1,4,4),(7,4,4)],82,42)
        self.assertTrue(t.equal(actual,expected));self.assertTrue(t.equal(clean,before))
