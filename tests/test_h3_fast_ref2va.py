import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research.h3_personal import fast_ref2va as f
from research.h3_personal.ref2va_runtime import request
from research.audit_h3_fast_ref2va import compare_headers,component_inventory

class FastRefTests(unittest.TestCase):
    def test_default_warm_forward_without_utilization_helper(self):
        import builtins
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name.endswith('utilization_policy'):
                raise ModuleNotFoundError(name)
            return original(name, *args, **kwargs)
        pipeline = SimpleNamespace()
        with patch.dict(os.environ, {}, clear=True), \
                patch('builtins.__import__', side_effect=missing), \
                patch.object(f, '_warm_forward', return_value='legacy') as forward:
            self.assertEqual(f.warm_forward(pipeline, None, None, None), 'legacy')
            forward.assert_called_once_with(pipeline, None, None, None)

    def test_sampler_and_config_are_not_uniform_or_50_forward(self):
        self.assertEqual(f.FastRef2VAConfig().validate().sigma_grid_points,5)
        self.assertEqual(len(f.raw_sigmas()),5)
        self.assertAlmostEqual(f.raw_sigmas()[-2],1/7,places=15)
        self.assertNotEqual(f.raw_sigmas(),[1,.75,.5,.25,0])
        self.assertEqual(f.shifted_sigmas(12)[0],1)
        self.assertEqual(f.shifted_sigmas(3)[-1],0)
        with self.assertRaises(ValueError):f.FastRef2VAConfig(sigma_grid_points=51).validate()

    def test_ref_segments_emit_every_frame_and_throughput_is_not_prebuffer(self):
        from research.h3_personal.realtime_segments import SegmentPipeline
        with SegmentPipeline(lambda job,cancel:dict(id=job),timing_factory=f.segment_timing) as pipeline:
            for i in range(4):pipeline.submit(i)
        self.assertEqual(sum(r['timing'].newly_emitted_frames for r in pipeline._reports),496)
        self.assertTrue(all(r['timing'].head_trim_frames==0 for r in pipeline._reports))
        generations=[(i*4,i*4+3) for i in range(4)]
        reports=[dict(encode_started_at=i*4+3,publication_finished_at=i*4+3.5) for i in range(4)]
        self.assertTrue(f.evaluate_warm(generations,reports)['server_gate_passed'])
        self.assertFalse(f.evaluate_warm([(i*4,i*4+6) for i in range(4)],reports)['server_gate_passed'])
        reports[1]['publication_finished_at']=10
        self.assertFalse(f.evaluate_warm(generations,reports)['server_gate_passed'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
    def test_warm_condition_cache_avoids_encoder_and_keeps_new_latent_stage(self):
        import torch,sys,types,json
        with tempfile.TemporaryDirectory() as d,torch.inference_mode():
            image=Path(d)/'own-reference';image.write_bytes(b'pixels1')
            batch=SimpleNamespace(references=[SimpleNamespace(media_type='image',source=image)],prompt='own person',
                height=256,width=448,num_frames=124,fps=24,prompt_embeds=[torch.tensor([1.])],
                extra={'minimax_h3_text_token_tags':torch.tensor([2])})
            key=f.condition_key(batch);batch.extra['_smd_condition_key']=key;stage=SimpleNamespace()
            f.store_condition(stage,batch);batch.prompt_embeds[0].fill_(99)
            self.assertEqual(stage._smd_condition_cache[1][0].item(),1)
            order=[]
            def step(name):
                def apply(b,args):
                    order.append(name)
                    if name=='latent_preparation_stage':
                        self.assertEqual(b.prompt_embeds[0].device.type,'meta')
                        self.assertEqual(b.extra['minimax_h3_text_token_tags'].device.type,'cpu')
                        tags=torch.zeros(3,dtype=torch.long)
                        tags[torch.tensor([1])]=b.extra['minimax_h3_text_token_tags'].to(torch.long)
                        self.assertEqual(tags.tolist(),[0,2,0])
                    return b
                return apply
            mapping={n:step(n) for n in ('input_preparation_stage','latent_preparation_stage','denoising_stage','video_decoding_stage','audio_decoding_stage')}
            mapping['conditioning_stage']=stage;pipeline=SimpleNamespace(_stage_name_mapping=mapping)
            base=types.ModuleType('fastvideo.pipelines.composed_pipeline_base');base.get_local_torch_device=lambda:'meta'
            with patch.dict(sys.modules,{'fastvideo.pipelines.composed_pipeline_base':base}),patch.dict(os.environ,{'H3_REF2VA_COVERAGE_DIR':d}),patch.object(torch.distributed,'get_rank',return_value=0):
                f.warm_forward(pipeline,batch,None,stage._smd_condition_cache)
            self.assertEqual(order[0],'input_preparation_stage');self.assertIn('latent_preparation_stage',order)
            self.assertEqual(json.loads((Path(d)/'cache-rank-0.json').read_text()),{'hits':1})
            batch.prompt_embeds[0].fill_(77);self.assertEqual(stage._smd_condition_cache[1][0].item(),1)
            image.write_bytes(b'pixels2');self.assertNotEqual(f.condition_key(batch),key)

    def test_deferred_adapter_is_applied_exactly_once_before_denoise(self):
        order=[];model=object()
        pipeline=SimpleNamespace(_smd_deferred_adapter=True,trainable_transformer_modules={},
            get_module=lambda name:model,lora_nickname='default',lora_path='derived',lora_strength=1.0)
        pipeline.convert_to_lora_layers=lambda:order.append('convert')
        def apply(*args,**kw):
            self.assertTrue(pipeline._setting_constructor_adapter);self.assertEqual(kw,{'strength':1.0});order.append('apply')
        pipeline.set_lora_adapter=apply
        f.load_deferred_adapter(pipeline);f.load_deferred_adapter(pipeline)
        self.assertEqual(order,['convert','apply']);self.assertFalse(pipeline._setting_constructor_adapter)
        self.assertEqual(pipeline.trainable_transformer_modules,{'transformer':model})

    def test_header_target_coverage_and_rejections(self):
        student={'__metadata__':{'alpha':'128'}};identity={}
        for i in range(312):
            student[f'x{i}.lora_A.default.weight']={'shape':[128,3],'dtype':'BF16'}
            student[f'x{i}.lora_B.default.weight']={'shape':[2,128],'dtype':'BF16'}
            if i<208:
                identity[f'x{i}.lora_A.weight']={'shape':[16,3],'dtype':'BF16'}
                identity[f'x{i}.lora_B.weight']={'shape':[2,16],'dtype':'BF16'}
        self.assertEqual(compare_headers(student,identity)['combined_rank'],144)
        bad=deepcopy(student);bad['x0.lora_A.default.weight']['shape'][1]=4
        with self.assertRaises(ValueError):compare_headers(bad,identity)
        bad=deepcopy(student);bad['__metadata__']['alpha']='8'
        with self.assertRaises(ValueError):compare_headers(bad,identity)

    def test_base_comparison_uses_payload_hash_not_revision_names(self):
        a=SimpleNamespace(siblings=[SimpleNamespace(rfilename='transformer_ref/x',size=2,lfs=SimpleNamespace(sha256='payload'),blob_id='pointer')])
        b=SimpleNamespace(siblings=[SimpleNamespace(rfilename='transformer_ref/x',size=2,lfs=SimpleNamespace(sha256='payload'),blob_id='different-pointer')])
        self.assertEqual(component_inventory(a),component_inventory(b))
        b.siblings[0].lfs.sha256='other'
        self.assertNotEqual(component_inventory(a),component_inventory(b))

    @unittest.skipUnless(os.environ.get('H3_FASTVIDEO_SOURCE_ARCHIVE'),'pinned archive required')
    def test_real_pinned_source_overlay_compiles_and_preserves_input(self):
        archive=Path(os.environ['H3_FASTVIDEO_SOURCE_ARCHIVE']);before=f.sha(archive)
        with tempfile.TemporaryDirectory() as d:
            with tarfile.open(archive) as t:t.extractall(d,filter='data')
            source=next(Path(d).iterdir());report=f.install(source)
            for p in (source/'fastvideo/pipelines/lora_pipeline.py',source/'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py',
                source/'fastvideo/pipelines/basic/minimax_h3/minimax_h3_pipeline.py',source/'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_conditioning.py'):
                ast.parse(p.read_text())
            self.assertNotEqual(report['schedule_original_sha256'],report['schedule_patched_sha256'])
            with self.assertRaises(ValueError):f.install(source)
        self.assertEqual(f.sha(archive),before)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),'CPU Torch+safetensors required')
    def test_derived312_serialization(self):
        import json,torch
        from safetensors.torch import save_file,load_file
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);student={};identity={}
            for i in range(312):
                for side,shape in (('A',(128,3)),('B',(2,128))):student[f'x{i}.lora_{side}.default.weight']=torch.ones(shape,dtype=torch.bfloat16)
                if i<208:
                    identity[f'x{i}.lora_A.weight']=torch.ones(16,3,dtype=torch.bfloat16)
                    identity[f'x{i}.lora_B.weight']=torch.ones(2,16,dtype=torch.bfloat16)
            s=root/'s.safetensors';p=root/'p.safetensors';c=root/'c.json';out=root/'new.safetensors'
            save_file(student,str(s));save_file(identity,str(p));c.write_text(json.dumps(dict(r=16,lora_alpha=16)))
            ss,ps=f.sha(s),f.sha(p)
            with patch.object(f,'WEIGHT_SHA',ss),patch.object(f,'IDENTITY_SHA',ps):report=f.build_adapter(s,p,c,out)
            self.assertEqual((f.sha(s),f.sha(p)),(ss,ps));self.assertTrue(report['strict_cpu_reload'])
            result=load_file(str(out));self.assertEqual(result['x0.lora_A.default.weight'].shape,(144,3))
            self.assertEqual(result['x311.lora_A.default.weight'].shape,(128,3))
