import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research.h3_personal import ref2va_runtime as r

class Ref2VAContractTests(unittest.TestCase):
    def test_request_uses_ordered_reference_not_fl2va(self):
        def obj(**kw):return SimpleNamespace(**kw)
        api=SimpleNamespace(**{k:obj for k in ('GenerationRequest','InputConfig','SamplingConfig','OutputConfig',
            'GeneratorConfig','PipelineSelection','ComponentConfig','EngineConfig','ParallelismConfig','OffloadConfig','CompileConfig')})
        request=r.request(api,obj,Path('/private/own-photo.png'),r.Ref2VAConfig())
        self.assertEqual(len(request.inputs.references),1)
        self.assertEqual(request.inputs.references[0].media_type,'image')
        self.assertNotIn('pil_image',vars(request.inputs));self.assertNotIn('image_path',vars(request.inputs))
        self.assertEqual(request.sampling.num_inference_steps,51)
        config=r.generator_config(api,'base','original64')
        self.assertEqual(config.pipeline.components.override_pipeline_cls_name,r.PIPELINE)
        self.assertEqual(config.pipeline.components.lora_path,'original64')
        self.assertEqual(config.pipeline.experimental['attention_backend'],'FLASH_ATTN')
        self.assertNotIn('VSA_sparsity',config.pipeline.experimental)

    def test_explicit_person_scene_request_no_crop_or_silent_fallback(self):
        def obj(**kw):return SimpleNamespace(**kw)
        api=SimpleNamespace(**{k:obj for k in ('GenerationRequest','InputConfig','SamplingConfig','OutputConfig')})
        selection={'references':[{'path':'/private/person.png'},{'path':'/private/scene.png'}]}
        # File validation has its own real synthetic PNG tests. Here isolate forwarding.
        with patch('research.h3_personal.ordered_references.verify_local_files',return_value=selection) as verify:
            request=r.request(api,obj,'/private/person.png',r.Ref2VAConfig(),ordered_references=selection)
            verify.assert_called_once_with(selection)
            self.assertEqual([x.source for x in request.inputs.references],['/private/person.png','/private/scene.png'])
            self.assertEqual([x.media_type for x in request.inputs.references],['image','image'])
            with self.assertRaises(ValueError):
                r.request(api,obj,'/private/other.png',r.Ref2VAConfig(),ordered_references=selection)
            with self.assertRaises(ValueError):
                r.request(api,obj,'/private/person.png',r.Ref2VAConfig(),face_crop='/private/crop.png',ordered_references=selection)
        with patch('research.h3_personal.ordered_references.verify_local_files',side_effect=ValueError('changed')):
            with self.assertRaises(ValueError):
                r.request(api,obj,'/private/person.png',r.Ref2VAConfig(),ordered_references=selection)

    def test_original_header_pairs_not_derived_rank80(self):
        header={}
        for i in range(208):
            header[f'x{i}.lora_A.weight']={'shape':[16,3],'dtype':'BF16'}
            header[f'x{i}.lora_B.weight']={'shape':[2,16],'dtype':'BF16'}
        self.assertEqual(r.validate_identity_header(header),208)
        for key,value in (('x0.lora_A.weight',{'shape':[80,3],'dtype':'BF16'}),
                          ('x0.lora_A.weight',{'shape':[16,3],'dtype':'F32'})):
            bad=deepcopy(header);bad[key]=value
            with self.assertRaises(ValueError):r.validate_identity_header(bad)
        del header['x0.lora_B.weight']
        with self.assertRaises(ValueError):r.validate_identity_header(header)

    def test_config_does_not_accept_distilled_fast_settings(self):
        for change in ({'sigma_grid_points':5},{'num_frames':125},{'seed':True}):
            with self.assertRaises(ValueError):r.Ref2VAConfig(**change).validate()

    def test_actual_ref_partition_and_complete_rank16_required(self):
        pipeline=SimpleNamespace(_ref2va=True,_extra_config_module_map={'transformer':'transformer_ref'})
        state={}
        for i in range(208):
            state[f'x{i}.lora_A']=SimpleNamespace(shape=(16,3))
            state[f'x{i}.lora_B']=SimpleNamespace(shape=(2,16))
        with patch.dict(os.environ,{},clear=True):
            r.assert_ref_loaded_coverage(pipeline,state,set(state),208)
            for bad in (SimpleNamespace(_ref2va=False),SimpleNamespace(_ref2va=True,_extra_config_module_map={'transformer':'transformer'})):
                with self.assertRaises(RuntimeError):r.assert_ref_loaded_coverage(bad,state,set(state),208)
            with self.assertRaises(RuntimeError):r.assert_ref_loaded_coverage(pipeline,state,set(),208)
            with self.assertRaises(RuntimeError):r.assert_ref_loaded_coverage(pipeline,state,set(state),362)
            state['x0.lora_A']=SimpleNamespace(shape=(80,3))
            with self.assertRaises(RuntimeError):r.assert_ref_loaded_coverage(pipeline,state,set(state),208)

    def fixture(self,root):
        shards=[f'diffusion_pytorch_model-{i:05d}-of-00014.safetensors' for i in range(1,15)]
        data={f'transformer_ref/{n}':b'fixture-not-real-weights' for n in shards}
        data.update({'transformer_ref/config.json':b'{}','model_index.json':b'{}',
            'transformer_ref/diffusion_pytorch_model.safetensors.index.json':json.dumps({'weight_map':{f'p{i}':n for i,n in enumerate(shards)}}).encode()})
        for directory in ('scheduler','audio_scheduler','text_encoder','tokenizer','processor','vae','audio_vae'):
            data[directory+('/scheduler_config.json' if 'scheduler' in directory else '/config.json')]=b'{}'
        files=[]
        for name,body in data.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
            files.append(dict(path=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
        return dict(repo=r.BASE_REPO,revision=r.BASE_REV,files=files)

    def test_full_hashes_index_no_t2va_or_13shard_substitution(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=self.fixture(root)
            self.assertTrue(r.verify_model(root,manifest)['full_hashes_verified'])
            bad=deepcopy(manifest);bad['files']=[f for f in bad['files'] if not f['path'].endswith('00014-of-00014.safetensors')]
            with self.assertRaises(ValueError):r.validate_model_manifest(bad)
            bad=deepcopy(manifest);bad['files'][0]['path']='transformer/other.safetensors'
            with self.assertRaises(ValueError):r.validate_model_manifest(bad)
            (root/'fasth3_inference.json').write_text('{}')
            with self.assertRaises(ValueError):r.verify_model(root,manifest)
            (root/'fasth3_inference.json').unlink()
            (root/manifest['files'][0]['path']).write_bytes(b'x'*manifest['files'][0]['bytes'])
            with self.assertRaises(ValueError):r.verify_model(root,manifest)

    def test_git_blobs_and_unsafe_or_incomplete_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=self.fixture(root)
            row=next(f for f in manifest['files'] if f['path']=='model_index.json')
            row['sha256']=None;row['git_blob']=hashlib.sha1(b'blob 2\0{}').hexdigest()
            r.verify_model(root,manifest)
            for change in ({'repo':'other'},{'revision':'main'}):
                bad=dict(manifest,**change)
                with self.assertRaises(ValueError):r.validate_model_manifest(bad)
            for name in ('../escape','/absolute','a\\b'):
                with self.assertRaises(ValueError):r.relative(name)

    @unittest.skipUnless(os.environ.get('H3_FASTVIDEO_SOURCE_ARCHIVE'),'pinned public source archive required')
    def test_actual_pinned_api_dataclasses_and_ref_entry(self):
        import tarfile,sys,types
        from research.run_h3_ref2va import ARCHIVE_SHA,FASTVIDEO_REV
        archive=Path(os.environ['H3_FASTVIDEO_SOURCE_ARCHIVE'])
        self.assertEqual(r.sha(archive),ARCHIVE_SHA)
        prefix='FastVideo-'+FASTVIDEO_REV+'/fastvideo/'
        with tarfile.open(archive) as t:
            schema=t.extractfile(prefix+'api/schema.py').read().decode()
            reference=t.extractfile(prefix+'pipelines/basic/minimax_h3/reference.py').read().decode()
            pipeline=t.extractfile(prefix+'pipelines/basic/minimax_h3/minimax_h3_pipeline.py').read().decode()
        module=types.ModuleType('h3_pinned_api_fixture')
        with patch.dict(sys.modules,{module.__name__:module}):
            exec(compile(schema,'pinned-schema','exec'),module.__dict__)
            tree=ast.parse(reference);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3Reference')
            imports=ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass\nfrom typing import Any,Literal\nfrom numbers import Real,Integral\n')
            imports.body.append(node);exec(compile(ast.fix_missing_locations(imports),'pinned-reference','exec'),module.__dict__)
            req=r.request(module,module.MiniMaxH3Reference,'/private/reference.png',r.Ref2VAConfig())
            cfg=r.generator_config(module,'/model','/original64')
            self.assertIsInstance(req.inputs.references[0],module.MiniMaxH3Reference)
            self.assertEqual(cfg.pipeline.components.override_pipeline_cls_name,r.PIPELINE)
        self.assertIn('_extra_config_module_map = {"transformer": "transformer_ref"}',pipeline)
        self.assertIn('_ref2va_default = True',pipeline)
        self.assertIn('class '+r.PIPELINE,pipeline)

    def test_runner_is_no_submit_no_upload_and_defaults_offline(self):
        import research.run_h3_ref2va as runner
        source=Path(runner.__file__).read_text();ast.parse(source)
        self.assertNotIn('run_job(',source);self.assertNotIn('upload_file(',source)
        self.assertIn("if not a.run:",source)
        self.assertIn("generated_audio_discarded=True",source)
        self.assertIn("generator.shutdown()",source)
