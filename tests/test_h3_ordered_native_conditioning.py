import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zlib

from research.h3_personal import conversation_conditioning as roles
from research.h3_personal import conditioning_experiment as cond
from research.h3_personal.accepted_profiles import choose_profile
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime


class OrderedNativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        refs=[]
        for role,w,h in (('person',9,16),('surroundings',14,11)):
            path=self.root/(role+'.png')
            from test_h3_ordered_references import png_header
            data=png_header(w,h)
            path.write_bytes(data)
            refs.append(dict(role=role,path=str(path),sha256=hashlib.sha256(data).hexdigest(),width=w,height=h))
        self.selected=dict(schema='h3-ordered-references-v1',references=refs)
        self.context=roles.ordered_context_record(self.selected)
        self.file=self.root/'context.json';self.file.write_text(json.dumps(self.context))
        self.policy=self.root/'policy.json';self.policy.write_text(json.dumps(dict(schema=cond.SCHEMA,short_edges=[1024,512],active_short_edge=1024)))
        self.env=patch.dict(os.environ,{roles.ORDERED_CONTEXT_ENV:str(self.file),cond.EXPERIMENT_ENV:str(self.policy)})
        self.env.start()
        for key in (roles.ROLES_ENV,roles.CURRENT_ROLE_ENV,cond.ACCEPTED_PROFILE_ENV):os.environ.pop(key,None)

    def tearDown(self):
        self.env.stop();self.tmp.cleanup()

    def batch(self,T=1000):
        refs=[NS(media_type='image',image=NS(size=size),num_latent_frames=1,latent_height=0,latent_width=0)
              for size in ((1024,1824),(1312,1024))]
        sources=tuple(NS(source=r['path'],media_type='image') for r in self.selected['references'])
        batch=NS(references=refs,extra={'_h3_conditioning_sources':sources,
            '_h3_conditioning_source_sha':tuple(r['sha256'] for r in self.selected['references'])})
        cond.capture_prepared_geometry(batch)
        for ref in refs:
            ref.latent_height=ref.image.size[1]//16;ref.latent_width=ref.image.size[0]//16;ref.image=None
        batch.references=None
        from test_h3_ordered_guide_layout import batch as layout_fixture
        native=layout_fixture();batch.extra.update(native.extra)
        batch.raw_latent_shape=native.raw_latent_shape;batch.num_frames=124
        layout=batch.extra['minimax_h3_layout'];layout.text_indices=NS(numel=lambda:T)
        layout.sequence_length=T+layout.num_condition_video_rows+layout.num_condition_audio_rows+4144+414
        batch.prompt=roles.role_prompt('speech')
        return batch

    def test_candidate_prompts_survive_native_strip_and_strict_binding(self):
        # Exact native VideoGenerator strips request prompt before ForwardBatch.
        policy=self.root/'roles.json';current=self.root/'current-role.json'
        policy.write_text(json.dumps(roles.roles_manifest(visual_intents=True)))
        with patch.dict(os.environ,{roles.ROLES_ENV:str(policy),roles.CURRENT_ROLE_ENV:str(current)}):
            for role in ('speech','listening'):
                current.write_text(json.dumps(roles.role_record(role,visual_intents=True)))
                prompt=roles.conditioning_prompt(role)
                self.assertEqual(prompt,prompt.strip())
                profile=choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                b=self.batch();b.prompt=prompt.strip();b.width=256;b.height=448;b.fps=24
                with patch.object(cond,'read_accepted_profile',return_value=profile):
                    cond.validate_accepted_batch(b)
                    b.prompt+=' altered'
                    with self.assertRaisesRegex(ValueError,'actual role prompt mismatch'):cond.validate_accepted_batch(b)

    def test_pure_engineering_admission_bounds(self):
        self.assertEqual(cond.ordered_admission_bounds(1024),dict(max_reference_rows=8192,max_tokens=8703))
        self.assertEqual(cond.ordered_admission_bounds(512),dict(max_reference_rows=2048,max_tokens=2559))
        for edge in (True,1024.0,2048,None):
            with self.assertRaises(ValueError):cond.ordered_admission_bounds(edge)

    def test_actual_native_geometry_not_singleton_or_doubled(self):
        proof=cond.ordered_geometry_evidence(self.batch(T=3260),R=3136,T=3260)
        self.assertEqual([r['latent_rows'] for r in proof['images']],[1824,1312])
        self.assertEqual([r['prepared_size'] for r in proof['images']],[[1024,1824],[1312,1024]])
        self.assertEqual(proof['candidate_visual_acceptance'],'unreviewed')
        self.assertEqual(proof['max_reference_rows'],8192)
        self.assertEqual(proof['max_tokens'],8703)
        self.assertIn('engineering',proof['admission_kind'])

    def test_qwen_T_not_assumed_equal_or_above_VAE_R(self):
        proof=cond.ordered_geometry_evidence(self.batch(),R=3136,T=1000)
        self.assertEqual(proof['T'],1000)
        self.assertEqual(proof['R'],3136)
        self.assertEqual(proof['guide_segments_excluded'],2)
        self.assertEqual(proof['native_layout_reference_segments'][0],
                         dict(media_type='image',rows=1824,latent_shape=[1,114,64]))
        self.assertEqual([s['media_type'] for s in proof['native_layout_reference_segments']],
                         ['image','image','audio','video'])
        b=self.batch();b.extra['minimax_h3_layout'].reference_segments=(('image',3136,(1,114,64)),)
        with self.assertRaises(RuntimeError):cond.ordered_geometry_evidence(b,R=3136,T=1000)

    def test_write_evidence_candidate_role_T_below_VAE_R(self):
        import sys
        b=self.batch();prompt_sha=hashlib.sha256(b.prompt.encode()).hexdigest()
        profile=choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256=prompt_sha)
        from research.h3_personal.ordered_qwen_tokens import capture_qwen_presentation,PROOF_KEY
        qbatch=NS(references=[NS(media_type='image',has_audio=False)]*2,prompt=b.prompt,extra=b.extra)
        tags=[1]*2+[0]*402+[1]*2+[0]*502+[1]*92
        qwen=capture_qwen_presentation(NS(),qbatch,[[1,20,80],[1,20,100]],2,[400,500],list(range(1000)),tags)
        class PackedTags:
            def __getitem__(self,index):return NS(tolist=lambda:tags)
        b.extra['minimax_h3_layout'].token_tags=PackedTags()
        # Match native forward: raw tags have been consumed and removed.
        self.assertNotIn('minimax_h3_text_token_tags',b.extra)
        b.extra['_h3_conditioning_cache_hit']=True
        parameter=object()
        proof=dict(role='speech',profile_digest=profile.digest,prompt_sha256=prompt_sha,T=1000,
                   ordered_reference_sha256=self.context['identity_sha256'],qwen_tokens=qwen)
        stage=NS(_h3_conditioning_sizes={1024:None,512:None},
                 _h3_conditioning_processor_digest='c'*64,
                 _h3_role_condition_proofs={'key':proof},
                 _h3_conditioning_cache={'key':('key',None,NS(shape=(1000,)))})
        pipeline=NS(_stage_name_mapping={'conditioning_stage':stage},_denoise_stages_ready=True,
                    get_module=lambda name:NS(parameters=lambda:[parameter]))
        os.environ['H3_REF2VA_COVERAGE_DIR']=str(self.root/'coverage')
        with patch.dict(sys.modules,torch=NS(distributed=NS(get_rank=lambda:0),cuda=NS(memory_allocated=lambda:100,memory_reserved=lambda:200))), \
             patch.object(cond,'validate_accepted_batch'), \
             patch.object(cond,'read_accepted_profile',return_value=profile), \
             patch.object(cond,'_receipt_sizes',return_value=[[1024,1824],[1312,1024]]), \
             patch.object(roles,'current_role',return_value='speech'), \
             patch.object(roles,'conditioning_sha',return_value=prompt_sha), \
             patch('research.h3_personal.fast_ref2va.condition_key',return_value='key'):
            result=cond.write_evidence(pipeline,b)
            tags[0]=0
            with self.assertRaisesRegex(ValueError,'tags differ from Qwen proof'):
                cond.write_evidence(pipeline,b)
        self.assertEqual((result['R'],result['T']),(3136,1000))
        self.assertEqual(result['candidate_visual_acceptance'],'unreviewed')
        self.assertEqual(result['candidate_memory'],dict(measurement='worker-current-all-models-not-reference-delta',
                                                       allocated_bytes=100,reserved_bytes=200))
        self.assertEqual(result['ordered_geometry']['reference_rows_source'],'native-layout-image-segments')

    def test_missing_latents_order_size_and_token_bound_rejected(self):
        for change in ('latents','size','rows','tokens'):
            b=self.batch();r,t=3136,3200
            if change=='latents':b.extra['_h3_ordered_prepared_references'][1].latent_width=0
            elif change=='size':
                binding,_=b.extra['_h3_conditioning_prepared_geometry'];bad=(binding[0],binding[1],((1312,1024),(1024,1824)))
                b.extra['_h3_conditioning_prepared_geometry']=(bad,cond.digest(bad))
            elif change=='rows':r=3648
            else:t=8192+512
            with self.assertRaises(RuntimeError):cond.ordered_geometry_evidence(b,R=r,T=t)

    def test_file_change_context_hash_or_acceptance_rejected(self):
        for key,value in (('identity_sha256','a'*64),('visual_acceptance','accepted')):
            bad=dict(self.context);bad[key]=value;self.file.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):roles.read_ordered_context()
        self.file.write_text(json.dumps(self.context))
        Path(self.selected['references'][0]['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):roles.read_ordered_context()

    def test_both_prompts_and_return_keep_same_two_reference_guidance(self):
        active=dict(schema=roles.VISUAL_INTENT_SCHEMA,instruction='Show the surroundings')
        returning=dict(schema=roles.VISUAL_INTENT_SCHEMA,instruction='Return to a frontal selfie')
        for role in ('speech','listening'):
            first=roles.visual_prompt(role,active);second=roles.visual_prompt(role,returning)
            self.assertIn('Reference image 2',first);self.assertIn('Reference image 2',second)
            self.assertIn('Reference image 1',first)
        before=roles.read_ordered_context()['identity_sha256']
        roles.resolve_visual_intent(active,dict(schema=roles.VISUAL_INTENT_SCHEMA,instruction=None))
        self.assertEqual(before,roles.read_ordered_context()['identity_sha256'])

    def test_base_profile_sources_unchanged_candidate_role_digest_bound(self):
        profile=choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256='b'*64)
        first=roles.role_profile_digest(profile,'speech')
        self.assertEqual(profile.ordered_source_sha256,('a'*64,))
        replacement=copy.deepcopy(self.selected)
        from test_h3_ordered_references import png_chunk
        path=Path(replacement['references'][1]['path']);original=path.read_bytes()
        data=original[:-12]+png_chunk(b'tEXt',b'Comment\x00x')+original[-12:];path.write_bytes(data)
        replacement['references'][1]['sha256']=hashlib.sha256(data).hexdigest()
        self.file.write_text(json.dumps(roles.ordered_context_record(replacement)))
        self.assertNotEqual(first,roles.role_profile_digest(profile,'speech'))

    def test_default_legacy_prompt_and_digest_unchanged(self):
        os.environ.pop(roles.ORDERED_CONTEXT_ENV)
        profile=choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256=roles.prompt_sha('speech'))
        self.assertEqual(roles.role_prompt('speech'),roles.SPEECH_PROMPT)
        self.assertEqual(roles.role_profile_digest(profile,'speech'),profile.digest)
        self.assertIsNone(cond.ordered_geometry_evidence(NS(),R=1824,T=1867))

    def test_candidate_batch_sources_bound_separately_from_base(self):
        from dataclasses import replace
        manifest=self.root/'roles.json';manifest.write_text(json.dumps(roles.roles_manifest(visual_intents=True)))
        current=self.root/'role.json';current.write_text(json.dumps(roles.role_record('speech',visual_intents=True)))
        os.environ[roles.ROLES_ENV]=str(manifest);os.environ[roles.CURRENT_ROLE_ENV]=str(current)
        profile=choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256=roles.prompt_sha('speech'))
        accepted=self.root/'accepted.json';accepted.write_text(json.dumps(profile.wire_record()))
        os.environ[cond.ACCEPTED_PROFILE_ENV]=str(accepted)
        sources=tuple(NS(source=r['path'],media_type='image') for r in self.selected['references'])
        b=NS(prompt=roles.role_prompt('speech'),width=256,height=448,num_frames=124,fps=24,
             extra={'_h3_conditioning_sources':sources,
                    '_h3_conditioning_source_sha':tuple(r['sha256'] for r in self.selected['references'])})
        cond.validate_accepted_batch(b)
        self.assertEqual(cond.read_accepted_profile().ordered_source_sha256,('a'*64,))
        b.extra['_h3_conditioning_sources']=tuple(reversed(sources))
        b.extra['_h3_conditioning_source_sha']=tuple(reversed(b.extra['_h3_conditioning_source_sha']))
        with self.assertRaises(ValueError):cond.validate_accepted_batch(b)
        b.extra['_h3_conditioning_sources']=sources
        b.extra['_h3_conditioning_source_sha']=tuple(r['sha256'] for r in self.selected['references'])
        b.prompt=roles.SPEECH_PROMPT
        with self.assertRaises(ValueError):cond.validate_accepted_batch(b)

    def test_runtime_explicit_candidate_binding_excludes_old_visual_acceptance(self):
        profile=choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256='b'*64)
        runtime=LiveVoiceRuntime(model='m',model_manifest='mm',identity='i',identity_config='ic',smd='s',
            reference=self.root/'original-baseline.png',archive='a',output=self.root/'output',
            continuation=True,conversation_roles=True,visual_intents=True,quality_profile=profile,
            ordered_references=self.selected)
        binding=runtime.candidate_reference_binding
        self.assertEqual(binding['content_visual_acceptance'],'unreviewed')
        self.assertEqual(binding['ordered_content_sha256'],runtime.ordered_reference_sha256)
        self.assertEqual(binding['binding_sha256'],runtime.candidate_binding_sha256)
        self.assertNotIn('quality_review_sha256',binding['base_settings'])
        self.assertNotIn('ordered_source_sha256',binding['base_settings'])
        self.assertNotIn('prompt_sha256',binding['base_settings'])
        self.assertIs(runtime.quality_profile,profile)
        self.assertEqual(runtime.quality_profile.ordered_source_sha256,('a'*64,))

    def test_runtime_rejects_implicit_nonvisual_candidate(self):
        with self.assertRaises(ValueError):
            LiveVoiceRuntime(model='m',model_manifest='mm',identity='i',identity_config='ic',smd='s',
                reference='r',archive='a',output='o',ordered_references=self.selected)

    def test_context_policy_cache_identity_changes_without_touching_base(self):
        key=cond.policy_key()
        os.environ.pop(roles.ORDERED_CONTEXT_ENV)
        self.assertNotEqual(key,cond.policy_key())

if __name__=='__main__':unittest.main()
