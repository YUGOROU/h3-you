"""Real cache-control functions with fake encodings; no model/processor claim."""
import json
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import test_h3_visual_intent_wiring as legacy
from research.h3_personal import conditioning_experiment as ce,conversation_conditioning as cc
from research.h3_personal.ordered_references import content_identity,content_sha256
from research.h3_personal.ordered_qwen_tokens import capture_qwen_presentation,restore_role_proof


class Tensor:
    def __init__(self,values):self.values=list(values);self.shape=(len(values),)
    def detach(self):return self
    def cpu(self):return self
    def clone(self):return Tensor(self.values)
    def tolist(self):return list(self.values)


class Tests(unittest.TestCase):
    batch=legacy.CacheTests.batch
    def setUp(self):
        legacy.CacheTests.setUp(self)
        config=dict(schema='h3-ordered-references-v1',references=[dict(role=r,path='/private/'+r+'.png',
            sha256=c*64,width=w,height=h) for r,c,w,h in [('person','a',9,16),('surroundings','b',14,11)]])
        self.context=dict(ordered_references=config,identity=content_identity(config),identity_sha256=content_sha256(config))
        for name,value in (('read_ordered_context',lambda **kw:self.context),):
            p=patch.object(cc,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(ce,'read_policy',return_value=dict(active_short_edge=1024,short_edges=[1024,512]));p.start();self.addCleanup(p.stop)
        p=patch('research.h3_personal.fast_ref2va.condition_key',lambda b:(b.prompt,ce.reference_short_edge()));p.start();self.addCleanup(p.stop)
        self.r.roles_file.write_text(json.dumps(cc.roles_manifest(visual_intents=True)))
        self.stage._h3_ordered_actual_geometry=dict(identity_sha256=self.context['identity_sha256'],R=3136)
        def forward(batch,args):
            self.calls+=1
            batch.references=[NS(media_type='image',has_audio=False,image=NS(size=(1024,1824))),
                              NS(media_type='image',has_audio=False,image=NS(size=(1312,1024)))]
            tags=[1]*2+[0]*6+[1]*2+[0]*10+[1]*3
            capture_qwen_presentation(self.stage,batch,[[1,4,4],[1,4,8]],2,[4,8],list(range(23)),tags)
            batch.prompt_embeds=[Tensor(range(23))];batch.extra['minimax_h3_text_token_tags']=Tensor(tags)
            return batch
        self.stage.forward=forward

    def test_cold_bank_and_exact_tag_restore(self):
        batch=self.batch('Prepare context')
        batch.extra['_h3_conditioning_sources']=[NS(source='person',media_type='image'),NS(source='scene',media_type='image')]
        ce._store_role_conditions(self.stage,batch,None)
        self.assertEqual(len(self.stage._h3_role_condition_proofs),4)
        key=next(k for k in self.stage._h3_role_condition_proofs if k[1]==1024)
        proof=self.stage._h3_role_condition_proofs[key]['qwen_tokens']
        batch.prompt=key[0];batch.extra['minimax_h3_text_token_tags']=self.stage._h3_conditioning_cache[key][2].clone()
        self.assertEqual(restore_role_proof(self.stage,batch,key,23),proof)
        batch.extra['minimax_h3_text_token_tags'].values[0]=0
        with self.assertRaises(ValueError):restore_role_proof(self.stage,batch,key,23)

    def test_refresh_proof_preserved_and_hit_does_not_encode(self):
        batch=self.batch('Show the surroundings')
        ce.refresh_visual_condition(self.pipeline,batch,None)
        self.assertEqual(self.calls,1)
        key=(batch.prompt,1024);proof=self.stage._h3_role_condition_proofs[key]['qwen_tokens']
        self.assertEqual(proof['image_token_counts'],[4,8])
        ce.refresh_visual_condition(self.pipeline,batch,None)
        self.assertEqual(self.calls,1)
        returning=self.batch('Return to the same frontal selfie',epoch=2)
        ce.refresh_visual_condition(self.pipeline,returning,None)
        self.assertEqual(self.calls,2)
        self.assertNotEqual(proof['prompt_sha256'],self.stage._h3_role_condition_proofs[(returning.prompt,1024)]['qwen_tokens']['prompt_sha256'])


if __name__=='__main__':unittest.main()
