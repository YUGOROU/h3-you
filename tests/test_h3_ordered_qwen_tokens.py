"""Synthetic processor/tokenizer/memo tests, not actual Qwen/Torch execution."""
import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from research.h3_personal import ordered_qwen_tokens as q
from research.h3_personal.ordered_references import content_identity,content_sha256


def context():
    config=dict(schema='h3-ordered-references-v1',references=[
        dict(role='person',path='/private/synthetic-person.png',sha256='a'*64,width=9,height=16),
        dict(role='surroundings',path='/private/synthetic-scene.png',sha256='b'*64,width=14,height=11)])
    return dict(ordered_references=config,identity=content_identity(config),identity_sha256=content_sha256(config))


def batch(prompt='return front'):
    return NS(prompt=prompt,references=[NS(media_type='image',has_audio=False),NS(media_type='image',has_audio=False)],extra={})


def presentation(counts=(8,12),overhead=9):
    lengths=[2,counts[0]+2,2,counts[1]+2,overhead-8]
    tags=[tag for tag,n in zip((1,0,1,0,1),lengths) for _ in range(n)]
    return list(range(len(tags))),tags


class Memo:
    def __init__(self):self.values={}
    def get_or_compute(self,key,compute):
        if key not in self.values:self.values[key]=compute()
        return self.values[key]


class Tests(unittest.TestCase):
    def setUp(self):
        self.c=context()
        p=patch('research.h3_personal.conversation_conditioning.read_ordered_context',return_value=self.c)
        p.start();self.addCleanup(p.stop)
        p=patch('research.h3_personal.conditioning_experiment.reference_short_edge',return_value=1024)
        p.start();self.addCleanup(p.stop)
        self.stage=NS(_ref2va_memo=Memo());self.b=batch()

    def capture(self,counts=(8,12),overhead=9):
        ids,tags=presentation(counts,overhead)
        return q.capture_qwen_presentation(self.stage,self.b,[[1,4,8],[1,6,8]],2,list(counts),ids,tags)

    def test_actual_grids_runs_overhead_and_detached_proof(self):
        proof=self.capture()
        self.assertEqual(proof['image_token_counts'],[8,12])
        self.assertEqual(proof['total_tokens'],29)
        self.assertEqual(proof['text_overhead'],9)
        expected=hashlib.sha256(self.b.prompt.encode()).hexdigest()
        self.assertEqual(q.validate_proof(proof,self.c,1024,expected,29),proof)
        proof['segments'][0]['tokens']=100
        self.assertEqual(self.b.extra[q.PROOF_KEY]['segments'][0]['tokens'],2)

    def test_grid_count_merge_tags_and_total_mismatch(self):
        ids,tags=presentation()
        bad=( ([[1,4,7],[1,6,8]],2,[8,12],ids,tags),
              ([[1,4,8],[1,6,8]],2,[9,12],ids,tags),
              ([[1,4,8],[1,6,8]],True,[8,12],ids,tags),
              ([[1,4,8],[1,6,8]],2,[8,12],ids,tags[:-1]),
              ([[1,4,8],[1,6,8]],2,[8,12],ids,[0]*len(tags)))
        for args in bad:
            with self.assertRaises(ValueError):q.capture_qwen_presentation(self.stage,self.b,*args)
        proof=self.capture();sha=hashlib.sha256(self.b.prompt.encode()).hexdigest()
        with self.assertRaises(ValueError):q.validate_proof(proof,self.c,1024,sha,30)
        with self.assertRaises(ValueError):q.validate_proof(proof,self.c,1024,'f'*64)
        wrong=copy.deepcopy(proof);wrong['policy']['max_text_overhead']=1000
        with self.assertRaises(ValueError):q.validate_proof(wrong,self.c,1024,sha)

    def test_overhead_at_limit_and_overflow_before_encoder(self):
        self.assertEqual(self.capture(overhead=511)['text_overhead'],511)
        calls=[]
        def encode():calls.append('encoder')
        with self.assertRaises(ValueError):
            calls.append('processor');ids,tags=presentation(overhead=512);calls.append('builder')
            q.capture_qwen_presentation(self.stage,self.b,[[1,4,8],[1,6,8]],2,[8,12],ids,tags)
            encode()
        self.assertEqual(calls,['processor','builder'])

    def test_total_at_limit_and_overflow(self):
        # Operational cap8703: grids represent exactly8694 image tokens.
        ids,tags=presentation((4347,4347),9)
        proof=q.capture_qwen_presentation(self.stage,self.b,[[1,4,4347],[1,4,4347]],2,[4347,4347],ids,tags)
        self.assertEqual(proof['total_tokens'],8703)
        ids,tags=presentation((4347,4348),9)
        with self.assertRaises(ValueError):q.capture_qwen_presentation(self.stage,self.b,[[1,4,4347],[1,4,4348]],2,[4347,4348],ids,tags)

    def test_reference_count_audio_and_changed_role_prompt_reject(self):
        for refs in ([NS(media_type='image',has_audio=False)],
                     [NS(media_type='image',has_audio=True),NS(media_type='image',has_audio=False)]):
            self.b.references=refs
            with self.assertRaises(ValueError):self.capture()
        self.b=batch();proof=self.capture()
        with self.assertRaises(ValueError):q.validate_proof(proof,self.c,1024,hashlib.sha256(b'listening').hexdigest())

    def test_native_memo_hit_restores_exact_proof_no_second_encode(self):
        calls=[]
        def compute():calls.append('encode');self.capture();return ('embeddings','tags')
        key=('device','prompt','two-images')
        first=q.memo_presentation(self.stage,self.b,key,compute)
        proof=copy.deepcopy(self.b.extra[q.PROOF_KEY]);self.b.extra.clear()
        self.assertEqual(q.memo_presentation(self.stage,self.b,key,compute),first)
        self.assertEqual(self.b.extra[q.PROOF_KEY],proof);self.assertEqual(calls,['encode'])
        self.b.prompt='changed prompt'
        with self.assertRaises(ValueError):q.memo_presentation(self.stage,self.b,key,compute)

    def test_memo_missing_proof_and_bounded_growth_fail_closed(self):
        self.stage._ref2va_memo.values['missing']='cached'
        with self.assertRaises(ValueError):q.memo_presentation(self.stage,self.b,'missing',lambda:None)
        def compute():self.capture();return 'encoded'
        for i in range(8):q.memo_presentation(self.stage,self.b,i,compute)
        with self.assertRaises(ValueError):q.memo_presentation(self.stage,self.b,8,compute)
        self.assertEqual(len(self.stage._h3_ordered_qwen_memo_proofs),8)

    def test_role_bank_restore_bound_to_prompt_total_context_and_fresh_copy(self):
        proof=self.capture();self.stage._h3_role_condition_proofs={'role-key':{'qwen_tokens':proof}}
        self.b.extra.clear()
        self.b.extra['minimax_h3_text_token_tags']=presentation()[1]
        restored=q.restore_role_proof(self.stage,self.b,'role-key',29)
        self.assertEqual(restored,proof)
        self.assertEqual(self.b.extra[q.PROOF_KEY],proof)
        restored['segments'][0]['tokens']=90
        self.assertEqual(proof['segments'][0]['tokens'],2)
        self.assertEqual(self.b.extra[q.PROOF_KEY]['segments'][0]['tokens'],2)
        with self.assertRaises(ValueError):q.restore_role_proof(self.stage,self.b,'role-key',30)
        self.assertNotIn(q.PROOF_KEY,self.b.extra)
        with self.assertRaises(ValueError):q.restore_role_proof(self.stage,self.b,'missing',29)
        self.b.prompt='changed-role'
        with self.assertRaises(ValueError):q.restore_role_proof(self.stage,self.b,'role-key',29)
        self.b.prompt='return front'
        altered=presentation()[1];altered[0]=0
        self.b.extra['minimax_h3_text_token_tags']=NS(tolist=lambda:altered)
        with self.assertRaises(ValueError):q.restore_role_proof(self.stage,self.b,'role-key',29)
        self.b.extra['minimax_h3_text_token_tags']=NS(tolist=lambda:presentation()[1])
        self.assertEqual(q.restore_role_proof(self.stage,self.b,'role-key',29),proof)
        self.stage._h3_role_condition_proofs['role-key'].pop('qwen_tokens')
        with self.assertRaises(ValueError):q.restore_role_proof(self.stage,self.b,'role-key',29)

    def test_legacy_no_context_unchanged(self):
        with patch('research.h3_personal.conversation_conditioning.read_ordered_context',return_value=None):
            self.assertIsNone(q.capture_qwen_presentation(self.stage,self.b,None,None,None,None,None))
            self.assertEqual(q.memo_presentation(self.stage,self.b,'legacy',lambda:17),17)
            self.assertIsNone(q.restore_role_proof(self.stage,self.b,'legacy',None))
        self.assertNotIn(q.PROOF_KEY,self.b.extra)

    @unittest.skipUnless(os.environ.get('H3_PINNED_QWEN_SOURCE'),'existing exact source AST opt-in')
    def test_pinned_native_builder_no_double_tokenization_call_order(self):
        data=Path(os.environ['H3_PINNED_QWEN_SOURCE']).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'eaae1a9116367ac06eb7548dfa7d269f8c014b571e8a2f486d257684c894f0d8')
        raw=data.decode();tree=ast.parse(raw)
        methods=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('_token_ids','build_ref2va_presentation')]
        constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                   and isinstance(node.value,ast.Constant)}
        constants.update(MINIMAX_H3_TEXT_TAG=1,MINIMAX_H3_VIDEO_TAG=0,torch=NS(Tensor=type('FakeTensor',(),{})),
            MINIMAX_H3_IMAGE_PAD_TOKEN='image-pad',MINIMAX_H3_VIDEO_PAD_TOKEN='video-pad',
            MINIMAX_H3_VISION_END_TOKEN='vision-end',MINIMAX_H3_VISION_START_TOKEN='vision-start')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*methods],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),'<pinned native presentation>','exec'),constants)
        calls=[]
        class Tokenizer:
            def __call__(self,text,**kw):calls.append(('tokenize',text));return NS(input_ids=[ord(c) for c in text])
            def convert_tokens_to_ids(self,name):return 7
        calls.append('processor')
        ids,tags=constants['build_ref2va_presentation'](Tokenizer(),self.b.prompt,self.b.references,[8,12],[])
        calls.append('builder')
        q.capture_qwen_presentation(self.stage,self.b,[[1,4,8],[1,6,8]],2,[8,12],ids,tags)
        calls.append('encoder')
        self.assertEqual([x for x in calls if isinstance(x,tuple)],[('tokenize','<Picture 1>: '),('tokenize','<Picture 2>: '),('tokenize',self.b.prompt)])
        self.assertLess(calls.index('builder'),calls.index('encoder'))
        calls.clear();self.b.prompt='p'*512
        with self.assertRaises(ValueError):
            calls.append('processor')
            ids,tags=constants['build_ref2va_presentation'](Tokenizer(),self.b.prompt,self.b.references,[8,12],[])
            calls.append('builder')
            q.capture_qwen_presentation(self.stage,self.b,[[1,4,8],[1,6,8]],2,[8,12],ids,tags)
            calls.append('encoder')
        self.assertNotIn('encoder',calls)
        self.assertEqual(sum(isinstance(x,tuple) for x in calls),3)


if __name__=='__main__':unittest.main()
