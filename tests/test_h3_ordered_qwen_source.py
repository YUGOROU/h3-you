"""Execute exact patched native Qwen control flow with fake CPU dependencies.

The real pinned installer and native presentation/tokenization code run here.
Processor, tokenizer, pixels and encoder are fakes: no actual Qwen processing,
neural execution, pixel decoding, memory fitness or GPU acceptance is claimed.
Set H3_FASTVIDEO_ARCHIVE to an existing archive; no download/install occurs.
"""
import ast
import hashlib
import math
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research.h3_personal.fast_ref2va import install
from research.h3_personal import ordered_qwen_tokens as admission
from research.h3_personal.ordered_references import content_identity, content_sha256
from test_h3_fast_continuation_stages import pinned_source


class Grid(list):
    def prod(self):return NS(item=lambda:math.prod(self))


class Grids(list):
    def tolist(self):return [list(row) for row in self]


@unittest.skipUnless(os.environ.get('H3_FASTVIDEO_ARCHIVE'),'existing pinned archive opt-in')
class NativeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Install into the fixture's private source copy, never the archive.
        with pinned_source() as source:
            install(source)
            base=source/'fastvideo/pipelines/basic/minimax_h3'
            tree=ast.parse((base/'stages/minimax_h3_conditioning.py').read_text())
            selected=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                      and node.name in ('_token_ids','build_ref2va_presentation')]
            native=next(node for node in tree.body if isinstance(node,ast.ClassDef)
                        and node.name=='MiniMaxH3ConditioningStage')
            selected.append(next(node for node in native.body if isinstance(node,ast.FunctionDef)
                                 and node.name=='_present_ref2va'))
            constants={}
            needed={'MINIMAX_H3_IMAGE_PAD_TOKEN','MINIMAX_H3_TEXT_TAG','MINIMAX_H3_VIDEO_PAD_TOKEN',
                    'MINIMAX_H3_VIDEO_TAG','MINIMAX_H3_VISION_END_TOKEN','MINIMAX_H3_VISION_START_TOKEN'}
            for node in ast.parse((base/'packing.py').read_text()).body:
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id in needed:
                            constants[target.id]=ast.literal_eval(node.value)
            if set(constants)!=needed:raise AssertionError('exact native presentation constants unavailable')
            module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                                   *selected],type_ignores=[])
            namespace=dict(constants,torch=NS(Tensor=type('UnusedFakeTensor',(),{})))
            exec(compile(ast.fix_missing_locations(module),'<exact patched native Qwen source>','exec'),namespace)
            cls.native=namespace['_present_ref2va']
            cls.builder=namespace['build_ref2va_presentation']
            cls.namespace=namespace

    def setUp(self):
        config=dict(schema='h3-ordered-references-v1',references=[
            dict(role='person',path='/private/synthetic-person.png',sha256='a'*64,width=9,height=16),
            dict(role='surroundings',path='/private/synthetic-surroundings.png',sha256='b'*64,width=14,height=11)])
        self.context=dict(ordered_references=config,identity=content_identity(config),identity_sha256=content_sha256(config))
        self.events=[];self.tokenized=[];self.processor_calls=[];self.encoder_calls=[];self.hooks=[]

    def run_native(self,grids=((1,4,6),(1,6,8)),prompt='Actual prompt'):
        events=self.events
        class Tokenizer:
            def __call__(_,text,add_special_tokens):
                self.assertFalse(add_special_tokens)
                self.tokenized.append(text);events.append('tokenize')
                return dict(input_ids=[ord(char)+100 for char in text])
            def convert_tokens_to_ids(_,token):
                return {'<|vision_start|>':1,'<|image_pad|>':2,'<|video_pad|>':3,'<|vision_end|>':4}[token]
        images=[object(),object()];pixel_values=object()
        def processor(*,images,return_tensors):
            self.processor_calls.append((images,return_tensors));events.append('processor')
            return dict(pixel_values=pixel_values,image_grid_thw=Grids(Grid(g) for g in grids))
        processor.merge_size=2
        def encoder(*args,**kwargs):
            events.append('encoder');self.encoder_calls.append((args,kwargs))
            self.assertIn(admission.PROOF_KEY,batch.extra)
            return 'fake-embeddings','fake-tags'
        stage=NS(processor=NS(image_processor=processor),tokenizer=Tokenizer(),_encode_tokens=encoder)
        batch=NS(prompt=prompt,extra={},references=[NS(media_type='image',image=image,has_audio=False) for image in images])
        native_builder=type(self).builder
        def counted_builder(*args,**kwargs):
            events.append('presentation')
            return native_builder(*args,**kwargs)
        real_capture=admission.capture_qwen_presentation
        def capture(*args):
            events.append('admission');self.hooks.append(args)
            return real_capture(*args)
        with patch('research.h3_personal.conversation_conditioning.read_ordered_context',return_value=self.context), \
             patch('research.h3_personal.conditioning_experiment.reference_short_edge',return_value=1024), \
             patch.object(admission,'capture_qwen_presentation',side_effect=capture), \
             patch.dict(type(self).namespace,build_ref2va_presentation=counted_builder):
            result=type(self).native(stage,batch,batch.references,'fake-cpu-device')
        return result,batch,images,pixel_values

    def test_one_actual_presentation_and_native_tokenization_before_encoder(self):
        result,batch,images,pixels=self.run_native()
        self.assertEqual(result,('fake-embeddings','fake-tags'))
        self.assertEqual(self.events,['processor','presentation','tokenize','tokenize','tokenize','admission','encoder'])
        self.assertEqual(self.tokenized,['<Picture 1>: ','<Picture 2>: ','Actual prompt'])
        self.assertEqual(len(self.processor_calls),1);self.assertEqual(len(self.encoder_calls),1)
        self.assertEqual(self.processor_calls[0],(images,'pt'))
        args,kwargs=self.encoder_calls[0];hook=self.hooks[0]
        # SAME actual arrays generated by native builder, not a retokenized copy.
        self.assertIs(hook[5],args[0]);self.assertIs(hook[6],args[1])
        self.assertIs(kwargs['pixel_values'],pixels)
        proof=batch.extra[admission.PROOF_KEY]
        self.assertEqual(proof['image_grid_thw'],[[1,4,6],[1,6,8]])
        self.assertEqual(proof['image_token_counts'],[6,12])
        self.assertEqual([s['tokens'] for s in proof['segments']],
                         [len('<Picture 1>: '),8,len('<Picture 2>: '),14,len('Actual prompt')])
        overhead=sum(map(len,self.tokenized))+4
        self.assertEqual(proof['text_overhead'],overhead)
        self.assertEqual(proof['total_tokens'],18+overhead)
        self.assertEqual(proof['prompt_sha256'],hashlib.sha256(b'Actual prompt').hexdigest())

    def test_actual_grid_token_overflow_rejects_before_encoder(self):
        with self.assertRaises(ValueError):self.run_native(grids=((1,2,40000),(1,6,8)))
        self.assertEqual(self.events.count('processor'),1)
        self.assertEqual(self.events.count('presentation'),1)
        self.assertEqual(self.events[-1],'admission')
        self.assertEqual(self.encoder_calls,[])
        self.assertEqual(len(self.tokenized),3)

    def test_actual_label_prompt_overhead_overflow_rejects_before_encoder(self):
        with self.assertRaises(ValueError):self.run_native(prompt='x'*512)
        self.assertEqual(self.encoder_calls,[])
        self.assertEqual(self.events[-1],'admission')
        self.assertEqual(len(self.tokenized),3)

    def test_nondivisible_actual_processor_grid_rejects_before_encoder(self):
        # Upstream integer division alone rounds down; admission must reject it.
        with self.assertRaises(ValueError):self.run_native(grids=((1,3,3),(1,6,8)))
        self.assertEqual(self.encoder_calls,[])
        self.assertEqual(self.events[-1],'admission')


if __name__=='__main__':unittest.main()
