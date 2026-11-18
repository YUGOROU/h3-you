"""Offline interruption boundaries using real owner progression and fake CPU tensors."""
import ast
import copy
import inspect
import unittest

import test_h3_streaming_native as native_tests
from research.h3_personal import streaming_native_stages as stages
from research.h3_personal.fast_continuation import SamplerContract
from research.h3_personal.continuation_plan import video_t
from test_h3_fast_continuation import Tensor


class SpeechInterruptTests(unittest.TestCase):
    owner = native_tests.NativeOwnerTests.owner
    start = native_tests.NativeOwnerTests.start
    commit = native_tests.NativeOwnerTests.commit

    def ready(self, role='speech', final=False):
        o=self.owner()
        self.commit(o,self.start(o,0,'old',0,role,final))
        return o

    def test_transition_preserves_state_without_fabricating_eos(self):
        o=self.ready()
        before=dict(o.__dict__)
        session=o.source_session
        o.interrupt_source('old',124)
        for key,value in before.items():
            if key not in ('_logical_interrupted','_interrupted_sources'):
                self.assertIs(o.__dict__[key],value,key)
        self.assertIs(o.source_session,session)
        self.assertFalse(o._logical_final)
        self.assertTrue(o._logical_interrupted)
        with self.assertRaises(ValueError):o.interrupt_source('old',124)
        with self.assertRaises(ValueError):self.start(o,1,'old',1,'speech',False)
        with self.assertRaises(ValueError):self.start(o,1,'new',1,'speech',False)
        t=self.start(o,1,'new',0,'speech',False)
        self.assertEqual((o.committedF,o.emittedF),(124,119))
        self.assertEqual(t.model.window.end,226)
        self.assertEqual(o.ops.digest(t.model.guide_video),before['_tail_digest'])
        self.assertFalse(o._logical_interrupted)
        self.commit(o,t)
        o.interrupt_source('new',226)
        with self.assertRaises(ValueError):self.start(o,2,'old',0,'speech',False)
        t=self.start(o,2,'idle',0,'listening',True)
        self.commit(o,t,final=True)
        self.assertEqual((o.committedF,o.emittedF),(328,328))

    def test_unsafe_boundaries_and_wrong_bindings(self):
        o=self.owner()
        with self.assertRaises(ValueError):o.interrupt_source('old',0)
        t=self.start(o,0,'old',0,'speech',False)
        with self.assertRaises(ValueError):o.interrupt_source('old',0)
        o.commit_window(t,clean_video=Tensor((video_t(124),96)),
            clean_audio=o.ops.copy(t.model.locked_audio),observed_sampler=SamplerContract())
        with self.assertRaises(ValueError):o.interrupt_source('old',124)
        o.ack_decoded_count(119)
        for source,frame in [('wrong',124),('old',123),('old',True),('old',124.0)]:
            with self.subTest(source=source,frame=frame),self.assertRaises(ValueError):
                o.interrupt_source(source,frame)
        self.assertFalse(o._logical_interrupted)
        o.interrupt_source('old',124)

    def test_non_speech_eos_fault_and_terminal_rejected(self):
        for role,final in [('listening',False),('speech',True)]:
            o=self.ready(role,final)
            with self.assertRaises(ValueError):o.interrupt_source('old',124)
        o=self.ready();o.source_session._fault=True
        with self.assertRaises(ValueError):o.interrupt_source('old',124)
        o=self.ready();o.abandon()
        with self.assertRaises(RuntimeError):o.interrupt_source('old',124)


class MarkerTests(unittest.TestCase):
    def manifest(self):
        return dict(source_index=0,index=0,total_frames=124,expected_windows=1,final=True,
            candidate=dict(audio_context='window-context-canonical-v1'),
            streaming_audio=dict(schema=stages.SCHEMA,id='c',profile_sha256='a'*64,
                sequence=1,logical_source_id='new',logical_index=0,role='listening',
                source_final=False,conversation_final=False,committed_before=124,
                max_windows=8,audio_offer=None))

    def marker(self):
        return dict(schema='h3-speech-interruption-v1',source_id='old',committed_frame=124)

    def test_old_manifest_and_explicit_marker(self):
        m=self.manifest()
        self.assertIs(stages.validate_manifest(m),m['streaming_audio'])
        m['streaming_audio']['previous_speech_interrupted']=self.marker()
        self.assertIs(stages.validate_manifest(m),m['streaming_audio'])

    def test_marker_shape_types_and_transition_validation(self):
        invalid=[None,False,{},dict(self.marker(),extra=1),
            dict(self.marker(),schema='other'),dict(self.marker(),source_id=''),
            dict(self.marker(),source_id=1),dict(self.marker(),source_id='new'),
            dict(self.marker(),committed_frame=True),dict(self.marker(),committed_frame=124.0),
            dict(self.marker(),committed_frame=123)]
        for marker in invalid:
            m=self.manifest();m['streaming_audio']['previous_speech_interrupted']=marker
            with self.subTest(marker=marker),self.assertRaises(ValueError):stages.validate_manifest(m)
        for updates in [dict(sequence=0,committed_before=0),dict(logical_index=1)]:
            m=copy.deepcopy(self.manifest());c=m['streaming_audio'];c.update(updates)
            c['previous_speech_interrupted']=dict(self.marker(),committed_frame=c['committed_before'])
            with self.assertRaises(ValueError):stages.validate_manifest(m)

    def test_marker_is_applied_inside_common_allocate_before_start(self):
        tree=ast.parse(inspect.getsource(stages.append_conditions))
        allocate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='allocate')
        calls=[n for n in ast.walk(allocate) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr in ('interrupt_source','start_window')]
        self.assertEqual(sorted((n.lineno,n.func.attr) for n in calls)[0][1],'interrupt_source')
        self.assertIn('common(device,allocate)',inspect.getsource(stages.append_conditions))


if __name__=='__main__':unittest.main()
