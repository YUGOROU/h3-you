import ast
import json
from pathlib import Path
import tempfile
import unittest

from research.launch_h3_conversation_scenario import closure,STREAMING_REQUIRED,ROOT,CONVERSATION,SPEECH
from research.hf_h3_streaming_trial import SECTION,validate_spec,read_evidence


class StreamingLaunchTests(unittest.TestCase):
    def test_streaming_dispatch_precedes_compatibility_section(self):
        tree=ast.parse((ROOT/'research/bootstrap_h3_fast_ref2va.py').read_text())
        streaming=[];old=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.If):
                test=ast.unparse(node.test)
                if test=="spec.get('streaming_trial') is not None":streaming.append(node.lineno)
                if test=="spec.get('conversation_scenario') is not None":old.append(node.lineno)
        self.assertEqual(len(streaming),1);self.assertEqual(len(old),1)
        self.assertLess(streaming[0],old[0])

    def test_transitive_runtime_closure_includes_all_streaming_code(self):
        with tempfile.TemporaryDirectory() as d:
            closure(d,['research.hf_h3_streaming_trial'])
            for name in STREAMING_REQUIRED:
                self.assertTrue((Path(d)/name).is_file(),name)
                self.assertEqual((Path(d)/name).read_bytes(),(ROOT/name).read_bytes())

    def test_explicit_spec_no_quality_or_producer_fallback(self):
        spec=dict(conversation_scenario=dict(CONVERSATION),streaming_trial=dict(SECTION),locked_speech=dict(SPEECH),
                  attention_mode='fa4-packed',layout='portrait',runtime_policy='resident-visual-cached')
        validate_spec(spec)
        for key,value in (('profile','reference512'),('expected_frames',532),('producer','live-tts'),('paced',1)):
            broken=dict(spec,streaming_trial=dict(SECTION,**{key:value}))
            with self.assertRaises(ValueError):validate_spec(broken)

    def test_both_rank_evidence_is_bound_to_current_prefix(self):
        import test_h3_streaming_windows as fixtures
        from research.h3_personal.streaming_admission import StreamingAdmission,NativeCommitEvidence
        from research.h3_personal.streaming_windows import ResamplingContext
        from research.h3_personal.streaming_trial import manifest_for,write_wav
        c=StreamingAdmission('c',context=ResamplingContext('same-rate-32k'))
        idle=c.reserve(c.inspect());c.commit(idle,NativeCommitEvidence(124,None,'a'*64));c.ack_decoded(idle,119)
        source=fixtures.source();fixtures.append_samples(source,136000);c.attach(source)
        ticket=c.reserve(c.inspect())
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);wav=d/'prefix.wav';write_wav(wav,b'\x01\x00'*136000,32000)
            m=manifest_for(ticket,conversation_id='c',profile_sha256='a'*64,wav=wav)
            for rank in (0,1):
                row=dict(schema=m['streaming_audio']['schema'],streaming=m['streaming_audio'],
                    native_window_id=m['session_id'],committed_frame=226,forwards=4,steps_held=4,
                    waveform_samples=165600,window_context_encodes=1,received_source_samples=136000,
                    canonical_sha256='b'*64,guide_digests=['c'*64]*3,
                    prefix_pcm_sha256=ticket.decision.audio.prefix_sha256,source_eos=False,padded_samples=0)
                (d/f'streaming-rank-{rank}-window-1.json').write_text(json.dumps(row))
                (d/f'h3-continuation-decode-rank-{rank}-window-0.json').write_text(json.dumps(dict(
                    session_id=m['session_id'],worker_rank=rank,committed_frames=226,emitted_frames=221,final=False)))
                (d/f'conditioning-experiment-{m["session_id"]}-window-0-rank-{rank}.json').write_text(json.dumps(dict(
                    current_role='speech',streaming_sequence=1,parameter_identity_preserved=True,cache_hit=True)))
            self.assertTrue(read_evidence(d,m)['evidence']['both_ranks'])
            path=d/'streaming-rank-1-window-1.json';row=json.loads(path.read_text());row['source_eos']=True
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):read_evidence(d,m)
