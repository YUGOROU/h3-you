import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from test_h3_fast_continuation import Tensor,Ops
from research.h3_personal.streaming_native_owner import StreamingWindowOwner
from research.h3_personal.conversation_continuation import ConversationOwner
from research.h3_personal.fast_continuation import SamplerContract
from research.h3_personal.continuation_plan import video_t
from research.h3_personal.streaming_windows import ResamplingContext,NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256
from research.h3_personal.streaming_trial import run_trial
from research.h3_personal.streaming_native_stages import validate_manifest


class CatOps(Ops):
    def cat_time(self,a,b):
        values=[]
        for c in range(2):
            values.extend(a.values[c*a.shape[1]*32:(c+1)*a.shape[1]*32])
            values.extend(b.values[c*b.shape[1]*32:(c+1)*b.shape[1]*32])
        return Tensor((2,a.shape[1]+b.shape[1],32),values)


class NativeOwnerTests(unittest.TestCase):
    def owner(self,limit=8):
        return StreamingWindowOwner('c',profile='p',height=2,width=2,
            max_windows=limit,model_healthy=True,enabled=True,ops=CatOps())

    def start(self,o,seq,logical,index,role,final):
        audio=Tensor((2,207,32),[seq+1]*(2*207*32))
        o.start_window(str(seq),canonical_audio=audio,logical_source_id=logical,
            logical_index=index,role=role,source_final=final,profile='p',height=2,width=2)
        return o.begin_window()

    def commit(self,o,t,final=False):
        result=o.commit_window(t,clean_video=Tensor((video_t(124),96)),
            clean_audio=o.ops.copy(t.model.locked_audio),observed_sampler=SamplerContract(),
            conversation_final=final)
        o.ack_decoded_count(o.committedF if final else o.committedF-5,final_flush=final)
        return result

    def test_six_windows_one_owner_prior_canonical_overlap_and_role_reset(self):
        o=self.owner()
        program=(('idle',0,'listening',False),('idle',1,'listening',False),
                 ('speech',0,'speech',False),('speech',1,'speech',False),
                 ('speech',2,'speech',True),('return',0,'listening',True))
        for seq,(source,index,role,last) in enumerate(program):
            token=self.start(o,seq,source,index,role,last)
            self.assertEqual(token.local.window.start,0) # new immutable Session
            self.assertEqual(token.model.window.end,124+seq*102)
            self.assertEqual(token.model.locked_audio.values[0],seq if index else seq+1)
            self.assertEqual(token.model.locked_audio.values[37*32],seq+1)
            self.commit(o,token,final=seq==5)
        self.assertEqual((o.committedF,o.emittedF),(634,634))
        self.assertEqual(o._sequence,6)

    def test_early_source_switch_and_bad_index_rejected(self):
        o=self.owner();self.commit(o,self.start(o,0,'speech',0,'speech',False))
        with self.assertRaises(ValueError):self.start(o,1,'return',0,'listening',True)
        with self.assertRaises(ValueError):self.start(o,1,'speech',2,'speech',False)
        with self.assertRaises(RuntimeError):o.start_source('fake')
        self.assertEqual(o.committedF,124)

    def test_defaults_limit_preserved_and_stream_limit_finite(self):
        with self.assertRaises(ValueError):ConversationOwner('c',profile='p',height=2,width=2,
            max_sessions=5,enabled=True,model_healthy=True,ops=CatOps())
        o=self.owner(1);self.commit(o,self.start(o,0,'idle',0,'listening',False))
        with self.assertRaises(ValueError):self.start(o,1,'idle',1,'listening',False)
        with self.assertRaises(ValueError):self.owner(145)


@unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
class TrialTests(unittest.TestCase):
    def test_real_headless_trial_before_eos_no_future_prefix_leak(self):
        pcm=b'\x01\x00'*577920;seen=[]
        def backend(m):
            c=validate_manifest(m);seq=c['sequence'];seen.append(m)
            from unittest.mock import patch
            from research.h3_personal.fast_continuation_stages import _manifest,ENV
            path=Path(m['wav']).parent/'current-manifest.json';path.write_text(json.dumps(m))
            with patch.dict('os.environ',{ENV:str(path)}):self.assertEqual(_manifest(),m)
            end=124+seq*102;emitted=end-(0 if c['conversation_final'] else 5)
            start=0 if not seq else (124+(seq-1)*102-5)
            now=time.monotonic()
            return dict(raw=bytes((emitted-start)*252*448*3),count=emitted-start,
                committed_frame=end,emitted_frame=emitted,interval=(now,now),
                receipt_sha256='b'*64,evidence=dict(fake_native=True))
        context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        with tempfile.TemporaryDirectory() as d:
            report=run_trial(root=Path(d)/'trial',speech_pcm=pcm,profile_sha256='a'*64,
                             context=context,backend=backend,paced=False)
            self.assertTrue(Path(report['output']).is_file()) # survives sink cleanup
            self.assertEqual(report['frames'],634)
            self.assertTrue(report['first_two_speech_before_eos'])
            self.assertEqual([m['source_samples'] for m in seen[2:5]],[204128,408128,577920])
            self.assertEqual([m['streaming_audio']['audio_offer']['eos'] for m in seen[2:5]],[False,False,True])
            self.assertEqual(report['admission']['phase'],'closed')
            self.assertLessEqual(report['av']['bounded_queue_highwater'],2)
            audio=subprocess.check_output(['ffmpeg','-v','error','-i',report['output'],
                '-map','0:a','-f','s16le','pipe:1'],timeout=30)
            self.assertEqual(len(audio),1268000*2)
            self.assertEqual(audio[452000*2:1029920*2],pcm)
            self.assertFalse(any(audio[:452000*2]));self.assertFalse(any(audio[1029920*2:]))
            broken=dict(seen[2],streaming_audio=dict(seen[2]['streaming_audio'],logical_index=1))
            with self.assertRaises(ValueError):validate_manifest(broken)
