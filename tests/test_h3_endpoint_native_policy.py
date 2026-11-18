"""CPU-only native policy and delivery guards; no Hub/device calls."""
import copy
import json
import subprocess
import sys
import time
import unittest

from research.hf_h3_live_voice import validate_request
from research.deliver_h3_endpoint_turn import request, verify_codecs, verify_output


class NativePolicyTests(unittest.TestCase):
    def result(self):
        return dict(origin='comet_process_audio_endpoint_asr',terminal_state='paused',
                    asr=dict(text='こんにちは',final=True,capture_stopped=True,on_device=False,
                             cleanup_complete=True,errors=[],events=[dict(final=True,text='こんにちは',endpoint_source='native_completed')]),
                    proposal=dict(model='Qwen/Qwen3.8-27B:cerebras',reply='こんにちは。'))

    def test_native(self):
        row=request(self.result(),'s')
        self.assertIs(validate_request(row,'s',1,'endpoint-native-v1'),row)

    def test_false_labels(self):
        row=request(self.result(),'s')
        for key,value in [('origin','discord_received_audio_ondevice_asr'),('asr_final',False),('asr_stopped',False),('asr_on_device',True),('endpoint_source','application_stable_partial'),('asr_revision',1),('asr_endpoint_source','native_completed')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_request(dict(row,**{key:value}),'s',1,'endpoint-native-v1')
        for key in ('asr_final','asr_stopped','asr_on_device','endpoint_source'):
            broken=dict(row);broken.pop(key)
            with self.assertRaises(ValueError):validate_request(broken,'s',1,'endpoint-native-v1')

    def test_no_legacy_relabel(self):
        row=request(self.result(),'s')
        for policy in ('apple-final','application-stream-v1'):
            with self.assertRaises(ValueError):validate_request(row,'s',1,policy)

    def test_legacy_unchanged(self):
        row=dict(schema='h3-live-voice-input-v1',session_id='s',sequence=1,origin='discord_received_audio_ondevice_asr',asr_stopped=True,asr_final=True,director_model='Qwen/Qwen3.8-27B:cerebras',reply='reply',expires_at=time.time()+45)
        validate_request(row,'s',1)
        row.update(asr_final=False,asr_endpoint_source='application_stable_partial',asr_revision=1)
        validate_request(row,'s',1,'application-stream-v1')

    def test_native_source_cleanup(self):
        for key,value in [('final',False),('capture_stopped',False),('on_device',True),('cleanup_complete',False),('errors',['bad']),('events',[])]:
            broken=self.result();broken['asr'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):request(broken,'s')
        broken=self.result();broken['proposal']['model']='fallback'
        with self.assertRaises(ValueError):request(broken,'s')

    def test_whole_input_delivery_binds_all_segments(self):
        r=self.result();a=r['asr']
        a.update(completion_policy='input-finalized',input_committed=True,
                 audio_sent_seconds=2.,finalized_audio_seconds=2.)
        a['events'].append(dict(final=True,text='質問です',endpoint_source='native_completed'))
        a['text']='こんにちは 質問です'
        request(r,'s')
        for key,value in [('input_committed',False),('finalized_audio_seconds',1.),
                          ('finalized_audio_seconds',float('nan')),('text','質問です')]:
            broken=copy.deepcopy(r);broken['asr'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):request(broken,'s')

    def test_codecs(self):
        probe=dict(streams=[dict(codec_type='video',codec_name='h264'),dict(codec_type='audio',codec_name='alac')])
        verify_codecs(probe)
        probe['streams'][1]['codec_name']='aac'
        with self.assertRaises(ValueError):verify_codecs(probe)

    def test_output_binding(self):
        import hashlib
        row=request(self.result(),'s')
        output=dict(schema='h3-live-voice-state-v1',status='output',session_id='s',sequence=1,previous_audio_reused=False,reply_sha256=hashlib.sha256(row['reply'].encode()).hexdigest())
        verify_output(output,'s',row)
        for key,value in [('sequence',2),('session_id','old'),('previous_audio_reused',True),('reply_sha256','wrong')]:
            with self.assertRaises(ValueError):verify_output(dict(output,**{key:value}),'s',row)

    def test_offline_nonexistent_paths(self):
        result=subprocess.run([sys.executable,'-m','research.deliver_h3_endpoint_turn','--directory','/nonexistent','--asr-result','/nonexistent/result','--obs-receipt','/nonexistent/obs'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout)['mode'],'offline')


if __name__=='__main__':unittest.main()
