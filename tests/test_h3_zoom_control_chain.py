"""Mocked two-turn control chain only: no meeting/native ASR/H3/AV/devices."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

from _numpy_shim import install as _install_numpy

np = _install_numpy()

from test_h3_relay_conversation import Transport,Director,Sink,Pump
from research.run_h3_relay_conversation import run_conversation
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.live_relay_worker import validate_turn
from research.h3_personal.zoom_input import source_contract
from research.h3_personal.zoom_turn_input import ZoomTurnInput

class Extractor:
    def reset(self):pass
    def push(self,x):return x*.5
    def flush(self):return np.zeros(0,np.float32)

class ZoomControlChainTests(unittest.TestCase):
    def test_two_turns_extraction_source_contract_pi_existing_transport_worker(self):
        source=source_contract(reference_sha256='a'*64,capture_source_sha256='b'*64)
        p=ZoomTurnInput(Extractor(),source,all_participants_consented=True,confirmation=lambda **kw:True,tail_validator=lambda **kw:True)
        base=dict(policy='endpoint-native-v1',endpoint_name='h3-asr-t4-7c926aae18',revision='c'*40,
                  endpoint_source='native_completed',completion_policy='input-finalized',capture_stopped=True,cleanup_complete=True)
        expected=dict(base,input_source=source)
        class Input:
            calls=notes=0
            def __call__(self,*,context,deadline):
                self.calls+=1;p.prepare_capture(deadline=deadline)
                r=p.wrap_capture(io.BytesIO(np.full(128000,1000,dtype='<i2').tobytes()),capture_seconds=8)
                delivered=b''
                while data:=r.read(5120):delivered+=data
                if len(delivered)!=256000:raise AssertionError('fixed window/sample count')
                receipt=p.finish_capture(dict(type='stopped',frames=129000,cleanup=True,app='zoom',mute_behavior='unmuted'),
                    dict(audio_sent_seconds=8),capture_seconds=8)
                return dict(text='private-user-input-'+str(self.calls),asr=dict(expected,zoom_capture=receipt))
            def note_reply_published(self):self.notes+=1;p.note_reply_published()
        inp=Input();tr=Transport();director=Director()
        binding=LiveAVBinding('fresh','a'*64,1,max_total_frames=248)
        ready=dict(status='ready',session='fresh',profile_sha256='a'*64,epoch=1,expires_at=time.time()+60)
        with tempfile.TemporaryDirectory() as d:
            result=run_conversation(tr,binding,Path(d),None,director,ready=ready,max_seconds=2,input_source=inp,
                                    sink_factory=Sink,pump_factory=Pump)
        self.assertTrue(result['complete']);self.assertEqual(inp.calls,2);self.assertEqual(inp.notes,2)
        turns=[r for r in tr.rows if r['kind']=='turn']
        self.assertEqual(len(turns),2)
        epochs=[]
        for row in turns:
            payload=row['payload'];self.assertEqual(payload['origin'],'zoom_process_voxlock_endpoint_asr')
            self.assertNotIn('private-user-input',json.dumps(payload))
            self.assertEqual(validate_turn(row,expected),'新しい返事')
            epochs.append(payload['asr']['zoom_capture']['input_epoch'])
        self.assertGreater(epochs[1],epochs[0]);self.assertTrue(p.guard.closed)
