"""Preparatory native-progress trigger, not heard-speech barge-in."""
import copy
import types
import unittest
from unittest.mock import patch

from research.run_h3_relay_conversation import wait_native_preroll
from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL
from research.h3_personal.voxcpm_endpoint_service import Gate
import time


def row(epoch=1, state='streaming', samples=48000, stamp=1, terminal=False):
    return dict(schema='h3-native-progress-v1',session='s',boot='b',cancel_protocol=CANCEL_PROTOCOL,
                epoch=epoch,state=state,samples=samples,chunks=1 if samples else 0,
                observed_monotonic_ns=stamp,terminal=terminal)


class Tests(unittest.TestCase):
    def run_wait(self, rows, *, first=99., limit=101., check=lambda:None):
        clock=[100.];calls=[];pending=list(rows)
        def progress(epoch, deadline):
            calls.append((epoch,deadline))
            self.assertEqual(epoch,1)
            return pending.pop(0) if pending else row(stamp=100)
        fake_time=types.SimpleNamespace(monotonic=lambda:clock[0],
                                       sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds))
        class Sink:
            @property
            def first_av_submission_monotonic(self):
                return first(clock[0]) if callable(first) else first
        with patch('research.run_h3_relay_conversation.time',fake_time):
            answer=wait_native_preroll(Sink(),progress,deadline=limit,reserve_samples=48000,check=check)
        return answer,calls

    def test_idle_prefirstyield_then_progress_is_observation_not_cancel(self):
        answer,calls=self.run_wait([row(0,'idle',0,1),row(samples=0,stamp=2),row(stamp=3)])
        self.assertEqual(len(calls),3)
        self.assertEqual(answer['outcome'],'eligible_snapshot')
        self.assertEqual(answer['source_epoch'],1)
        self.assertNotIn('native_cancel_exercised',answer)

    def test_waits_for_actual_local_av_not_source_audio_output(self):
        answer,calls=self.run_wait([row(stamp=i) for i in range(1,8)],
                                  first=lambda now:99. if now>=100.1 else None)
        self.assertGreaterEqual(len(calls),3)
        self.assertEqual(answer['outcome'],'eligible_snapshot')

    def test_eos_before_trigger_is_unexercised_and_not_repolled(self):
        answer,calls=self.run_wait([row(state='iterator_exhausted',terminal=True)],
                                  first=lambda now:99. if now>=100.1 else None)
        self.assertEqual(len(calls),1)
        self.assertEqual(answer['outcome'],'already_terminal_or_cancel_pending')
        self.assertTrue(answer['native_progress']['terminal'])

    def test_eos_can_win_after_eligible_snapshot(self):
        answer,_=self.run_wait([row()])
        self.assertFalse(answer['native_progress']['terminal'])
        # Snapshot is explicitly not a cancelled-terminal receipt.
        self.assertEqual(answer['outcome'],'eligible_snapshot')

    def test_stale_epoch_timestamp_or_rewound_prefix_rejected(self):
        cases=([row(epoch=2)], [row(samples=1,stamp=5),row(stamp=4)],
               [row(samples=40000,stamp=1),row(samples=20000,stamp=2)])
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):self.run_wait(rows)

    def test_no_av_or_cancelled_consumer_is_bounded(self):
        with self.assertRaises(TimeoutError):self.run_wait([row()],first=None,limit=100.2)
        def stopped():raise InterruptedError('test consumer stopped')
        with self.assertRaises(InterruptedError):self.run_wait([row()],check=stopped)

    def test_existing_controller_invalidates_and_sends_before_pi2_without_old_voice_wait(self):
        import tempfile
        from pathlib import Path
        from test_h3_relay_conversation import Transport, Pump, rpc_proposal
        from research.run_h3_relay_conversation import run_conversation
        from research.h3_personal.paired_av import LiveAVBinding
        from research.h3_personal.epoch_av_sink import EpochAVSink
        events=[];holder=[]
        class Wire(Transport):
            def send_input(self,raw):
                value=super().send_input(raw)
                events.append(self.rows[-1]['kind'])
                return value
        class Sink(EpochAVSink):
            def __init__(self,root,binding,**kwargs):
                self.binding=binding;self.first=time.monotonic();self.cut=None;holder.append(self)
            @property
            def first_av_submission_monotonic(self):return self.first
            def wait_for_speech_output(self,*args):raise AssertionError('must not wait for old voice')
            def invalidate(self,new_epoch,**kwargs):
                events.append('sink_invalidate');now=time.monotonic()
                self.cut=dict(actual_decoded_sink_observed=True,detected_monotonic=now,
                    old_sink_eof_monotonic=now+.0001,stop_upper_bound_seconds=.0001,
                    old_sink_last_pcm_monotonic=None)
            @property
            def timing(self):
                return dict(discontinuities=[self.cut],decoded_source_output={'2':dict(
                    first_monotonic=self.cut['detected_monotonic']+.01)})
            def cancel(self):pass
        class Director:
            calls=0
            def turn(self,text):
                self.calls+=1
                if self.calls==2:
                    events.append('pi2')
                    case.assertLess(events.index('sink_invalidate'),events.index('interrupt_turn'))
                    case.assertLess(events.index('interrupt_turn'),events.index('pi2'))
                return rpc_proposal()
            def close(self):pass
        case=self;transport=Wire();binding=LiveAVBinding('fresh','a'*64,1,max_total_frames=248)
        ready=dict(status='ready',session='fresh',profile_sha256='a'*64,epoch=1,
            expires_at=time.time()+30,native_tts_cancel=True,native_cancel_trigger='native-preroll-v1')
        with tempfile.TemporaryDirectory() as tmp,patch('research.h3_personal.epoch_av_sink.EpochAVSink',Sink):
            result=run_conversation(transport,binding,Path(tmp),['one','two'],Director(),ready=ready,
                headless_interrupt=True,pump_factory=Pump,native_cancel_trigger='native-preroll-v1',
                native_progress=lambda *args:row(),native_preroll_samples=48000)
        self.assertEqual(result['native_trigger']['outcome'],'eligible_snapshot')
        self.assertEqual(result['interruption_kind'],'generation_preparation_cancel_not_heard_speech_barge_in')
        self.assertTrue(result['presentation_interrupted'])
        self.assertEqual([r['kind'] for r in transport.rows].count('interrupt_turn'),1)

    def test_remote_progress_binding_and_stale_rejection(self):
        import json
        from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
        for mutation in (None,{'epoch':2},{'boot':'wrong'},{'samples':True},
                         {'terminal':1},{'state':'idle'},{'schema':'other'}):
            data=row()
            if mutation:data.update(mutation)
            calls=[]
            def transport(method,path,body,headers,deadline):
                calls.append((method,path,body,headers))
                return 200,json.dumps(data).encode()
            client=RemoteVoxCPM('https://test.endpoints.huggingface.cloud',token='fake-token',key='fake-key',
                session='s',boot='b',expires_at=time.time()+60,cancel_protocol=CANCEL_PROTOCOL,transport=transport)
            if mutation is None:
                self.assertEqual(client.progress(1,time.monotonic()+2),data)
                self.assertEqual(calls[0][:3],('GET','/progress',None))
                self.assertEqual(calls[0][3]['X-H3-TTS-Key'],'fake-key')
            else:
                with self.assertRaises(RuntimeError):client.progress(1,time.monotonic()+2)
            client.close()

    def test_gate_progress_records_prefix_and_natural_terminal(self):
        class Candidate:
            sample_rate=48000
            def stream(self,text):yield [.25];yield [.5]
        gate=Gate(Candidate(),key='k',session='s',boot='b',expires_at=time.time()+60,
                  cancel_protocol=CANCEL_PROTOCOL)
        self.assertEqual(gate.progress()['state'],'idle')
        seen=[]
        def write(data):
            if data[:1]==b'P':seen.append(gate.progress())
        gate.stream(dict(session='s',boot='b',epoch=1,text='private',cancel_protocol=CANCEL_PROTOCOL),write)
        self.assertEqual(seen[0]['samples'],0)  # current write not acknowledged yet
        self.assertEqual(seen[1]['samples'],1)
        self.assertEqual(gate.progress()['samples'],2)
        self.assertTrue(gate.progress()['terminal'])
        self.assertEqual(gate.progress()['state'],'iterator_exhausted')
        self.assertNotIn('private',str(gate.progress()))


if __name__=='__main__':unittest.main()
