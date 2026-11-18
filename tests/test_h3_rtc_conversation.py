"""Offline owner/loop contracts. No cloud, browser, model or capture operations."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from research.h3_personal.webrtc_conversation import run_conversation,collect_delivery
from research.run_h3_rtc_conversation import run,main

class RTCConversationTests(unittest.TestCase):
    def test_offline_cli_never_launches(self):
        with patch('research.run_h3_rtc_conversation.run',side_effect=AssertionError),patch('builtins.print'):
            self.assertEqual(main([]),0)

    def test_owner_dispatch_preserves_native_and_no_http_sink(self):
        config={'test':'config'}
        def coupled(root,*,rtc_conversation):return rtc_conversation(Path(root),config)
        with patch('research.run_h3_native_coupled.run',side_effect=coupled),patch('research.launch_h3_direct_conversation.run',return_value='done') as direct:
            self.assertEqual(run('/not-created',on_browser=Mock(),turns=['new 1','new 2']),'done')
        kw=direct.call_args.kwargs
        self.assertEqual(kw['media_transport'],'webrtc-rtp-v1');self.assertIs(kw['native_tts'],config)
        self.assertTrue(kw['five_minute_test']);self.assertNotIn('local_sink_factory',kw)
        self.assertFalse(kw.get('continuous_av',False));self.assertFalse(kw['capture_comet'])

    def test_capture_requires_explicit_parent_and_no_typed_replay(self):
        with patch('research.run_h3_native_coupled.run',side_effect=AssertionError):
            for kw in (dict(capture_comet=True),dict(asr_parent=Path('/private')),
                       dict(capture_comet=True,asr_parent=Path('/private'),turns=['replay','old'])):
                with self.subTest(kw=kw),self.assertRaises(ValueError):run('/not-created',on_browser=Mock(),**kw)

    def fixture(self):
        ready=dict(expires_at=time.time()+120,rtc_boot='b'*32,ice_profile='hf-cloudflare-turn-tls-v1')
        binding=SimpleNamespace(session='s')
        status=dict(ready,session='s',boot=ready['rtc_boot']);status.pop('rtc_boot')
        return ready,binding,status

    def test_receiver_presentation_precedes_pi_and_source_turns(self):
        import tempfile
        ready,binding,status=self.fixture();proxy=Mock();proxy.url='private';proxy.receipts=({'event':'frame-presented'},)
        factory=Mock();factory.return_value.__enter__=Mock(return_value=proxy);factory.return_value.__exit__=Mock(return_value=False)
        completion=Mock(return_value=dict(source_complete=True,media_complete=False))
        director=Mock();director.deadline=time.monotonic()+120
        with tempfile.TemporaryDirectory() as root,patch('research.h3_personal.webrtc_conversation.BrowserProxy',factory),\
             patch('research.h3_personal.webrtc_conversation.request',return_value=status),\
             patch('research.h3_personal.webrtc_conversation.fetch_turn_credentials',return_value={}),\
             patch('research.h3_personal.webrtc_conversation.proposal_reply',return_value={'text':'reply'}),\
             patch('research.h3_personal.webrtc_conversation.Commands') as command:
            result=run_conversation(Mock(),binding,root,['fresh1','fresh2'],director,ready=ready,
                token='HF',key='KEY',job_id='a'*24,on_browser=Mock(),completion=completion)
        self.assertEqual(director.turn.call_count,2)
        self.assertEqual([c.args[0] for c in command.return_value.send.call_args_list],['turn','turn','stop'])
        self.assertFalse(result['all_media_webrtc']);self.assertFalse(result['audibility_verified'])
        completion.assert_called_once()
        self.assertEqual(factory.call_args.kwargs['signaling_path'],'/rtc/offer')

    def test_identity_failure_before_turn_credentials_or_pi(self):
        ready,binding,status=self.fixture();status['boot']='wrong'
        with patch('research.h3_personal.webrtc_conversation.request',return_value=status),\
             patch('research.h3_personal.webrtc_conversation.fetch_turn_credentials',side_effect=AssertionError):
            with self.assertRaises(ValueError):run_conversation(Mock(),binding,'/never',['fresh'],Mock(),ready=ready,
                token='HF',key='KEY',job_id='a'*24,on_browser=Mock(),completion=Mock())

    def test_native_evidence_remains_required_without_remote_pcm_claim(self):
        import tempfile
        from test_h3_native_coupled import Coupled
        from research.run_h3_rtc_conversation import validate_evidence
        with tempfile.TemporaryDirectory() as tmp:
            config,evidence,_=Coupled().evidence_fixture(Path(tmp))
            result=dict(source_complete=True,turns=2,media_complete=False,remote_original_pcm_verified=False)
            proof=validate_evidence(tmp,config,evidence,result)
            self.assertTrue(proof['native_natural_stop_verified']);self.assertFalse(proof['remote_original_pcm_verified'])
            evidence['evidence']['completions'][0]['completion']['reason']='max_len'
            with self.assertRaises(ValueError):validate_evidence(tmp,config,evidence,result)

    def test_delivery_is_session_bound_and_lossy_not_exact(self):
        row=dict(type='h3_rtc_delivery',session='s',source_complete=True,media_complete=False,complete=False,
                 closing_tail_unverified=True,local_ipc_source_verified=True,remote_original_pcm_verified=False)
        api=Mock();job=SimpleNamespace(id='a'*24)
        api.fetch_job_logs.return_value=[json.dumps(row)]
        self.assertEqual(collect_delivery(api,job,'s',time.monotonic()+1),row)
        for change in ({'session':'wrong'},{'remote_original_pcm_verified':True},{'complete':True},{'source_complete':1}):
            api.fetch_job_logs.return_value=[json.dumps(dict(row,**change))]
            with self.subTest(change=change),self.assertRaises(ValueError):collect_delivery(api,job,'s',time.monotonic()+1)

if __name__=='__main__':unittest.main()
