"""Continuous carrier strict opt-in, READY binding and real wire regression."""
import ast
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,MagicMock,patch
from research import launch_h3_direct_conversation as launcher
from research import hf_h3_direct_conversation as worker
from test_h3_direct_conversation_spec import spec

class ContinuousWiringTests(unittest.TestCase):
    def test_validation_and_cli(self):
        for invalid in (1,None,'true',{}):
            item=spec();item['live_direct_session']['continuous_av']=invalid
            with self.assertRaises(ValueError):worker.validate_spec(item)
            with self.assertRaises(ValueError):launcher.run('/unused',['fresh'],continuous_av=invalid)
        item=spec();item['continuous_av']=True
        with self.assertRaises(ValueError):worker.validate_spec(item)
        with patch.object(launcher,'run') as run,patch('builtins.input',return_value='fresh'),patch('builtins.print'):
            with self.assertRaises(SystemExit):launcher.main(['--directory','/unused','--continuous-av'])
            run.assert_not_called()
            launcher.main(['--directory','/unused','--run','--continuous-av'])
            self.assertIs(run.call_args.kwargs['continuous_av'],True)
            launcher.main(['--directory','/unused','--run'])
            self.assertNotIn('continuous_av',run.call_args.kwargs)

    def test_immutable_row_and_ready(self):
        tree=ast.parse(inspect.getsource(launcher.run));suite=next(n for n in tree.body[0].body if isinstance(n,ast.Try)).body
        start=next(i for i,n in enumerate(suite) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets))
        end=next(i for i,n in enumerate(suite) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ready' for t in n.targets))
        code=compile(ast.fix_missing_locations(ast.Module(body=suite[start:end+1],type_ignores=[])),'<continuous launch>','exec')
        for enabled in (False,True):
            prepare=Mock(return_value='stage');ready=Mock()
            scope=dict(headless_interrupt=False,visual_intents=False,review_av=False,rtc=False,h3_adoption=None,stable_selfie=False,compute_profile=None,job_api=Mock(),job_token='fake',job_namespace='YUGOROU',continuous_av=enabled,http_keepalive=True,overlap_send=True,tts_mode='complete',session='a'*32,
                windows=16,frames=1654,timeout=1200,five_minute_test=False,ten_minute_test=False,idle_anchor=False,
                conversation_anchor=False,source=None,directory='/unused',api=Mock(),token='fake',pk='fake',ck='fake',
                attended_fields={},long_session=False,ordered_metadata={},native_tts_cancel=False,native_cancel_trigger=None,ordered_references=None,confirm_private_reference_upload=False,
                time=time,prepare=prepare,submit=Mock(return_value=(Mock(),'image','command')),await_ready=ready)
            exec(code,scope);row=prepare.call_args.args[1]
            worker.validate_spec(dict(spec(),live_direct_session=row))
            self.assertEqual(row.get('continuous_av',False),enabled)
            self.assertEqual(ready.call_args.kwargs.get('continuous_av',False),enabled)

    def test_server_and_ready_wiring(self):
        tree=ast.parse(inspect.getsource(worker.run_session));block=next(n for n in ast.walk(tree) if isinstance(n,ast.With))
        block.body=[ast.Expr(n.value) if isinstance(n,ast.Return) else n for n in block.body]
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'<continuous session>','exec')
        for enabled in (False,True):
            row=spec()['live_direct_session'];row['continuous_av']=enabled
            binding=NS(session=row['session'],profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654)
            fake=Mock(side_effect=lambda *a,**kw:kw['on_ready']());server=MagicMock()
            scope=dict(rtc=None,rtc_options={},native=False,row=row,binding=binding,expires=time.time()+600,json=json,ResamplingContext=Mock(),
                NATIVE_TORCH='x',NATIVE_TORCHAUDIO='y',REFERENCE_SHA256='z',
                CONVERSATION_ANCHOR_POLICY=worker.CONVERSATION_ANCHOR_POLICY,JobAVServer=server,config=Mock(),
                run_worker=fake,runtime=Mock(),tts=Mock(),output=Path('/unused'),deadline=999)
            with patch('builtins.print') as emit:exec(code,scope)
            self.assertEqual(server.call_args.kwargs.get('continuous_av',False),enabled)
            ready=json.loads(emit.call_args.args[0]);launcher.validate_ready(ready,row['session'],continuous_av=enabled)
            with self.assertRaises(ValueError):launcher.validate_ready(ready,row['session'],continuous_av=not enabled)
            if enabled:
                for invalid in (1,False,None):
                    with self.assertRaises(ValueError):launcher.validate_ready(dict(ready,continuous_av=invalid),row['session'],continuous_av=True)

    def test_real_paired_wire_to_ffmpeg_exact_pcm(self):
        import test_h3_live_relay_integration as fixtures
        from research.h3_personal.continuous_av_client import ContinuousConsumerAdapter
        from research.h3_personal.live_relay_client import SocketAdapter
        from research.live_relay_server import build_server
        case=fixtures.RelayIntegrationTests('test_http_relay_to_pumps_to_headless_exact_pcm_pts')
        def client(*args,**kwargs):
            return (ContinuousConsumerAdapter if kwargs['role']=='consumer' else SocketAdapter)(*args,**kwargs)
        with patch.object(fixtures,'build_server',side_effect=lambda *a,**kw:build_server(*a,**kw,continuous_av=True)), \
             patch.object(fixtures,'SocketAdapter',side_effect=client):
            case.setUp()
            try:case.test_http_relay_to_pumps_to_headless_exact_pcm_pts()
            finally:case.tearDown()

if __name__=='__main__':unittest.main()
