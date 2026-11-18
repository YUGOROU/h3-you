"""Immutable keepalive mode remains explicit/default-off and READY-bound."""
import ast
from copy import deepcopy
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

class KeepaliveWiringTests(unittest.TestCase):
    def test_validation_and_cli(self):
        item=spec();before=deepcopy(item);worker.validate_spec(item);self.assertEqual(item,before)
        for invalid in (1,None,'true',{}):
            item=spec();item['live_direct_session']['http_keepalive']=invalid
            with self.assertRaises(ValueError):worker.validate_spec(item)
            with self.assertRaises(ValueError):launcher.run('/unused',['fresh'],http_keepalive=invalid)
        item=spec();item['http_keepalive']=True
        with self.assertRaises(ValueError):worker.validate_spec(item)
        with patch.object(launcher,'run') as run,patch('builtins.input',return_value='fresh'),patch('builtins.print'):
            with self.assertRaises(SystemExit):launcher.main(['--directory','/unused','--http-keepalive'])
            run.assert_not_called()
            launcher.main(['--directory','/unused','--run','--http-keepalive'])
            self.assertIs(run.call_args.kwargs['http_keepalive'],True)
            launcher.main(['--directory','/unused','--run'])
            self.assertNotIn('http_keepalive',run.call_args.kwargs)

    def test_immutable_row_and_expected_ready(self):
        tree=ast.parse(inspect.getsource(launcher.run))
        suite=next(n for n in tree.body[0].body if isinstance(n,ast.Try)).body
        start=next(i for i,n in enumerate(suite) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets))
        end=next(i for i,n in enumerate(suite) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ready' for t in n.targets))
        code=compile(ast.fix_missing_locations(ast.Module(body=suite[start:end+1],type_ignores=[])),'<keepalive launch>','exec')
        for enabled in (False,True):
            prepare=Mock(return_value='stage');ready=Mock()
            scope=dict(headless_interrupt=False,visual_intents=False,review_av=False,rtc=False,h3_adoption=None,continuous_av=False,http_keepalive=enabled,attended_fields={},ordered_metadata={},long_session=False,native_tts_cancel=False,native_cancel_trigger=None,ordered_references=None,confirm_private_reference_upload=False,stable_selfie=False,compute_profile=None,job_api=Mock(),job_token='fake',job_namespace='YUGOROU',overlap_send=True,tts_mode='complete',session='a'*32,windows=16,frames=1654,timeout=1200,
                five_minute_test=False,ten_minute_test=False,idle_anchor=False,conversation_anchor=False,source=None,
                directory='/unused',api=Mock(),token='fake',pk='fake',ck='fake',time=time,prepare=prepare,submit=Mock(return_value=(Mock(),'image','command')),await_ready=ready)
            exec(code,scope)
            row=prepare.call_args.args[1]
            worker.validate_spec(dict(spec(),live_direct_session=row))
            self.assertEqual(row.get('http_keepalive',False),enabled)
            self.assertEqual(ready.call_args.kwargs.get('http_keepalive',False),enabled)

    def test_actual_server_constructor_wiring_and_ready(self):
        tree=ast.parse(inspect.getsource(worker.run_session))
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.With))
        block.body=[ast.Expr(n.value) if isinstance(n,ast.Return) else n for n in block.body]
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'<keepalive session>','exec')
        for enabled in (False,True):
            row=spec()['live_direct_session'];row['http_keepalive']=enabled
            binding=NS(session=row['session'],profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654)
            fake=Mock(side_effect=lambda *a,**kw:kw['on_ready']())
            server=MagicMock()
            scope=dict(rtc=None,rtc_options={},native=False,row=row,binding=binding,expires=time.time()+600,json=json,ResamplingContext=Mock(),
                NATIVE_TORCH='x',NATIVE_TORCHAUDIO='y',REFERENCE_SHA256='z',
                CONVERSATION_ANCHOR_POLICY=worker.CONVERSATION_ANCHOR_POLICY,JobAVServer=server,config=Mock(),
                run_worker=fake,runtime=Mock(),tts=Mock(),output=Path('/unused'),deadline=999)
            with patch('builtins.print') as emit:exec(code,scope)
            self.assertEqual(server.call_args.kwargs.get('keepalive',False),enabled)
            ready=json.loads(emit.call_args.args[0])
            launcher.validate_ready(ready,row['session'],http_keepalive=enabled)
            with self.assertRaises(ValueError):launcher.validate_ready(ready,row['session'],http_keepalive=not enabled)
            if enabled:
                for invalid in (1,False,None):
                    with self.assertRaises(ValueError):launcher.validate_ready(dict(ready,http_keepalive=invalid),row['session'],http_keepalive=True)

if __name__=='__main__':unittest.main()
