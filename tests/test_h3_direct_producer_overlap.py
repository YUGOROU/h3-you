"""Offline immutable overlap flag/READY/session wiring tests."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from research import launch_h3_direct_conversation as launcher
from research import hf_h3_direct_conversation as worker
from test_h3_direct_conversation_spec import spec


class WiringTests(unittest.TestCase):
    def test_spec_strict_default_unchanged(self):
        item=spec();before=deepcopy(item)
        worker.validate_spec(item);self.assertEqual(item,before)
        for value in (True,False):
            item=spec();item['live_direct_session']['overlap_send']=value
            self.assertIs(worker.validate_spec(item)['overlap_send'],value)
        for value in (1,0,None,'true',{},[]):
            item=spec();item['live_direct_session']['overlap_send']=value
            with self.assertRaises(ValueError):worker.validate_spec(item)
        item=spec();item['overlap_send']=True
        with self.assertRaises(ValueError):worker.validate_spec(item)

    def test_cli_and_early_type_validation(self):
        with patch.object(launcher,'run') as run, patch('builtins.input',return_value='fresh'),patch('builtins.print'):
            with self.assertRaises(SystemExit):
                launcher.main(['--directory','/unused','--overlap-send'])
            run.assert_not_called()
            launcher.main(['--directory','/unused','--run','--overlap-send'])
            self.assertIs(run.call_args.kwargs['overlap_send'],True)
            launcher.main(['--directory','/unused','--run'])
            self.assertNotIn('overlap_send',run.call_args.kwargs)
        with patch.object(launcher,'canonical',side_effect=AssertionError('network')):
            for value in (1,None,'true'):
                with self.assertRaises(ValueError):launcher.run('/unused',['fresh'],overlap_send=value)

    def test_immutable_row_and_ready_expectation(self):
        tree=ast.parse(inspect.getsource(launcher.run))
        suite=next(n for n in tree.body[0].body if isinstance(n,ast.Try)).body
        start=next(i for i,n in enumerate(suite) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets))
        end=next(i for i,n in enumerate(suite) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ready' for t in n.targets))
        code=compile(ast.fix_missing_locations(ast.Module(body=suite[start:end+1],type_ignores=[])),'<immutable overlap>','exec')
        for enabled in (False,True):
            prepare=Mock(return_value='stage');ready=Mock()
            scope=dict(continuous_av=False,http_keepalive=False,overlap_send=enabled,tts_mode='complete',session='a'*32,windows=16,frames=1654,timeout=1200,rtc=False,headless_interrupt=False,visual_intents=False,review_av=False,h3_adoption=None,attended_fields={},ordered_metadata={},long_session=False,native_tts_cancel=False,native_cancel_trigger=None,ordered_references=None,confirm_private_reference_upload=False,stable_selfie=False,compute_profile=None,job_api=Mock(),job_token='fake',job_namespace='YUGOROU',
                five_minute_test=False,ten_minute_test=False,idle_anchor=False,conversation_anchor=False,
                source=None,directory='/unused',api=Mock(),token='fake',pk='fake',ck='fake',time=time,
                prepare=prepare,submit=Mock(return_value=(Mock(),'image','command')),await_ready=ready)
            exec(code,scope)
            row=prepare.call_args.args[1]
            worker.validate_spec(dict(spec(),live_direct_session=row))
            self.assertEqual(row.get('overlap_send',False),enabled)
            self.assertEqual(ready.call_args.kwargs.get('overlap_send',False),enabled)

    def test_worker_and_ready_fail_closed(self):
        tree=ast.parse(inspect.getsource(worker.run_session))
        body=next(n for n in ast.walk(tree) if isinstance(n,ast.With)).body
        body=[ast.Expr(n.value) if isinstance(n,ast.Return) else n for n in body]
        code=compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<overlap session>','exec')
        for enabled in (False,True):
            row=spec()['live_direct_session']
            if enabled:row['overlap_send']=True
            binding=NS(session=row['session'],profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654)
            fake=Mock(side_effect=lambda *a,**kw:kw['on_ready']())
            scope=dict(row=row,binding=binding,expires=time.time()+600,json=json,ResamplingContext=Mock(),rtc=None,native=False,
                NATIVE_TORCH='x',NATIVE_TORCHAUDIO='y',REFERENCE_SHA256='z',
                CONVERSATION_ANCHOR_POLICY=worker.CONVERSATION_ANCHOR_POLICY,
                run_worker=fake,runtime=Mock(),tts=Mock(),server=Mock(),output=Path('/unused'),deadline=999)
            with patch('builtins.print') as emit:exec(code,scope)
            ready=json.loads(emit.call_args.args[0])
            launcher.validate_ready(ready,row['session'],overlap_send=enabled)
            self.assertEqual(fake.call_args.kwargs.get('overlap_send',False),enabled)
            with self.assertRaises(ValueError):launcher.validate_ready(ready,row['session'],overlap_send=not enabled)
            if enabled:
                for invalid in (1,False,None):
                    bad=dict(ready,overlap_send=invalid)
                    with self.assertRaises(ValueError):launcher.validate_ready(bad,row['session'],overlap_send=True)

if __name__=='__main__':unittest.main()
