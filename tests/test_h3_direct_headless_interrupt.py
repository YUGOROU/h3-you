"""Existing entrypoint integration; no provider/device/model operation."""
import ast
import copy
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, MagicMock, patch

from research import hf_h3_direct_conversation as worker
from research import launch_h3_direct_conversation as launcher
from research.h3_personal.native_tts_backend import config_sha256, config_preroll
from test_h3_native_tts_backend import config
from test_h3_direct_conversation_spec import spec


class IntegratedHeadlessTests(unittest.TestCase):
    def selected(self):
        value=spec(); value['live_direct_session'].update(tts_mode='native_voxcpm',
            native_tts=config(), continuous_av=True, turn_interruptions=True)
        return value

    def test_config_is_explicit_native_https_and_ready_binds_interruption(self):
        value=self.selected(); row=worker.validate_spec(value)
        ready=dict(type='h3_direct_ready',status='ready',session=row['session'],
            profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654,
            port=7860,expires_at=time.time()+500,continuous_av=True,
            turn_interruptions=True,tts_mode='native_voxcpm',
            native_tts_config_sha256=config_sha256(row['native_tts']))
        launcher.validate_ready(ready,row['session'],tts_mode='native_voxcpm',
            native_tts=row['native_tts'],continuous_av=True,turn_interruptions=True)
        with self.assertRaises(ValueError):
            launcher.validate_ready(ready,row['session'],tts_mode='native_voxcpm',
                native_tts=row['native_tts'],continuous_av=True)
        for name, invalid in (('turn_interruptions',1),('continuous_av',False),
                              ('tts_mode','complete')):
            altered=copy.deepcopy(value);altered['live_direct_session'][name]=invalid
            with self.subTest(name=name),self.assertRaises(ValueError):worker.validate_spec(altered)

    def test_worker_real_constructor_path_receives_opt_in(self):
        tree=ast.parse(inspect.getsource(worker.run_session))
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.With))
        block.body=[ast.Expr(n.value) if isinstance(n,ast.Return) else n for n in block.body]
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),
                     '<headless existing worker>','exec')
        row=self.selected()['live_direct_session'];row.update(visual_intents=True,paired_cuts=True);calls=Mock(side_effect=lambda *a,**kw:kw['on_ready']())
        scope=dict(row=row,native=True,rtc=None,rtc_options={},binding=NS(session=row['session'],
            profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654),
            expires=time.time()+500,json=json,ResamplingContext=Mock(),
            NATIVE_TORCH='x',NATIVE_TORCHAUDIO='y',REFERENCE_SHA256='z',
            CONVERSATION_ANCHOR_POLICY=worker.CONVERSATION_ANCHOR_POLICY,
            JobAVServer=MagicMock(),config=Mock(),run_worker=calls,runtime=Mock(),
            tts=Mock(),output=Path('/unused'),deadline=999,
            config_sha256=config_sha256,config_preroll=config_preroll)
        with patch('builtins.print') as emit:exec(code,scope)
        self.assertIs(calls.call_args.kwargs['turn_interruptions'],True)
        self.assertIs(calls.call_args.kwargs['visual_intents'],True)
        self.assertIs(calls.call_args.kwargs['paired_cuts'],True)
        self.assertIs(json.loads(emit.call_args.args[0])['paired_cuts'],True)
        self.assertIs(json.loads(emit.call_args.args[0])['visual_intents'],True)
        self.assertIs(json.loads(emit.call_args.args[0])['turn_interruptions'],True)

    def test_launcher_immutable_row_and_await_ready_are_connected(self):
        tree=ast.parse(inspect.getsource(launcher.run))
        body=next(n for n in tree.body[0].body if isinstance(n,ast.Try)).body
        def assignment(node,name):
            return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        start=next(i for i,n in enumerate(body) if assignment(n,'row'))
        end=next(i for i,n in enumerate(body) if assignment(n,'ready'))
        code=compile(ast.fix_missing_locations(ast.Module(body=body[start:end+1],type_ignores=[])),
                     '<headless existing launch>','exec')
        prepare=Mock(return_value='stage'); ready=Mock()
        scope=dict(headless_interrupt=True,visual_intents=True,review_av=True,native_tts_cancel=False,native_cancel_trigger=None,ordered_metadata={},ordered_references=None,rtc=False,h3_adoption=None,continuous_av=True,
            http_keepalive=True,overlap_send=True,tts_mode='native_voxcpm',native_tts=config(),
            session='a'*32,windows=16,frames=1654,timeout=1200,five_minute_test=False,
            ten_minute_test=False,idle_anchor=False,conversation_anchor=False,source=None,
            attended_fields={},long_session=False,stable_selfie=False,
            directory='/unused',api=Mock(),token='fake',pk='fake',ck='fake',time=time,
            # Legacy main-account aliases: no compute profile on this route.
            compute_profile=None,job_api=Mock(),job_token='fake',job_namespace='YUGOROU',
            prepare=prepare,submit=Mock(return_value=(Mock(),'image','command')),await_ready=ready)
        exec(code,scope)
        self.assertIs(prepare.call_args.args[1]['turn_interruptions'],True)
        self.assertIs(ready.call_args.kwargs['turn_interruptions'],True)
        self.assertIs(prepare.call_args.args[1]['visual_intents'],True)
        self.assertIs(ready.call_args.kwargs['visual_intents'],True)
        self.assertIs(prepare.call_args.args[1]['paired_cuts'],True)
        self.assertIs(ready.call_args.kwargs['paired_cuts'],True)

    def test_continuous_progress_allowance_is_headless_ordered_candidate_only(self):
        tree=ast.parse(inspect.getsource(launcher.run))
        body=next(n for n in tree.body[0].body if isinstance(n,ast.Try)).body
        names={'candidate_progress','consumer'}
        nodes=[n for n in body if isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
        self.assertEqual(len(nodes),2)
        code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                     '<existing continuous consumer constructor>','exec')
        for continuous in (False,True):
            for headless in (False,True):
                for candidate in (False,True):
                    with self.subTest(continuous=continuous,headless=headless,candidate=candidate):
                        constructor=Mock()
                        scope=dict(continuous_av=continuous,headless_interrupt=headless,
                            ordered_references={} if candidate else None,adapter_type=constructor,
                            # stable_selfie is a real run() option; the device-free branch is under test.
                            stable_selfie=False,attended_visual=False,
                            permit=NS(base_url='https://example.invalid'),session='s',
                            ready={'expires_at':123},ck='key',token='fake',
                            # Legacy main-account aliases resolved before the consumer.
                            job_token='fake',job_api=Mock(),job_namespace='YUGOROU')
                        exec(code,scope)
                        kwargs=constructor.call_args.kwargs
                        if continuous and headless and candidate:
                            self.assertEqual(kwargs['http_timeout'],120)
                            self.assertEqual(kwargs['stream_progress_timeout'],120)
                        else:
                            self.assertNotIn('stream_progress_timeout',kwargs)
                            self.assertNotIn('http_timeout',kwargs)
                        self.assertEqual(kwargs['not_after'],123)

    def test_visual_selection_requires_exact_ready_and_latest_tail(self):
        value=self.selected();value['live_direct_session']['visual_intents']=True
        row=worker.validate_spec(value)
        ready=dict(type='h3_direct_ready',status='ready',session=row['session'],profile_sha256='b'*64,
            epoch=1,max_segments=16,max_total_frames=1654,port=7860,expires_at=time.time()+500,
            continuous_av=True,turn_interruptions=True,visual_intents=True,tts_mode='native_voxcpm',
            native_tts_config_sha256=config_sha256(row['native_tts']))
        kwargs=dict(tts_mode='native_voxcpm',native_tts=row['native_tts'],continuous_av=True,turn_interruptions=True)
        launcher.validate_ready(ready,row['session'],visual_intents=True,**kwargs)
        with self.assertRaises(ValueError):launcher.validate_ready(ready,row['session'],**kwargs)
        missing=dict(ready);missing.pop('visual_intents')
        with self.assertRaises(ValueError):launcher.validate_ready(missing,row['session'],visual_intents=True,**kwargs)
        for name,bad in (('visual_intents',1),('anchor_policy',worker.CONVERSATION_ANCHOR_POLICY),('turn_interruptions',False)):
            altered=copy.deepcopy(value);altered['live_direct_session'][name]=bad
            with self.subTest(name=name),self.assertRaises(ValueError):worker.validate_spec(altered)
        with patch('builtins.print'),self.assertRaises(SystemExit):
            launcher.main(['--directory','/unused','--visual-intents'])
        with patch('builtins.print'):
            launcher.main(['--directory','/unused','--headless-interrupt','--vox-parent','/unused','--visual-intents'])

    def test_review_requires_visual_and_matching_ready(self):
        with self.assertRaises(ValueError):launcher.run('/unused',['one','two'],review_av=True)
        with patch('builtins.print'),self.assertRaises(SystemExit):
            launcher.main(['--directory','/unused','--review-av'])
        value=self.selected();value['live_direct_session'].update(visual_intents=True,paired_cuts=True)
        row=worker.validate_spec(value)
        ready=dict(type='h3_direct_ready',status='ready',session=row['session'],profile_sha256='b'*64,
            epoch=1,max_segments=16,max_total_frames=1654,port=7860,expires_at=time.time()+500,
            continuous_av=True,turn_interruptions=True,visual_intents=True,paired_cuts=True,
            tts_mode='native_voxcpm',native_tts_config_sha256=config_sha256(row['native_tts']))
        kw=dict(tts_mode='native_voxcpm',native_tts=row['native_tts'],continuous_av=True,
                turn_interruptions=True,visual_intents=True)
        launcher.validate_ready(ready,row['session'],paired_cuts=True,**kw)
        with self.assertRaises(ValueError):launcher.validate_ready(ready,row['session'],**kw)
        ready.pop('paired_cuts')
        with self.assertRaises(ValueError):launcher.validate_ready(ready,row['session'],paired_cuts=True,**kw)
        value['live_direct_session'].pop('visual_intents')
        with self.assertRaises(ValueError):worker.validate_spec(value)

    def test_same_cli_finalizes_result_without_duplicate_keywords_or_device_launch(self):
        import contextlib, tempfile
        from research.h3_personal import voxcpm_https_lease as owned
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); directory=root/'campaign'
            @contextlib.contextmanager
            def lease(*args,**kwargs):
                yield dict(token='fake',key='fake',config=config())
            outcome=dict(complete=True,devices_used=False,transport_complete=True,
                         stop_criteria={'passed':True})
            with patch.object(owned,'PRIVATE',root),patch.object(owned,'lease',side_effect=lease), \
                 patch.object(launcher,'run',return_value=outcome) as run, \
                 patch('builtins.input',side_effect=['typed one','typed two']),patch('builtins.print'):
                launcher.main(['--directory',str(directory),'--run','--headless-interrupt',
                               '--vox-parent',str(root/'parent')])
            saved=json.loads((directory/'result.json').read_text())
            self.assertTrue(saved['complete']);self.assertFalse(saved['devices_used'])
            self.assertTrue(run.call_args.kwargs['headless_interrupt'])
            self.assertEqual(run.call_args.args[0],directory/'h3')

    def test_cli_offline_and_device_guard(self):
        with patch.object(launcher,'run') as run,patch('builtins.print'):
            launcher.main(['--directory','/unused','--headless-interrupt','--vox-parent','/unused-parent'])
        run.assert_not_called()
        with self.assertRaises(SystemExit):
            launcher.main(['--directory','/unused','--headless-interrupt','--vox-parent','/unused',
                           '--capture-comet','--asr-parent','/unused'])
        with self.assertRaises(ValueError):
            launcher.run('/unused',['one','two'],headless_interrupt=True,allow_obs=True)


if __name__=='__main__':unittest.main()
