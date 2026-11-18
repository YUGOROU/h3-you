"""Offline explicit E2E candidate gates and actual selection proofs."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import hf_h3_direct_conversation as worker
from research import launch_h3_direct_conversation as launcher
from research.h3_personal.live_native_backend import LiveNativeBackend
from test_h3_direct_conversation_spec import spec


class ConversationAnchorTests(unittest.TestCase):
    def test_strict_row_and_cli(self):
        item=spec(); item['live_direct_session']['anchor_policy']=worker.CONVERSATION_ANCHOR_POLICY
        worker.validate_spec(item)
        for value in ('epoch-idle-every3-v1', None, True, {}, 'unknown'):
            item['live_direct_session']['anchor_policy']=value
            with self.assertRaises(ValueError):worker.validate_spec(item)
        for args in (['--conversation-anchor'], ['--run','--conversation-anchor','--idle-anchor']):
            with self.assertRaises(SystemExit):launcher.main(['--directory','/unused',*args])
        with patch.object(launcher,'run') as run, patch('builtins.input',return_value='fresh'), patch('builtins.print'):
            launcher.main(['--directory','/unused','--run','--conversation-anchor'])
            self.assertTrue(run.call_args.kwargs['conversation_anchor'])
        for kwargs in ({'conversation_anchor':1}, {'conversation_anchor':True,'idle_anchor':True}):
            with self.assertRaises(ValueError):launcher.run('/unused',['fresh'],**kwargs)

    def test_environment_reject_and_cleanup(self):
        for name in (worker.ANCHOR_ENV, worker.CONVERSATION_ANCHOR_ENV):
            with patch.dict(os.environ,{name:'unexpected'}):
                with self.assertRaises(ValueError):worker.run_session(None,None,spec(),None,None,0)
                with self.assertRaises(ValueError):launcher.run('/unused',['fresh'],conversation_anchor=True)
                self.assertEqual(os.environ[name],'unexpected')

    def test_candidate_lifecycle_cleanup(self):
        import ast
        import inspect
        from unittest.mock import Mock
        tree=ast.parse(inspect.getsource(worker.run_session)); fn=tree.body[0]
        start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.ImportFrom) and n.module=='research.hf_h3_direct_conversation' and any(a.name=='ANCHOR_POLICY' for a in n.names))
        fn.name='lifecycle'; fn.args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[ast.arg(arg='deadline')],kw_defaults=[ast.parse("time.time()+3600",mode='eval').body],defaults=[]);fn.body=[ast.parse("rtc=None\nordered_temporary=None\noutput=Path('/unused')\nroot=Path('/unused')").body[0:]] and ast.parse("rtc=None\nordered_temporary=None\noutput=Path('/unused')\nroot=Path('/unused')\ntts=None").body+fn.body[start:]
        code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<lifecycle>','exec')
        for close_fails in (False,True):
            with patch.dict(os.environ,{},clear=True):
                seen=[]; runtime=Mock()
                def start_runtime():
                    seen.append(os.environ.get(worker.CONVERSATION_ANCHOR_ENV));raise RuntimeError('start')
                def close():
                    seen.append(os.environ.get(worker.CONVERSATION_ANCHOR_ENV))
                    if close_fails:raise RuntimeError('close')
                runtime.start.side_effect=start_runtime;runtime.close.side_effect=close
                scope=dict(row={'anchor_policy':worker.CONVERSATION_ANCHOR_POLICY},os=os,Path=Path,time=__import__('time'),native=False,server=Mock(),expires=0,ANCHOR_ENV=worker.ANCHOR_ENV,JobAVServer=Mock(),LiveAVBinding=Mock(),RTCGateway=Mock(),RelayConfig=Mock(),LiveRTCSidecar=Mock(),NATIVE_TORCH='x',NATIVE_TORCHAUDIO='y',REFERENCE_SHA256='z',ResamplingContext=Mock(),config_preroll=lambda *a,**kw:{},config_sha256=lambda *a,**kw:'c',role_profile_digest=lambda *a,**kw:'r',json=__import__('json'),run_worker=Mock(),on_ready=None,runtime=runtime,tts_future=Mock(),close_tts_future=Mock())
                exec(code,scope)
                with self.assertRaises(RuntimeError):scope['lifecycle']()
                self.assertEqual(seen,[worker.CONVERSATION_ANCHOR_SCHEMA]*2)
                self.assertNotIn(worker.CONVERSATION_ANCHOR_ENV,os.environ)
                self.assertNotIn(worker.ANCHOR_ENV,os.environ)

    def test_actual_both_rank_immutable_global_selection(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            backend=object.__new__(LiveNativeBackend);backend.runtime=SimpleNamespace(coverage=Path(d))
            backend._visual_policy=worker.CONVERSATION_ANCHOR_POLICY;backend._visual_anchor=None
            def write(seq,alter=None):
                backend._sequence=seq
                e=dict(policy=backend._visual_policy,sequence=seq,kind='unconditioned' if seq==0 else 'anchor' if seq%3==0 else 'latest',anchor_sha256=None if seq==0 else 'a'*64,selected_sha256=None if seq==0 else ('a'*64 if seq in (1,3,6) else 'b'*64),anchor_sequence=None if seq==0 else 1)
                m=dict(session_id='test',streaming_audio={'sequence':seq})
                for rank in (0,1):
                    r=dict(native_window_id='test',streaming=m['streaming_audio'],guide_digests=[e['selected_sha256'],None,None],visual_ancestry_evidence=dict(e))
                    if rank==1 and alter:alter(r)
                    (Path(d)/f'streaming-rank-{rank}-window-{seq}.json').write_text(json.dumps(r))
                return m
            for seq in range(7):backend._read_visual_ancestry(write(seq))
            for alter in (lambda r:r.pop('visual_ancestry_evidence'), lambda r:r['visual_ancestry_evidence'].update(policy='wrong'),lambda r:r['streaming'].update(sequence=9),lambda r:r['visual_ancestry_evidence'].update(anchor_sha256='c'*64)):
                with self.assertRaises(ValueError):backend._read_visual_ancestry(write(6,alter))


if __name__=='__main__':unittest.main()
