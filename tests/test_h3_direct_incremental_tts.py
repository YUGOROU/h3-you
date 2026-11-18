"""CPU-only immutable sentence TTS selection; no native/provider startup."""
import ast
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import hf_h3_direct_conversation as worker
from research import launch_h3_direct_conversation as launcher
from test_h3_direct_conversation_spec import spec


class IncrementalTTSTests(unittest.TestCase):
    def test_spec_strict_and_default_unchanged(self):
        item = spec(); before = deepcopy(item)
        worker.validate_spec(item)
        self.assertEqual(item, before)
        self.assertNotIn('tts_mode', item['live_direct_session'])
        for mode in ('complete', 'sentence_incremental'):
            item['live_direct_session']['tts_mode'] = mode
            self.assertEqual(worker.validate_spec(item)['tts_mode'], mode)
        for mode in (None, True, 1, [], {}, '', 'sentence', 'COMPLETE'):
            item['live_direct_session']['tts_mode'] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                worker.validate_spec(item)
        item = spec(); item['live_direct_session']['unknown'] = True
        with self.assertRaises(ValueError): worker.validate_spec(item)
        for mode in ('complete', 'sentence_incremental', None):
            item = spec(); item['tts_mode'] = mode
            with self.assertRaises(ValueError): worker.validate_spec(item)

    def test_ambient_and_invalid_run_rejected_before_provider(self):
        with patch.object(launcher, 'canonical', side_effect=AssertionError('provider preflight')):
            for mode in (None, True, 1, {}, 'other'):
                with self.assertRaises(ValueError): launcher.run('/unused', ['fresh'], tts_mode=mode)
            with patch.dict(os.environ, {worker.TTS_MODE_ENV: 'sentence_incremental'}):
                with self.assertRaises(ValueError): launcher.run('/unused', ['fresh'])
                with self.assertRaises(ValueError): worker.run_session(None, None, spec(), None, None, 0)

    def test_cli_requires_run_and_forwards_optin(self):
        with patch.object(launcher, 'run') as run, patch('builtins.input', return_value='fresh'), patch('builtins.print'):
            with self.assertRaises(SystemExit) as error:
                launcher.main(['--directory', '/unused', '--sentence-incremental-tts'])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            launcher.main(['--directory', '/unused', '--run', '--sentence-incremental-tts'])
            self.assertEqual(run.call_args.kwargs['tts_mode'], 'sentence_incremental')
            launcher.main(['--directory', '/unused', '--run'])
            self.assertNotIn('tts_mode', run.call_args.kwargs)

    def test_launcher_immutable_row_and_ready_expectation(self):
        tree = ast.parse(inspect.getsource(launcher.run))
        suite = next(n for n in tree.body[0].body if isinstance(n, ast.Try)).body
        start = next(i for i, n in enumerate(suite) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'row' for t in n.targets))
        end = next(i for i, n in enumerate(suite) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ready' for t in n.targets))
        code = compile(ast.fix_missing_locations(ast.Module(body=suite[start:end+1], type_ignores=[])), '<launcher wiring>', 'exec')
        for mode in ('complete', 'sentence_incremental'):
            prepare = Mock(return_value='immutable-stage'); submit = Mock(return_value=('job', 'image', 'command')); ready = Mock()
            scope = dict(rtc=False, continuous_av=False, http_keepalive=False, overlap_send=False, tts_mode=mode, session='a'*32, windows=16, frames=1654, timeout=1200,
                headless_interrupt=False, visual_intents=False, review_av=False, h3_adoption=None, attended_fields={},ordered_metadata={},long_session=False,native_tts_cancel=False,native_cancel_trigger=None,ordered_references=None,confirm_private_reference_upload=False,stable_selfie=False,compute_profile=None,job_api=Mock(),job_token='fake',job_namespace='YUGOROU',
                five_minute_test=False, ten_minute_test=False, idle_anchor=False, conversation_anchor=False,
                source=None, directory='/unused', api=object(), token='fake', pk='fake', ck='fake',
                
                time=time, prepare=prepare, submit=submit, await_ready=ready)
            exec(code, scope)
            row = prepare.call_args.args[1]
            worker.validate_spec(dict(spec(), live_direct_session=row))
            self.assertEqual('tts_mode' in row, mode != 'complete')
            self.assertEqual(row.get('tts_mode', 'complete'), mode)
            self.assertEqual(ready.call_args.kwargs.get('tts_mode', 'complete'), mode)
            self.assertEqual(submit.call_args.args[1], 'immutable-stage')

    def test_real_session_wiring_and_ready_callback(self):
        # Execute the actual run_session wiring block with a fake worker, without
        # importing/starting native runtime. Includes every intervening option.
        tree = ast.parse(inspect.getsource(worker.run_session))
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.With))
        body = block.body
        body = [n if not isinstance(n, ast.Return) else ast.Expr(n.value) for n in body]
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<session wiring>', 'exec')
        for mode in ('complete', 'sentence_incremental'):
            row = spec()['live_direct_session']
            if mode != 'complete': row['tts_mode'] = mode
            binding = SimpleNamespace(session=row['session'], profile_sha256='b'*64, epoch=1, max_segments=16, max_total_frames=1654)
            fake = Mock(side_effect=lambda *a, **kw: kw['on_ready']())
            scope = dict(native=False, rtc=None, row=row, binding=binding, expires=time.time()+600, json=json,
                ResamplingContext=Mock(), NATIVE_TORCH='x', NATIVE_TORCHAUDIO='y', REFERENCE_SHA256='z',
                CONVERSATION_ANCHOR_POLICY=worker.CONVERSATION_ANCHOR_POLICY,
                run_worker=fake, runtime=object(), tts=object(), server=Mock(), output=Path('/unused'), deadline=999)
            with patch('builtins.print') as emit:
                exec(code, scope)
            ready = json.loads(emit.call_args.args[0])
            launcher.validate_ready(ready, row['session'], tts_mode=mode)
            self.assertEqual(fake.call_args.kwargs.get('tts_mode', 'complete'), mode)
            self.assertEqual('tts_mode' in fake.call_args.kwargs, mode != 'complete')
            self.assertEqual('tts_mode' in ready, mode != 'complete')
            opposite = 'complete' if mode != 'complete' else 'sentence_incremental'
            with self.assertRaises(ValueError): launcher.validate_ready(ready, row['session'], tts_mode=opposite)
            api = Mock(); api.fetch_job_logs.return_value = [json.dumps(ready)]
            job = SimpleNamespace(id='fake')
            httpx = SimpleNamespace(NetworkError=ConnectionError, TimeoutException=TimeoutError, RemoteProtocolError=BrokenPipeError)
            with patch.dict('sys.modules', httpx=httpx):
                self.assertEqual(launcher.await_ready(api, job, row['session'], time.monotonic()+1, tts_mode=mode), ready)
                with self.assertRaises(ValueError):
                    launcher.await_ready(api, job, row['session'], time.monotonic()+1, tts_mode=opposite)
            api.inspect_job.assert_not_called()


if __name__ == '__main__':
    unittest.main()
