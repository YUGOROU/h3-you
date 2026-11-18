"""CPU-only opt-in contracts; no cloud or device operations."""
import ast
import inspect
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from research import hf_h3_direct_conversation as worker
from research import launch_h3_direct_conversation as launcher
from test_h3_direct_conversation_spec import spec, asr_binding


class IdleAnchorTests(unittest.TestCase):
    def test_optional_combinations_and_unknown_rejection(self):
        for anchor in (False, True):
            for asr in (False, True):
                item = spec()
                row = item['live_direct_session']
                if anchor:
                    row['anchor_policy'] = worker.ANCHOR_POLICY
                if asr:
                    row['expected_asr'] = asr_binding()
                self.assertIs(worker.validate_spec(item), row)
        for value in (None, False, 3, [], {}, '', 'other'):
            item = spec(); item['live_direct_session']['anchor_policy'] = value
            with self.assertRaises(ValueError):
                worker.validate_spec(item)
        for key in ('unexpected', 'anchor_policy'):
            item = spec()
            if key == 'anchor_policy':
                item[key] = worker.ANCHOR_POLICY
            else:
                item['live_direct_session'][key] = True
            with self.assertRaises(ValueError): worker.validate_spec(item)
        item = spec(); item['live_direct_session']['anchor_policy'] = worker.ANCHOR_POLICY
        item['live_relay_session'] = {}
        with self.assertRaises(ValueError): worker.validate_spec(item)

    def test_arguments(self):
        with self.assertRaises(SystemExit):
            launcher.main(['--directory', '/unused', '--idle-anchor'])
        with patch.object(launcher, 'run') as run, patch('builtins.input', return_value='fresh'), patch('builtins.print'):
            launcher.main(['--directory', '/unused', '--run', '--idle-anchor'])
            self.assertIs(run.call_args.kwargs['idle_anchor'], True)
            launcher.main(['--directory', '/unused', '--run'])
            self.assertNotIn('idle_anchor', run.call_args.kwargs)
        with self.assertRaises(ValueError): launcher.run('/unused', ['fresh'], idle_anchor=1)

    def test_preexisting_and_malformed_environment_rejected_before_imports(self):
        for value in ('', 'other', worker.ANCHOR_POLICY):
            with patch.dict(os.environ, {worker.ANCHOR_ENV: value}):
                with self.assertRaises(ValueError): worker.run_session(None, None, spec(), None, None, 0)
                with self.assertRaises(ValueError): launcher.run('/unused', ['fresh'])
                self.assertEqual(os.environ[worker.ANCHOR_ENV], value)
        item = spec(); item['live_direct_session']['anchor_policy'] = 'bad'
        with self.assertRaises(ValueError): worker.run_session(None, None, item, None, None, 0)

    def test_real_worker_scope_restored_on_start_and_cleanup_failures(self):
        # Execute the actual lifecycle tail, excluding heavyweight setup imports.
        tree = ast.parse(inspect.getsource(worker.run_session))
        start = next(i for i, node in enumerate(tree.body[0].body)
                     if isinstance(node, ast.ImportFrom) and node.module == 'research.hf_h3_direct_conversation' and any(a.name == 'ANCHOR_POLICY' for a in node.names))
        function = tree.body[0]
        function.name = 'lifecycle'
        function.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[ast.arg(arg='deadline')], kw_defaults=[ast.parse("time.time()+3600", mode='eval').body], defaults=[])
        function.body = ast.parse("rtc = None\nordered_temporary = None\noutput = Path('/unused')\nroot = Path('/unused')\ntts = None").body + tree.body[0].body[start:]
        code = compile(ast.fix_missing_locations(ast.Module(body=[function,
            ast.Expr(value=ast.Call(func=ast.Name(id='lifecycle', ctx=ast.Load()), args=[], keywords=[]))], type_ignores=[])), '<lifecycle>', 'exec')
        for anchor in (False, True):
            for close_fails in (False, True):
                with patch.dict(os.environ, {}, clear=True):
                    row = {'anchor_policy': worker.ANCHOR_POLICY} if anchor else {}
                    seen = []
                    def check():
                        seen.append(os.environ.get(worker.ANCHOR_ENV))
                    runtime = Mock()
                    runtime.start.side_effect = lambda: (check(), (_ for _ in ()).throw(RuntimeError('start')))
                    def close():
                        check()
                        if close_fails: raise RuntimeError('close')
                    runtime.close.side_effect = close
                    cleanup = Mock()
                    with self.assertRaises(RuntimeError):
                        exec(code, dict(row=row, os=os, Path=Path, time=__import__('time'), native=False, server=Mock(), expires=0, ANCHOR_ENV=worker.ANCHOR_ENV,
                            JobAVServer=Mock(), LiveAVBinding=Mock(), RTCGateway=Mock(), RelayConfig=Mock(), LiveRTCSidecar=Mock(),
                            NATIVE_TORCH='x', NATIVE_TORCHAUDIO='y', REFERENCE_SHA256='z', ResamplingContext=Mock(), json=__import__('json'),
                            config_preroll=lambda *a, **kw: {}, config_sha256=lambda *a, **kw: 'c', role_profile_digest=lambda *a, **kw: 'r',
                            run_worker=Mock(), on_ready=None, runtime=runtime, tts_future=Mock(), close_tts_future=cleanup))
                    self.assertEqual(seen, [worker.ANCHOR_POLICY if anchor else None] * 2)
                    self.assertNotIn(worker.ANCHOR_ENV, os.environ)
                    cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
