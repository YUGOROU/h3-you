import io
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from research.h3_personal import webrtc_sidecar as rtc
from research.h3_personal.paired_av import LiveAVBinding


class SidecarTests(unittest.TestCase):
    def owner(self, root, **kwargs):
        return rtc.LiveRTCSidecar(root, LiveAVBinding('session', 'a' * 64, 0),
                                  time.time() + 200, 'k' * 32, 'private-token',
                                  time.monotonic() + 180, **kwargs)

    def test_parent_has_no_rtc_imports(self):
        import ast
        tree = ast.parse(Path(rtc.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name in ('aiortc', 'av') for n in node.names))

    def test_installer_commands_are_bounded_and_secret_free(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as z:
            z.writestr('uv/uv', b'fake')
        body = data.getvalue()
        import hashlib
        with tempfile.TemporaryDirectory() as tmp, patch.object(rtc, 'UV_SHA256', hashlib.sha256(body).hexdigest()), \
                patch.object(rtc.urllib.request, 'urlopen', return_value=io.BytesIO(body)), \
                patch.object(rtc.subprocess, 'run') as run, \
                patch.dict(os.environ, HF_TOKEN='secret', H3_RTC_KEY='secret', OTHER_SECRET='secret'):
            deadline = time.monotonic() + 30
            python = rtc.bootstrap(Path(tmp), Path(tmp), deadline)
            self.assertEqual(python, Path(tmp) / 'venv/bin/python')
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                self.assertLessEqual(call.kwargs['timeout'], 30)
                self.assertNotIn('HF_TOKEN', call.kwargs['env'])
                self.assertNotIn('OTHER_SECRET', call.kwargs['env'])
            self.assertIn('3.12', run.call_args_list[0].args[0])
            self.assertIn('--require-hashes', run.call_args_list[2].args[0])

    def test_setup_cancellation_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.owner(root)
            with patch.object(rtc, 'bootstrap', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    owner.__enter__()
            self.assertIsNone(owner._scratch)
            self.assertIsNone(owner._secrets)
            owner.close()

    def test_workspace_in_project_refused(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.owner(root, workspace=root)
            with patch.object(rtc, 'bootstrap') as install:
                with self.assertRaises(ValueError):
                    owner.__enter__()
                install.assert_not_called()

    def test_fake_child_ready_and_redaction(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.owner(root)
            children = []
            class Process:
                alive = True
                def poll(self): return None if self.alive else 0
                def terminate(self): self.alive = False
                def wait(self, timeout): return 0
            def spawn(args, **kw):
                self.assertNotIn('private-token', repr(args))
                self.assertEqual(kw['env']['HF_TOKEN'], 'private-token')
                self.assertEqual(len(kw['pass_fds']), 1)
                config = Path(args[-2])
                self.assertEqual(config.stat().st_mode & 0o777, 0o600)
                child = socket.socket(fileno=os.dup(kw['pass_fds'][0]))
                children.append(child)
                child.sendall(rtc.wire._pack(dict(schema='h3-rtc-sidecar-ready-v1',
                    session='session', boot='b' * 32, port=7861, expires_at=owner.expires_at)))
                return Process()
            with patch.object(rtc, 'bootstrap', return_value=Path('/fake/python')), \
                    patch.object(rtc.subprocess, 'Popen', side_effect=spawn):
                try:
                    with owner as running:
                        self.assertEqual(running.boot, 'b' * 32)
                        self.assertIsNone(running._child)
                        self.assertIsNone(running._secrets)
                    owner.close()
                    self.assertIsNone(owner.media_adapter)
                    self.assertIsNone(owner.process)
                finally:
                    for child in children: child.close()

    def test_bad_startup_reaps_and_closes_both_fds(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.owner(root)
            from unittest.mock import Mock
            process = Mock()
            process.poll.side_effect = [None, 0]
            with patch.object(rtc, 'bootstrap', return_value=Path('/fake/python')), \
                    patch.object(rtc.subprocess, 'Popen', return_value=process), \
                    patch.object(rtc.wire, '_json', return_value={}):
                with self.assertRaises(ValueError): owner.__enter__()
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
            self.assertIsNone(owner.media_adapter)
            self.assertIsNone(owner._child)
            self.assertIsNone(owner._scratch)

    def test_kill_after_terminate_timeout(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as root:
            owner = self.owner(root)
            process = Mock()
            process.poll.side_effect = [None, 0]
            process.wait.side_effect = [subprocess.TimeoutExpired('child', 5), 0]
            owner.process = process
            owner.close()
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_args.kwargs['timeout'], 3)


if __name__ == '__main__':
    unittest.main()
