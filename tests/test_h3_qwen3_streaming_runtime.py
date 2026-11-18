import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from research.h3_personal import qwen3_streaming_runtime as r


class RuntimeTests(unittest.TestCase):
    def test_explicit_source_and_precision_gate(self):
        with self.assertRaises(RuntimeError):
            r.command('/tmp', '/tmp', accepted_revision='7d224', accepted_bf16=True)
        with self.assertRaises(RuntimeError):
            r.command('/tmp', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=False)

    def test_actual_cli(self):
        argv = r.command('/tmp', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=True)
        self.assertEqual(argv[:2], ['vllm', 'serve'])
        self.assertIn('--omni', argv)
        self.assertEqual(argv[argv.index('--host') + 1], '127.0.0.1')
        self.assertNotIn('--quantization', argv)

    def test_ports_and_paths(self):
        for p in (0, True, 65536):
            with self.assertRaises(ValueError):
                r.command('/tmp', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=True, port=p)
        with self.assertRaises(ValueError):
            r.command('relative', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=True)

    def test_local_reference_directory_is_explicit_and_not_root(self):
        with tempfile.TemporaryDirectory() as d:
            argv=r.command('/tmp','/tmp',accepted_revision=r.RELEASE_REVISION,accepted_bf16=True,reference_directory=d)
            self.assertEqual(argv[argv.index('--allowed-local-media-path')+1],d)
        with self.assertRaises(ValueError):
            r.command('/tmp','/tmp',accepted_revision=r.RELEASE_REVISION,accepted_bf16=True,reference_directory='/tmp')

    def test_offline_full_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / r.MODEL_REVISION
            (p / 'speech_tokenizer').mkdir(parents=True)
            for name in ('config.json', 'model.safetensors', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors'):
                (p / name).write_bytes(b'x')
            download = Mock(return_value=str(p))
            self.assertEqual(r.stage_model(snapshot_download=download), p.resolve())
            download.assert_called_once_with(repo_id=r.MODEL_ID, revision=r.MODEL_REVISION, local_files_only=True)
            (p / 'speech_tokenizer/model.safetensors').unlink()
            with self.assertRaises(RuntimeError):
                r.stage_model(snapshot_download=download)

    def test_version_and_source_gate(self):
        with self.assertRaises(RuntimeError):
            r.verify_install('/tmp', version=lambda _: '0.28.0')
        with tempfile.TemporaryDirectory() as d:
            with patch.object(r, 'SOURCE_HASHES', {'test.py': hashlib.sha256(b'good').hexdigest()}):
                (Path(d) / 'test.py').write_bytes(b'bad')
                with self.assertRaises(RuntimeError):
                    r.verify_install(d, version=lambda _: '0.30.0')
                (Path(d) / 'test.py').write_bytes(b'good')
                r.verify_install(d, version=lambda _: '0.30.0')

    def test_reap_kills_workers_even_when_leader_exited(self):
        p = Mock(pid=123)
        with patch.object(r.os, 'killpg') as kill:
            r._reap(p)
        self.assertEqual(kill.call_count, 2)
        self.assertEqual(p.wait.call_count, 2)

    def test_reap_escalates(self):
        p = Mock(pid=123)
        p.wait.side_effect = [subprocess.TimeoutExpired('x', 3), 0]
        with patch.object(r.os, 'killpg') as kill:
            r._reap(p)
        self.assertEqual(kill.call_args.args[1], r.signal.SIGKILL)

    def test_deadlines_before_launch(self):
        for lifetime in (float('inf'), 0, 901):
            with self.assertRaises(ValueError), patch.object(r.subprocess, 'Popen') as launch:
                with r.server('/tmp', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=True, lifetime=lifetime):
                    pass
            launch.assert_not_called()

    def test_exit_before_ready_reaped(self):
        p = Mock(pid=123)
        p.poll.return_value = 1
        with patch.object(r.subprocess, 'Popen', return_value=p), patch.object(r, '_reap') as reap:
            with self.assertRaises(RuntimeError):
                with r.server('/tmp', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=True):
                    self.fail('not ready')
            reap.assert_called_once_with(p)

    def test_ready_and_caller_failure_reaped(self):
        p = Mock(pid=123)
        p.poll.return_value = None
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(r.subprocess, 'Popen', return_value=p) as launch, patch.object(r, '_reap') as reap, patch.object(r.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaisesRegex(RuntimeError, 'caller'):
                with r.server('/tmp', '/tmp', accepted_revision=r.RELEASE_REVISION, accepted_bf16=True):
                    raise RuntimeError('caller')
            reap.assert_called_once_with(p)
            self.assertEqual(launch.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertTrue(launch.call_args.kwargs['start_new_session'])


if __name__ == '__main__':
    unittest.main()
