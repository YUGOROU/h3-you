"""Device/network-free tests for the explicit CPU preview scaffold."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal.core_audio_output import output_status
from research import preview_h3_browser as preview


class PreviewTests(unittest.TestCase):
    def test_offline_has_no_side_effects(self):
        with patch.object(preview, 'serve', side_effect=AssertionError), \
             patch.object(preview, 'output_status', side_effect=AssertionError), \
             patch.object(preview.subprocess, 'Popen', side_effect=AssertionError), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(preview.main([]), 0)
        self.assertEqual(json.loads(out.getvalue())['mode'], 'offline_plan')

    def test_status_contract(self):
        for value, reason in [(True, 'verified'), (False, 'not_builtin_speaker')]:
            self.assertEqual(output_status(query=lambda: value, platform='darwin'),
                             dict(default_is_builtin_speaker=value, reason=reason))
        def fail():
            raise RuntimeError('secret unknown UID')
        self.assertEqual(output_status(query=fail, platform='darwin')['reason'], 'query_failed')
        self.assertEqual(output_status(query=fail, platform='linux')['reason'], 'unsupported_platform')

    def test_encoder_is_paced_silent_and_finite(self):
        args = preview.encoder_arguments('/private/media')
        self.assertEqual(args.count('-re'), 2)
        self.assertIn('anullsrc=r=48000:cl=mono', args)
        self.assertIn('testsrc2=size=252x448:rate=24', args)
        self.assertEqual(args[args.index('-t') + 1], '8')
        self.assertIn('fmp4', args)
        self.assertNotIn('-hls_flags', args)

    def test_output_denial_never_starts_server_or_encoder(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent).resolve() / 'fresh'
            with self.assertRaises(RuntimeError):
                preview.serve(root, status=lambda: {'default_is_builtin_speaker': False},
                              server_factory=lambda *a, **k: self.fail('server started'),
                              popen=lambda *a, **k: self.fail('encoder started'))
            self.assertEqual(json.loads((root / 'preview-result.json').read_text())['outcome'], 'failed')
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)

    def test_synthetic_lifecycle_with_mock_server_and_encoder(self):
        import threading
        order = []
        class Server:
            def __init__(self, root, status, **kwargs):
                self.media_root = root / 'media'
                self.client_ready = threading.Event()
                self.client_ready.set()
                self.playback_done = threading.Event()
                self.playback_done.set()
                self.stream_state = 'waiting'
                self.url = 'http://127.0.0.1:1234/'
                self.root = root
                self.kwargs = kwargs
            def __enter__(self):
                order.append('server')
                return self
            def __exit__(self, *args):
                order.append('closed')
            def validate_output(self):
                pass
            def finish(self, state):
                order.append(state)
                self.stream_state = state
        class Process:
            returncode = 0
            def poll(self):
                return 0
            def wait(self, **kwargs):
                order.append('reaped')
        def spawn(args, **kwargs):
            order.append('encoder')
            Path(args[-1]).write_bytes(b'#EXTM3U\n#EXT-X-ENDLIST\n')
            return Process()
        with tempfile.TemporaryDirectory() as parent, contextlib.redirect_stdout(io.StringIO()):
            result = preview.serve(Path(parent).resolve() / 'fresh',
                status=lambda: {'default_is_builtin_speaker': True},
                server_factory=Server, popen=spawn)
        self.assertEqual(result['outcome'], 'ended')
        self.assertEqual(order[:3], ['server', 'streaming', 'encoder'])
        self.assertLess(order.index('reaped'), order.index('closed'))
        self.assertLess(order.index('encoder'), order.index('eos'))

    def test_fresh_directory_required(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                preview._private_root(Path(root).resolve())

    def test_repo_directory_rejected(self):
        with self.assertRaises(ValueError):
            preview._private_root(Path(__file__).resolve().parents[1] / 'preview-private')


if __name__ == '__main__':
    unittest.main()
