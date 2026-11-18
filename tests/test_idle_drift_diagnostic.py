"""CPU-only orchestration acceptance: no model/cloud/devices or extra deps."""
import hashlib
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from research.h3_personal.idle_drift_diagnostic import run_diagnostic
from research.h3_personal.streaming_native_stages import validate_manifest


class Backend:
    instances = []

    def __init__(self, runtime, root, **kwargs):
        root.mkdir()
        self._retired = False
        self._id = None
        self.receipts = []
        self.calls = []
        self.abandoned = False
        self.instances.append(self)

    def __call__(self, manifest):
        c = validate_manifest(manifest)
        self._id = c['id']
        self.calls.append(c)
        end = 124 + c['sequence'] * 102
        emitted = end - (0 if c['conversation_final'] else 5)
        previous = 0 if not self.receipts else self.receipts[-1]['emitted']
        self.receipts.append(dict(role=c['role'], emitted=emitted))
        return dict(video=b'fake-video', count=emitted-previous,
                    committed_frame=end, emitted_frame=emitted,
                    conditioning_prefix_sha256=None, receipt_sha256='a'*64)

    def retire(self):
        assert self.calls[-1]['conversation_final']
        self._retired = True

    def abandon(self, **kwargs):
        self.abandoned = True
        self._retired = True

    def close_unused(self):
        self._retired = True


def capture(root, deadline, records):
    # Capture seam: orchestration test intentionally does not assert pixel quality.
    records.extend(dict(mae=0., max_error=0, native_sha256='a'*64,
                        encoded_sha256='b'*64) for _ in range(42))
    return None


class DiagnosticTests(unittest.TestCase):
    def test_full_native_roundtrip_and_partial_manifest(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy not installed')
        from research.h3_personal.idle_drift_diagnostic import _retain_native
        # Fixture only: numpy is a runtime dependency of _retain_native itself.
        frames = np.arange(5*8*10*3, dtype=np.uint8).reshape(5, 8, 10, 3)
        raw = frames.tobytes()
        entries = []
        def ffmpeg(args, **kwargs):
            if 'input' in kwargs:
                self.assertEqual(kwargs['input'], raw)
                self.assertIn('ffv1', args)
                self.assertIn('bgr0', args)
                self.assertEqual(args[args.index('-threads')+1], '1')
                Path(args[-1]).write_bytes(b'fake ffv1')
                return subprocess.CompletedProcess(args, 0, b'')
            return subprocess.CompletedProcess(args, 0, raw)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('research.h3_personal.idle_drift_diagnostic.subprocess.run', ffmpeg):
                _retain_native(root, time.monotonic()+10, frames, 0, entries)
            manifest = json.loads((root/'native-manifest.json').read_text())
            self.assertEqual(manifest[0]['shape'], [5, 8, 10, 3])
            self.assertEqual(manifest[0]['rgb_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest[0]['status'], 'verified')
            self.assertTrue((root/'segments/native-000.mkv').exists())
            def corrupt(args, **kwargs):
                result = ffmpeg(args, **kwargs)
                if 'input' not in kwargs:
                    result.stdout = raw[:-1]
                return result
            with patch('research.h3_personal.idle_drift_diagnostic.subprocess.run', corrupt):
                with self.assertRaisesRegex(ValueError, 'hash differs'):
                    _retain_native(root, time.monotonic()+10, frames, 1, entries)
            manifest = json.loads((root/'native-manifest.json').read_text())
            self.assertEqual([e['status'] for e in manifest], ['verified', 'failed'])

    def test_real_admission_final_fourteenth(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('research.h3_personal.live_native_backend.LiveNativeBackend', Backend), \
                patch('research.h3_personal.conversation_conditioning.role_profile_digest', return_value='c'*64), \
                patch('research.h3_personal.idle_drift_diagnostic._capture_export', capture):
            runtime = type('Runtime', (), {'quality_profile': object()})()
            report = run_diagnostic(runtime, Path(tmp)/'review', deadline=time.monotonic()+301)
            backend = Backend.instances[-1]
            self.assertEqual(len(backend.calls), 14)
            self.assertEqual([c['conversation_final'] for c in backend.calls], [False]*13+[True])
            self.assertTrue(backend._retired)
            self.assertEqual(report['emitted_frames'], 1450)
            self.assertEqual(report['final_packet_frames'], 107)
            self.assertEqual(report['native_samples'], 42)
            with self.assertRaises(FileExistsError):
                run_diagnostic(runtime, Path(tmp)/'review', deadline=time.monotonic()+301)

    def test_failure_abandons(self):
        class Broken(Backend):
            def __call__(self, manifest):
                super().__call__(manifest)
                raise RuntimeError('injected')
        with tempfile.TemporaryDirectory() as tmp, \
                patch('research.h3_personal.live_native_backend.LiveNativeBackend', Broken), \
                patch('research.h3_personal.conversation_conditioning.role_profile_digest', return_value='c'*64), \
                patch('research.h3_personal.idle_drift_diagnostic._capture_export', capture):
            runtime = type('Runtime', (), {'quality_profile': object()})()
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                run_diagnostic(runtime, Path(tmp)/'review', deadline=time.monotonic()+301)
            self.assertTrue(Backend.instances[-1].abandoned)


if __name__ == '__main__':
    unittest.main()
