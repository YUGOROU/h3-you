"""Offline controller contract tests: never contact HF or devices."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research import launch_h3_relay_conversation as launch


class LaunchTests(unittest.TestCase):
    def test_offline_main(self):
        with patch.object(launch, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            launch.main(['--directory','/tmp/not-created-live-relay'])
        run.assert_not_called()

    def test_inputs_before_network(self):
        for turns in ([], [''], ['x'*2001], ['a','b','c']):
            with self.assertRaises(ValueError):
                launch.run('/tmp/not-created-live-relay', turns)

    def test_endpoint_override(self):
        with patch.dict(os.environ, {'HF_ENDPOINT':'https://example.com'}):
            with self.assertRaises(ValueError):
                launch.canonical()

    def test_obs_approval(self):
        with self.assertRaises(PermissionError):
            launch.run('/tmp/not-created-live-relay',['hello'],obs_port=1234)

    def test_ready_only(self):
        import time
        row = dict(type='h3_relay_ready',status='ready',session='a'*32,
                   profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654,
                   expires_at=time.time()+100,origin='fresh_typed_text',asr_used=False,tts_streaming=False)
        api = NS(fetch_job_logs=lambda **kw: iter(['private unrelated log',json.dumps(row)]))
        self.assertEqual(launch.await_ready(api,NS(id='job'),'a'*32,time.monotonic()+1),row)
        row['extra'] = 'not scalar allowlist'
        with self.assertRaises(ValueError):
            launch.await_ready(api,NS(id='job'),'a'*32,time.monotonic()+1)

    def test_full_current_sources_and_manifest(self):
        # A fake shared preparer stages only the files needed by this contract;
        # all current modules must overwrite it, including dynamic imports.
        import research.prepare_h3_fast_ref2va_gpu as base
        import research.launch_h3_live_voice as voice
        import research.hf_h3_relay_conversation as worker
        def fake(stage, *args, **kwargs):
            assert kwargs['locked_speech'] is None
            assert kwargs['timeout_seconds'] == 1200
            stage.mkdir()
            (stage/'spec.json').write_text('{}')
            (stage/'upload-allowlist.json').write_text('{"schema":"test","files":[]}')
        with tempfile.TemporaryDirectory() as tmp:
            private=Path(tmp)/'private'; private.mkdir()
            source=private/'gpu-hf-irodori-diagnostic-v1'; source.mkdir()
            manifest={k:[] for k in ('source_revision','uv_lock_sha256','uv_version','models','license')}
            body=json.dumps(manifest).encode(); (source/'request.json').write_bytes(body)
            real_hash=launch.hashlib.sha256
            def hashed(data=b''):
                if data == body:
                    return NS(hexdigest=lambda:'b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002')
                return real_hash(data)
            with patch.object(base,'prepare',fake), patch.object(launch,'PRIVATE',private), patch.object(launch.hashlib,'sha256',hashed), patch.object(voice,'inspect'), patch.object(worker,'validate_spec'):
                stage=launch.prepare(Path(tmp)/'gpu',{'session':'a'*32},None)
            for name in ('live_relay_worker','live_native_backend','streaming_windows'):
                self.assertEqual((stage/f'research/h3_personal/{name}.py').read_bytes(),(launch.ROOT/f'research/h3_personal/{name}.py').read_bytes())
            spec=json.loads((stage/'spec.json').read_text())
            self.assertEqual(spec['live_voice_session']['max_turns'],2)
            self.assertEqual(json.loads((stage/'tts-manifest.json').read_text()),manifest)


if __name__ == '__main__':
    unittest.main()
