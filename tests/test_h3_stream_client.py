"""Mocked opt-in live runner; no real capture, inference, upload, playback or GPU."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from research import run_h3_live_voice as live


class StreamClientTests(unittest.TestCase):
    def test_stream_endpoint_reaches_backend_without_forging_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'deployment').mkdir()
            (root/'deployment/spec.json').write_text(json.dumps(dict(
                live_voice_session=dict(id='session', max_turns=16),
                publication_repo='private/repo', asr_policy='application-stream-v1',
                config=dict(timeout_seconds=1500))))
            (root/'submission.json').write_text(json.dumps(dict(job_id='job')))
            obs = root/'obs.json'
            obs.write_text(json.dumps(dict(pid=123, host='127.0.0.1', port=59513,
                                           audio_monitor_device='BlackHole2ch_UID')))
            data = b'unit-test-generated-media'
            media = root/'remote.mp4'
            media.write_bytes(data)
            state = root/'state.json'
            state.write_text(json.dumps(dict(session_id='session', status='ready', expires_at=10**12)))
            result = root/'result.json'
            result.write_text(json.dumps(dict(session_id='session', sequence=1,
                reply_sha256=hashlib.sha256('reply'.encode()).hexdigest(), previous_audio_reused=False,
                files={'live/turn-001/review.mp4': hashlib.sha256(data).hexdigest()}, generated_seconds=1)))
            uploads = []
            class API:
                def repo_info(self, *a, **k):
                    return types.SimpleNamespace(private=True, sha='revision')
                def list_repo_files(self, *a, **k):
                    return ['live/state.json', 'live/turn-001/result.json']
                def inspect_job(self, **k):
                    return types.SimpleNamespace(status=types.SimpleNamespace(stage='RUNNING'))
                def create_commit(self, *a, **k):
                    uploads.append(json.loads(k['operations'][0].path_or_fileobj.getvalue()))
            hub = types.ModuleType('huggingface_hub')
            hub.HfApi = lambda **k: API()
            hub.CommitOperationAdd = lambda **k: types.SimpleNamespace(**k)
            hub.hf_hub_download = lambda repo, name, **k: str(state if name.endswith('state.json') else result if name.endswith('result.json') else media)
            class Stream:
                def __init__(self, propose):
                    self.propose = propose
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass
                def capture(self, command):
                    self.command = command
                    return dict(type='utterance_endpoint', text='hello', revision=2,
                                on_device=True, final=False, capture_stopped=True,
                                endpoint_source='application_stable_partial'), self.propose('hello')
            def run(args, **kwargs):
                if args[0] == 'python3':
                    return types.SimpleNamespace(returncode=0, stdout=json.dumps(dict(
                        type='turn', model='Qwen/Qwen3.8-27B:cerebras', reply='reply')))
                self.assertEqual(args[0], 'ffmpeg')
                self.assertEqual(len(uploads), 1)  # never play before validated upload/output
                return types.SimpleNamespace(returncode=0)
            with patch.dict(sys.modules, huggingface_hub=hub), \
                 patch('research.provision_modal_identity_secret.cached_token', return_value='test-placeholder'), \
                 patch('research.h3_personal.streaming_asr.StreamingASR', Stream), \
                 patch.object(live.subprocess, 'run', side_effect=run), \
                 patch.object(live.subprocess, 'check_output', return_value='123\n'), \
                 patch.object(live.time, 'sleep'), \
                 patch.object(sys, 'argv', ['run', '--directory', str(root), '--turn', '--obs-receipt', str(obs)]):
                live.main()
            self.assertEqual(len(uploads), 1)
            self.assertIs(uploads[0]['asr_final'], False)
            self.assertIs(uploads[0]['asr_stopped'], True)
            self.assertEqual(uploads[0]['asr_endpoint_source'], 'application_stable_partial')
            self.assertEqual(uploads[0]['asr_revision'], 2)
            receipt = json.loads((root/'turn-001/delivered.json').read_text())
            self.assertIs(receipt['apple_final'], False)
            self.assertEqual(receipt['asr_policy'], 'application-stream-v1')


if __name__ == '__main__':
    unittest.main()
