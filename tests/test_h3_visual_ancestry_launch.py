"""CPU-only ancestry launcher checks; no provider calls."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch
import wave

from research import launch_h3_visual_ancestry_diagnostic as launch


def wav_bytes(rate=48000, channels=1, width=2, samples=48):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setparams((channels, width, rate, samples, 'NONE', 'not compressed'))
        wav.writeframes(b'\0' * samples * channels * width)
    return out.getvalue()


class LaunchTests(unittest.TestCase):
    def test_pcm_digest_and_invalid_speech(self):
        body = wav_bytes()
        row = launch.speech_metadata(body)
        self.assertEqual(row['samples'], 48)
        self.assertEqual(row['pcm_sha256'], hashlib.sha256(b'\0'*96).hexdigest())
        for body in (b'bad', wav_bytes(rate=44100), wav_bytes(channels=2),
                     wav_bytes(width=1), wav_bytes(samples=0),
                     wav_bytes(samples=1440001), wav_bytes()[:-2]):
            with self.subTest(size=len(body)), self.assertRaises(ValueError):
                launch.speech_metadata(body)

    def test_inspection_section_and_speech_binding(self):
        worker = ModuleType(launch.WORKER)
        worker.validate_spec = lambda spec: None
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            (stage/'diagnostic-speech.wav').write_bytes(wav_bytes())
            spec = dict(visual_ancestry_comparison=dict(launch.SECTION),
                        diagnostic_speech=launch.speech_metadata(wav_bytes()),
                        publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-ancestry-'+'a'*32,
                        config={'timeout_seconds':1200}, locked_speech=None,
                        attention_mode='fa4-packed', layout='portrait',
                        runtime_policy='resident-visual-cached')
            name = launch.WORKER.replace('.', '/')+'.py'
            (stage/name).parent.mkdir(parents=True)
            (stage/name).write_text('pass\n')
            inventory = {'files':[{'path':name}, {'path':'diagnostic-speech.wav'}]}
            with patch.dict(sys.modules, {launch.WORKER:worker}), patch(
                    'research.launch_h3_fast_ref2va.inspect', return_value=(spec, inventory)):
                launch.inspect(stage)
                spec['visual_ancestry_comparison'] = {}
                with self.assertRaises(ValueError): launch.inspect(stage)
                spec['visual_ancestry_comparison'] = dict(launch.SECTION)
                spec['diagnostic_speech']['wav_sha256'] = '0'*64
                with self.assertRaises(ValueError): launch.inspect(stage)

    def test_submit_uses_local_inspect_before_api(self):
        class FakeAPI:
            def whoami(self):
                raise AssertionError('must not reach API')
        original = launch.idle.inspect
        with patch.object(launch.idle, 'canonical'), patch.object(
                launch, 'inspect', side_effect=ValueError('tampered')):
            # Lazy dependency import must remain offline even without the SDK.
            hub = ModuleType('huggingface_hub')
            hub.CommitOperationAdd = object
            with patch.dict(sys.modules, {'huggingface_hub':hub, 'httpx':ModuleType('httpx')}):
                with self.assertRaisesRegex(ValueError, 'tampered'):
                    launch.submit(FakeAPI(), Path('/unused'), 'token')
        self.assertIs(launch.idle.inspect, original)

    def test_single_post_unknown_outcome_never_replayed(self):
        class FakeClient:
            calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                return SimpleNamespace(status_code=201, json=lambda: {'id':'invalid'})
        client = FakeClient()
        payload = dict(flavor='h200x2', timeoutSeconds=1200, attempts=1)
        with self.assertRaisesRegex(RuntimeError, 'without replay'):
            launch.idle.submit_job(client, 'token', payload, lambda job: None)
        self.assertEqual(client.calls, 1)
        payload['expose'] = []
        with self.assertRaises(ValueError):
            launch.idle.submit_job(client, 'token', payload, lambda job: None)
        self.assertEqual(client.calls, 1)


if __name__ == '__main__':
    unittest.main()
