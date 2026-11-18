"""Runtime-level finite conversation scenario with a stubbed native generator.

The numeric numpy stack step is injected; every other step (manifest order,
both-rank receipts, role switch, sequence, AV sink, publication, abandon) is the
real production code. No GPU, no model reload, no device access.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import wave
from unittest.mock import patch

from research.h3_personal.accepted_profiles import AcceptedQualityProfile
from research.h3_personal.conversation_conditioning import SPEECH_PROMPT
from research.h3_personal.conversation_audio import audio_origin
from research.h3_personal.conversation_scenario import plan as scenario_plan
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime


RATE = 48000
PROFILE = AcceptedQualityProfile('reference1024', ('a' * 64,), 'b' * 64)


@dataclass
class _Config:
    prompt: str


class FakeFrames:
    def __init__(self, count):
        self.count = count
        self.ndim = 4
        self.shape = (count, 448, 256, 3)

    def __getitem__(self, key):
        return FakeExported(self.count)


class FakeExported:
    def __init__(self, count):
        self.count = count
        self.shape = (count, 448, 252, 3)
        self.flags = {'C_CONTIGUOUS': False}

    def copy(self, order):
        if order != 'C':
            raise ValueError('C-order export required')
        self.flags['C_CONTIGUOUS'] = True
        return self

    def __getitem__(self, key):
        return self

    def tobytes(self):
        return bytes(self.count * 448 * 252 * 3)


def write_wav(path, pcm):
    with wave.open(str(path), 'wb') as handle:
        handle.setparams((1, 2, RATE, 0, 'NONE', 'x'))
        handle.writeframes(pcm)
    return path


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class ConversationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.root.chmod(0o700)
        self.output = self.root / 'output'
        self.output.mkdir(mode=0o700)
        self.turn = self.output / 'turn'
        self.listening = (write_wav(self.root / 'l0.wav', bytes(9600)),
                          write_wav(self.root / 'l2.wav', bytes(9600)))
        self.speech_pcm = bytes((i * 13 + 7) % 251 for i in range(9600 * 2))
        self.speech = write_wav(self.root / 'speech.wav', self.speech_pcm)
        self.runtime = LiveVoiceRuntime(model=self.root / 'm', model_manifest=self.root / 'mm',
            identity=self.root / 'i', identity_config=self.root / 'ic', smd=self.root / 's',
            reference=self.root / 'r', archive=self.root / 'a', output=self.output,
            continuation=True, quality_profile=PROFILE, conversation_roles=True)
        self.runtime._state = 'ready'
        self.runtime.config = _Config(SPEECH_PROMPT)
        self.coverage = self.output / 'coverage'
        self.coverage.mkdir(mode=0o700)
        self.runtime.coverage = self.coverage
        self.runtime.continuation_file = self.coverage / 'window.json'
        self.runtime.continuation_file.write_text(json.dumps(dict(mode='independent')))
        self.runtime.quality_profile_file = self.coverage / 'profile.json'
        self.runtime.quality_profile_file.write_text('{}')
        self.runtime.current_role_file = self.coverage / 'role.json'
        self.runtime.current_role_file.write_text(json.dumps(dict(role='speech', prompt_sha256='b' * 64)))
        self.roles = []
        self.requests = 0
        self.abandons = []
        self.scenario = None
        self.fail_at = None

    def tearDown(self):
        self.tmp.cleanup()

    def scenario_for(self):
        if self.scenario is None:
            self.scenario = scenario_plan(conversation_id='conversation-test',
                                          profile_sha256=PROFILE.digest,
                                          listening_wavs=self.listening, speech_wav=self.speech)
        return self.scenario

    def generate(self, *, continuation=False):
        manifest = json.loads(self.runtime.continuation_file.read_text())
        if 'conversation_control' in manifest:
            self.abandons.append(manifest['conversation_control'])
            return None, (time.monotonic(), time.monotonic())
        self.assertNotEqual(manifest.get('mode'), 'independent', 'manifest not published')
        epoch = manifest['conversation']['source_epoch']
        index = manifest['index']
        scenario = self.scenario_for()
        self.requests += 1
        self.runtime._requests = self.requests
        if self.fail_at == (epoch, index):
            raise RuntimeError('native inference failed')
        source = scenario.sources[epoch]
        committed = source.committed_endpoints[index]
        for rank in (0, 1):
            (self.coverage / f'h3-conversation-rank-{rank}-epoch-{epoch}-window-{index}.json'
             ).write_text(json.dumps(dict(schema='h3-model-conversation-v1',
                conversation=manifest['conversation'], committed_frames=committed,
                source_id=source.source_id, canonical_audio_sha256='f' * 64,
                audio_origin=audio_origin(source.onset_frame, source.total_frames),
                guide_digests=[None, None, None] if epoch == 0 else ['a' * 64, 'b' * 64, 'c' * 64])))
        count = [e[2] for e in scenario.window_emissions()
                 if (e[0], e[1]) == (epoch, index)][0]
        begin = time.monotonic()
        return type('Result', (), {'frames': [None] * count})(), (begin, begin + 0.1)

    def switch(self, role):
        self.roles.append(role)

    def render(self):
        from research.h3_personal import live_voice_runtime as module
        self.runtime._generate = self.generate
        self.runtime._switch_role = self.switch
        original = module._ConversationBackend.__init__

        def init(backend, runtime, scenario, stack=None):
            original(backend, runtime, scenario, stack=lambda frames: FakeFrames(len(frames)))
        with patch.object(module._ConversationBackend, '__init__', init):
            return self.runtime.render_conversation_av(self.listening, self.speech,
                self.turn, self.output / 'continuous.nut', timeout=180, paced=False)

    def test_finite_scenario_end_to_end(self):
        receipt = self.render()
        self.assertEqual(receipt['frames'], 532)
        self.assertEqual([s['source_epoch'] for s in receipt['segments']], [0, 1, 1, 1, 2])
        self.assertEqual([s['frame_stop'] for s in receipt['segments']], [119, 221, 323, 425, 532])
        self.assertEqual(self.roles, ['listening', 'speech', 'listening'])
        self.assertEqual(self.abandons, [])
        published = Path(receipt['output'])
        self.assertEqual(published.stat().st_mode & 0o777, 0o400)
        self.assertEqual(published.name, 'continuous.nut')
        self.assertEqual(len(receipt['output_sha256']), 64)
        phases = [t['phase'] for t in receipt['sequence']['transitions']]
        self.assertEqual(phases[0], 'listening')
        self.assertIn('preparing_speech', phases)
        self.assertEqual(phases[-1], 'closed')
        self.assertEqual(json.loads(self.runtime.continuation_file.read_text()), dict(mode='independent'))
        self.assertEqual(self.runtime._state, 'ready')
        audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(published),
            '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=60)
        from research.h3_personal.conversation_timeline import ConversationBinding
        boundary = lambda n: n * RATE // 24
        self.assertEqual(audio[:boundary(124) * 2], bytes(boundary(124) * 2))
        self.assertEqual(audio[boundary(124) * 2:boundary(124) * 2 + len(self.speech_pcm)],
                         self.speech_pcm)

    def test_inference_failure_abandons_without_restart(self):
        self.fail_at = (1, 1)
        with self.assertRaises(RuntimeError):
            self.render()
        self.assertEqual(len(self.abandons), 1)
        self.assertEqual(self.abandons[0]['action'], 'abandon')
        self.assertIn(self.abandons[0]['reason'],
                      ('local-sink-failure', 'local-preparation-failure'))
        self.assertFalse((self.output / 'continuous.nut').exists())
        self.assertEqual(json.loads(self.runtime.continuation_file.read_text()), dict(mode='independent'))
        # No restart/reload: the same owner object is still used and no new start.
        self.assertEqual(self.runtime._state, 'ready')
        self.assertEqual(self.runtime.generator, None)

    def test_requires_explicit_conversation_owner(self):
        runtime = LiveVoiceRuntime(model=self.root / 'm', model_manifest=self.root / 'mm',
            identity=self.root / 'i', identity_config=self.root / 'ic', smd=self.root / 's',
            reference=self.root / 'r', archive=self.root / 'a', output=self.output)
        runtime._state = 'ready'
        with self.assertRaises(RuntimeError):
            runtime.render_conversation_av(self.listening, self.speech, self.turn,
                                           self.output / 'x.nut', timeout=30)



if __name__ == '__main__':
    unittest.main()


class OrderingTests(unittest.TestCase):
    def test_roles_env_is_published_after_layout_config(self):
        """Roles must not read config.prompt before config_for_layout exists."""
        import inspect
        from research.h3_personal import live_voice_runtime
        source = inspect.getsource(live_voice_runtime.LiveVoiceRuntime._start)
        config = source.index("self.config = config_for_layout('portrait')")
        roles = source.index("if getattr(self, 'conversation_roles', False):")
        self.assertLess(config, roles)
        generator = source.index('self.generator = ')
        self.assertLess(roles, generator)
        self.assertIn('ROLES_ENV', source[roles:generator])


class FixedPilotTests(unittest.TestCase):
    def test_only_declared_roles_may_change_the_prompt(self):
        from dataclasses import replace
        from research.h3_personal.conversation_conditioning import (
            LISTENING_PROMPT, SPEECH_PROMPT)
        from research.h3_personal.fast_ref2va import config_for_layout
        base = config_for_layout('portrait')
        self.assertIs(base.validate(), base)
        self.assertIsNotNone(replace(base, prompt=LISTENING_PROMPT).validate())
        self.assertIsNotNone(replace(base, prompt=SPEECH_PROMPT).validate())
        for changes in ({'prompt': 'arbitrary'},
                        {'num_frames': 120}, {'fps': 25}, {'seed': 1},
                        {'sigma_grid_points': 4}, {'height': 512}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    replace(base, **changes).validate()
