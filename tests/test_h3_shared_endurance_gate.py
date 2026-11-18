"""Offline shared-inspect timeout gate; all manifests are temporary fixtures."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import launch_h3_live_voice as shared
from research.h3_personal.fast_ref2va import config_for_layout
from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA
from research.hf_h3_direct_conversation import validate_spec


class SharedEnduranceGateTests(unittest.TestCase):
    def spec(self, timeout=1200):
        session = 'a' * 32
        return dict(
            config={'timeout_seconds': timeout}, execution={'timeout_seconds': timeout},
            live_voice_session=dict(id=session, max_turns=2, fresh_input_required=True),
            live_direct_session=dict(schema='h3-live-direct-session-v1', session=session,
                                     port=7860, max_windows=70, max_frames=7200,
                                     lease_seconds=1200, five_minute_test=True),
            asr_policy='endpoint-native-v1', layout='portrait',
            runtime_policy='resident-visual-cached', attention_mode='fa4-packed',
            fast_config=asdict(config_for_layout('portrait')),
            identity_prefix=IDENTITY_PREFIX, reference_sha256=REFERENCE_SHA)

    def inspect(self, spec):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / 'private'
            request = private / 'gpu-hf-irodori-diagnostic-v1/request.json'
            request.parent.mkdir(parents=True)
            manifest = dict(source_revision='fixture', uv_lock_sha256='fixture',
                            uv_version='fixture', models=[], license='fixture')
            request.write_text(json.dumps(manifest))
            stage = root / 'stage'
            stage.mkdir()
            (stage / 'tts-manifest.json').write_text(json.dumps(manifest))
            inventory = {'files': [{'path': name} for name in (*shared.EXTRA, 'tts-manifest.json')]}
            with patch.object(shared, 'PRIVATE', private), patch.object(
                    shared, 'base_inspect', return_value=(spec, inventory)):
                return shared.inspect(stage)

    def test_valid_1200_no_shared_exception(self):
        spec = self.spec()
        self.assertIs(self.inspect(spec)[0], spec)
        validate_spec(spec)
        with self.assertRaises(ValueError):
            self.inspect(self.spec(300))

    def test_generic_or_missing_flag_300_rejected(self):
        for mode in ('generic', 'missing', 'false'):
            with self.subTest(mode=mode):
                spec = self.spec(300)
                if mode == 'generic':
                    del spec['live_direct_session']
                elif mode == 'missing':
                    del spec['live_direct_session']['five_minute_test']
                else:
                    spec['live_direct_session']['five_minute_test'] = False
                with self.assertRaises(ValueError):
                    self.inspect(spec)

    def test_malformed_flags_rejected(self):
        for flag in (1, 'true', None, [], {}):
            with self.subTest(flag=flag):
                spec = self.spec()
                spec['live_direct_session']['five_minute_test'] = flag
                with self.assertRaises(ValueError):
                    validate_spec(spec)

    def test_invalid_limits_and_policy_rejected(self):
        for section, key, value in (
                ('live_direct_session', 'max_windows', 71),
                ('live_direct_session', 'max_frames', 7201),
                ('live_direct_session', 'lease_seconds', 301),
                ('execution', 'timeout_seconds', 300),
                ('live_voice_session', 'max_turns', 16),
                ('fast_config', 'width', 512)):
            with self.subTest(section=section, key=key):
                spec = self.spec()
                spec[section][key] = value
                with self.assertRaises(ValueError):
                    validate_spec(spec)

    def test_legacy_timeouts_unchanged(self):
        for timeout in (660, 1200, 1500):
            for policy in shared.ASR_POLICIES:
                with self.subTest(timeout=timeout, policy=policy):
                    spec = self.spec(timeout)
                    del spec['live_direct_session']
                    spec['asr_policy'] = policy
                    with patch('research.hf_h3_direct_conversation.validate_spec') as check:
                        self.assertIs(self.inspect(spec)[0], spec)
                        check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
