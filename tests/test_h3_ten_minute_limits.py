"""Offline ten-minute opt-in contracts; no cloud/device operations."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

from research.hf_h3_direct_conversation import endurance_limits, validate_spec
from research.launch_h3_direct_conversation import main, validate_ready, run
from research.h3_personal.fast_ref2va import config_for_layout
from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA


class TenMinuteLimitsTests(unittest.TestCase):
    def spec(self):
        return dict(config={'timeout_seconds': 1200}, execution={'timeout_seconds': 1200},
            live_voice_session=dict(id='a'*32, max_turns=2, fresh_input_required=True),
            live_direct_session=dict(schema='h3-live-direct-session-v1', session='a'*32,
                port=7860, max_windows=142, max_frames=14506, lease_seconds=1200,
                ten_minute_test=True), asr_policy='endpoint-native-v1', layout='portrait',
            runtime_policy='resident-visual-cached', attention_mode='fa4-packed',
            fast_config=asdict(config_for_layout('portrait')),
            identity_prefix=IDENTITY_PREFIX, reference_sha256=REFERENCE_SHA)

    def test_limits_and_strict_flags(self):
        self.assertEqual(endurance_limits(), (16, 1654, 1200))
        self.assertEqual(endurance_limits(True), (70, 7200, 1200))
        self.assertEqual(endurance_limits(ten_minute_test=True), (142, 14506, 1200))
        for flags in ((True, True), (1, False), (False, 1), (None, False), (False, 'true')):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                endurance_limits(*flags)
        with self.assertRaises(ValueError):
            run('/unused', ten_minute_test=1)

    def test_immutable_spec(self):
        validate_spec(self.spec())
        for section, key, value in (
            ('config', 'timeout_seconds', 600), ('execution', 'timeout_seconds', 600),
            ('live_direct_session', 'lease_seconds', 600),
            ('live_direct_session', 'max_windows', 70),
            ('live_direct_session', 'max_frames', 7200),
            ('live_direct_session', 'ten_minute_test', 1),
            ('live_direct_session', 'five_minute_test', True)):
            spec = self.spec()
            spec[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_spec(spec)

    def test_ready_mode_binding(self):
        row = dict(type='h3_direct_ready', status='ready', session='a'*32,
            profile_sha256='b'*64, epoch=1, max_segments=142, max_total_frames=14506,
            port=7860, expires_at=2000)
        with patch('research.launch_h3_direct_conversation.time.time', return_value=1000):
            self.assertIs(validate_ready(row, 'a'*32, ten_minute_test=True), row)
            for kwargs in ({}, {'five_minute_test': True}):
                with self.assertRaises(ValueError):
                    validate_ready(row, 'a'*32, **kwargs)
            for key, value in (('max_segments', 70), ('max_total_frames', 7200), ('session', 'c'*32)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_ready(dict(row, **{key: value}), 'a'*32, ten_minute_test=True)

    def test_cli_requires_run_and_excludes_five(self):
        for flags in (['--ten-minute-test'], ['--run', '--ten-minute-test', '--five-minute-test']):
            with self.assertRaises(SystemExit) as error:
                main(['--directory', '/unused', *flags])
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
