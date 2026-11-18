"""Pure endurance contracts; no cloud, device, or personal inputs."""
import ast
from copy import deepcopy
from dataclasses import asdict
import inspect
import unittest
from unittest.mock import patch

from research import launch_h3_direct_conversation as launch
from research.hf_h3_direct_conversation import endurance_limits, validate_spec
from research.h3_personal.fast_ref2va import config_for_layout
from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA


def fixture(enabled=False):
    windows, frames, timeout = endurance_limits(enabled)
    row = dict(schema='h3-live-direct-session-v1', session='a'*32, port=7860,
               max_windows=windows, max_frames=frames, lease_seconds=timeout)
    if enabled:
        row['five_minute_test'] = True
    return dict(image='pinned-image', live_direct_session=row, config=dict(timeout_seconds=timeout),
                execution=dict(timeout_seconds=timeout),
                live_voice_session=dict(id='a'*32, max_turns=2, fresh_input_required=True),
                fast_config=asdict(config_for_layout('portrait')), layout='portrait',
                attention_mode='fa4-packed', runtime_policy='resident-visual-cached',
                identity_prefix=IDENTITY_PREFIX, reference_sha256=REFERENCE_SHA)


class EnduranceLimitsTests(unittest.TestCase):
    def test_defaults_and_opt_in(self):
        self.assertEqual(endurance_limits(), (16, 1654, 1200))
        self.assertEqual(endurance_limits(True), (70, 7200, 1200))
        for enabled in (False, True):
            item = fixture(enabled)
            before = deepcopy(item)
            self.assertEqual(validate_spec(item), item['live_direct_session'])
            self.assertEqual(item, before)
        for value in (1, None, 'true'):
            with self.assertRaises(ValueError):
                endurance_limits(value)

    def test_reject_unbound_limits(self):
        for enabled in (False, True):
            for key in ('max_windows', 'max_frames'):
                item = fixture(enabled)
                item['live_direct_session'][key] += 1
                with self.assertRaises(ValueError):
                    validate_spec(item)
        for key in ('config', 'execution', 'lease_seconds'):
            item = fixture(True)
            if key == 'lease_seconds':
                item['live_direct_session'][key] = 300
            else:
                item[key]['timeout_seconds'] = 300
            with self.assertRaises(ValueError):
                validate_spec(item)
        item = fixture(True)
        del item['live_direct_session']['five_minute_test']
        with self.assertRaises(ValueError):
            validate_spec(item)

    def test_ready_binding(self):
        for enabled in (False, True):
            windows, frames, timeout = endurance_limits(enabled)
            ready = dict(type='h3_direct_ready', status='ready', session='a'*32,
                         profile_sha256='b'*64, epoch=1, max_segments=windows,
                         max_total_frames=frames, port=7860, expires_at=1000+timeout)
            with patch.object(launch.time, 'time', return_value=1000):
                launch.validate_ready(ready, 'a'*32, five_minute_test=enabled)
                ready['expires_at'] += 1
                with self.assertRaises(ValueError):
                    launch.validate_ready(ready, 'a'*32, five_minute_test=enabled)
                ready['expires_at'] = 1001
                ready['max_segments'] += 1
                with self.assertRaises(ValueError):
                    launch.validate_ready(ready, 'a'*32, five_minute_test=enabled)

    def test_provider_payload_uses_staged_timeout(self):
        tree = ast.parse(inspect.getsource(launch.submit))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
        for enabled in (False, True):
            namespace = dict(spec=fixture(enabled), command=[], token='', producer_key='',
                             consumer_key='', commit=type('Commit', (), {'oid': 'x'})(),
                             native_secrets={})
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<payload>', 'exec'), namespace)
            self.assertEqual(namespace['payload']['timeoutSeconds'], 1200)

    def test_mac_clock_starts_on_first_actual_av(self):
        source = inspect.getsource(launch.run)
        self.assertIn("'hold_listening': True, 'start_on_first_av': True", source)
        self.assertIn('max_seconds=min(10800 if long_session else 300 if five_minute_test else 600', source)
        self.assertIn("ready['expires_at'] - time.time() < (620 if ten_minute_test else 320)", source)
        self.assertIn('time.monotonic()+min(1050, timeout)', source)

    def test_flag_requires_run_without_io(self):
        with patch.object(launch, 'run') as run:
            with self.assertRaises(SystemExit) as error:
                launch.main(['--directory', '/unused', '--five-minute-test'])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
