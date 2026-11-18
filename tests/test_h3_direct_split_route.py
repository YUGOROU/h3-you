"""Offline split route consent and immutable binary preflight."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research import launch_h3_direct_conversation as launcher
from research.run_h3_relay_conversation import direct_audio_pin


class SplitRouteTests(unittest.TestCase):
    def test_pin_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp)/'output'
            binary.write_bytes(b'not an executable format')
            binary.chmod(0o700)
            pin = direct_audio_pin(binary)
            self.assertEqual(pin['binary_sha256'], hashlib.sha256(binary.read_bytes()).hexdigest())
            self.assertEqual(pin['audio_delay_seconds'], 0.2)
            link = Path(tmp)/'link'
            link.symlink_to(binary)
            with self.assertRaises(ValueError):
                direct_audio_pin(link)
            for delay in (True, -1, 1.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    direct_audio_pin(binary, delay)

    def test_invalid_cli_before_run(self):
        cases = [ ['--audio-delay-seconds', '0.2'],
                  ['--direct-blackhole-audio', '/missing'],
                  ['--run', '--allow-obs', '--obs-port', '1234',
                   '--direct-blackhole-audio', '/missing'] ]
        with patch.object(launcher, 'run') as run:
            for args in cases:
                with self.assertRaises(SystemExit):
                    launcher.main(['--directory', '/tmp/unused', *args])
            run.assert_not_called()

    def test_cli_passes_explicit_route_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp)/'output'
            binary.write_bytes(b'x')
            binary.chmod(0o700)
            with patch.object(launcher, 'run') as run, patch('builtins.input', side_effect=['hello', '']):
                launcher.main(['--directory', tmp, '--run', '--allow-obs', '--obs-port', '1234',
                               '--direct-blackhole-audio', str(binary)])
                self.assertEqual(run.call_args.kwargs['direct_audio_binary'], str(binary.resolve()))
                self.assertEqual(run.call_args.kwargs['audio_delay_seconds'], 0.2)

    def test_direct_requires_both_consent_and_video(self):
        for kwargs in ({}, {'allow_obs': True}, {'obs_port': 1234}):
            with self.assertRaises(PermissionError):
                launcher.run('/tmp/unused', ['hello'], direct_audio_binary='/missing', **kwargs)


if __name__ == '__main__':
    unittest.main()
