"""Regression tests for the 2026-11-05 attended E2E failure audit.

Covers three defects found in `three-hour-attended-20261105-4`
(see docs/h3-last-e2e-failure-audit-20261110-v1.md):

F1 the attended route accepted --review-av and silently dropped it;
F2 an all-zero capture source was reported as ordinary no-speech forever;
F3 the relay's rejection stage/flags were discarded, leaving a bare HTTP 400.

CPU/file only: no GPU, network, device or capture execution.
"""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research.h3_personal import local_av_diagnostics as diagnostics
from research.h3_personal import resident_endpoint as r


def _lease(root):
    lease = r.ResidentEndpoint()
    lease.root = root
    lease.closed = False
    lease.owned = True
    lease._token = 'test-token'
    lease._key = 'memory-only'
    lease._path = '/owned'
    lease.lease_id = 'fresh-id'
    lease.deadline = time.time() + 1800
    lease.receipt = {'terminal_state': 'unconfirmed'}
    lease._binary = root / 'binary'
    lease._pins = {'capture_binary': 'pin'}
    lease._intent = {}
    lease._url = 'https://example.endpoints.huggingface.cloud'
    lease._silent_windows = 0
    return lease


class SilentInputPathTests(unittest.TestCase):
    """F2: consecutive all-zero PCM is a dead input path, not a quiet human."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def capture(self, lease, name, *, nonzero):
        def spawn(*args, **kwargs):
            kwargs['stderr'].write((json.dumps(dict(type='stopped', cleanup=True,
                frames=128000, nonzero_samples=nonzero)) + '\n').encode())
            return SimpleNamespace(stdout=io.BytesIO(bytes(256000)))

        def transcribe(*args, **kwargs):
            kwargs['pcm']()
            kwargs['on_input_stop']()
            return dict(final=False, errors=['no_input_audio'], cleanup_complete=True,
                        input_committed=False, finalized_audio_seconds=0,
                        audio_sent_seconds=0)

        with patch.object(r, 'control', return_value={'status': {'state': 'running'}}), \
             patch.object(r, 'assert_deployment'), \
             patch.object(r, 'sha', return_value='pin'), \
             patch.object(r.subprocess, 'Popen', side_effect=spawn), \
             patch.object(r, 'stop_capture'), \
             patch.object(r, 'transcribe', side_effect=transcribe), \
             patch.object(r, 'pause_owned') as pause:
            try:
                lease.capture(self.root / name, None)
            finally:
                pass
        pause.assert_not_called()
        return json.loads((self.root / name / 'result.json').read_text())

    def test_threshold_is_three_consecutive_windows(self):
        self.assertEqual(r.SILENT_INPUT_WINDOWS, 3)

    def test_input_path_silent_is_not_a_turn_rejection(self):
        # DirectASRInput continues on TurnRejected/NoSpeech and would spin forever.
        self.assertFalse(issubclass(r.InputPathSilent, r.TurnRejected))
        self.assertFalse(issubclass(r.InputPathSilent, r.NoSpeech))

    def test_first_two_silent_windows_remain_ordinary_no_speech(self):
        lease = _lease(self.root)
        for index in (1, 2):
            with self.subTest(window=index):
                with self.assertRaises(r.NoSpeech):
                    self.capture(lease, f'silent-{index}', nonzero=0)
                self.assertEqual(lease._silent_windows, index)
        saved = json.loads((self.root / 'silent-2' / 'result.json').read_text())
        self.assertEqual(saved['skipped'], 'no_speech')

    def test_third_consecutive_silent_window_stops_the_conversation(self):
        lease = _lease(self.root)
        for index in (1, 2):
            with self.assertRaises(r.NoSpeech):
                self.capture(lease, f'silent-{index}', nonzero=0)
        with self.assertRaises(r.InputPathSilent) as caught:
            self.capture(lease, 'silent-3', nonzero=0)
        self.assertIn('all-zero PCM', str(caught.exception))
        self.assertEqual(lease.receipt['last_rejected_turn_code'], 'input_path_silent')
        self.assertEqual(lease.receipt['silent_windows'], 3)
        saved = json.loads((self.root / 'silent-3' / 'result.json').read_text())
        self.assertEqual(saved['skipped'], 'input_path_silent')
        self.assertEqual(saved['silent_windows'], 3)

    def test_audible_window_resets_the_counter(self):
        lease = _lease(self.root)
        for index in (1, 2):
            with self.assertRaises(r.NoSpeech):
                self.capture(lease, f'silent-{index}', nonzero=0)
        self.assertEqual(lease._silent_windows, 2)
        # A window carrying any nonzero sample proves the source is alive.
        with self.assertRaises(r.NoSpeech):
            self.capture(lease, 'audible', nonzero=100)
        self.assertEqual(lease._silent_windows, 0)
        # Two more silent windows must not trip the threshold again.
        for index in (4, 5):
            with self.assertRaises(r.NoSpeech):
                self.capture(lease, f'silent-{index}', nonzero=0)
        self.assertEqual(lease._silent_windows, 2)


class RelayRejectEvidenceTests(unittest.TestCase):
    """F3: a rejected request must be attributable to a specific server check."""

    def test_reject_stage_and_flags_are_retained(self):
        evidence = diagnostics.failure_evidence(dict(
            phase='boot-identity', http_status=400, error_type='RelayError',
            lane='abort', method='POST', reject_stage='route-auth', reject_flags='3'))
        self.assertEqual(evidence['phase'], 'boot-identity')
        self.assertEqual(evidence['http_status'], 400)
        self.assertEqual(evidence['reject_stage'], 'route-auth')
        self.assertEqual(evidence['reject_flags'], 3)

    def test_unknown_stage_is_dropped(self):
        evidence = diagnostics.failure_evidence(dict(
            phase='boot-identity', http_status=400,
            reject_stage='/v1/input?token=secret', reject_flags='3'))
        self.assertNotIn('reject_stage', evidence)
        self.assertEqual(evidence['reject_flags'], 3)

    def test_non_numeric_or_oversized_flags_are_dropped(self):
        for flags in ('abc', '-1', '9' * 12, '', None, 3):
            with self.subTest(flags=flags):
                evidence = diagnostics.failure_evidence(dict(
                    phase='boot-identity', http_status=400,
                    reject_stage='route-auth', reject_flags=flags))
                self.assertNotIn('reject_flags', evidence)

    def test_every_server_stage_is_accepted(self):
        # Mirrors the stage= assignments in research/live_relay_server.py.
        for stage in ('path', 'headers-duplicate', 'headers-framing', 'length', 'route-auth'):
            with self.subTest(stage=stage):
                evidence = diagnostics.failure_evidence(dict(
                    phase='boot-identity', http_status=400, reject_stage=stage))
                self.assertEqual(evidence['reject_stage'], stage)

    def test_client_failure_before_headers_has_no_reject_fields(self):
        # A transport failure can happen before any response header exists, so the
        # client must not reference an unset reject variable in its handler.
        source = (Path(__file__).resolve().parents[1]
                  / 'research/h3_personal/live_relay_client.py').read_text()
        self.assertIn('reject_stage=None;reject_flags=None', source)


class AttendedReviewAvTests(unittest.TestCase):
    """F1: the attended route must not accept and silently drop --review-av."""

    def test_run_rejects_review_av_on_attended_route(self):
        from research.launch_h3_direct_conversation import run
        with self.assertRaisesRegex(ValueError, 'review retention is unavailable'):
            run(Path('/nonexistent'), None, attended_visual=True, visual_intents=True,
                review_av=True, capture_comet=True, asr_parent=Path('/nonexistent'),
                allow_obs=True, obs_port=48124, direct_audio_binary=Path('/nonexistent'))

    def test_cli_rejects_review_av_with_attended_visual(self):
        from research.launch_h3_direct_conversation import main
        with patch('sys.argv', ['launch', '--directory', '/tmp/h3-audit-cli',
                                '--attended-visual', '--visual-intents', '--review-av',
                                '--capture-comet', '--asr-parent', '/tmp/none',
                                '--allow-obs', '--obs-port', '48124',
                                '--vox-parent', '/tmp/none']):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)

    def test_cli_still_accepts_review_av_on_headless_route(self):
        from research.launch_h3_direct_conversation import main
        with patch('sys.argv', ['launch', '--directory', '/tmp/h3-audit-cli-headless',
                                '--headless-interrupt', '--visual-intents', '--review-av',
                                '--vox-parent', '/tmp/none']):
            # Offline inspection returns normally and reports the review mode.
            main()


if __name__ == '__main__':
    unittest.main()
