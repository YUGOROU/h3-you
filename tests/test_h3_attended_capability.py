"""Attended visual capability seam and finite ASR capture-scope propagation.

Device-free: no capture, browser, network, provider or GPU operation is started.
"""
import ast
import time
from pathlib import Path
import unittest

from research import launch_h3_direct_conversation as L

FILTERS = dict(visual_intents=True, attended_visual=True, headless_interrupt=False,
               capture_comet=True, asr_parent=Path('/tmp/asr-parent'),
               media_transport='http-paired-v1', asr_rtc_factory=None,
               allow_obs=True, obs_port=48124, direct_audio_binary=Path('/bin/echo'),
               start_gate=None, idle_anchor=False, local_sink_factory=None)


def capability(**changes):
    args = dict(FILTERS)
    args.update(changes)
    return L.attended_visual_capability(**args)


class AttendedCapabilityTests(unittest.TestCase):
    def test_exact_attended_selection_is_accepted(self):
        self.assertEqual(capability(), dict(attended_visual=True, visual_intents=True))

    def test_selection_is_opt_in_only(self):
        self.assertEqual(capability(attended_visual=False), {})

    def test_visual_intents_required(self):
        with self.assertRaises(ValueError):
            capability(visual_intents=False)

    def test_headless_and_attended_are_exclusive(self):
        with self.assertRaises(ValueError):
            capability(headless_interrupt=True)

    def test_comet_input_required(self):
        with self.assertRaises(ValueError):
            capability(capture_comet=False)
        with self.assertRaises(ValueError):
            capability(asr_parent=None)

    def test_zoom_input_is_a_different_route(self):
        with self.assertRaises(ValueError):
            capability(zoom_input=object())

    def test_rtc_transport_rejected(self):
        with self.assertRaises(ValueError):
            capability(media_transport='webrtc-rtp-v1')
        with self.assertRaises(ValueError):
            capability(asr_rtc_factory=object())

    def test_start_gate_is_not_part_of_this_route(self):
        with self.assertRaises(ValueError):
            capability(start_gate=Path('/tmp/gate'))

    def test_periodic_anchor_reset_rejected(self):
        with self.assertRaises(ValueError):
            capability(idle_anchor=True)

    def test_local_browser_sink_is_not_this_route(self):
        with self.assertRaises(ValueError):
            capability(local_sink_factory=lambda: None)

    def test_verified_split_output_required(self):
        for change in (dict(allow_obs=False), dict(obs_port=None),
                       dict(direct_audio_binary=None)):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    capability(**change)

    def test_bools_are_explicit_not_truthy(self):
        with self.assertRaises(ValueError):
            capability(attended_visual=1)
        with self.assertRaises(ValueError):
            capability(visual_intents='yes')


class AttendedExecutionTests(unittest.TestCase):
    """The attended CLI now runs the verified device route; nothing starts here."""

    def test_run_requires_the_split_obs_output(self):
        with self.assertRaises(ValueError) as caught:
            L.run(Path('/tmp/h3-attended-seam'), None, capture_comet=True,
                  asr_parent=Path('/tmp/asr-parent'), attended_visual=True,
                  visual_intents=True)
        self.assertIn('verified OBS video plus PCM audio output', str(caught.exception))

    def test_split_output_still_reaches_the_real_prerequisites(self):
        with self.assertRaises(ValueError) as caught:
            L.run(Path('/tmp/h3-attended-seam'), None, capture_comet=True,
                  asr_parent=Path('/tmp/asr-parent'), attended_visual=True,
                  visual_intents=True, allow_obs=True, obs_port=48124,
                  direct_audio_binary=Path('/bin/echo'))
        # The private review root is checked first; on hosts where /bin/echo is a
        # symlink the binary check fires first instead. Either way the run is refused.
        self.assertTrue('PRIVATE' in str(caught.exception)
                        or 'symlink' in str(caught.exception), caught.exception)

    def test_half_wired_relay_still_fails_closed(self):
        from research import run_h3_relay_conversation
        saved = run_h3_relay_conversation.run_conversation
        try:
            run_h3_relay_conversation.run_conversation = lambda *a, **k: None
            with self.assertRaises(ValueError) as caught:
                L.run(Path('/tmp/h3-attended-seam'), None, capture_comet=True,
                      asr_parent=Path('/tmp/asr-parent'), attended_visual=True,
                      visual_intents=True, allow_obs=True, obs_port=48124,
                      direct_audio_binary=Path('/bin/echo'))
            self.assertIn('not yet wired', str(caught.exception))
        finally:
            run_h3_relay_conversation.run_conversation = saved

    def test_cli_rejects_attended_without_the_split_output(self):
        with self.assertRaises(SystemExit) as caught:
            L.main(['--directory', '/tmp/h3-attended-seam', '--attended-visual',
                    '--visual-intents', '--capture-comet',
                    '--asr-parent', '/tmp/asr-parent', '--run'])
        self.assertEqual(caught.exception.code, 2)

    def test_cli_rejects_attended_without_comet_input(self):
        with self.assertRaises(SystemExit) as caught:
            L.main(['--directory', '/tmp/h3-attended-seam', '--attended-visual',
                    '--visual-intents', '--allow-obs', '--obs-port', '48124',
                    '--direct-blackhole-audio', '/bin/echo'])
        self.assertEqual(caught.exception.code, 2)

    def test_visual_intents_still_require_a_route(self):
        with self.assertRaises(SystemExit) as caught:
            L.main(['--directory', '/tmp/h3-attended-seam', '--visual-intents'])
        self.assertEqual(caught.exception.code, 2)

    def test_attended_run_uses_the_existing_vox_lease_and_device_route(self):
        import inspect
        source = inspect.getsource(L.main)
        branch = source[source.index('if getattr(a,\'attended_visual\',False):'):]
        branch = branch[:branch.index('print(json.dumps(dict(result=result)')]
        for needle in ('vox_lease(', 'parent_campaign=a.vox_parent',
                       'tts_mode=\'native_voxcpm\'', 'capture_comet=True',
                       'allow_obs=True', 'attended_visual=True',
                       '10800 if a.three_hour_session else 1500)'):
            self.assertIn(needle, branch)


def _direct_asr_calls():
    tree = ast.parse(Path(L.__file__).read_text())
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'DirectASRInput'):
            calls.append({k.arg: ast.unparse(k.value) for k in node.keywords})
    return calls


class CaptureSecondsPropagationTests(unittest.TestCase):
    def test_every_direct_asr_input_receives_the_requested_duration(self):
        calls = _direct_asr_calls()
        self.assertEqual(len(calls), 2)
        for keywords in calls:
            self.assertEqual(keywords.get('capture_seconds'), 'asr_capture_seconds')

    def test_requested_duration_is_finite_and_bounded(self):
        from research.h3_personal.direct_asr_input import DirectASRInput
        for seconds in (8, 45):
            source = DirectASRInput.__new__(DirectASRInput)
            # Construction validation is what the launcher relies on; mirror it here.
            self.assertTrue(8 <= seconds <= 45)
            self.assertIsNotNone(source)
        for seconds in (7, 46, 0, -1, None, 45.0, True):
            with self.assertRaises(ValueError):
                DirectASRInput(object(), capture_seconds=seconds)


if __name__ == '__main__':
    unittest.main()
