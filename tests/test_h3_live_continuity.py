"""Continuity admission contracts only; no generation, frames, or quality claim."""
import threading
import unittest
from unittest.mock import Mock
from research.h3_personal.live_voice_runtime import (
    LiveVoiceRuntime, UnsupportedTemporalContinuation, continuity_contract,
)


class ContinuityTests(unittest.TestCase):
    def test_no_shared_head_or_hidden_audio_video_time_equivalence(self):
        row = continuity_contract(3)
        self.assertFalse(row['temporal_state_carried'])
        self.assertFalse(row['previous_tail_conditioning'])
        self.assertFalse(row['motion_continuity_verified'])
        self.assertEqual(row['overlap_frames'], 0)
        self.assertEqual(row['windows'][1]['first_frame'], 124)
        self.assertEqual(row['windows'][0]['last_frame'], 123)
        self.assertAlmostEqual(row['windows'][1]['video_start_seconds'], 124/24)
        self.assertAlmostEqual(row['windows'][1]['audio_start_seconds'], 5.175)
        self.assertNotEqual(row['windows'][1]['video_start_seconds'], 5.175)

    def runtime(self, windows):
        runtime = LiveVoiceRuntime.__new__(LiveVoiceRuntime)
        runtime._lock = threading.Lock()
        runtime._state = 'ready'
        runtime._plan = Mock(return_value=dict(windows_required=windows))
        runtime._render = Mock(return_value='asset')
        runtime._shutdown = Mock()
        return runtime

    def test_unsupported_request_preserves_warm_runtime_without_generation(self):
        runtime = self.runtime(3)
        with self.assertRaises(UnsupportedTemporalContinuation):
            runtime.render('/private/input.wav', 'digest', '/private/turn', require_continuity=True)
        runtime._render.assert_not_called()
        runtime._shutdown.assert_not_called()
        self.assertEqual(runtime._state, 'ready')

    def test_single_window_does_not_claim_continuity_verified(self):
        runtime = self.runtime(1)
        self.assertEqual(runtime.render('/private/input.wav', 'digest', '/private/turn', require_continuity=True), 'asset')
        self.assertFalse(continuity_contract(1, require_continuity=True)['motion_continuity_verified'])

    def test_legacy_diagnostic_mode_preserved_not_silently_reconditioned(self):
        runtime = self.runtime(3)
        self.assertEqual(runtime.render('/private/input.wav', 'digest', '/private/turn'), 'asset')
        runtime._plan.assert_not_called()
        runtime._render.assert_called_once()

    def test_invalid_requirements_fail_before_gpu(self):
        for value in ('yes', 1, None):
            runtime = self.runtime(3)
            with self.assertRaises(ValueError):
                runtime.render('/private/input.wav', 'digest', '/private/turn', require_continuity=value)
            runtime._render.assert_not_called();runtime._shutdown.assert_not_called()
        for count in (0, 5, True):
            with self.assertRaises(ValueError):continuity_contract(count)


if __name__ == '__main__':unittest.main()
