"""CPU policy checks only; do not establish real TTS/H3/adapter quality."""
import unittest

from research.hf_infer_personal import frame_count, packed_video_channels, comparison_cases, set_adapter_strength


class InferenceDurationTests(unittest.TestCase):
    def test_buckets_preserve_entire_speech_and_tail(self):
        for spoken in [.1, 4.2, 5, 7, 10, 14]:
            with self.subTest(spoken=spoken):
                frames = frame_count(spoken)
                self.assertEqual((frames - 5) % 17, 0)
                self.assertGreaterEqual(frames, 124)
                self.assertLessEqual(frames / 24, 15)
                self.assertGreaterEqual(frames / 24, spoken + .25)
                if frames > 124:
                    self.assertLess((frames - 17) / 24, spoken + .25)

    def test_raw_channels_are_patchified_before_noise_initialization(self):
        self.assertEqual(packed_video_channels(24, (1, 2, 2)), 96)
        self.assertEqual(packed_video_channels(24, (1, 1, 1)), 24)
        with self.assertRaises(ValueError):
            packed_video_channels(24, (1, 0, 2))

    def test_comparison_keeps_lora_on_and_changes_one_factor(self):
        cases = comparison_cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({c['name'] for c in cases}), 4)
        self.assertEqual([(c['lora_scale'], c['schedule_points']) for c in cases],
                         [(1, 50), (.5, 50), (1.5, 50), (1, 100)])
        self.assertTrue(all(c['lora_scale'] > 0 for c in cases))

    def test_rejects_disabled_or_invalid_adapter_strength(self):
        for scale in [0, -1, float('nan'), float('inf'), 2.1]:
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    set_adapter_strength(None, scale)

    def test_refuses_invalid_or_overlong_speech(self):
        for spoken in [0, -1, float('nan'), float('inf'), 15]:
            with self.subTest(spoken=spoken):
                with self.assertRaises(ValueError):
                    frame_count(spoken)


if __name__ == '__main__':
    unittest.main()
