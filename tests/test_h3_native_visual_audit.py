import unittest
from research.audit_h3_native_video import frame_context,first_decode_influence,GROUPS

class NativeVisualAuditTests(unittest.TestCase):
    def test_first_clip_has_no_temporal_crossfade(self):
        for frame in range(17):
            c=frame_context(frame)
            self.assertEqual(c['decode_clip'],0);self.assertEqual(c['decode_latents'],[0,7])
            self.assertFalse(c['temporal_overlap_slot']);self.assertFalse(c['two_clip_blend'])
        self.assertEqual(frame_context(12)['seconds'],'1/2')
    def test_overlap_weights_and_tail(self):
        self.assertTrue(frame_context(17)['temporal_overlap_slot'])
        self.assertFalse(frame_context(17)['two_clip_blend'])
        self.assertEqual(frame_context(21)['incoming_clip_weight'],'4/5')
        self.assertFalse(frame_context(22)['temporal_overlap_slot'])
        for frame in range(680,685):
            self.assertTrue(frame_context(frame)['tail']);self.assertEqual(frame_context(frame)['decode_clip'],39)
            self.assertFalse(frame_context(frame)['two_clip_blend'])
    def test_append_influence_envelope_from_actual_decode_windows(self):
        for latent in range(202):
            expected=next(i*17 for i in range(40) if i*5<=latent<i*5+7)
            self.assertEqual(first_decode_influence(latent),expected)
        self.assertEqual([first_decode_influence(n) for n in (82,157)],[272,527])
    def test_bounded_inputs_and_selected_frames(self):
        for value in (-1,685,True,1.1):
            with self.assertRaises(ValueError):frame_context(value)
        for value in (-1,202,True,1.1):
            with self.assertRaises(ValueError):first_decode_influence(value)
        self.assertTrue(all(0<=n<683 for group in GROUPS.values() for n in group))
