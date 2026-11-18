import math
import unittest

from research.h3_personal import conversation_conditioning as roles
from research.h3_personal.listening_motion import (
    BOUNDS, MAX_FRAMES, MotionPolicy, propose_window, sample,
)


class ListeningMotionTests(unittest.TestCase):
    def test_default_unchanged(self):
        before = roles.roles_manifest()
        for role in ('speech', 'listening'):
            self.assertIsNone(propose_window(MotionPolicy(), role, 0))
        self.assertEqual(before, roles.roles_manifest())
        self.assertEqual(roles.role_prompt('speech'), roles.SPEECH_PROMPT)
        self.assertEqual(roles.role_prompt('listening'), roles.LISTENING_PROMPT)

    def test_determinism_and_binding(self):
        p = MotionPolicy(True)
        a = propose_window(p, 'listening', 22)
        self.assertEqual(a, propose_window(p, 'listening', 22))
        self.assertNotEqual(a['spec_sha256'], propose_window(p, 'speech', 22)['spec_sha256'])
        self.assertFalse(a['spec']['runtime_integrated'])
        self.assertFalse(a['spec']['anchor_reset'])
        self.assertEqual(a['spec']['ancestry_policy'], 'conversation-tail-every3-v1')

    def test_finite_bounds_and_gradual(self):
        p = MotionPolicy(True, True)
        previous = None
        for f in range(MAX_FRAMES + 1):
            row = sample(p, 'listening', f)
            for key, value in row.items():
                self.assertTrue(math.isfinite(value))
                self.assertLessEqual(abs(value), BOUNDS[key])
                if previous is not None and key != 'blink':
                    self.assertLess(abs(value - previous[key]), .03)
            self.assertGreaterEqual(row['blink'], 0)
            previous = row
        self.assertTrue(any(sample(p, 'listening', f)['blink'] > .9 for f in range(2400)))

    def test_global_role_transition_no_reset(self):
        p = MotionPolicy(True)
        expected = propose_window(p, 'listening', 20)
        propose_window(p, 'speech', 19)
        self.assertEqual(expected, propose_window(p, 'listening', 20))
        self.assertNotEqual(expected['spec']['samples'][0], sample(p, 'listening', 0))
        self.assertTrue(all(v == 0 for v in sample(p, 'speech', 2040).values()))
        self.assertEqual(propose_window(p, 'speech', 20)['spec']['directions'], '')
        self.assertEqual(propose_window(p, 'listening', 0)['spec']['stop_frame'],
                         propose_window(p, 'listening', 1)['spec']['start_frame'])

    def test_light_explicit_and_validation(self):
        self.assertEqual(sample(MotionPolicy(True), 'listening', 200)['light_fraction'], 0)
        for args in ((1, False), (False, True), (True, 'yes')):
            with self.assertRaises(ValueError):
                MotionPolicy(*args)
        for frame in (-1, MAX_FRAMES + 1, True, 1.0, float('nan')):
            with self.assertRaises(ValueError):
                sample(MotionPolicy(), 'listening', frame)
        for seq in (-1, 144, True, 1.0):
            with self.assertRaises(ValueError):
                propose_window(MotionPolicy(), 'listening', seq)
        for role in ('idle', None, []):
            with self.assertRaises(ValueError):
                sample(MotionPolicy(), role, 0)


if __name__ == '__main__':
    unittest.main()
