"""Mac-only ten minute opt-in: fake time, no devices or cloud operations."""
import inspect
import unittest
from unittest.mock import patch

from test_h3_relay_conversation import Transport

import test_h3_endurance_first_av_clock as first_av
from research.run_h3_relay_conversation import run_conversation


class TenMinuteClockTests(unittest.TestCase):
    # Reuse the deterministic receiver/clock harness, not its test methods.
    exercise = first_av.FirstAVClockTests.exercise

    def ten(self, **options):
        kwargs = dict(max_seconds=600, ten_minute_test=True)
        kwargs.update(options)
        return self.exercise(**kwargs)

    def test_explicit_600_excludes_first_av_delay(self):
        self.assertEqual(self.ten(), {'complete': True})
        self.assertEqual(self.initial_deadline, 1000.)
        self.assertEqual(len(self.inputs), 1)
        self.assertGreaterEqual(self.inputs[0][0], 110.)
        self.assertEqual(self.inputs[0][1], 710.)
        self.assertEqual(self.director.deadline, 710.)
        self.assertEqual(self.pump.deadline, 710.)
        self.assertEqual(self.pump.receiver.deadline, 710.)
        self.assertAlmostEqual(self.now, 708.)
        self.assert_clean()

    def assert_clean(self):
        self.assertTrue(self.sink.canceled)
        self.assertTrue(self.director.closed)
        self.assertTrue(self.transport._closed)

    def test_strict_opt_in_and_exact_duration(self):
        for options in (
                {'ten_minute_test': False}, {'ten_minute_test': 1},
                {'ten_minute_test': None}, {'hold_listening': False},
                {'start_on_first_av': False}, {'max_seconds': True},
                {'max_seconds': float('nan')}, {'max_seconds': float('inf')},
                {'max_seconds': 599}, {'max_seconds': 601}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.ten(**options)

    def test_full_lease_reservation_and_120_second_first_wait(self):
        for lease, expected in ((1000., 220.), (750., 150.)):
            with self.subTest(lease=lease), self.assertRaises(TimeoutError):
                self.ten(first=None, lease=lease)
            self.assertAlmostEqual(self.now, expected)
            self.assertEqual(self.inputs, [])
            self.assert_clean()

    def test_cancel_or_receiver_end_cleans_up_without_capture(self):
        for mode in ('cancel', 'error', 'result'):
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                self.ten(first=None, mode=mode)
            self.assertLess(self.now, 101.)
            self.assertEqual(self.inputs, [])
            self.assert_clean()

    def test_hard_deadline_aborts_stalled_drain(self):
        send = Transport.send_input

        def stalled(transport, raw):
            count = send(transport, raw)
            transport.done.clear()
            return count

        with patch.object(Transport, 'send_input', stalled):
            with self.assertRaises(TimeoutError):
                self.ten()
        self.assertAlmostEqual(self.now, 710.)
        self.assertTrue(self.transport.aborted)
        self.assert_clean()

    def test_existing_defaults_and_caps_unchanged(self):
        params = inspect.signature(run_conversation).parameters
        for name, expected in dict(max_seconds=120, max_turns=2,
                                   hold_listening=False,
                                   start_on_first_av=False,
                                   ten_minute_test=False).items():
            self.assertEqual(params[name].default, expected)
        with self.assertRaises(ValueError):
            self.exercise(max_seconds=301)
        self.exercise(first=None, hold_listening=False,
                      start_on_first_av=False, max_seconds=600)
        self.assertEqual(self.inputs, [(100., 700.)])


if __name__ == '__main__':
    unittest.main()
