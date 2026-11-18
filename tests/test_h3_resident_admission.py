"""No devices/network: capture retries cannot expand the paid Pi allowance."""
import unittest
from research.h3_personal.resident_admission import (
    admit_capture, reserve_pi_request, MAX_CAPTURE_ATTEMPTS, MAX_PI_REQUESTS,
)


class AdmissionTests(unittest.TestCase):
    def test_silence_does_not_consume_pi_slots(self):
        state = dict(pi_requests=0)
        for attempt in range(MAX_CAPTURE_ATTEMPTS):
            admit_capture(attempt, state['pi_requests'])
        self.assertEqual(state['pi_requests'], 0)
        self.assertEqual(reserve_pi_request(state), 1)

    def test_capture_attempts_still_finite(self):
        with self.assertRaises(ValueError):admit_capture(MAX_CAPTURE_ATTEMPTS, 0)

    def test_unknown_ack_cannot_refund_pi_request(self):
        state = dict(pi_requests=0)
        for index in range(MAX_PI_REQUESTS):
            self.assertEqual(reserve_pi_request(state), index+1)
        with self.assertRaises(ValueError):reserve_pi_request(state)
        with self.assertRaises(ValueError):admit_capture(4, state['pi_requests'])
        self.assertEqual(state['pi_requests'], MAX_PI_REQUESTS)

    def test_no_invalid_counter_bypass(self):
        for attempts, requests in ((True, 0), (0, False), (-1, 0), (0, -1), (0, 1.5)):
            with self.subTest(attempts=attempts, requests=requests), self.assertRaises(ValueError):
                admit_capture(attempts, requests)


if __name__ == '__main__':unittest.main()
