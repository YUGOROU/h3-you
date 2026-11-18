"""CPU-only: half-duplex capture gate. No device, browser, network or GPU use."""
import math
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.h3_personal.half_duplex_gate import (
    EVIDENCE_SCHEMA, GateState, HalfDuplexCaptureGate,
)

SESSION = 'session-abc'


def evidence(*, epoch=1, session=SESSION, completed_at=None, output_seconds=4.5,
             schema=EVIDENCE_SCHEMA):
    return {'schema': schema, 'session': session, 'epoch': epoch,
            'completed_at_monotonic': (time.monotonic() if completed_at is None
                                       else completed_at),
            'output_seconds': output_seconds}


class HalfDuplexGateTests(unittest.TestCase):
    def gate(self, *, deadline=None, max_turns=2):
        return HalfDuplexCaptureGate(session=SESSION,
                                     deadline=(time.monotonic() + 600 if deadline is None
                                               else deadline),
                                     max_turns=max_turns)

    def test_construction_rejects_unbounded_parameters(self):
        for kwargs in ({'session': '', 'deadline': time.monotonic() + 60},
                       {'session': SESSION, 'deadline': math.inf},
                       {'session': SESSION, 'deadline': None},
                       {'session': SESSION, 'deadline': time.monotonic() - 1},
                       {'session': SESSION, 'deadline': time.monotonic() + 60, 'max_turns': 0},
                       {'session': SESSION, 'deadline': time.monotonic() + 60, 'max_turns': True}):
            with self.assertRaises(ValueError):
                HalfDuplexCaptureGate(**kwargs)

    def test_first_turn_capture_is_explicit_and_once(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        self.assertFalse(gate.request_capture(1))
        self.assertFalse(gate.request_capture(2))
        self.assertEqual(gate.snapshot().state, GateState.CAPTURE_ACTIVE)

    def test_publication_alone_never_enables_capture(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        self.assertEqual(gate.snapshot().state, GateState.WAITING_OUTPUT)
        self.assertFalse(gate.request_capture(2))
        self.assertEqual(gate.snapshot().state, GateState.WAITING_OUTPUT)

    def test_ack_style_evidence_is_rejected(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        for bogus in ({'schema': 'h3-turn-ack-v1', 'session': SESSION, 'epoch': 1},
                      {'schema': EVIDENCE_SCHEMA, 'session': SESSION, 'epoch': 1,
                       'completed_at_monotonic': time.monotonic()},
                      {'schema': EVIDENCE_SCHEMA, 'session': SESSION, 'epoch': 1,
                       'completed_at_monotonic': time.monotonic(), 'output_seconds': 1.0,
                       'ack': True},
                      'presented'):
            with self.assertRaises(ValueError):
                gate.note_presentation_complete(bogus)
        self.assertEqual(gate.snapshot().state, GateState.WAITING_OUTPUT)
        self.assertFalse(gate.request_capture(2))
        self.assertEqual(gate.snapshot().rejections, 4)

    def test_completion_enables_exactly_one_next_capture(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        self.assertTrue(gate.note_presentation_complete(evidence()))
        self.assertEqual(gate.snapshot().state, GateState.OUTPUT_PRESENTED)
        self.assertTrue(gate.request_capture(2))
        self.assertFalse(gate.request_capture(2))
        self.assertFalse(gate.request_capture(3))  # max_turns bound, no third capture
        gate.note_publication(2)
        self.assertFalse(gate.request_capture(3))

    def test_duplicate_completion_blocks(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        self.assertTrue(gate.note_presentation_complete(evidence()))
        with self.assertRaises(ValueError):
            gate.note_presentation_complete(evidence())
        snap = gate.snapshot()
        self.assertEqual((snap.state, snap.reason), (GateState.BLOCKED, 'unexpected_completion'))
        self.assertFalse(gate.request_capture(2))

    def test_stale_epoch_completion_blocks(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        with self.assertRaises(ValueError):
            gate.note_presentation_complete(evidence(epoch=2))
        snap = gate.snapshot()
        self.assertEqual((snap.state, snap.reason), (GateState.BLOCKED, 'epoch_mismatch'))
        self.assertFalse(gate.request_capture(2))

    def test_publication_out_of_order_blocks(self):
        gate = self.gate()
        with self.assertRaises(ValueError):
            gate.note_publication(1)  # no capture started for this epoch
        self.assertEqual(gate.snapshot().reason, 'publication_order')
        self.assertFalse(gate.request_capture(1))

    def test_wrong_session_never_accepts(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        with self.assertRaises(ValueError):
            gate.note_presentation_complete(evidence(session='other'))
        self.assertEqual(gate.snapshot().state, GateState.WAITING_OUTPUT)

    def test_future_or_zero_duration_evidence_rejected(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        for kwargs in ({'completed_at': time.monotonic() + 60},
                       {'output_seconds': 0},
                       {'output_seconds': math.nan},
                       {'completed_at': 0}):
            with self.assertRaises(ValueError):
                gate.note_presentation_complete(evidence(**kwargs))
        self.assertEqual(gate.snapshot().state, GateState.WAITING_OUTPUT)

    def test_cancel_blocks_capture_and_late_completion(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        self.assertTrue(gate.cancel())
        self.assertFalse(gate.cancel())
        snap = gate.snapshot()
        self.assertEqual((snap.state, snap.reason), (GateState.STOPPED, 'canceled'))
        self.assertFalse(gate.request_capture(2))
        with self.assertRaises(ValueError):
            gate.note_presentation_complete(evidence())
        self.assertEqual(gate.snapshot().reason, 'post_cancel_completion')

    def test_deadline_expiry_fails_closed_without_timer(self):
        gate = self.gate(deadline=time.monotonic() + 0.05)
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        time.sleep(0.08)
        snap = gate.snapshot()
        self.assertEqual((snap.state, snap.reason), (GateState.BLOCKED, 'deadline'))
        self.assertEqual(snap.deadlines_left, 0.0)
        self.assertFalse(gate.request_capture(2))
        with self.assertRaises(ValueError):
            gate.note_presentation_complete(evidence())

    def test_blocked_gate_never_replays_queue(self):
        gate = self.gate()
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        with self.assertRaises(ValueError):
            gate.note_presentation_complete(evidence(epoch=9))
        for _ in range(3):
            self.assertFalse(gate.request_capture(2))
        snap = gate.snapshot()
        self.assertEqual(snap.state, GateState.BLOCKED)
        self.assertEqual((snap.capture_epoch, snap.output_epoch), (1, 1))
        self.assertFalse(gate.request_capture(1))
        self.assertEqual(gate.snapshot().capture_epoch, 1)


if __name__ == '__main__':
    unittest.main()
