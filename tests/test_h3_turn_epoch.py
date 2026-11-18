"""CPU state-machine tests only; no native, transport, or playback acceptance."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal.turn_epoch import TurnEpoch, TurnReason, TurnState


class TurnEpochTests(unittest.TestCase):
    def guard(self, **kwargs):
        return TurnEpoch(deadline=time.monotonic() + 60, **kwargs)

    def capture(self, guard):
        epoch = guard.speech_onset()
        self.assertTrue(guard.finish_interrupt(epoch))
        return epoch

    def test_interrupt_all_processing_phases_and_stale_callbacks(self):
        # Pi/TTS/H3/transfer/sink share an epoch, not fictitious native states.
        for phase in ('listening', 'capturing', 'pi', 'tts', 'h3',
                      'transfer', 'sink', 'interrupting'):
            with self.subTest(phase=phase):
                guard = self.guard()
                old = self.capture(guard)
                if phase == 'listening':
                    guard.finish(old, TurnReason.FAILURE)
                elif phase in ('pi', 'tts', 'h3', 'transfer', 'sink'):
                    guard.commit(old)
                    if phase in ('h3', 'transfer', 'sink'):
                        guard.speaking(old)
                elif phase == 'interrupting':
                    old = guard.speech_onset()
                new = guard.speech_onset()
                # The caller has not yet run any cancellation/output effects.
                self.assertGreater(new, old)
                self.assertEqual(guard.snapshot().state, TurnState.INTERRUPTING)
                self.assertEqual(guard.snapshot().reason, TurnReason.INTERRUPT)
                self.assertFalse(guard.is_current(old))
                self.assertFalse(guard.commit(old))
                self.assertFalse(guard.speaking(old))
                self.assertFalse(guard.finish(old, TurnReason.EOS))
                self.assertFalse(guard.finish(old, TurnReason.FAILURE))
                self.assertFalse(guard.finish_interrupt(old))
                self.assertTrue(guard.finish_interrupt(new))
                self.assertTrue(guard.is_current(new))
                self.assertTrue(guard.commit(new))

    def test_concurrent_onset_and_commit(self):
        guard = self.guard()
        count = 16
        barrier = threading.Barrier(count)

        def onset(_):
            barrier.wait(timeout=5)
            return guard.speech_onset()

        with ThreadPoolExecutor(max_workers=count) as pool:
            epochs = list(pool.map(onset, range(count)))
            self.assertEqual(sorted(epochs), list(range(1, count + 1)))
            self.assertEqual(guard.snapshot().epoch, count)
            for epoch in epochs:
                self.assertEqual(guard.finish_interrupt(epoch), epoch == count)
            commits = list(pool.map(lambda _: guard.commit(count), range(count)))
        self.assertEqual(sum(commits), 1)
        self.assertTrue(guard.snapshot().committed)
        self.assertFalse(guard.commit(count))

    def test_eos_failure_false_onset_and_stop_are_distinct(self):
        guard = self.guard()
        epoch = self.capture(guard)
        self.assertFalse(guard.finish(epoch, TurnReason.EOS))
        self.assertTrue(guard.commit(epoch))
        self.assertTrue(guard.speaking(epoch))
        self.assertTrue(guard.finish(epoch, TurnReason.EOS))
        self.assertFalse(guard.finish(epoch, TurnReason.EOS))
        self.assertEqual(guard.snapshot().reason, TurnReason.EOS)
        self.assertFalse(guard.is_current(epoch))
        epoch = self.capture(guard)
        self.assertTrue(guard.finish(epoch, TurnReason.FAILURE))
        self.assertEqual(guard.snapshot().reason, TurnReason.FAILURE)
        epoch = guard.speech_onset()
        self.assertTrue(guard.finish_interrupt(epoch, capture=False))
        self.assertEqual(guard.snapshot().state, TurnState.LISTENING)
        self.assertFalse(guard.is_current(epoch))
        epoch = self.capture(guard)
        self.assertFalse(guard.close())
        self.assertTrue(guard.stop())
        self.assertFalse(guard.stop())
        self.assertFalse(guard.is_current(epoch))
        self.assertFalse(guard.commit(epoch))
        self.assertEqual(guard.snapshot().state, TurnState.DRAINING)
        self.assertEqual(guard.snapshot().reason, TurnReason.STOP)
        with self.assertRaises(RuntimeError):
            guard.speech_onset()
        self.assertTrue(guard.close())
        self.assertFalse(guard.close())
        self.assertEqual(guard.snapshot().state, TurnState.CLOSED)
        self.assertEqual(guard.snapshot().reason, TurnReason.STOP)

    def test_stop_at_every_state(self):
        for phase in TurnState:
            with self.subTest(phase=phase):
                guard = self.guard()
                epoch = 0
                if phase is not TurnState.LISTENING:
                    epoch = guard.speech_onset()
                if phase in (TurnState.CAPTURING, TurnState.THINKING, TurnState.SPEAKING):
                    guard.finish_interrupt(epoch)
                if phase in (TurnState.THINKING, TurnState.SPEAKING):
                    guard.commit(epoch)
                if phase is TurnState.SPEAKING:
                    guard.speaking(epoch)
                if phase in (TurnState.DRAINING, TurnState.CLOSED):
                    guard.stop()
                if phase is TurnState.CLOSED:
                    guard.close()
                self.assertEqual(guard.snapshot().state, phase)
                guard.stop()
                self.assertFalse(guard.is_current(epoch))
                self.assertFalse(guard.finish_interrupt(epoch))
                self.assertFalse(guard.commit(epoch))
                self.assertFalse(guard.speaking(epoch))
                self.assertFalse(guard.finish(epoch, TurnReason.FAILURE))

    def test_deadline_and_bounded_turns(self):
        with patch('research.h3_personal.turn_epoch.time.monotonic', return_value=100):
            guard = TurnEpoch(deadline=110, max_turns=2)
            first = self.capture(guard)
            second = guard.speech_onset()
            self.assertGreater(second, first)
            with self.assertRaises(RuntimeError):
                guard.speech_onset()
            self.assertTrue(guard.finish_interrupt(second))
        with patch('research.h3_personal.turn_epoch.time.monotonic', return_value=110):
            self.assertFalse(guard.commit(second))
            self.assertTrue(guard.expire())
            self.assertFalse(guard.is_current(second))
            self.assertFalse(guard.stop())
            self.assertEqual(guard.snapshot().reason, TurnReason.DEADLINE)
            self.assertTrue(guard.close())

    def test_invalid_configuration_and_reasons(self):
        for deadline in (float('inf'), float('nan'), -1, time.monotonic() - 1):
            with self.assertRaises(ValueError):
                TurnEpoch(deadline=deadline)
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.guard(max_turns=limit)
        guard = self.guard()
        epoch = self.capture(guard)
        for reason in (TurnReason.STOP, TurnReason.INTERRUPT, TurnReason.DEADLINE, 'eos'):
            with self.assertRaises(ValueError):
                guard.finish(epoch, reason)
        self.assertFalse(guard.is_current(True))
        self.assertFalse(guard.commit(True))
        self.assertFalse(guard.finish_interrupt(True))


if __name__ == '__main__':
    unittest.main()
