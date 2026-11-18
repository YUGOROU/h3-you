"""Deterministic CPU-only first-submission clock; no devices/network."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_h3_relay_conversation import Transport, Director, Sink
from research.h3_personal.paired_av import LiveAVBinding
from research.run_h3_relay_conversation import run_conversation


class FirstAVClockTests(unittest.TestCase):
    def exercise(self, *, first=110., lease=1000., mode=None, **options):
        self.now = 100.
        self.transport = Transport()
        self.director = Director()
        self.director.deadline = 2000.
        self.inputs = []
        owner = self

        class ClockSink(Sink):
            @property
            def first_av_submission_monotonic(self):
                return first if first is not None and owner.now >= first else None

        class ClockPump:
            def __init__(self, transport, binding, root, sink, *, deadline):
                self.deadline = deadline
                self.receiver = SimpleNamespace(deadline=deadline)
                owner.pump = self
                owner.sink = sink
                owner.initial_deadline = deadline

            def run(self):
                if mode == 'error':
                    raise ValueError('private error')
                return {'complete': True}

        class Thread:
            def __init__(self, target, **kw):
                self.target = target
                self.alive = False

            def start(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, seconds):
                if not owner.transport.done.is_set():
                    owner.now += seconds
                if mode == 'cancel':
                    cancel.set()
                if owner.transport.done.is_set() or mode in ('error', 'result'):
                    self.target()
                    self.alive = False

        import threading
        cancel = threading.Event()

        def source(**kw):
            self.inputs.append((self.now, kw['deadline']))
            return None

        ready = dict(status='ready', session='fresh', profile_sha256='a'*64,
                     epoch=1, expires_at=lease)
        kwargs = dict(max_seconds=300, hold_listening=True,
                      start_on_first_av=True)
        kwargs.update(options)
        with tempfile.TemporaryDirectory() as root, \
                patch('research.run_h3_relay_conversation.time.monotonic',
                      side_effect=lambda: self.now), \
                patch('research.run_h3_relay_conversation.time.time', return_value=100.), \
                patch('research.run_h3_relay_conversation.threading.Thread', Thread), \
                patch('builtins.print'):
            return run_conversation(
                self.transport, LiveAVBinding('fresh', 'a'*64, 1,
                                               max_total_frames=248),
                Path(root), None, self.director, ready=ready,
                sink_factory=ClockSink, pump_factory=ClockPump,
                input_source=source, cancel=cancel, **kwargs)

    def test_full_300_starts_at_actual_submission_not_ready(self):
        self.exercise()
        self.assertEqual(self.initial_deadline, 1000.)
        self.assertGreaterEqual(self.inputs[0][0], 110.)
        self.assertEqual(self.inputs[0][1], 410.)
        self.assertEqual(self.director.deadline, 410.)
        self.assertEqual(self.pump.deadline, 410.)
        self.assertEqual(self.pump.receiver.deadline, 410.)
        self.assertGreaterEqual(self.now, 408.)
        self.assertLess(self.now, 409.)

    def test_no_first_timeout_and_full_duration_reservation(self):
        for lease, expected in ((1000., 220.), (450., 150.)):
            with self.assertRaises(TimeoutError):
                self.exercise(first=None, lease=lease)
            self.assertAlmostEqual(self.now, expected)
            self.assertEqual(self.inputs, [])
            self.assertTrue(self.sink.canceled)
            self.assertTrue(self.director.closed)
            self.assertTrue(self.transport._closed)

    def test_error_result_cancel_interrupt_wait_and_cleanup(self):
        for mode in ('error', 'result', 'cancel'):
            with self.assertRaises(RuntimeError):
                self.exercise(first=None, mode=mode)
            self.assertLess(self.now, 101.)
            self.assertEqual(self.inputs, [])
            self.assertTrue(self.sink.canceled)
            self.assertTrue(self.director.closed)
            self.assertTrue(self.transport._closed)

    def test_strict_options(self):
        for options in ({'start_on_first_av': 1},
                        {'hold_listening': False}, {'max_seconds': 301}):
            with self.assertRaises(ValueError):
                self.exercise(**options)

    def test_default_clock_does_not_require_first_property(self):
        self.exercise(first=None, start_on_first_av=False,
                      hold_listening=False, max_seconds=10)
        self.assertEqual(self.inputs, [(100., 110.)])


if __name__ == '__main__':
    unittest.main()
