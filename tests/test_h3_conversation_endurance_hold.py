"""CPU-only finite listening hold contracts; no capture or cloud adapters."""
from pathlib import Path
import tempfile
import threading
import time
import unittest

from test_h3_relay_conversation import Transport, Director, Sink, Pump
from research.h3_personal.paired_av import LiveAVBinding
from research.run_h3_relay_conversation import run_conversation


class HoldTests(unittest.TestCase):
    def run_case(self, *, seconds=2.15, pump=Pump, turns=None, **kwargs):
        self.transport = Transport()
        self.director = Director()
        self.started = time.monotonic()
        self.stop_times = []
        send = self.transport.send_input

        def timed_send(raw):
            result = send(raw)
            if self.transport.rows[-1]['kind'] == 'stop':
                self.stop_times.append(time.monotonic())
            return result

        self.transport.send_input = timed_send
        binding = LiveAVBinding('fresh', 'a'*64, 1, max_total_frames=248)
        ready = dict(status='ready', session='fresh', profile_sha256='a'*64,
                     epoch=1, expires_at=time.time()+60)
        with tempfile.TemporaryDirectory() as root:
            return run_conversation(
                self.transport, binding, Path(root), turns, self.director,
                ready=ready, max_seconds=seconds, sink_factory=Sink,
                pump_factory=pump, **kwargs)

    def test_two_turns_hold_then_stop_before_deadline(self):
        self.run_case(turns=['one', 'two'], hold_listening=True)
        self.assertEqual(self.director.inputs, ['one', 'two'])
        self.assertEqual([r['kind'] for r in self.transport.rows],
                         ['turn', 'turn', 'stop'])
        elapsed = self.stop_times[0] - self.started
        self.assertGreaterEqual(elapsed, .14)
        self.assertLess(elapsed, .6)

    def test_source_none_called_once_and_no_extra_capture(self):
        calls = []
        def source(**kw):
            calls.append(kw)
            return None
        self.run_case(input_source=source, hold_listening=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.director.inputs, [])
        self.assertGreaterEqual(self.stop_times[0] - self.started, .14)

    def test_default_and_explicit_false_stop_without_hold(self):
        for options in ({}, {'hold_listening': False}):
            self.run_case(turns=['one'], **options)
            self.assertLess(self.stop_times[0] - self.started, .12)
            self.assertEqual([r['kind'] for r in self.transport.rows], ['turn', 'stop'])

    def test_failure_and_early_end_interrupt_hold(self):
        for fail in (False, True):
            class Ends(Pump):
                def run(self):
                    time.sleep(.03)
                    if fail:
                        raise ValueError('receiver error')
                    return {'complete': True}
            with self.assertRaises(RuntimeError):
                self.run_case(seconds=3, turns=['one'], pump=Ends,
                              hold_listening=True)
            self.assertLess(time.monotonic() - self.started, .5)
            self.assertEqual([r['kind'] for r in self.transport.rows],
                             ['turn', 'cancel_pending', 'stop'])

    def test_cancel_interrupts_hold(self):
        cancel = threading.Event()
        timer = threading.Timer(.04, cancel.set)
        timer.start()
        try:
            with self.assertRaises(RuntimeError):
                self.run_case(seconds=3, turns=['one'], cancel=cancel,
                              hold_listening=True)
        finally:
            timer.join()
        self.assertLess(time.monotonic() - self.started, .5)
        self.assertEqual([r['kind'] for r in self.transport.rows],
                         ['turn', 'cancel_pending', 'stop'])

    def test_hold_bool_and_five_minute_validation(self):
        for value in (None, 0, 1, 'true'):
            with self.assertRaises(ValueError):
                self.run_case(turns=['one'], hold_listening=value)
            self.assertEqual(self.transport.rows, [])
        with self.assertRaises(ValueError):
            self.run_case(seconds=301, turns=['one'], hold_listening=True)

    def test_short_duration_reserves_all_remaining_time_for_drain(self):
        self.run_case(seconds=.3, turns=['one'], hold_listening=True)
        self.assertLess(self.stop_times[0] - self.started, .12)


if __name__ == '__main__':
    unittest.main()
