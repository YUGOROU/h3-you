"""Exclusive sink-first control and stale Pi races, no devices/network."""
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest

from research.h3_personal.epoch_av_sink import EpochAVSink
from research.h3_personal.live_relay_commands import Commands
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.turn_control import TurnController


class Transport:
    def __init__(self):self.rows = []
    def send_input(self, data):
        self.rows.append(json.loads(data[4:]))
        return len(data)


class TurnControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.sink = EpochAVSink(Path(self.tmp.name), LiveAVBinding('session', 'a'*64, 1))
        self.addCleanup(self.sink.close)
        self.transport = Transport()
        self.commands = Commands(self.transport, 'session', turn_interruptions=True)
        self.control = TurnController(self.commands, self.sink, deadline=time.monotonic()+30)

    def test_old_pi_callback_never_publishes_after_new_onset(self):
        first = self.control.begin_input()
        self.assertTrue(self.control.commit_input(first))
        self.assertFalse(self.control.commit_input(first))
        second = self.control.begin_input()
        self.assertFalse(self.control.publish_reply(first, {'private': 'stale proposal'}))
        self.assertTrue(self.control.commit_input(second))
        self.assertTrue(self.control.publish_reply(second, {'reply': 'current proposal'}))
        self.assertEqual([(r['kind'], r['epoch']) for r in self.transport.rows], [('turn', 2)])
        self.assertFalse(self.sink.timing['discontinuities'])  # no remote speech was published

    def test_sink_invalidation_precedes_remote_interrupt_and_new_publication(self):
        first = self.control.begin_input(); self.control.commit_input(first)
        self.assertTrue(self.control.publish_reply(first, {'reply': 'first'}))
        original = self.transport.send_input
        def send(data):
            row = json.loads(data[4:])
            if row['kind'] == 'interrupt_turn':
                self.assertEqual(self.sink.timing['discontinuities'][0]['epoch'], 1)
                self.assertFalse(self.control.current(first))
            return original(data)
        self.transport.send_input = send
        second = self.control.begin_input(); self.control.commit_input(second)
        self.assertTrue(self.control.publish_reply(second, {'reply': 'second'}))
        self.assertEqual([r['kind'] for r in self.transport.rows], ['turn', 'interrupt_turn', 'turn'])
        self.assertEqual(self.transport.rows[1]['payload']['presentation_epoch'], 1)
        self.assertEqual(self.control.snapshot()['published_turn_epoch'], 2)

    def test_inflight_control_send_cannot_delay_local_invalidation(self):
        entered, release = threading.Event(), threading.Event(); outcomes = []
        original = self.transport.send_input
        def send(data):
            row = json.loads(data[4:])
            if row['kind'] == 'turn' and row['epoch'] == 1:
                entered.set(); release.wait(2)
            return original(data)
        self.transport.send_input = send
        first = self.control.begin_input(); self.control.commit_input(first)
        publication = threading.Thread(target=lambda: outcomes.append(
            self.control.publish_reply(first, {'reply': 'old'})))
        publication.start(); self.assertTrue(entered.wait(1))
        inputs = []
        onset = threading.Thread(target=lambda: inputs.append(self.control.begin_input()))
        onset.start()
        limit = time.monotonic()+1
        while not self.sink.timing['discontinuities'] and time.monotonic() < limit:
            time.sleep(.001)
        self.assertTrue(self.sink.timing['discontinuities'])
        self.assertFalse(self.control.current(first))
        self.assertTrue(onset.is_alive())  # control lane may wait, local output is already invalid
        release.set(); publication.join(2); onset.join(2)
        self.assertFalse(publication.is_alive()); self.assertFalse(onset.is_alive())
        self.assertEqual(outcomes, [False]); self.assertEqual(inputs, [2])
        self.assertEqual([r['kind'] for r in self.transport.rows], ['turn', 'interrupt_turn'])

    def test_uncertain_send_is_terminal_without_replay(self):
        first = self.control.begin_input(); self.control.commit_input(first)
        self.transport.send_input = lambda _: 0
        with self.assertRaises(ValueError):
            self.control.publish_reply(first, {'reply': 'uncertain'})
        self.assertTrue(self.control.snapshot()['failed'])
        self.assertFalse(self.control.publish_reply(first, {'reply': 'retry refused'}))
        with self.assertRaises(RuntimeError):self.control.begin_input()

    def test_stop_is_session_stop_not_turn_interrupt(self):
        self.control.begin_input()
        self.assertTrue(self.control.stop()); self.assertFalse(self.control.stop())
        self.assertEqual(self.control.snapshot()['reason'], 'stop')
        self.assertEqual([r['kind'] for r in self.transport.rows], ['stop'])


if __name__ == '__main__':
    unittest.main()
