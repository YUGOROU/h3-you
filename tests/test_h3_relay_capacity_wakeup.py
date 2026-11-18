"""CPU contracts: opt-in prefetch wiring and relay capacity wake-up.

No GPU, provider, device or external network.
"""
import hashlib
import threading
import time
import unittest

from research.h3_personal.live_relay import Relay, RelayConfig, RelayError, CAPS
from research.h3_personal.paired_av_pump import ReceiverPump
from research.run_h3_relay_conversation import build_receiver_pump


class RecordingFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, transport, binding, root, sink, *, deadline, **kwargs):
        self.calls.append(dict(deadline=deadline, **kwargs))
        return 'pump'


class PrefetchWiringTests(unittest.TestCase):
    def test_default_factory_receives_explicit_prefetch(self):
        factory = RecordingFactory()
        self.assertEqual(build_receiver_pump(factory, 't', 'b', 'r', 's', deadline=1,
                                            prefetch_receive=False), 'pump')
        self.assertEqual(factory.calls, [dict(deadline=1)])
        factory.calls.clear()
        build_receiver_pump(factory, 't', 'b', 'r', 's', deadline=1, prefetch_receive=True)
        # An injected factory is never silently reconfigured.
        self.assertEqual(factory.calls, [dict(deadline=1)])

    def test_real_default_factory_opt_in_is_bounded_prefetch(self):
        import tempfile, socket
        from pathlib import Path
        from test_h3_paired_av_pump import DummySink
        from research.h3_personal.paired_av import LiveAVBinding
        with tempfile.TemporaryDirectory() as root:
            binding = LiveAVBinding('wiring', 'a' * 64, 1)
            a, c = socket.socketpair()
            try:
                sink = DummySink(root, binding)
                pump = build_receiver_pump(ReceiverPump, a, binding, root, sink,
                                           deadline=time.monotonic() + 5, prefetch_receive=True)
                self.assertTrue(pump.prefetch)
                plain = build_receiver_pump(ReceiverPump, c, binding, root, sink,
                                            deadline=time.monotonic() + 5, prefetch_receive=False)
                self.assertFalse(plain.prefetch)
            finally:
                a.close(); c.close()

    def test_non_bool_prefetch_rejected(self):
        factory = RecordingFactory()
        for value in (1, 'yes', None):
            with self.assertRaises(ValueError):
                build_receiver_pump(factory, 't', 'b', 'r', 's', deadline=1,
                                    prefetch_receive=value)
        self.assertEqual(factory.calls, [])


class NotifyingCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.notifications = 0

    def notify_all(self):
        self.notifications += 1
        super().notify_all()


class CapacityWakeupTests(unittest.TestCase):
    def setUp(self):
        from research.h3_personal.job_av_server import LocalProducer
        self.relay = Relay(RelayConfig('wake', time.time() + 30, 'a' * 64, 'b' * 64,
                                       av_capacity=CAPS['av']))
        self.relay._cv = NotifyingCondition()
        self.producer = LocalProducer(self.relay)

    def test_take_notifies_capacity_waiter(self):
        chunk = b'\x01' * 1024
        # Fill the relay completely so the next send must wait for capacity.
        with self.relay._cv:
            while True:
                try:
                    ticket = self.relay.reserve('av', len(chunk))
                except RelayError as exc:
                    self.assertEqual(exc.status, 429)
                    break
                self.relay.commit(ticket, chunk)
        self.producer.settimeout(5)
        blocked, errors, results = threading.Event(), [], []

        def send():
            blocked.set()
            try:
                results.append(self.producer.send(chunk))
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=send)
        worker.start()
        self.assertTrue(blocked.wait(2))
        time.sleep(0.05)
        self.assertEqual(results, [])
        with self.relay._cv:
            taken = self.relay.take('av')
        self.assertGreaterEqual(len(taken), len(chunk))
        self.assertEqual(taken[:len(chunk)], chunk)
        self.assertGreaterEqual(self.relay._cv.notifications, 1)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(results, [len(chunk)])

    def test_terminal_take_does_not_report_success_bytes(self):
        with self.relay._cv:
            self.relay._end()
        with self.assertRaises(RelayError):
            with self.relay._cv:
                self.relay.take('av')


if __name__ == '__main__':
    unittest.main()
