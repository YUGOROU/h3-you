"""No models, provider, network, device, or native runtime startup."""
import threading
import unittest

from research.h3_personal.warm_conversation_service import (
    AdmissionError, CleanupAck, DeadlineExceeded, ReleaseAck, State,
    WarmConversationService,
)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Adapter:
    def __init__(self):
        self.starts = self.cleanups = self.releases = 0
        self.resource = object()
        self.cleanup_ack = CleanupAck(True, True)
        self.release_ack = ReleaseAck(True)

    def start(self, deadline):
        self.starts += 1
        return self.resource

    def cleanup(self, resources, deadline):
        assert resources is self.resource
        self.cleanups += 1
        return self.cleanup_ack

    def release(self, resources, deadline):
        assert resources is self.resource
        self.releases += 1
        return self.release_ack


class WarmServiceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.adapter = Adapter()
        self.service = self.make()

    def make(self, **overrides):
        args = dict(startup_seconds=10, session_seconds=5,
                    service_seconds=30, idle_seconds=20, clock=self.clock)
        args.update(overrides)
        return WarmConversationService(self.adapter, **args)

    def test_ready_only_after_single_startup_and_retain_across_sessions(self):
        s = self.service
        self.assertEqual(s.state, State.STARTING)
        with self.assertRaises(AdmissionError):
            s.run_session(lambda *_: None)
        s.start()
        for i in range(3):
            def session(resource, deadline):
                self.assertIs(resource, self.adapter.resource)
                self.assertEqual(s.state, State.BUSY)
                return i
            self.assertEqual(s.run_session(session), i)
            self.assertEqual(s.state, State.READY)
        with self.assertRaises(AdmissionError):
            s.start()
        self.assertEqual((self.adapter.starts, self.adapter.cleanups), (1, 3))
        s.close()
        s.close()
        self.assertEqual(s.state, State.CLOSED)
        self.assertEqual(self.adapter.releases, 1)

    def test_invalid_limits(self):
        for key in ('startup_seconds', 'session_seconds', 'service_seconds', 'idle_seconds'):
            for value in (0, -1, float('inf'), float('nan'), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.make(**{key: value})

    def test_startup_timeout_and_no_replay(self):
        def start(deadline):
            self.clock.now = deadline.expires_at
            return self.adapter.resource
        self.adapter.start = start
        with self.assertRaises(DeadlineExceeded):
            self.service.start()
        self.assertEqual(self.service.state, State.QUARANTINED)
        with self.assertRaises(AdmissionError):
            self.service.start()

    def test_startup_exception_unknown_partial_resource(self):
        def fail(_):
            raise RuntimeError('partial startup')
        self.adapter.start = fail
        with self.assertRaises(RuntimeError):
            self.service.start()
        self.service.close()
        self.assertEqual(self.service.state, State.QUARANTINED)
        self.assertEqual(self.adapter.releases, 0)

    def test_expiry_before_start(self):
        self.clock.now = 10
        self.assertEqual(self.service.state, State.DRAINING)
        with self.assertRaises(AdmissionError):
            self.service.start()

    def test_deadlines_do_not_slide_with_admission(self):
        s = self.service
        s.start()
        for now in (1, 7, 14, 19):
            self.clock.now = now
            s.run_session(lambda _, d: self.assertEqual(d.expires_at, min(now + 5, 20)))
        self.assertEqual((s.startup_deadline, s.idle_deadline, s.service_deadline), (10, 20, 30))
        self.clock.now = 20
        self.assertEqual(s.state, State.DRAINING)
        with self.assertRaises(AdmissionError):
            s.run_session(lambda *_: self.fail('admitted after idle'))

    def test_service_caps_every_deadline(self):
        s = self.make(service_seconds=3)
        self.assertEqual(s.startup_deadline, 3)
        self.assertEqual(s.idle_deadline, 3)
        s.start()
        s.run_session(lambda _, d: self.assertEqual(d.expires_at, 3))
        self.clock.now = 3
        self.assertEqual(s.tick(), State.DRAINING)

    def test_late_session_quarantines_without_cleanup_or_replay(self):
        s = self.service
        s.start()
        def late(_, deadline):
            self.clock.now = deadline.expires_at
            self.assertEqual(s.tick(), State.QUARANTINED)
        with self.assertRaises(DeadlineExceeded):
            s.run_session(late)
        self.assertEqual(self.adapter.cleanups, 0)
        with self.assertRaises(AdmissionError):
            s.run_session(late)

    def test_session_failure_never_replayed(self):
        s = self.service
        s.start()
        calls = []
        def fail(*_):
            calls.append(1)
            raise ValueError('unknown native state')
        with self.assertRaises(ValueError):
            s.run_session(fail)
        self.assertEqual(s.state, State.QUARANTINED)
        self.assertEqual(calls, [1])
        self.assertEqual(self.adapter.cleanups, 0)

    def test_cleanup_requires_explicit_both_acknowledgements(self):
        for ack in (None, True, {}, CleanupAck(False, True),
                    CleanupAck(True, False), CleanupAck(1, True)):
            with self.subTest(ack=ack):
                s = self.make()
                s.start()
                self.adapter.cleanup_ack = ack
                with self.assertRaises(RuntimeError):
                    s.run_session(lambda *_: 1)
                self.assertEqual(s.state, State.QUARANTINED)

    def test_cleanup_failure_or_overrun(self):
        for overrun in (False, True):
            s = self.make()
            s.start()
            def cleanup(_, deadline):
                if not overrun:
                    raise RuntimeError('cleanup unknown')
                self.clock.now = deadline.expires_at
                return CleanupAck(True, True)
            self.adapter.cleanup = cleanup
            with self.assertRaises((RuntimeError, DeadlineExceeded)):
                s.run_session(lambda *_: None)
            self.assertEqual(s.state, State.QUARANTINED)

    def test_threaded_concurrent_admission_and_drain(self):
        s = self.service
        s.start()
        entered, finish = threading.Event(), threading.Event()
        errors = []
        def session(*_):
            entered.set()
            if not finish.wait(2):
                raise RuntimeError('test wait exceeded')
        def worker():
            try:
                s.run_session(session)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(AdmissionError):
                s.run_session(lambda *_: self.fail('concurrent'))
            s.close()
            self.assertEqual(s.state, State.DRAINING)
            self.assertEqual(self.adapter.releases, 0)
        finally:
            finish.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], AdmissionError)
        self.assertEqual(s.state, State.QUARANTINED)
        s.close()
        self.assertEqual(s.state, State.CLOSED)

    def test_release_unknown_or_exception_not_retried(self):
        for ack in (None, False, ReleaseAck(False), ReleaseAck(1), 'exception'):
            s = self.make()
            s.start()
            calls = []
            def release(*_):
                calls.append(1)
                if ack == 'exception':
                    raise RuntimeError('native release unknown')
                return ack
            self.adapter.release = release
            with self.assertRaises(RuntimeError):
                s.close()
            s.close()
            self.assertEqual(calls, [1])
            self.assertEqual(s.state, State.QUARANTINED)

    def test_expired_release_is_not_fake_closed(self):
        s = self.service
        s.start()
        self.clock.now = 20
        with self.assertRaises(DeadlineExceeded):
            s.close()
        self.assertEqual(s.state, State.QUARANTINED)
        self.assertEqual(self.adapter.releases, 0)
        s.close()
        self.assertEqual(s.state, State.QUARANTINED)

    def test_unstarted_close(self):
        self.service.close()
        self.assertEqual(self.service.state, State.CLOSED)
        self.assertEqual(self.adapter.starts, 0)


if __name__ == '__main__':
    unittest.main()
