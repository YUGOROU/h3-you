"""CPU-only real socket continuous download contract tests."""
import threading
import time
import unittest
from unittest import mock

from research.live_relay_server import build_server
from research.h3_personal.live_relay import RelayConfig
from research.h3_personal.job_av_server import LocalProducer
from research.h3_personal.continuous_av_client import ContinuousConsumerAdapter
from research.h3_personal.live_relay_client import RelayError


class ContinuousTests(unittest.TestCase):
    def setup_carrier(self, enabled=True, timeout=1., progress_timeout=10., lease=8):
        cfg = RelayConfig('continuous-test', time.time()+lease, 'a'*64, 'b'*64,
                          wait_timeout=.02)
        server = build_server(('127.0.0.1', 0), cfg, continuous_av=enabled)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        producer = LocalProducer(server.relay)
        client = ContinuousConsumerAdapter('http://127.0.0.1:%d' % server.server_port,
            session=cfg.session, not_after=cfg.not_after, role='consumer', role_key='b'*64,
            hf_token='test', allow_test_loopback=True, http_timeout=timeout,
            stream_progress_timeout=progress_timeout)
        client.open_claim()
        def cleanup():
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(1)
            self.assertFalse(thread.is_alive())
        self.addCleanup(cleanup)
        return server, producer, client

    def test_prefix_fragmentation_and_separate_control(self):
        _, producer, client = self.setup_carrier()
        producer.send(b'prefix')
        self.assertEqual(client.recv(65536), b'prefix')  # No remainder exists yet.
        client.send(b'ACK')
        self.assertEqual(producer.recv(20), b'ACK')
        client.send_input(b'control')
        self.assertEqual(producer.read_input(), b'control')
        producer.send(b'abcdef')
        got = b''
        while len(got) < 6:
            got += client.recv(2)
        self.assertEqual(got, b'abcdef')

    def test_progress_timeout(self):
        _, producer, client = self.setup_carrier(timeout=.15)
        producer.send(b'a')
        self.assertEqual(client.recv(1), b'a')
        time.sleep(.09)
        producer.send(b'b')
        self.assertEqual(client.recv(1), b'b')
        start = time.monotonic()
        with self.assertRaises(RelayError):
            client.recv(1)
        self.assertLess(time.monotonic()-start, .8)
        self.assertEqual(client.last_failure['error_type'], 'TimeoutError')

    def test_explicit_progress_budget_survives_more_than_ten_seconds(self):
        # Virtual progress clock only; real socket/lease clocks stay untouched.
        _, producer, client = self.setup_carrier(timeout=20., progress_timeout=12., lease=30)
        producer.send(b'a')
        with mock.patch('research.h3_personal.continuous_av_client.time') as clock:
            clock.monotonic.return_value = 100.
            self.assertEqual(client.recv(1), b'a')
            self.assertEqual(client._progress_end, 112.)
            clock.monotonic.return_value = 111.
            self.assertGreater(client._stream_remaining(), 0)
            clock.monotonic.return_value = 112.
            with self.assertRaises(TimeoutError):
                client._stream_remaining()

    def test_default_budget_unchanged_and_explicit_budget_cannot_escape_bounds(self):
        _, producer, client = self.setup_carrier(timeout=20., lease=30)
        self.assertEqual(client._stream_progress_timeout, 10.)
        producer.send(b'a')
        with mock.patch('research.h3_personal.continuous_av_client.time') as clock:
            clock.monotonic.return_value = 100.
            self.assertEqual(client.recv(1), b'a')
            clock.monotonic.return_value = 111.
            with self.assertRaises(TimeoutError):
                client._stream_remaining()
        _, _, capped = self.setup_carrier(timeout=.2, progress_timeout=120.)
        self.assertEqual(capped._stream_progress_timeout, .2)
        _, _, lease_capped = self.setup_carrier(timeout=120., progress_timeout=120., lease=3)
        self.assertLessEqual(lease_capped._stream_progress_timeout, 3)
        # Absolute lease always wins even before an explicitly longer progress end.
        lease_capped._progress_end = time.monotonic()+120
        lease_capped._deadline = time.monotonic()-1
        with self.assertRaises(TimeoutError):
            lease_capped._stream_remaining()

    def test_invalid_progress_budget(self):
        for value in (True, None, '12', 0, -1, 121, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContinuousConsumerAdapter(role='consumer', stream_progress_timeout=value)

    def test_terminal_no_replay(self):
        server, producer, client = self.setup_carrier()
        producer.send(b'a')
        self.assertEqual(client.recv(1), b'a')
        server.relay.abort()
        with self.assertRaises(RelayError):
            client.recv(1)
        with self.assertRaises(RelayError):
            client.recv(1)

    def test_midstream_eof_no_replay(self):
        _, producer, client = self.setup_carrier()
        producer.send(b'prefix')
        self.assertEqual(client.recv(10), b'prefix')
        client._stream_socket.shutdown(2)  # Missing HTTP terminal chunk.
        with self.assertRaises(RelayError):
            client.recv(1)
        self.assertTrue(client._stream_attempted)
        with self.assertRaises(RelayError):
            client.recv(1)

    def test_empty_polls_are_not_eof(self):
        _, producer, client = self.setup_carrier()
        producer.send(b'a')
        self.assertEqual(client.recv(1), b'a')
        time.sleep(.08)  # Multiple server empty polls, no protocol heartbeat.
        producer.send(b'b')
        self.assertEqual(client.recv(1), b'b')

    def test_disabled(self):
        _, _, client = self.setup_carrier(enabled=False)
        with self.assertRaises(RelayError):
            client.recv(1)

    def test_flags_and_role(self):
        cfg = RelayConfig('flags', time.time()+5, 'a'*64, 'b'*64)
        for value in (1, None, 'true'):
            with self.assertRaises(ValueError):
                build_server(('127.0.0.1', 0), cfg, continuous_av=value)
        with self.assertRaises(ValueError):
            ContinuousConsumerAdapter(role='producer')

    def test_close_interrupts_receive(self):
        _, producer, client = self.setup_carrier()
        producer.send(b'a')
        client.recv(1)
        done = threading.Event()
        def read():
            try:
                client.recv(1)
            except RelayError:
                pass
            finally:
                done.set()
        thread = threading.Thread(target=read)
        thread.start()
        time.sleep(.03)
        client.close()
        thread.join(.8)
        self.assertTrue(done.is_set())
        self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
