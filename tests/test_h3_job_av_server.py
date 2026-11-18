import socket
import threading
import time
import unittest

from research.h3_personal.job_av_server import JobAVServer
from research.h3_personal.live_relay import MAX_CHUNK, RelayConfig, RelayError
from research.h3_personal.live_relay_client import SocketAdapter


def config(**kw):
    return RelayConfig('job-test', time.time() + 10, 'a' * 64, 'b' * 64, **kw)


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class JobAVServerTests(unittest.TestCase):
    def consumer(self, job):
        c = job.server.relay.config
        return SocketAdapter(
            'http://127.0.0.1:' + str(job.server.server_port),
            session=c.session, not_after=c.not_after, role='consumer',
            role_key=c.consumer_key, hf_token='test-only',
            allow_test_loopback=True).open_claim()

    def test_http_av_ack_input_and_split_cache(self):
        with JobAVServer(config(wait_timeout=.01), port=port()) as job:
            consumer = self.consumer(job)
            try:
                p = job.producer
                p.settimeout(2)
                self.assertEqual(p.send(b'video+original-pcm'), 18)
                self.assertEqual(consumer.recv(100), b'video+original-pcm')
                consumer.send(b'ack-one/ack-two')
                self.assertEqual(p.recv(3), b'ack')
                self.assertEqual(p.recv(100), b'-one/ack-two')
                consumer.send_input(b'fresh-input')
                self.assertEqual(p.read_input(), b'fresh-input')
                with self.assertRaises(RelayError):
                    job.server.relay.claim('producer')
            finally:
                consumer.close()
        self.assertFalse(job._thread.is_alive())
        self.assertTrue(all(not q for q in job.server.relay._queues.values()))

    def test_chunking_and_backpressure_terminal_timeout(self):
        with JobAVServer(config(av_capacity=MAX_CHUNK), port=port()) as job:
            p = job.producer
            self.assertEqual(p.send(b'x' * (MAX_CHUNK + 10)), MAX_CHUNK)
            p.settimeout(.02)
            with self.assertRaises(TimeoutError):
                p.send(b'next')
            with self.assertRaises(RelayError):
                job.server.relay.status()
            self.assertTrue(all(not q for q in job.server.relay._queues.values()))

    def test_empty_is_not_eof_and_abort_wakes_reader(self):
        with JobAVServer(config(wait_timeout=0), port=port()) as job:
            p = job.producer
            p.settimeout(.02)
            with self.assertRaises(TimeoutError):
                p.recv(1)
            p.settimeout(2)
            errors = []
            def read():
                try:
                    p.read_input()
                except RelayError as exc:
                    errors.append(exc.status)
            t = threading.Thread(target=read)
            t.start()
            p.abort()
            t.join(1)
            self.assertFalse(t.is_alive())
            self.assertEqual(errors, [410])

    def test_ttl_and_explicit_port(self):
        with self.assertRaises(ValueError):
            JobAVServer(config(), port=0)
        c = RelayConfig('short', time.time() + .08, 'a' * 64, 'b' * 64)
        with JobAVServer(c, port=port()) as job:
            time.sleep(.1)
            with self.assertRaises(RelayError):
                job.producer.recv(1)


if __name__ == '__main__':
    unittest.main()
