"""Retain interrupted source bytes without inventing EOS. CPU/fake only."""
import hashlib
import io
import struct
import time
import unittest

from research.h3_personal.relay_native_tts import produce
from research.h3_personal.streaming_pcm import PCMStreamBinding, PCMChunk, PCMIngress
from research.h3_personal.voxcpm_endpoint_service import Gate
from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL, decode


class Candidate:
    sample_rate = 48000
    def stream(self, text):
        yield [.25]
        yield [-.5, .125]


class Transport:
    def __init__(self, first='pcm'):
        self.gate = Gate(Candidate(), key='key', session='s', boot='b',
                         expires_at=time.time()+60, cancel_protocol=CANCEL_PROTOCOL)
        self.first = first
        self.closed = False
        self.last_completion = None
    def close(self): self.closed = True
    def stream(self, text, epoch, deadline):
        parts = []
        request = dict(session='s', boot='b', epoch=epoch, text=text, cancel_protocol=CANCEL_PROTOCOL)
        def write(data):
            parts.append(data)
            if epoch == 1 and data[:1] == (b'R' if self.first == 'ready' else b'P'):
                self.gate.cancel({k:v for k,v in request.items() if k != 'text'})
        self.gate.stream(request, write)
        self.last_completion = yield from decode(io.BytesIO(b''.join(parts)).read,
            dict(session='s', boot='b', expires_at=self.gate.expires_at, epoch=epoch),
            cancel_protocol=CANCEL_PROTOCOL)


class Tests(unittest.TestCase):
    def binding(self, epoch): return PCMStreamBinding('s', f'source-{epoch}', epoch, 'a'*64)
    def run_source(self, tts, epoch, published, **kw):
        return produce(tts, 'hello', epoch, self.binding(epoch), time.monotonic()+5,
                       published.append, lambda: True, **kw)

    def test_interrupted_prefix_readable_next_source_complete_and_exact(self):
        tts = Transport(); published = []
        old = self.run_source(tts, 1, published, allow_interrupted=True)
        self.assertEqual(old.completion, 'interrupted_protocol_END')
        self.assertEqual(old.stream.snapshot()['state'], 'interrupted')
        self.assertEqual(old.stream.read(0, 1).pcm, struct.pack('<h', 8192))
        self.assertEqual(old.stream.prefix_digest(1), old.converted_sha256)
        self.assertEqual(old.stream.read(0, 2).status, 'wait')
        with self.assertRaises(RuntimeError): old.stream.complete_source()
        with self.assertRaises(RuntimeError): old.stream.append(PCMChunk(old.stream.binding, 1, 1, b'\x00\x00'))
        with self.assertRaises(RuntimeError): old.stream.seal(total_samples=1, sha256=old.converted_sha256)
        new = self.run_source(tts, 2, published, allow_interrupted=True)
        self.assertEqual(new.completion, 'protocol_END')
        self.assertEqual(new.stream.complete_source().pcm, struct.pack('<hhh', 8192, -16384, 4096))
        self.assertEqual(new.converted_samples, 3)
        self.assertEqual(old.stream.read(0, 1).pcm, struct.pack('<h', 8192))
        self.assertFalse(tts.closed)
        for receipt in (old, new): receipt.stream.cancel()

    def test_zero_prefix_cancel_stays_unpublished_not_empty_complete_source(self):
        tts = Transport(first='ready'); published = []
        receipt = self.run_source(tts, 1, published, allow_interrupted=True)
        self.assertEqual(receipt.native_samples, 0)
        self.assertEqual(receipt.completion, 'interrupted_protocol_END')
        self.assertEqual(published, [])
        self.assertEqual(receipt.stream.read(0, 1).status, 'wait')
        with self.assertRaises(RuntimeError): receipt.stream.complete_source()
        receipt.stream.cancel()

    def test_default_rejects_cancelled_terminal_not_fake_full_completion(self):
        tts = Transport(); published = []
        with self.assertRaisesRegex(ValueError, 'unaccepted interrupted'):
            self.run_source(tts, 1, published)
        self.assertTrue(tts.closed)
        with self.assertRaises(RuntimeError): published[0].snapshot()

    def test_interrupted_retention_expires_and_wrong_hash_fails_closed(self):
        now = [0.]
        for bad in (True, False):
            stream = PCMIngress(self.binding(1), timeout=1, clock=lambda: now[0])
            stream.append(PCMChunk(stream.binding, 0, 0, b'\x00\x00'))
            if bad:
                with self.assertRaises(ValueError): stream.finish_interrupted(total_samples=1, sha256='a'*64)
                with self.assertRaises(RuntimeError): stream.snapshot()
            else:
                stream.finish_interrupted(total_samples=1, sha256=hashlib.sha256(b'\x00\x00').hexdigest())
                now[0] = 2.
                with self.assertRaises(TimeoutError): stream.read(0, 1)
                with self.assertRaises(RuntimeError): stream.snapshot()


if __name__ == '__main__': unittest.main()
