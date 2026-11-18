import hashlib
import struct
import time
import unittest
from unittest.mock import patch

from research.h3_personal import relay_native_tts as bridge
from research.h3_personal.streaming_pcm import PCMStreamBinding


def floats(*values):
    return struct.pack('<' + 'f'*len(values), *values)


class Transport:
    def __init__(self, chunks=(), terminal=None, close_error=None):
        self.chunks = chunks
        self.terminal = terminal
        self.close_error = close_error
        self.closed = self.finished = self.generator_closed = False
        self.calls = []

    def stream(self, *args):
        self.calls.append(args)
        try:
            yield from self.chunks
            if self.terminal:
                raise self.terminal
            self.finished = True
        finally:
            self.generator_closed = True

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.binding = PCMStreamBinding('conversation', 'source', 7, 'a'*64)
        self.published = []

    def run_bridge(self, tts, **kw):
        args = dict(tts=tts, reply='Hello! Full text。', epoch=7,
                    binding=self.binding, deadline=time.monotonic()+60,
                    publish=self.published.append, is_current=lambda: True)
        args.update(kw)
        return bridge.produce(**args)

    def test_early_once_and_hash_conversion(self):
        values = [-2, -1, -0.5, -1/65536, 0, 1/65536,
                  3/65536, 0.5, 1, 2]
        raw = floats(*values)
        tts = Transport([b'', raw[:12], raw[12:]])
        def publish(stream):
            self.assertFalse(tts.finished)
            self.assertEqual(stream.snapshot()['state'], 'open')
            with self.assertRaises(RuntimeError):
                stream.complete_source()
            self.published.append(stream)
        result = self.run_bridge(tts, publish=publish)
        expected = struct.pack('<10h', -32768, -32768, -16384, 0, 0, 0,
                               2, 16384, 32767, 32767)
        self.assertEqual(result.stream.complete_source().pcm, expected)
        self.assertEqual(result.native_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(result.converted_sha256, hashlib.sha256(expected).hexdigest())
        self.assertEqual((result.native_samples, result.converted_samples), (10, 10))
        self.assertEqual(result.saturation_count, 3)
        self.assertEqual(len(self.published), 1)
        self.assertEqual(tts.calls[0][:2], ('Hello! Full text。', 7))
        self.assertTrue(tts.finished and tts.generator_closed)
        self.assertFalse(tts.closed)
        self.assertEqual(result.completion, 'protocol_END')
        self.assertFalse(result.watermark or result.model_eos_verified)

    def test_explicit_reserve_delays_publication_without_changing_pcm(self):
        from research.h3_personal.tts_preroll import PCMPrerollPolicy
        tts=Transport([floats(.25,.5),floats(-.25,-.5),floats(0)])
        def publish(stream):
            self.assertFalse(tts.finished)
            self.assertEqual(stream.snapshot()['received_samples'],4)
            self.published.append(stream)
        result=self.run_bridge(tts,publish=publish,preroll=PCMPrerollPolicy(reserve_samples=4))
        self.assertEqual(len(self.published),1)
        self.assertEqual(result.preroll.status,'ready')
        self.assertFalse(result.preroll.short_sealed)
        self.assertEqual(result.stream.complete_source().pcm,struct.pack('<5h',8192,16384,-8192,-16384,0))

    def test_short_source_publishes_only_after_verified_seal(self):
        from research.h3_personal.tts_preroll import PCMPrerollPolicy
        tts=Transport([floats(.25)])
        def publish(stream):
            self.assertTrue(tts.finished)
            self.assertEqual(stream.snapshot()['state'],'sealed')
            self.published.append(stream)
        result=self.run_bridge(tts,publish=publish,preroll=PCMPrerollPolicy())
        self.assertTrue(result.preroll.short_sealed)
        self.assertEqual(len(self.published),1)

    def test_reserve_does_not_publish_on_end_failure(self):
        from research.h3_personal.tts_preroll import PCMPrerollPolicy
        tts=Transport([floats(.25)],terminal=ValueError('bad END'))
        with self.assertRaisesRegex(ValueError,'bad END'):
            self.run_bridge(tts,preroll=PCMPrerollPolicy())
        self.assertEqual(self.published,[])
        self.assertTrue(tts.closed)

    def test_chunk_boundaries(self):
        raw = floats(.25)*65536
        result = self.run_bridge(Transport([raw, floats(-.25)]))
        self.assertEqual(result.native_samples, 65537)
        self.assertEqual(result.stream.snapshot()['chunks'], 4)
        self.assertEqual(result.stream.complete_source().pcm,
                         struct.pack('<h', 8192)*65536 + struct.pack('<h', -8192))

    def test_invalid_chunks(self):
        for chunk in (b'x', bytearray(4), 'xxxx', b'\0'*(256*1024+4),
                      floats(float('nan')), floats(float('inf')),
                      floats(float('-inf'))):
            with self.subTest(chunk_type=type(chunk), length=len(chunk)):
                tts = Transport([floats(0), chunk])
                self.published.clear()
                with self.assertRaises(ValueError):
                    self.run_bridge(tts)
                self.assertTrue(tts.closed and tts.generator_closed)
                with self.assertRaises(RuntimeError):
                    self.published[0].complete_source()

    def test_empty_and_empty_chunk_flood_fail_closed(self):
        for chunks,error in [((),ValueError),([b'']*1025,BufferError)]:
            tts=Transport(chunks)
            with self.assertRaises(error):self.run_bridge(tts)
            self.assertTrue(tts.closed and tts.generator_closed)
            self.assertEqual(self.published,[])

    def test_aggregate_limit(self):
        block = floats(0)*65536
        tts = Transport([block]*22)  # 1,441,792 > 30 seconds
        with self.assertRaises(BufferError):
            self.run_bridge(tts)
        self.assertTrue(tts.closed and tts.generator_closed)

    def test_exact_aggregate_limit(self):
        block = floats(0)*65536
        result = self.run_bridge(Transport([block]*21 + [floats(0)*63744]))
        self.assertEqual(result.native_samples, 1440000)

    def test_deadline_and_stale_initial(self):
        for kwargs, error in ((dict(deadline=0), TimeoutError),
                              (dict(is_current=lambda: False), InterruptedError)):
            tts = Transport([floats(0)])
            with self.assertRaises(error):
                self.run_bridge(tts, **kwargs)
            self.assertTrue(tts.closed)
            self.assertFalse(tts.calls)

    def test_deadline_after_remote_call(self):
        tts = Transport([floats(0)])
        with patch.object(bridge.time, 'monotonic', side_effect=[1, 1, 1, 1, 100]):
            with self.assertRaises(TimeoutError):
                self.run_bridge(tts, deadline=50)
        self.assertTrue(tts.closed and tts.generator_closed)

    def test_stale_after_publication(self):
        tts = Transport([floats(0), floats(1)])
        with self.assertRaises(InterruptedError):
            self.run_bridge(tts, is_current=lambda: not self.published)
        self.assertTrue(tts.closed and tts.generator_closed)

    def test_primary_errors_and_generator_close(self):
        for primary in (RuntimeError('publish'), KeyboardInterrupt()):
            tts = Transport([floats(0)], close_error=OSError('close'))
            def publish(stream):
                self.published.append(stream)
                raise primary
            with self.assertRaises(type(primary)) as caught:
                self.run_bridge(tts, publish=publish)
            self.assertIs(caught.exception, primary)
            self.assertTrue(tts.closed and tts.generator_closed)

    def test_terminal_end_error(self):
        primary = ValueError('missing verified END')
        tts = Transport([floats(0)], terminal=primary, close_error=OSError())
        with self.assertRaises(ValueError) as caught:
            self.run_bridge(tts)
        self.assertIs(caught.exception, primary)
        self.assertTrue(tts.closed)
        with self.assertRaises(RuntimeError):
            self.published[0].complete_source()

    def test_iterator_close_failure_priority(self):
        primary = OSError('iterator close')
        class Iterator:
            def __init__(self):
                self.sent = False
            def __iter__(self):
                return self
            def __next__(self):
                if self.sent:
                    raise StopIteration
                self.sent = True
                return floats(0)
            def close(self):
                raise primary
        tts = Transport(close_error=RuntimeError('transport close'))
        tts.stream = lambda *args: Iterator()
        with self.assertRaises(OSError) as caught:
            self.run_bridge(tts)
        self.assertIs(caught.exception, primary)
        self.assertTrue(tts.closed)

    def test_empty_and_invalid_request(self):
        for reply in ('', 'x'*161, '{}'):
            tts = Transport()
            with self.assertRaises(ValueError):
                self.run_bridge(tts, reply=reply)
            self.assertTrue(tts.closed)
        tts = Transport([b''])
        with self.assertRaises(ValueError):
            self.run_bridge(tts)
        self.assertTrue(tts.closed)
        self.assertFalse(self.published)


if __name__ == '__main__':
    unittest.main()
