import unittest
from dataclasses import replace
from fractions import Fraction as F
from hashlib import sha256

from research.h3_personal.conversation_state import SpeechSource
from research.h3_personal.av_delivery import (
    Binding, Segment, Delivery, Backpressure, FakeClock, HeadlessConsumer,
)


class AVDeliveryTests(unittest.TestCase):
    def setup_delivery(self, rate=48000, samples=None):
        samples = samples if samples is not None else rate * 301 // 25
        pcm = bytes(range(256)) * (samples * 2 // 256) + bytes(range(samples * 2 % 256))
        source = SpeechSource(sha256(pcm).hexdigest(), rate, samples)
        binding = Binding('session', 'original', source, 'a' * 64)
        return Delivery(binding, pcm), pcm

    def segments(self, delivery):
        start = 0
        for index, frames in enumerate((119, 102, 107)):
            yield Segment(delivery.binding, index, start, start + frames, 'b' * 64)
            start += frames

    def test_original_pcm_and_padding(self):
        for rate in (48000, 44100):
            d, pcm = self.setup_delivery(rate)
            original = []
            padding = 0
            previous = 0
            for s in self.segments(d):
                self.assertEqual(s.sample_start, previous)
                self.assertEqual(s.binding.boundary(s.frame_stop), s.frame_stop * rate // 24)
                d.enqueue(s)
                lease = d.lease()
                original.append(lease.original_pcm)
                padding += s.padding_samples
                self.assertLess(abs(s.audio_pts - s.video_pts), F(1, rate))
                d.ack(lease)
                previous = s.sample_stop
            self.assertEqual(b''.join(original), pcm)
            self.assertEqual(padding, 328 * rate // 24 - d.binding.source.samples)
            d.close_producer()
            self.assertTrue(d.final_flush())
            with self.assertRaises(RuntimeError):
                d.lease()

    def test_no_per_window_rounding(self):
        d, _ = self.setup_delivery(44100)
        segments = list(self.segments(d))
        self.assertEqual(segments[0].sample_stop, 218662)
        self.assertEqual(segments[1].sample_stop, 406087)
        # Rational non-integer frame rate uses the same global-boundary rule.
        b = replace(d.binding, fps=F(24000, 1001))
        self.assertEqual(b.boundary(221), (F(221 * 44100 * 1001, 24000)).__floor__())

    def test_queue_lease_backpressure_stale_ack(self):
        d, _ = self.setup_delivery()
        a, b, c = self.segments(d)
        d.enqueue(a)
        lease = d.lease()
        d.enqueue(b)
        self.assertEqual(d.outstanding, 2)
        with self.assertRaises(Backpressure):
            d.enqueue(c)
        with self.assertRaises(RuntimeError):
            d.lease()
        with self.assertRaises(ValueError):
            d.ack(replace(lease))
        d.ack(lease)
        with self.assertRaises(ValueError):
            d.ack(lease)
        d.enqueue(c)
        d.close_producer()
        self.assertFalse(d.final_flush())
        for _ in range(2):
            d.ack(d.lease())
        self.assertTrue(d.final_flush())

    def test_cancel_failure_uncertain_never_advance(self):
        for operation in ('cancel', 'sink_failed', 'uncertain'):
            d, _ = self.setup_delivery()
            d.enqueue(next(self.segments(d)))
            lease = d.lease()
            if operation == 'uncertain':
                d.ack(lease, certain=False)
            else:
                getattr(d, operation)()
            self.assertEqual((d.acked_frame, d.acked_sample), (0, 0))
            self.assertEqual(d.outstanding, 0)
            for call in (lambda: d.ack(lease), d.lease, d.close_producer, d.final_flush):
                with self.assertRaises(RuntimeError):
                    call()

    def test_binding_committed_and_monotonic(self):
        d, pcm = self.setup_delivery()
        a, b, _ = self.segments(d)
        for binding in (replace(d.binding, session_id='other'),
                        replace(d.binding, source_id='other'),
                        replace(d.binding, profile_sha256='c' * 64),
                        replace(d.binding, source=replace(d.binding.source, digest='d' * 64))):
            with self.assertRaises(ValueError):
                d.enqueue(replace(a, binding=binding))
        with self.assertRaises(ValueError):
            Delivery(d.binding, pcm[:-2])
        with self.assertRaises(ValueError):
            Delivery(d.binding, b'0' * len(pcm))
        with self.assertRaises(ValueError):
            d.enqueue(b)
        for committed in (False, 1, 'true'):
            with self.assertRaises(ValueError):
                replace(a, committed=committed)
        d.enqueue(a)
        with self.assertRaises(ValueError):
            d.enqueue(a)
        with self.assertRaises(ValueError):
            d.close_producer()
        with self.assertRaises(RuntimeError):
            d.final_flush()

    def test_strict_bool_ints(self):
        d, pcm = self.setup_delivery()
        a = next(self.segments(d))
        for field in ('index', 'frame_start', 'frame_stop'):
            with self.assertRaises(ValueError):
                replace(a, **{field: True})
        with self.assertRaises(ValueError):
            replace(d.binding, source_frame=True)
        with self.assertRaises(ValueError):
            replace(d.binding, fps=True)
        for field in ('max_segments', 'max_frames'):
            with self.assertRaises(ValueError):
                Delivery(d.binding, pcm, **{field: True})
        d.enqueue(a)
        lease = d.lease()
        with self.assertRaises(ValueError):
            d.ack(lease, certain=1)
        d.ack(lease)

    def test_delay_shared_clock_and_final_flush(self):
        d, _ = self.setup_delivery()
        a, b, c = self.segments(d)
        clock = FakeClock()
        sink = HeadlessConsumer(d, clock)
        self.assertFalse(sink.poll())
        clock.advance(F(2))
        d.enqueue(a)
        self.assertTrue(sink.poll())
        self.assertEqual(sink.first_av_at, F(2))
        self.assertFalse(d.closed)  # first AV precedes final producer completion
        clock.advance(F(118, 24))
        self.assertFalse(sink.poll())
        self.assertEqual((d.acked_frame, d.acked_sample), (0, 0))
        clock.advance(F(1, 24))
        self.assertFalse(sink.poll())
        self.assertEqual(d.acked_frame, 119)
        clock.advance(F(3, 2))
        d.enqueue(b)
        self.assertTrue(sink.poll())
        self.assertEqual(sink.stall, F(3, 2))
        d.enqueue(c)
        d.close_producer()
        clock.advance(F(102, 24))
        self.assertTrue(sink.poll())
        clock.advance(F(107, 24))
        self.assertFalse(sink.poll())
        self.assertEqual(d.acked_sample, 577920)
        self.assertTrue(d.final_flush())

    def test_finite_and_clock_refusal(self):
        d, _ = self.setup_delivery()
        a = next(self.segments(d))
        with self.assertRaises(ValueError):
            d.enqueue(replace(a, frame_stop=120))
        clock = FakeClock()
        sink = HeadlessConsumer(d, clock)
        clock.advance(F(1))
        sink.poll()
        clock.now = F(0)
        with self.assertRaises(ValueError):
            sink.poll()
        with self.assertRaises(ValueError):
            clock.advance(True)


if __name__ == '__main__':
    unittest.main()
