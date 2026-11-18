import unittest
from dataclasses import replace

from research.h3_personal.paired_av import (
    LiveAVBinding, AudioRegion, VerifiedReceipt, encode, ProtocolError, CreditError)
from research.h3_personal.paired_av_sender import Sender, Ticket


class SenderTests(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.b = LiveAVBinding('s', 'a' * 64, 1)
        self.s = Sender(self.b, 10, lambda: self.now)

    def envelope(self, i=0, final=False, binding=None):
        return encode(binding or self.b, i, i, i + 1, b'v', bytes(4000),
                      (AudioRegion('silence', i * 2000, (i + 1) * 2000),), final)

    def wire(self, e, stage='verified'):
        return dict(schema='h3-paired-av-ack-v1', session=e.binding.session,
                    epoch=e.binding.epoch, profile_sha256=e.binding.profile_sha256,
                    index=e.index, video_sha256=e.video_sha256,
                    pcm_sha256=e.pcm_sha256, stage=stage)

    def test_two_credits_and_recovery(self):
        a = self.s.reserve(self.envelope())
        self.s.reserve(self.envelope(1))
        with self.assertRaises(CreditError):
            self.s.reserve(self.envelope(2))
        self.s.verified(a)
        with self.assertRaises(CreditError):
            self.s.reserve(self.envelope(2))
        self.s.sink_ack(a, certainty=True)
        self.s.reserve(self.envelope(2))
        self.assertEqual(self.s.snapshot()['pending'], 2)

    def test_verification_not_release(self):
        e = self.envelope()
        self.s.reserve(e)
        self.s.verified(VerifiedReceipt(e))
        self.assertEqual(self.s.snapshot()['inflight_bytes'], 4001)
        self.s.sink_ack(VerifiedReceipt(e), certainty=True)
        self.assertEqual(self.s.snapshot()['inflight_bytes'], 0)

    def test_fifo(self):
        self.s.reserve(self.envelope())
        b = self.s.reserve(self.envelope(1))
        self.s.verified(b)
        with self.assertRaises(ProtocolError):
            self.s.sink_ack(b, certainty=True)
        self.assertEqual(self.s.snapshot()['pending'], 0)

    def test_wire_spoofs(self):
        for field, value in [('epoch', True), ('epoch', 2),
                             ('video_sha256', 'b' * 64), ('index', False),
                             ('session', 'other')]:
            with self.subTest(field=field, value=value):
                s = Sender(self.b, 10, lambda: 1)
                e = self.envelope()
                s.reserve(e)
                msg = self.wire(e)
                msg[field] = value
                with self.assertRaises(ProtocolError):
                    s.ack(msg)
                self.assertIsNotNone(s.snapshot()['terminal'])

    def test_wire_success(self):
        e = self.envelope()
        self.s.reserve(e)
        self.s.ack(self.wire(e))
        self.s.ack(self.wire(e, 'sink-accepted'))
        self.assertEqual(self.s.snapshot()['acked'], 1)

    def test_ambiguity_and_unverified(self):
        for verified, certainty in [(False, True), (True, False), (True, 1)]:
            s = Sender(self.b, 10, lambda: 1)
            t = s.reserve(self.envelope())
            if verified:
                s.verified(t)
            with self.assertRaises(ProtocolError):
                s.sink_ack(t, certainty=certainty)
            self.assertEqual(s.snapshot()['inflight_bytes'], 0)

    def test_clock(self):
        for now in [10, float('nan'), float('inf'), 0]:
            s = Sender(self.b, 10, lambda: self.now)
            self.now = now
            with self.assertRaises(ProtocolError):
                s.reserve(self.envelope())
            self.now = 1
        values = iter([1, 2, 10])
        s = Sender(self.b, 10, lambda: next(values))
        with self.assertRaises(ProtocolError):
            s.reserve(self.envelope())
        self.assertEqual(s.snapshot()['pending'], 0)

    def test_replay_and_forgery(self):
        for action in ['reserve', 'ticket', 'duplicate']:
            s = Sender(self.b, 10, lambda: 1)
            e = self.envelope()
            t = s.reserve(e)
            with self.assertRaises(ProtocolError):
                if action == 'reserve':
                    s.reserve(e)
                elif action == 'ticket':
                    s.verified(Ticket())
                else:
                    s.verified(t)
                    s.verified(t)

    def test_finality_and_limits(self):
        self.s.reserve(self.envelope(final=True))
        with self.assertRaises(ProtocolError):
            self.s.reserve(self.envelope(1))
        b = replace(self.b, max_segments=1)
        s = Sender(b, 10, lambda: 1)
        with self.assertRaises(ProtocolError):
            s.reserve(self.envelope(binding=b))

    def test_byte_capacity(self):
        b = replace(self.b, max_inflight_bytes=4001)
        s = Sender(b, 10, lambda: 1)
        t = s.reserve(self.envelope(binding=b))
        with self.assertRaises(CreditError):
            s.reserve(self.envelope(1, binding=b))
        s.verified(t)
        s.sink_ack(t, certainty=True)
        s.reserve(self.envelope(1, binding=b))

    def test_metadata_and_cleanup(self):
        e = self.envelope()
        t = self.s.reserve(e)
        object.__setattr__(e, 'video_sha256', 'b' * 64)
        with self.assertRaises(ProtocolError):
            self.s.verified(VerifiedReceipt(e))
        for method in ['close', 'cancel', 'disconnect', 'failed']:
            s = Sender(self.b, 10, lambda: 1)
            t = s.reserve(self.envelope())
            getattr(s, method)(t) if method == 'failed' else getattr(s, method)()
            self.assertEqual(s.snapshot()['inflight_bytes'], 0)
            with self.assertRaises(ProtocolError):
                s.verified(t)


if __name__ == '__main__':
    unittest.main()
