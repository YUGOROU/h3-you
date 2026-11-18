"""CPU wire/credit contracts only; synthetic bytes are not playable media."""
from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import socket
import time
import unittest

from research.h3_personal.paired_av import (
    LiveAVBinding, AudioRegion, PresentationCut, ProtocolError, Receiver,
    ReceivedAVLease, encode,
)
from research.h3_personal.paired_av_sender import Sender
from research.h3_personal import paired_av_wire as wire


def sha(body):
    return hashlib.sha256(body).hexdigest()


class PartialCutTests(unittest.TestCase):
    def setUp(self):
        self.binding = LiveAVBinding('cut-session', 'a'*64, 9)
        self.deadline = time.monotonic()+10
        self.video = b'verified synthetic video'
        self.pcm = b'\x01\x00'*2000 + b'\x02\x00'*4000
        self.regions = (
            AudioRegion('speech', 0, 2000, 'old', 1, 0, sha(self.pcm[:4000])),
            AudioRegion('speech', 2000, 6000, 'new', 2, 0, sha(self.pcm[4000:])),
        )
        self.cut = PresentationCut(1, 0, 1)
        self.e = encode(self.binding, 0, 0, 3, self.video, self.pcm,
                        self.regions, False, presentation_epoch=1,
                        presentation_cut=self.cut)

    def next_pair(self):
        pcm = b'\x03\x00'*4000
        regions = (AudioRegion('speech', 6000, 10000, 'new', 2, 4000, sha(pcm)),)
        return encode(self.binding, 1, 3, 5, self.video, pcm, regions, True,
                      presentation_epoch=1), pcm

    def test_strict_frozen_cut_and_inside_bounds(self):
        for args in ((True, 0, 1), (1, False, 1), (1, 0, True), (0, 0, 1),
                     (14710, 0, 1), (1, -1, 1), (1, 0, 2**63)):
            with self.subTest(args=args), self.assertRaises(ProtocolError):
                PresentationCut(*args)
        with self.assertRaises(AttributeError):
            self.cut.frame = 2
        for cut in (PresentationCut(3, 0, 1), PresentationCut(1, 1, 1)):
            with self.assertRaises(ProtocolError):
                replace(self.e, presentation_cut=cut)
        with self.assertRaises(ProtocolError):
            replace(self.e, presentation_epoch=0)
        with self.assertRaises(ProtocolError):
            replace(self.e, presentation_cut=asdict(self.cut))

    def test_cut_cannot_omit_new_head_or_leave_old_speech(self):
        # Extending the cut omits the beginning of source2 despite full hashes.
        with self.assertRaisesRegex(ProtocolError, 'new speech head'):
            replace(self.e, presentation_cut=PresentationCut(2, 0, 1))
        crossing = (
            AudioRegion('speech', 0, 3000, 'old', 1, 0, 'b'*64),
            AudioRegion('speech', 3000, 6000, 'new', 2, 0, 'c'*64),
        )
        with self.assertRaisesRegex(ProtocolError, 'extends after cut'):
            replace(self.e, regions=crossing)
        # Forged retired epoch would reclassify new speech as cancelled.
        with self.assertRaises(ProtocolError):
            replace(self.e, presentation_cut=PresentationCut(1, 0, 2))

    def test_full_hash_and_old_region_still_verified(self):
        for position in (0, len(self.pcm)-1):
            pcm = bytearray(self.pcm); pcm[position] ^= 1
            r = Receiver(self.binding, deadline=self.deadline)
            with self.assertRaises(ProtocolError):
                r.offer(self.e, self.video, bytes(pcm))
            self.assertIsNotNone(r.snapshot()['terminal'])
        # Updating the full-body hash still cannot bypass old-region digest.
        pcm = b'\x09'+self.pcm[1:]
        e = replace(self.e, pcm_sha256=sha(pcm))
        with self.assertRaises(ProtocolError):
            Receiver(self.binding, deadline=self.deadline).offer(e, self.video, pcm)

    def test_default_wire_encoding_unchanged_and_null_rejected(self):
        plain = replace(self.e, presentation_epoch=0, presentation_cut=None)
        record = wire.envelope_record(plain)
        self.assertNotIn('presentation_cut', record)
        self.assertNotIn('presentation_epoch', record)
        expected = asdict(plain)
        expected.pop('presentation_epoch'); expected.pop('presentation_cut')
        expected['regions'] = [asdict(r) for r in plain.regions]
        self.assertEqual(record, expected)
        self.assertEqual(wire.parse_envelope(dict(schema='h3-paired-av-wire-v1', envelope=record)), plain)
        for cut in (None, {}, dict(asdict(self.cut), forged=True),
                    dict(asdict(self.cut), frame=True)):
            value = dict(record, presentation_epoch=1, presentation_cut=cut)
            with self.assertRaises(ProtocolError):
                wire.parse_envelope(dict(schema='h3-paired-av-wire-v1', envelope=value))

    def test_full_pair_wire_roundtrip_and_ack_metadata(self):
        r = Receiver(self.binding, deadline=self.deadline)
        left, right = socket.socketpair()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                sent = pool.submit(wire.send_pair, left, self.e, self.video, self.pcm,
                                   deadline=self.deadline)
                receipt = wire.receive_pair(right, r, deadline=self.deadline)
                sent.result(timeout=10)
            self.assertEqual(receipt.envelope, self.e)
            lease = r.lease()
            self.assertEqual(lease.audio_pcm, self.pcm)  # no wire truncation
            for stage in ('verified', 'partially-presented-interrupted', 'discarded-interrupted'):
                wire.send_ack(right, self.e, stage=stage, deadline=self.deadline)
                message = wire.receive_ack(left, self.e, stage=stage, deadline=self.deadline)
                self.assertEqual(message['presentation_cut'], asdict(self.cut))
            with self.assertRaises(ProtocolError):
                wire.send_ack(right, self.e, stage='sink-accepted', deadline=self.deadline)
        finally:
            left.close(); right.close()

    def test_partial_fifo_releases_full_credits_and_retains_full_source_clock(self):
        e1, pcm1 = self.next_pair()
        r = Receiver(self.binding, deadline=self.deadline)
        s = Sender(self.binding, self.deadline)
        t0 = s.reserve(self.e); t1 = s.reserve(e1)
        r.offer(self.e, self.video, self.pcm); r.offer(e1, self.video, pcm1)
        s.verified(t0); s.verified(t1)
        lease = r.lease(); self.assertIsNone(r.lease())
        r.ack_partial(lease, certain=True); s.partial_ack(t0, certainty=True)
        self.assertEqual(r.snapshot()['inflight_bytes'], len(self.video)+len(pcm1))
        self.assertEqual(s.snapshot()['inflight_bytes'], len(self.video)+len(pcm1))
        r.ack(r.lease(), certain=True); s.sink_ack(t1, certainty=True)
        for state in (r.snapshot(), s.snapshot()):
            self.assertEqual(state['partial_acked'], 1)
            self.assertEqual(state['disposed'], 2)
            self.assertEqual(state['frame_stop'], 5)
            self.assertEqual(state['inflight_bytes'], 0)
            self.assertTrue(state['transport_complete'])
            self.assertFalse(state['presentation_complete'])
        self.assertEqual(r.snapshot()['sink_acked'], 1)
        self.assertEqual(s.snapshot()['acked'], 1)

    def test_invalid_partial_uncertain_nonfifo_unverified_or_forged_is_terminal(self):
        for mode in ('unverified', 'uncertain', 'nonfifo', 'full-cut'):
            with self.subTest(mode=mode):
                s = Sender(self.binding, self.deadline)
                t0 = s.reserve(self.e)
                t1 = s.reserve(self.next_pair()[0])
                if mode != 'unverified':
                    s.verified(t0); s.verified(t1)
                with self.assertRaises(ProtocolError):
                    if mode == 'full-cut':s.sink_ack(t0, certainty=True)
                    else:s.partial_ack(t1 if mode == 'nonfifo' else t0,
                                       certainty=mode != 'uncertain')
                self.assertIsNotNone(s.snapshot()['terminal'])
        for mode in ('forged', 'uncertain', 'full-cut'):
            with self.subTest(mode=mode):
                r = Receiver(self.binding, deadline=self.deadline)
                r.offer(self.e, self.video, self.pcm); lease = r.lease()
                with self.assertRaises(ProtocolError):
                    if mode == 'full-cut':r.ack(lease)
                    else:r.ack_partial(ReceivedAVLease(lease.segment, lease.video, lease.audio_pcm)
                                       if mode == 'forged' else lease, certain=mode != 'uncertain')
                self.assertIsNotNone(r.snapshot()['terminal'])

    def test_partial_on_no_cut_rejected_but_whole_discard_of_cut_allowed(self):
        plain = replace(self.e, presentation_cut=None)
        for receiver in (True, False):
            owner = Receiver(self.binding, deadline=self.deadline) if receiver else Sender(self.binding, self.deadline)
            if receiver:
                owner.offer(plain, self.video, self.pcm); ref = owner.lease()
            else:
                ref = owner.reserve(plain); owner.verified(ref)
            with self.assertRaises(ProtocolError):
                if receiver:owner.ack_partial(ref, certain=True)
                else:owner.partial_ack(ref, certainty=True)
        s = Sender(self.binding, self.deadline); t = s.reserve(self.e); s.verified(t)
        s.discard_ack(t, certainty=True)
        r = Receiver(self.binding, deadline=self.deadline); r.offer(self.e, self.video, self.pcm)
        r.discard_interrupted(r.lease(), certain=True)
        self.assertEqual(s.snapshot()['discarded'], 1)
        self.assertEqual(r.snapshot()['discarded'], 1)
        self.assertEqual(r.snapshot()['partial_acked'], 0)

    def test_wire_forged_or_stale_ack_cut_does_not_release_credit(self):
        for field, value in (('frame', 2), ('previous_epoch', 1), ('interrupted_source_epoch', 0)):
            with self.subTest(field=field):
                s = Sender(self.binding, self.deadline); s.reserve(self.e)
                msg = wire.ack_record(self.e, 'verified')
                msg['presentation_cut'][field] = value
                with self.assertRaises(ProtocolError):s.ack(msg)
                self.assertIsNotNone(s.snapshot()['terminal'])
        s = Sender(self.binding, self.deadline); s.reserve(self.e)
        s.ack(wire.ack_record(self.e, 'verified'))
        s.ack(wire.ack_record(self.e, 'partially-presented-interrupted'))
        self.assertEqual(s.snapshot()['partial_acked'], 1)
        for change in ({'presentation_cut': None}, {'presentation_epoch': 0},
                       {'stage': 'sink-accepted'}):
            msg = dict(wire.ack_record(self.e, 'partially-presented-interrupted'), **change)
            with self.assertRaises(ProtocolError):wire.parse_ack_message(msg)
        no_cut = wire.ack_record(replace(self.e, presentation_cut=None), 'verified')
        no_cut['stage'] = 'partially-presented-interrupted'
        with self.assertRaises(ProtocolError):wire.parse_ack_message(no_cut)


if __name__ == '__main__':
    unittest.main()
