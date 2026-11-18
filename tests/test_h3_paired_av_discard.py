"""Explicit interrupted credit disposition; no model, device, or WAN access."""
from dataclasses import asdict, replace
import hashlib
import json
import socket
import time
import unittest

from research.h3_personal.paired_av import (
    AudioRegion, LiveAVBinding, ProtocolError, Receiver, VerifiedReceipt, encode)
from research.h3_personal.paired_av_sender import Sender
from research.h3_personal import paired_av_wire as wire


class DiscardTests(unittest.TestCase):
    def setUp(self):
        self.binding = LiveAVBinding('session', 'a'*64, 73, max_segments=6,
                                     max_total_frames=6, max_inflight_bytes=9000)
        self.deadline = time.monotonic()+10

    def pair(self, index, *, final=False, source_epoch=None, source_offset=0,
             presentation_epoch=0):
        pcm = bytes(4000) if source_epoch is None else source_epoch.to_bytes(2, 'little')*2000
        region = (AudioRegion('silence', index*2000, (index+1)*2000)
                  if source_epoch is None else
                  AudioRegion('speech', index*2000, (index+1)*2000,
                              f'source-{source_epoch}', source_epoch,
                              source_offset, hashlib.sha256(pcm).hexdigest()))
        video = b'video'
        envelope = encode(self.binding, index, index, index+1, video, pcm, (region,), final,
                          presentation_epoch=presentation_epoch)
        return envelope, video, pcm

    def receiver(self):
        return Receiver(self.binding, deadline=self.deadline)

    def sender(self):
        return Sender(self.binding, self.deadline)

    def message(self, envelope, stage):
        result = dict(schema='h3-paired-av-ack-v1', session=envelope.binding.session,
                      epoch=envelope.binding.epoch, profile_sha256=envelope.binding.profile_sha256,
                      index=envelope.index, video_sha256=envelope.video_sha256,
                      pcm_sha256=envelope.pcm_sha256, stage=stage)
        if envelope.presentation_epoch:
            result['presentation_epoch'] = envelope.presentation_epoch
        return result

    def test_discard_releases_credit_not_presentation_ack_or_clock_reset(self):
        receiver, sender = self.receiver(), self.sender()
        first, second = self.pair(0), self.pair(1, final=True)
        ticket0, ticket1 = sender.reserve(first[0]), sender.reserve(second[0])
        receipt0, receipt1 = receiver.offer(*first), receiver.offer(*second)
        sender.verified(receipt0); sender.verified(receipt1)
        lease = receiver.lease()
        receiver.discard_interrupted(lease, certain=True)
        sender.discard_ack(ticket0, certainty=True)
        for snapshot in (receiver.snapshot(), sender.snapshot()):
            self.assertEqual(snapshot['discarded'], 1)
            self.assertEqual(snapshot['disposed'], 1)
            self.assertEqual(snapshot['inflight_bytes'], 4005)
            self.assertEqual(snapshot['frame_stop'], 2)
            self.assertIsNone(snapshot['terminal'])
            self.assertFalse(snapshot['transport_complete'])
            self.assertFalse(snapshot['presentation_complete'])
        self.assertEqual(receiver.snapshot()['sink_acked'], 0)
        self.assertEqual(sender.snapshot()['acked'], 0)
        receiver.ack(receiver.lease(), certain=True)
        sender.sink_ack(ticket1, certainty=True)
        for snapshot in (receiver.snapshot(), sender.snapshot()):
            self.assertEqual(snapshot['inflight_bytes'], 0)
            self.assertTrue(snapshot['transport_complete'])
            self.assertFalse(snapshot['presentation_complete'])
        # Completed transport drain must not wait for EOF after final discard.
        # Presentation completion remains a separate, false claim.
        self.assertTrue(receiver.snapshot()['complete'])
        self.assertEqual(receiver.binding.epoch, 73)
        self.assertEqual(sender.binding.epoch, 73)

    def test_final_pair_can_be_discarded_but_never_claims_presented_completion(self):
        receiver = self.receiver()
        receiver.offer(*self.pair(0, final=True))
        receiver.discard_interrupted(receiver.lease(), certain=True)
        state = receiver.snapshot()
        self.assertTrue(state['transport_complete'])
        self.assertTrue(state['complete'])
        self.assertFalse(state['presentation_complete'])
        self.assertEqual(state['sink_acked'], 0)

    def test_legacy_v1_no_discard_completion_unchanged(self):
        receiver, sender = self.receiver(), self.sender()
        pair = self.pair(0, final=True)
        ticket = sender.reserve(pair[0]); receipt = receiver.offer(*pair)
        sender.verified(receipt)
        receiver.ack(receiver.lease()); sender.sink_ack(ticket, certainty=True)
        self.assertTrue(receiver.snapshot()['complete'])
        self.assertTrue(receiver.snapshot()['presentation_complete'])
        self.assertTrue(receiver.snapshot()['transport_complete'])
        self.assertTrue(sender.snapshot()['presentation_complete'])
        self.assertEqual(receiver.snapshot()['discarded'], 0)
        self.assertEqual(sender.snapshot()['discarded'], 0)

    def test_receiver_requires_exact_leased_capability_and_certainty(self):
        for case in ('uncertain', 'not-bool', 'unleased', 'copy', 'duplicate'):
            with self.subTest(case=case):
                receiver = self.receiver(); receipt = receiver.offer(*self.pair(0))
                lease = receiver.lease()
                if case == 'duplicate':
                    receiver.discard_interrupted(lease, certain=True)
                candidate = (receipt if case == 'unleased' else
                             replace(lease) if case == 'copy' else lease)
                certain = False if case == 'uncertain' else 1 if case == 'not-bool' else True
                with self.assertRaises(ProtocolError):
                    receiver.discard_interrupted(candidate, certain=certain)
                self.assertIsNotNone(receiver.snapshot()['terminal'])
                self.assertEqual(receiver.snapshot()['inflight_bytes'], 0)

    def test_sender_requires_verified_fifo_certain_disposition(self):
        for case in ('unverified', 'non-fifo', 'uncertain', 'not-bool', 'duplicate'):
            with self.subTest(case=case):
                sender = self.sender()
                a, b = sender.reserve(self.pair(0)[0]), sender.reserve(self.pair(1)[0])
                if case != 'unverified':
                    sender.verified(a); sender.verified(b)
                if case == 'duplicate':
                    sender.discard_ack(a, certainty=True)
                certainty = False if case == 'uncertain' else 1 if case == 'not-bool' else True
                with self.assertRaises(ProtocolError):
                    sender.discard_ack(b if case == 'non-fifo' else a, certainty=certainty)
                self.assertIsNotNone(sender.snapshot()['terminal'])
                self.assertEqual(sender.snapshot()['inflight_bytes'], 0)

    def test_discard_preserves_logical_source_offset_and_freshness(self):
        receiver, sender = self.receiver(), self.sender()
        first = self.pair(0, source_epoch=1)
        a = sender.reserve(first[0]); sender.verified(receiver.offer(*first))
        receiver.discard_interrupted(receiver.lease(), certain=True)
        sender.discard_ack(a, certainty=True)
        # Same source's retained tail remains contiguous after explicit discard.
        second = self.pair(1, source_epoch=1, source_offset=2000)
        b = sender.reserve(second[0]); sender.verified(receiver.offer(*second))
        receiver.discard_interrupted(receiver.lease(), certain=True)
        sender.discard_ack(b, certainty=True)
        third = self.pair(2, source_epoch=2, final=True)
        c = sender.reserve(third[0]); sender.verified(receiver.offer(*third))
        receiver.ack(receiver.lease()); sender.sink_ack(c, certainty=True)
        self.assertTrue(receiver.snapshot()['transport_complete'])
        self.assertEqual(receiver.snapshot()['discarded'], 2)
        self.assertEqual(receiver.snapshot()['sink_acked'], 1)

    def test_discard_does_not_allow_replay_gap_or_binding_epoch_change(self):
        for change in ('replay', 'gap', 'epoch'):
            for kind in ('receiver', 'sender'):
                with self.subTest(change=change, kind=kind):
                    receiver, sender = self.receiver(), self.sender()
                    first = self.pair(0)
                    ticket = sender.reserve(first[0]); sender.verified(receiver.offer(*first))
                    receiver.discard_interrupted(receiver.lease(), certain=True)
                    sender.discard_ack(ticket, certainty=True)
                    pair = self.pair(0 if change == 'replay' else 2 if change == 'gap' else 1)
                    if change == 'epoch':
                        pair = (replace(pair[0], binding=replace(self.binding, epoch=74)), *pair[1:])
                    with self.assertRaises(ProtocolError):
                        receiver.offer(*pair) if kind == 'receiver' else sender.reserve(pair[0])

    def test_wire_new_stage_keeps_exact_v1_identity_and_disposition(self):
        sender = self.sender(); pair = self.pair(0, final=True)
        sender.reserve(pair[0])
        a, b = socket.socketpair()
        self.addCleanup(a.close); self.addCleanup(b.close)
        for stage in ('verified', 'discarded-interrupted'):
            wire.send_ack(a, pair[0], stage=stage, deadline=self.deadline)
            message = wire.receive_ack(b, pair[0], stage=stage, deadline=self.deadline)
            self.assertEqual(message, self.message(pair[0], stage))
            sender.ack(message)
        self.assertEqual(sender.snapshot()['discarded'], 1)
        self.assertEqual(sender.snapshot()['acked'], 0)
        self.assertTrue(sender.snapshot()['transport_complete'])
        self.assertFalse(sender.snapshot()['presentation_complete'])

    def test_discard_wire_identity_drift_is_terminal(self):
        envelope = self.pair(0)[0]
        for field, value in (('session', 'foreign'), ('epoch', 74), ('epoch', True),
                             ('profile_sha256', 'b'*64), ('index', 1),
                             ('video_sha256', 'c'*64), ('pcm_sha256', 'd'*64),
                             ('schema', 'h3-paired-av-ack-v2'), ('extra', 'ignored')):
            with self.subTest(field=field, value=value):
                sender = self.sender(); ticket = sender.reserve(envelope); sender.verified(ticket)
                message = dict(self.message(envelope, 'discarded-interrupted'), **{field: value})
                with self.assertRaises(ProtocolError):
                    sender.ack(message)
                self.assertIsNotNone(sender.snapshot()['terminal'])

    def test_presentation_epoch_is_explicit_and_bounded_not_binding_epoch(self):
        for generation in (0, 1, 2**63-1):
            envelope = self.pair(0, presentation_epoch=generation)[0]
            self.assertEqual(envelope.presentation_epoch, generation)
            self.assertEqual(envelope.binding.epoch, 73)
        for generation in (-1, 2**63, True, False, 1.0, '1', None):
            with self.subTest(generation=generation), self.assertRaises(ProtocolError):
                self.pair(0, presentation_epoch=generation)

    def test_default_wire_header_is_exact_legacy_record_without_zero(self):
        pair = self.pair(0)
        legacy = json.loads(json.dumps(asdict(pair[0])))
        del legacy['presentation_epoch']; del legacy['presentation_cut']
        self.assertEqual(wire.envelope_record(pair[0]), legacy)
        a, b = socket.socketpair()
        self.addCleanup(a.close); self.addCleanup(b.close)
        wire.send_pair(a, *pair, deadline=self.deadline)
        header = wire._json(b, self.deadline)
        self.assertEqual(header, dict(schema='h3-paired-av-wire-v1', envelope=legacy))
        self.assertNotIn('presentation_epoch', header['envelope'])
        self.assertEqual(wire.parse_envelope(header), pair[0])

    def test_positive_presentation_epoch_media_and_discard_ack_roundtrip(self):
        pair = self.pair(0, final=True, presentation_epoch=19)
        a, b = socket.socketpair()
        self.addCleanup(a.close); self.addCleanup(b.close)
        receiver, sender = self.receiver(), self.sender()
        sender.reserve(pair[0])
        wire.send_pair(a, *pair, deadline=self.deadline)
        receipt = wire.receive_pair(b, receiver, deadline=self.deadline)
        self.assertEqual(receipt.envelope.presentation_epoch, 19)
        self.assertEqual(receipt.envelope.binding.epoch, 73)
        for stage in ('verified', 'discarded-interrupted'):
            if stage == 'discarded-interrupted':
                receiver.discard_interrupted(receiver.lease(), certain=True)
            wire.send_ack(b, receipt.envelope, stage=stage, deadline=self.deadline)
            message = wire.receive_ack(a, pair[0], stage=stage, deadline=self.deadline)
            self.assertEqual(message['presentation_epoch'], 19)
            sender.ack(message)
        self.assertTrue(receiver.snapshot()['transport_complete'])
        self.assertFalse(receiver.snapshot()['presentation_complete'])
        self.assertEqual(sender.snapshot()['discarded'], 1)

    def test_wire_epoch_field_has_exact_optional_positive_encoding(self):
        envelope = self.pair(0)[0]
        legacy = wire.envelope_record(envelope)
        for value in (0, -1, True, False, 2**63, '1', 1.0, None):
            header = dict(schema='h3-paired-av-wire-v1',
                          envelope=dict(legacy, presentation_epoch=value))
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                wire.parse_envelope(header)
        header = dict(schema='h3-paired-av-wire-v1',
                      envelope=dict(legacy, presentation_epoch=9))
        self.assertEqual(wire.parse_envelope(header).presentation_epoch, 9)
        header['envelope']['unknown'] = 'no extension guessing'
        with self.assertRaises(ProtocolError):
            wire.parse_envelope(header)

    def test_presentation_epoch_ack_identity_cannot_be_omitted_or_relabelled(self):
        for stage in ('verified', 'sink-accepted', 'discarded-interrupted'):
            for generation in (None, 0, True, 20, 2**63):
                with self.subTest(stage=stage, generation=generation):
                    sender = self.sender(); envelope = self.pair(0, presentation_epoch=19)[0]
                    ticket = sender.reserve(envelope)
                    if stage != 'verified':
                        sender.verified(ticket)
                    message = self.message(envelope, stage)
                    if generation is None:
                        del message['presentation_epoch']
                    else:
                        message['presentation_epoch'] = generation
                    with self.assertRaises(ProtocolError):
                        sender.ack(message)
                    self.assertIsNotNone(sender.snapshot()['terminal'])
        # Explicit zero ACK is not an alternative encoding of default generation.
        message = dict(self.message(self.pair(0)[0], 'verified'), presentation_epoch=0)
        with self.assertRaises(ProtocolError):
            wire.parse_ack_message(message)

    def test_bound_receive_ack_rejects_generation_even_with_same_media_hashes(self):
        envelope = self.pair(0, presentation_epoch=19)[0]
        for replacement in (replace(envelope, presentation_epoch=20),
                            replace(envelope, presentation_epoch=0)):
            a, b = socket.socketpair()
            try:
                wire.send_ack(a, replacement, stage='discarded-interrupted', deadline=self.deadline)
                with self.assertRaises(ProtocolError):
                    wire.receive_ack(b, envelope, stage='discarded-interrupted', deadline=self.deadline)
            finally:
                a.close(); b.close()

    def test_wire_rejects_sink_acceptance_substitution(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close); self.addCleanup(b.close)
        envelope = self.pair(0)[0]
        wire.send_ack(a, envelope, stage='discarded-interrupted', deadline=self.deadline)
        with self.assertRaises(ProtocolError):
            wire.receive_ack(b, envelope, stage='sink-accepted', deadline=self.deadline)
        for stage in ('discarded', 'interrupted', 'sink-discarded', 1, None):
            with self.subTest(stage=stage), self.assertRaises(ProtocolError):
                wire.send_ack(a, envelope, stage=stage, deadline=self.deadline)


if __name__ == '__main__':
    unittest.main()
