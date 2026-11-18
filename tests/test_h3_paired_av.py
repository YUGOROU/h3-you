import dataclasses
import hashlib
import time
import unittest
from unittest.mock import patch

from research.h3_personal.paired_av import (
    LiveAVBinding, AudioRegion, AVEnvelope, Receiver, encode,
    ProtocolError, CreditError,
)


class PairedAVTests(unittest.TestCase):
    def setUp(self):
        self.binding = LiveAVBinding('session', 'a'*64, 1)

    def receiver(self, binding=None):
        return Receiver(binding or self.binding, deadline=time.monotonic()+30)

    def pair(self, index=0, start=0, frames=1, final=False, binding=None, speech=False, offset=0):
        pcm = (b'\x01\x00' if speech else b'\x00\x00') * (frames*2000)
        if speech:
            region = AudioRegion('speech', start*2000, (start+frames)*2000,
                                 'voice', 1, offset, hashlib.sha256(pcm).hexdigest())
        else:
            region = AudioRegion('silence', start*2000, (start+frames)*2000)
        video = b'video'
        return encode(binding or self.binding, index, start, start+frames,
                      video, pcm, (region,), final), video, pcm

    def test_actual_frame_sequence_exact_audio_and_final(self):
        r = self.receiver()
        start = 0
        emitted = []
        expected = []
        for i, n in enumerate((119, 102, 107)):
            pair = self.pair(i, start, n, i == 2, speech=True, offset=start*2000)
            receipt = r.offer(*pair)
            self.assertEqual(receipt.envelope.sample_stop, (start+n)*2000)
            self.assertEqual(r.snapshot()['sink_acked'], i)
            lease = r.lease()
            self.assertEqual(lease.segment.binding, self.binding)
            self.assertEqual(lease.segment.padding_samples, 0)
            emitted.append(lease.audio_pcm)
            expected.append(pair[2])
            r.ack(lease)
            start += n
        self.assertEqual(b''.join(emitted), b''.join(expected))
        self.assertEqual(len(b''.join(emitted)), start*4000)
        self.assertTrue(r.snapshot()['complete'])
        with self.assertRaises(ProtocolError):
            r.offer(*self.pair(3, start))

    def test_credits_only_ack_releases(self):
        r = self.receiver()
        r.offer(*self.pair())
        lease = r.lease()
        r.offer(*self.pair(1, 1))
        third = self.pair(2, 2)
        with self.assertRaises(CreditError):
            r.offer(*third)
        self.assertEqual(r.snapshot()['inflight_bytes'], 8010)
        self.assertIsNone(r.lease())
        r.ack(lease)
        r.offer(*third)
        self.assertEqual(r.snapshot()['pending'], 2)

    def test_byte_credit(self):
        b = dataclasses.replace(self.binding, max_inflight_bytes=5000)
        r = self.receiver(b)
        r.offer(*self.pair(binding=b))
        with self.assertRaises(CreditError):
            r.offer(*self.pair(1, 1, binding=b))
        self.assertIsNone(r.snapshot()['terminal'])

    def test_binding_type_fuzz(self):
        for field, values in {
            'epoch': [True, -1, 1.0, None],
            'max_segments': [True, 0, 145, float('inf')],
            'max_total_frames': [0, 14711, '1'],
            'max_inflight_bytes': [0, float('inf'), float('nan'), True],
            'sample_rate': [True, 44100, 48000.0],
            'fps': [24.0, True, 25], 'channels': [True, 2],
            'profile_sha256': ['A'*64, 'g'*64, '', None],
            'session': ['', None, 1],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ProtocolError):
                    dataclasses.replace(self.binding, **{field: value})

    def test_header_fuzz(self):
        h, _, _ = self.pair()
        for field, values in {
            'index': [True, -1, 128, 0.0], 'frame_start': [True, -1, 1],
            'frame_stop': [0, 125, 1.0], 'sample_start': [1, False],
            'sample_stop': [1999, 2001], 'pcm_size': [3999, 4001],
            'video_size': [0, 16*1024*1024+1], 'conversation_final': [1, None],
            'padding_samples': [True, 1], 'audio_format': ['f32le'],
            'regions': [[], (), (AudioRegion('silence', 1, 2000),)],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ProtocolError):
                    dataclasses.replace(h, **{field: value})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            h.index = 3

    def test_hash_size_order_stale_terminal(self):
        good = self.pair()
        bads = [self.pair(1), self.pair(start=1),
                self.pair(binding=dataclasses.replace(self.binding, epoch=2)),
                (good[0], b'Video', good[2]), (good[0], good[1], good[2][:-1]),
                (dataclasses.replace(good[0], video_sha256='b'*64), good[1], good[2]),
                (good[0], bytearray(good[1]), good[2])]
        for bad in bads:
            r = self.receiver()
            with self.assertRaises(ProtocolError):
                r.offer(*bad)
            self.assertIsNotNone(r.snapshot()['terminal'])
            with self.assertRaises(ProtocolError):
                r.offer(*good)
        r = self.receiver()
        r.offer(*good)
        with self.assertRaises(ProtocolError):
            r.offer(*good)
        self.assertEqual(r.snapshot()['inflight_bytes'], 0)

    def test_source_replay_and_silence_validation(self):
        r = self.receiver()
        r.offer(*self.pair(speech=True))
        r.ack(r.lease())
        with self.assertRaises(ProtocolError):
            r.offer(*self.pair(1, 1, speech=True, offset=0))
        h, v, p = self.pair(speech=True)
        bad = dataclasses.replace(h, regions=(AudioRegion('silence', 0, 2000),))
        with self.assertRaises(ProtocolError):
            self.receiver().offer(bad, v, p)
        region = dataclasses.replace(h.regions[0], source_sha256='0'*64)
        with self.assertRaises(ProtocolError):
            self.receiver().offer(dataclasses.replace(h, regions=(region,)), v, p)

    def test_new_source_must_start_at_zero_and_retired_source_cannot_resume(self):
        with self.assertRaises(ProtocolError):self.receiver().offer(*self.pair(speech=True,offset=2000))
        r=self.receiver();r.offer(*self.pair(speech=True));r.ack(r.lease())
        h,v,p=self.pair(1,1,speech=True)
        h=dataclasses.replace(h,regions=(dataclasses.replace(h.regions[0],source_id='second',source_epoch=2),))
        r.offer(h,v,p);r.ack(r.lease())
        with self.assertRaises(ProtocolError):r.offer(*self.pair(2,2,speech=True,offset=2000))

    def test_metadata_regions_are_independently_bounded(self):
        h,_,_=self.pair()
        regions=tuple(AudioRegion('silence',i*2000//9,(i+1)*2000//9) for i in range(9))
        with self.assertRaises(ProtocolError):dataclasses.replace(h,regions=regions)

    def test_uncertainty_disconnect_cancel_drop(self):
        for mode in ('ambiguous', 'foreign', 'disconnect', 'cancel', 'drop'):
            r = self.receiver()
            r.offer(*self.pair())
            lease = r.lease()
            if mode in ('ambiguous', 'foreign'):
                with self.assertRaises(ProtocolError):
                    r.ack(lease if mode == 'ambiguous' else dataclasses.replace(lease),
                          certain=mode != 'ambiguous')
            else:
                getattr(r, mode)()
            self.assertEqual(r.snapshot()['inflight_bytes'], 0)
            with self.assertRaises(ProtocolError):
                r.ack(lease)

    def test_deadline_no_renewal(self):
        for deadline in (True, float('inf'), float('nan'), -1, '30'):
            with self.assertRaises(ProtocolError):
                Receiver(self.binding, deadline=deadline)
        with patch('research.h3_personal.paired_av.time.monotonic', return_value=100):
            r = Receiver(self.binding, deadline=110)
            r.offer(*self.pair())
            lease = r.lease()
        with patch('research.h3_personal.paired_av.time.monotonic', return_value=110):
            with self.assertRaises(ProtocolError):
                r.ack(lease)
            self.assertEqual(r.snapshot()['deadline'], 110)
            self.assertEqual(r.snapshot()['inflight_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
