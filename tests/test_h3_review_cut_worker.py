"""CPU metadata/PCM safety; these fixtures do not establish native visual quality."""
import hashlib
import time
import unittest
from dataclasses import replace

from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal.live_conversation import Packet
from research.h3_personal.live_relay_worker import _quarantine_suffix


class QuarantineCutTests(unittest.TestCase):
    def packet(self, *, prefix_epoch=1, body_speech=False):
        b=LiveAVBinding('s','a'*64,1)
        old=b'\x01\x00'*10000
        silence=bytes(194000*2)
        regions=(AudioRegion('speech',238000,248000,'old',prefix_epoch,238000,
                             hashlib.sha256(old).hexdigest()),
                 AudioRegion('speech',248000,442000,'new',2,0,hashlib.sha256(silence).hexdigest())
                 if body_speech else AudioRegion('silence',248000,442000))
        pcm=old+silence;video=b'full-native-video'
        e=encode(b,1,119,221,video,pcm,regions,False)
        return Packet(e,video,pcm)

    def cut(self,p,previous=None):
        return _quarantine_suffix(p,dict(sequence=0,committed_frame=124,emitted_frame=119) if previous is None else previous,
                                  0,dict(presentation_epoch=1,invalidated_turn_epoch=1),source_epoch=1,
                                  frontier=dict(reserved=False,windows=1,committed_frame=124,emitted_frame=119),
                                  current=dict(sequence=1,role='listening',emitted_frame=221,
                                               committed_before=124,receipt_sha256='f'*64))

    def test_derived_cut_keeps_full_verified_wire_bytes_and_clock(self):
        p=self.packet();q=self.cut(p)
        self.assertEqual(q.envelope.presentation_cut.frame,124)
        self.assertEqual(q.envelope.presentation_epoch,1)
        self.assertEqual((q.envelope.frame_start,q.envelope.frame_stop),(119,221))
        self.assertEqual(q.pcm,p.pcm);self.assertEqual(q.video,p.video)
        self.assertEqual(q.envelope.pcm_sha256,p.envelope.pcm_sha256)

    def test_unknown_decoder_frontier_is_not_assumed_five(self):
        p=self.packet()
        for prior in ({},dict(sequence=0,committed_frame=124,emitted_frame=118),
                      dict(sequence=0,committed_frame=119,emitted_frame=119),
                      dict(sequence=2,committed_frame=124,emitted_frame=119)):
            with self.subTest(prior=prior),self.assertRaises(ValueError):self.cut(p,prior)

    def test_quarantine_cannot_be_reclassified_as_new_speech(self):
        with self.assertRaises(ValueError):self.cut(self.packet(body_speech=True))

    def test_new_source_head_in_omitted_prefix_is_rejected(self):
        with self.assertRaises(ValueError):self.cut(self.packet(prefix_epoch=2))

    def test_midstep_second_interrupt_does_not_promote_generation(self):
        import tempfile
        from pathlib import Path
        from research.h3_personal.epoch_av_sink import EpochAVSink
        from research.h3_personal.paired_av import ReceivedAVLease
        p=self.packet();q=self.cut(p)  # frozen generation1/source1, not latest gate2
        with tempfile.TemporaryDirectory() as tmp:
            with EpochAVSink(Path(tmp).resolve(),p.envelope.binding,paired_cuts=True) as sink:
                sink.invalidate(1,interrupted_source_epoch=1)
                self.assertFalse(sink.should_discard(ReceivedAVLease(q.envelope,q.video,q.pcm)))
                sink.invalidate(2,interrupted_source_epoch=2)
                self.assertEqual(q.envelope.presentation_epoch,1)
                self.assertEqual(q.envelope.presentation_cut.interrupted_source_epoch,1)
                self.assertTrue(sink.should_discard(ReceivedAVLease(q.envelope,q.video,q.pcm)))

    def test_cancel_during_decoder_construction_is_not_terminal_failure(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from research.h3_personal.epoch_av_sink import EpochAVSink,PresentationInvalidated
        from research.h3_personal.paired_av import ReceivedAVLease
        b=LiveAVBinding('s','a'*64,1);pcm=bytes(4000);video=b'v'
        e=encode(b,0,0,1,video,pcm,(AudioRegion('silence',0,2000),),False)
        lease=ReceivedAVLease(e,video,pcm)
        class Child:
            def __init__(self,*args,**kwargs):self.first_av_submission_monotonic=time.monotonic()
            def cancel(self):pass
            close=cancel
        with tempfile.TemporaryDirectory() as tmp:
            with EpochAVSink(Path(tmp).resolve(),b,sink_factory=Child,observe_output=True) as sink:
                def constructor(*args,**kwargs):
                    sink.invalidate(1,interrupted_source_epoch=1)
                    raise FileNotFoundError('canceled NUT')
                with patch('research.h3_personal.epoch_av_sink._DecodedOutput',side_effect=constructor):
                    with self.assertRaises(PresentationInvalidated):sink.accept('unused',lease)
                sink.discard(lease)
                self.assertEqual(sink.timing['global_frame'],1)
                self.assertFalse(sink.timing['discontinuities'][0]['actual_decoded_sink_observed'])

    def test_full_bytes_checked_even_when_only_suffix_will_be_used(self):
        p=self.packet()
        with self.assertRaises(ValueError):self.cut(replace(p,pcm=b'\x02'+p.pcm[1:]))


class CoverageTimingTests(unittest.TestCase):
    def exercise(self,chunks):
        import struct
        from research.h3_personal.relay_native_tts import produce
        from research.h3_personal.streaming_pcm import PCMStreamBinding
        from research.h3_personal.streaming_windows import (inspect_audio_window,ResamplingContext,
            NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        seen=[]
        class TTS:
            def stream(self,*args):return iter(chunks)
            def close(self):pass
        def observe(stream):
            offer=inspect_audio_window(stream,onset_frame=124,index=0,context=context)
            seen.append((stream.snapshot()['state'],offer.status,offer.required_source_samples))
            other=inspect_audio_window(stream,onset_frame=940,index=0,context=context)
            self.assertEqual(offer.required_source_samples,other.required_source_samples)
        receipt=produce(TTS(),'短い返答です。',1,PCMStreamBinding('s','one',1,'a'*64),
            time.monotonic()+10,lambda s:None,lambda:True,observe=observe)
        count=sum(len(x)//4 for x in chunks)
        self.assertEqual(receipt.converted_samples,count)
        self.assertEqual(receipt.converted_sha256,
                         hashlib.sha256(struct.pack('<h',3277)*count).hexdigest())
        return seen

    def test_full_coverage_observed_before_end_without_consuming_pcm(self):
        import struct
        seen=self.exercise([struct.pack('<f',.1)*32768]*7)
        self.assertTrue(any(state=='open' and status=='ready' for state,status,n in seen))
        self.assertTrue(all(n==204128 for state,status,n in seen[:-1]))

    def test_short_source_requires_verified_seal_not_fake_silence(self):
        import struct
        seen=self.exercise([struct.pack('<f',.1)*1000])
        self.assertEqual([x[1] for x in seen],['wait','ready'])
        self.assertEqual(seen[-1][0],'sealed')
        self.assertEqual(seen[-1][2],1000)


if __name__=='__main__':unittest.main()
