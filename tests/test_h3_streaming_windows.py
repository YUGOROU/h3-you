import dataclasses
import hashlib
import unittest
from research.h3_personal.streaming_pcm import PCMIngress, PCMChunk, PCMStreamBinding
from research.h3_personal.streaming_windows import (
    ResamplingContext, inspect_audio_window, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256,
)


def source(rate=32000, epoch=1, name='speech'):
    return PCMIngress(PCMStreamBinding('c',name,epoch,'a'*64,sample_rate=rate),
        max_bytes=2_000_000,max_chunk_bytes=65536,max_read_bytes=1_000_000,max_samples=1_000_000)


def append_samples(stream,n):
    for start in range(0,n,32000):
        snap=stream.snapshot();count=min(n-start,32000)
        stream.append(PCMChunk(stream.binding,snap['chunks'],snap['received_samples'],b'\x01\x00'*count))


def seal(stream):
    count=stream.snapshot()['received_samples']
    stream.seal(total_samples=count,sha256=stream.prefix_digest(count))


def measured():
    return ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)


class WindowReadinessTests(unittest.TestCase):
    def test_first_window_waits_for_4_25s_speech_not_first_byte(self):
        stream=source();ctx=ResamplingContext('same-rate-32k')
        append_samples(stream,1)
        offer=inspect_audio_window(stream,onset_frame=124,index=0,context=ctx)
        self.assertEqual(offer.status,'wait')
        self.assertEqual(offer.required_source_samples,136000)
        append_samples(stream,135999)
        offer=inspect_audio_window(stream,onset_frame=124,index=0,context=ctx)
        self.assertEqual(offer.status,'ready')
        self.assertEqual((offer.window.start,offer.window.end),(102,226))
        self.assertFalse(offer.eos);self.assertEqual(offer.eos_padding_native_samples,0)
        self.assertEqual(offer.prefix_sha256,hashlib.sha256(b'\x01\x00'*136000).hexdigest())

    def test_measured_resampler_requires_right_context(self):
        stream=source(48000);append_samples(stream,204000)
        before=inspect_audio_window(stream,onset_frame=124,index=0,context=measured())
        self.assertEqual(before.status,'wait');self.assertEqual(before.required_source_samples,204128)
        append_samples(stream,128)
        after=inspect_audio_window(stream,onset_frame=124,index=0,context=measured())
        self.assertEqual(after.status,'ready')
        append_samples(stream,1000)
        self.assertEqual(after,inspect_audio_window(stream,onset_frame=124,index=0,context=measured()))

    def test_unverified_resampler_waits_eos_without_silence(self):
        stream=source(48000);append_samples(stream,300000)
        offer=inspect_audio_window(stream,onset_frame=124,index=0,context=ResamplingContext())
        self.assertEqual(offer.status,'wait');self.assertIsNone(offer.prefix_sha256)
        seal(stream)
        offer=inspect_audio_window(stream,onset_frame=124,index=0,context=ResamplingContext())
        self.assertEqual(offer.status,'ready')
        self.assertEqual(offer.required_source_samples,300000) # full-source EOS resampling
        self.assertFalse(offer.final_source_window)

    def test_eos_is_the_only_permission_to_pad(self):
        stream=source();append_samples(stream,32000);ctx=ResamplingContext('same-rate-32k')
        self.assertEqual(inspect_audio_window(stream,onset_frame=124,index=0,context=ctx).eos_padding_native_samples,0)
        seal(stream)
        offer=inspect_audio_window(stream,onset_frame=124,index=0,context=ctx)
        self.assertEqual(offer.status,'ready');self.assertTrue(offer.final_source_window)
        self.assertEqual(offer.eos_padding_native_samples,104000)
        end=inspect_audio_window(stream,onset_frame=124,index=1,context=ctx)
        self.assertEqual(end.status,'end')

    def test_short_eos_does_not_demand_missing_resampler_context(self):
        stream=source(48000);append_samples(stream,204000);seal(stream)
        offer=inspect_audio_window(stream,onset_frame=226,index=0,context=measured())
        self.assertEqual(offer.status,'ready');self.assertTrue(offer.final_source_window)
        self.assertEqual(offer.eos_padding_native_samples,0)
        self.assertEqual(offer.required_source_samples,204000)

    def test_global_lattice_and_successive_new_intervals(self):
        stream=source();append_samples(stream,500000)
        for onset in (124,226,430):
            for index in range(3):
                offer=inspect_audio_window(stream,onset_frame=onset,index=index,context=ResamplingContext('same-rate-32k'))
                self.assertEqual(offer.required_native_samples,(index+1)*136000)
                self.assertEqual(offer.window.guide_audio_t,37)
                self.assertEqual(offer.status,'ready')

    def test_policy_and_invalid_epoch_geometry_fail(self):
        with self.assertRaises(ValueError):dataclasses.replace(measured(),torch_version='unknown')
        stream=source(48000)
        with self.assertRaises(ValueError):inspect_audio_window(stream,onset_frame=124,index=0,context=ResamplingContext('same-rate-32k'))
        for onset,index in ((0,0),(123,0),(124,True),(124,144)):
            with self.assertRaises(ValueError):inspect_audio_window(source(),onset_frame=onset,index=index,context=ResamplingContext())

    def test_long_finite_source_can_be_retired_after_64_windows(self):
        # 272s is within the explicit300s ingress ceiling. Do not reject the
        # EOS inspection at index64 just because model preparation is bounded.
        from research.h3_personal.streaming_pcm import PCMIngress,PCMStreamBinding
        s=PCMIngress(PCMStreamBinding('c','long',1,'a'*64,sample_rate=1),
                     max_bytes=1024,max_chunk_bytes=1024,max_read_bytes=1024)
        s.append(PCMChunk(s.binding,0,0,b'\x00\x00'*272));seal(s)
        offer=inspect_audio_window(s,onset_frame=124,index=64,context=ResamplingContext())
        self.assertEqual(offer.status,'end')

    def test_cancelled_ingress_never_produces_ready_offer(self):
        stream=source();append_samples(stream,136000);stream.cancel()
        with self.assertRaises(RuntimeError):inspect_audio_window(stream,onset_frame=124,index=0,context=ResamplingContext('same-rate-32k'))
