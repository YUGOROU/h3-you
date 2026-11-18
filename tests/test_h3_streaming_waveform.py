import dataclasses
import unittest
import test_h3_streaming_windows as ingress
from test_h3_conversation_audio import WaveOps
from research.h3_personal.conversation_audio import WaveHistory
from research.h3_personal.streaming_windows import ResamplingContext,inspect_audio_window
from research.h3_personal.streaming_waveform import assemble_window,read_original_prefix


class WaveformTests(unittest.TestCase):
    def test_streamed_fixed_windows_equal_batch_slices(self):
        s=ingress.source();ctx=ResamplingContext('same-rate-32k');ops=WaveOps()
        speech=[[i+1 for i in range(385280)] for _ in range(2)]
        prefix=[[0]*29600 for _ in range(2)]
        batch=ops.cat([prefix,speech,ops.zeros(speech,22720)])
        history=WaveHistory(124,prefix)
        for index,received in enumerate((136000,272000,385280)):
            ingress.append_samples(s,received-s.snapshot()['received_samples'])
            if index==2:ingress.seal(s)
            offer=inspect_audio_window(s,onset_frame=124,index=index,context=ctx)
            result=assemble_window([row[:received] for row in speech],offer,history,ops)
            self.assertEqual(result.waveform,ops.slice(batch,index*136000,index*136000+165600))
            self.assertEqual(result.padding_native_samples,22720 if index==2 else 0)
            self.assertEqual(result.history.end_frame,offer.window.end)
        self.assertTrue(any(result.history.tail[0]))
        self.assertEqual(result.history.tail[0][-1],0)

    def test_missing_future_and_stale_history_fail(self):
        s=ingress.source();ingress.append_samples(s,136000)
        offer=inspect_audio_window(s,onset_frame=124,index=0,context=ResamplingContext('same-rate-32k'))
        wave=[[1]*136000 for _ in range(2)];ops=WaveOps()
        for history in (None,WaveHistory(123,[[0]*29600]*2)):
            with self.assertRaises(ValueError):assemble_window(wave,offer,history,ops)
        with self.assertRaises(ValueError):assemble_window([[1]*135999]*2,offer,WaveHistory(124,[[0]*29600]*2),ops)
        with self.assertRaises(ValueError):assemble_window(wave,dataclasses.replace(offer,eos_padding_native_samples=1),WaveHistory(124,[[0]*29600]*2),ops)

    def test_prefix_read_uses_bounded_reads_and_hash(self):
        s=ingress.source();s.max_read_bytes=1024
        ingress.append_samples(s,136000)
        offer=inspect_audio_window(s,onset_frame=124,index=0,context=ResamplingContext('same-rate-32k'))
        self.assertEqual(read_original_prefix(s,offer),b'\x01\x00'*136000)
        with self.assertRaises(ValueError):read_original_prefix(s,dataclasses.replace(offer,prefix_sha256='f'*64))
        s.cancel()
        with self.assertRaises(RuntimeError):read_original_prefix(s,offer)
