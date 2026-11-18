"""CPU admission integration only; not native H3 generation or playback."""
import hashlib
import struct
import time
import unittest

from research.h3_personal.relay_native_tts import produce
from research.h3_personal.streaming_pcm import PCMStreamBinding
from research.h3_personal.streaming_admission import StreamingAdmission, NativeCommitEvidence
from research.h3_personal.streaming_windows import (ResamplingContext, NATIVE_TORCH,
    NATIVE_TORCHAUDIO, REFERENCE_SHA256, inspect_audio_window)
from research.h3_personal.tts_preroll import PCMPrerollPolicy


class NativePrerollHandoff(unittest.TestCase):
    def test_h3_lookahead_and_reserve_are_max_not_sum(self):
        context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        admission=StreamingAdmission('c',context=context)
        decision=admission.inspect()
        ticket=admission.reserve(decision)
        admission.commit(ticket,NativeCommitEvidence(124,None,'b'*64))
        admission.ack_decoded(ticket,119)
        binding=PCMStreamBinding('c','native',1,'a'*64)
        chunk=struct.pack('<f',.25)*7680
        published=[]; speech=[]; needed=[]
        def publish(stream):
            published.append(stream)
            self.assertGreaterEqual(stream.snapshot()['received_samples'],48000)
            self.assertLess(stream.snapshot()['received_samples'],96000)
            admission.attach(stream)
            offer=inspect_audio_window(stream,onset_frame=124,index=0,context=context)
            needed.append(offer.required_source_samples)
            self.assertEqual(admission.inspect().role,'listening')
        class TTS:
            closed=False
            def stream(inner,*args):
                for _ in range(36):
                    yield chunk
                    if published and not speech:
                        d=admission.inspect()
                        if d.status=='ready' and d.role=='speech':
                            speech.append(published[0].snapshot()['received_samples'])
            def close(inner):inner.closed=True
        tts=TTS()
        receipt=produce(tts,'One full utterance.',1,binding,time.monotonic()+20,publish,lambda:True,
                        preroll=PCMPrerollPolicy())
        expected=((needed[0]+7679)//7680)*7680
        self.assertEqual(speech,[expected])
        # Starting does not require an extra reserve on top of the native window.
        self.assertLess(speech[0],needed[0]+48000)
        pcm=struct.pack('<h',8192)*(36*7680)
        self.assertEqual(receipt.stream.complete_source().pcm,pcm)
        self.assertEqual(receipt.converted_sha256,hashlib.sha256(pcm).hexdigest())
        self.assertFalse(tts.closed)


if __name__=='__main__':unittest.main()
