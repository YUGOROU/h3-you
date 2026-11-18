"""Actual CPU driver/ingress/bridge handoff; fake native frames, no GPU/playback."""
import hashlib
from pathlib import Path
import queue
import struct
import tempfile
import threading
import time
import unittest

from research.h3_personal.live_conversation import LiveConversation
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.relay_native_tts import produce
from research.h3_personal.streaming_pcm import PCMStreamBinding
from research.h3_personal.streaming_windows import (ResamplingContext,NATIVE_TORCH,
    NATIVE_TORCHAUDIO,REFERENCE_SHA256)
from research.h3_personal.tts_preroll import PCMPrerollPolicy


class ActualDriverNativeTTS(unittest.TestCase):
    def test_growing_source_waits_for_end_and_preserves_tail_pcm(self):
        for bad_end in (False,True):
            with self.subTest(bad_end=bad_end),tempfile.TemporaryDirectory() as tmp:
                available=threading.Event(); release=threading.Event(); published=queue.Queue()
                results=queue.Queue(); errors=queue.Queue();closed=threading.Event()
                raw=struct.pack('<f',.25)*7680
                class TTS:
                    def stream(self,*args):
                        try:
                            for _ in range(36):yield raw
                            available.set()
                            if not release.wait(3):raise TimeoutError('test did not release END')
                            if bad_end:raise ValueError('invalid END')
                        finally:closed.set()
                    def close(self):closed.set()
                binding=LiveAVBinding('native-driver','a'*64,1,max_segments=24)
                pcm_binding=PCMStreamBinding(binding.session,'native-1',1,binding.profile_sha256)
                def synthesize():
                    try:results.put(produce(TTS(),'One complete text.',1,pcm_binding,
                        time.monotonic()+10,published.put,lambda:True,preroll=PCMPrerollPolicy()))
                    except BaseException as error:errors.put(error)
                thread=threading.Thread(target=synthesize)
                context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
                driver=LiveConversation(Path(tmp)/'driver',binding=binding,context=context)
                packets=[]
                def step():
                    prepared=driver.prepare();decision=prepared.ticket.decision
                    committed=decision.window.end
                    emitted=committed-(0 if decision.conversation_final else 5)
                    packets.append(driver.complete_native(prepared,dict(video=b'fake-native-frames',
                        count=emitted-driver._cursor,committed_frame=committed,emitted_frame=emitted,
                        conditioning_prefix_sha256=None if decision.audio is None else decision.audio.prefix_sha256,
                        receipt_sha256='b'*64)))
                try:
                    thread.start();stream=published.get(timeout=3)
                    self.assertTrue(available.wait(3))
                    self.assertEqual(stream.snapshot()['state'],'open')
                    driver.attach(stream)
                    while driver.inspect().status=='ready':step()
                    self.assertEqual(driver.inspect().status,'wait')
                    self.assertEqual(driver.inspect().role,'speech')
                    release.set();thread.join(timeout=3)
                    self.assertFalse(thread.is_alive());self.assertTrue(closed.is_set())
                    if bad_end:
                        self.assertIsInstance(errors.get_nowait(),ValueError)
                        self.assertTrue(results.empty())
                        with self.assertRaises(RuntimeError):stream.complete_source()
                        with self.assertRaises(RuntimeError):driver.inspect()
                    else:
                        receipt=results.get_nowait();self.assertTrue(errors.empty())
                        while driver.inspect().status=='ready':step()
                        self.assertEqual(driver.inspect().status,'end')
                        driver.retire_speech();driver.request_stop();step()
                        self.assertTrue(packets[-1].envelope.conversation_final)
                        speech=[]
                        for packet in packets:
                            for region in packet.envelope.regions:
                                if region.kind=='speech':
                                    lo=(region.sample_start-packet.envelope.sample_start)*2
                                    hi=(region.sample_stop-packet.envelope.sample_start)*2
                                    speech.append(packet.pcm[lo:hi])
                        expected=struct.pack('<h',8192)*(36*7680)
                        self.assertEqual(b''.join(speech),expected)
                        self.assertEqual(receipt.converted_sha256,hashlib.sha256(expected).hexdigest())
                finally:
                    release.set();thread.join(timeout=3);driver.close()


if __name__=='__main__':unittest.main()
