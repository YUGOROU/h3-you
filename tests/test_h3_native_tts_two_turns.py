"""Two-turn native worker + real CPU conversation driver; native video is fake."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from research.h3_personal.live_relay_worker import run_worker,PI_MODEL
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import (ResamplingContext,NATIVE_TORCH,
    NATIVE_TORCHAUDIO,REFERENCE_SHA256)
from research.h3_personal.tts_preroll import PCMPrerollPolicy


class NativeTwoTurns(unittest.TestCase):
    def test_two_native_epochs_exact_audio_and_receipts(self):
        all_pcm={i:threading.Event() for i in (1,2)}
        speech_started={i:threading.Event() for i in (1,2)}
        packets=[];calls=[];closed=[];last_speech=[0]
        class Commands:
            def __init__(self,*args):self.index=0
            def receive(self):
                self.index+=1
                if self.index>2:return dict(kind='stop',epoch=2)
                epoch=self.index
                return dict(kind='turn',epoch=epoch,payload=dict(schema='h3-relay-typed-turn-v1',
                    origin='fresh_typed_text',asr_used=False,reply=f'reply {epoch}',director_model=PI_MODEL,
                    reasoning_effort='none',previous_audio_reused=False))
            def close(self):pass
        class TTS:
            def stream(self,text,epoch,deadline):
                calls.append((text,epoch))
                try:
                    raw=struct.pack('<f',epoch/4)*7680
                    for _ in range(36):yield raw
                    all_pcm[epoch].set()
                    if not speech_started[epoch].wait(3):raise TimeoutError('speech handoff did not happen')
                finally:closed.append(epoch)
            def close(self):pass
        class Backend:
            def __init__(self,*args,**kwargs):self.retired=False
            def __call__(self,manifest):
                c=manifest['streaming_audio']
                if c['role']=='speech':
                    epoch=int(c['logical_source_id'].split('-')[-1])
                    last_speech[0]=epoch;speech_started[epoch].set()
                elif last_speech[0]<2:
                    # Deterministic CPU surrogate for time spent generating an
                    # idle native window. No measured speed/playback claim.
                    if not all_pcm[last_speech[0]+1].wait(3):raise TimeoutError('source did not arrive')
                end=c['committed_before']+(124 if c['sequence']==0 else 102)
                emitted=end-(0 if c['conversation_final'] else 5)
                before=0 if c['sequence']==0 else c['committed_before']-5
                return dict(video=b'fake-native-video',count=emitted-before,committed_frame=end,
                    emitted_frame=emitted,conditioning_prefix_sha256=None if c['audio_offer'] is None
                    else c['audio_offer']['prefix_sha256'],receipt_sha256='b'*64)
            def retire(self):self.retired=True
            def close_unused(self):raise AssertionError('healthy used backend must not be abandoned')
        class Pump:
            def __init__(self,*args,**kwargs):pass
            def run(self,iterator):
                packets.extend(iterator)
                return dict(complete=True,final=True)
        binding=LiveAVBinding('native-two-turn','a'*64,1,max_segments=16,max_total_frames=1654)
        context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        with tempfile.TemporaryDirectory() as root,contextlib.redirect_stdout(io.StringIO()), \
                patch('research.h3_personal.live_relay_worker.Commands',Commands), \
                patch('research.h3_personal.live_relay_worker.LiveNativeBackend',Backend), \
                patch('research.h3_personal.live_relay_worker.ProducerPump',Pump), \
                patch('research.h3_personal.live_relay_worker.role_profile_digest',return_value='a'*64):
            result=run_worker(SimpleNamespace(quality_profile={}),TTS(),Mock(),binding=binding,
                context=context,root=root,deadline=time.monotonic()+15,
                tts_mode='native_voxcpm',tts_preroll=PCMPrerollPolicy())
            receipts=json.loads((Path(root)/'native-tts-receipts.json').read_text())
        self.assertEqual(result['turns'],2)
        self.assertEqual(result['native_tts_receipts'],receipts)
        self.assertEqual(calls,[('reply 1',1),('reply 2',2)])
        self.assertEqual(closed,[1,2])
        self.assertEqual([r['epoch'] for r in receipts],[1,2])
        self.assertTrue(packets[-1].envelope.conversation_final)
        for epoch,receipt in enumerate(receipts,1):
            original=struct.pack('<f',epoch/4)*(36*7680)
            expected=struct.pack('<h',epoch*8192)*(36*7680)
            parts=[]
            for packet in packets:
                for region in packet.envelope.regions:
                    if region.kind=='speech' and region.source_epoch==epoch:
                        lo=(region.sample_start-packet.envelope.sample_start)*2
                        hi=(region.sample_stop-packet.envelope.sample_start)*2
                        parts.append(packet.pcm[lo:hi])
            self.assertEqual(b''.join(parts),expected)
            self.assertEqual(receipt['source_id'],f'typed-{epoch}')
            self.assertEqual(receipt['native_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(receipt['converted_sha256'],hashlib.sha256(expected).hexdigest())
            self.assertEqual(receipt['native_samples'],36*7680)
            self.assertEqual(receipt['preroll_policy']['reserve_samples'],48000)
            self.assertFalse(receipt['watermark_applied'] or receipt['model_eos_verified'])


if __name__=='__main__':unittest.main()
