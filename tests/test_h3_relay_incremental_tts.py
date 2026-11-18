import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research.h3_personal.director_irodori import SOURCE, MODEL, CODEC
from research.h3_personal.live_relay_worker import run_worker, complete_pcm, PI_MODEL
from research.h3_personal.relay_incremental_tts import produce, sentences
from research.h3_personal.streaming_pcm import PCMStreamBinding
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import ResamplingContext


def convert(path, deadline):
    with wave.open(str(path), 'rb') as wav:
        return wav.readframes(wav.getnframes())


class FakeTTS:
    def __init__(self, second=lambda: None, rate=48000, bad=False, frames=48000):
        self.calls = 0
        self.second, self.rate, self.bad, self.frames = second, rate, bad, frames

    def synthesize(self, text, turn_id, directory, deadline):
        self.calls += 1
        if self.calls == 2:
            self.second()
        directory.mkdir()
        path = directory/'reply.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setparams((1, 2, self.rate, 0, 'NONE', 'not compressed'))
            wav.writeframes(bytes([self.calls, 0])*self.frames)
        return dict(turn_id=turn_id, wav=str(path), receipt=dict(
            watermark_applied=not self.bad, source_revision=SOURCE,
            model_revision=MODEL, codec_revision=CODEC,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


class IncrementalTests(unittest.TestCase):
    def test_publish_before_second_complete_and_seal_order_hash(self):
        published=[]
        def second():
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0].snapshot()['state'], 'open')
        with tempfile.TemporaryDirectory() as root:
            stream=produce(FakeTTS(second), '一。二。', 'typed-1',
                PCMStreamBinding('s','typed-1',1,'a'*64),root,time.monotonic()+5,
                convert,complete_pcm,published.append,lambda: True)
            expected=b'\x01\x00'*48000+b'\x02\x00'*48000
            self.assertEqual(stream.complete_source().pcm,expected)
            self.assertEqual(stream.prefix_digest(96000),hashlib.sha256(expected).hexdigest())
            self.assertEqual(stream.snapshot()['state'],'sealed')

    def test_rejections_and_cancellation_after_blocked_call(self):
        for mode in (False, [], None, 'native_streaming'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                run_worker(None,None,None,binding=None,context=None,
                    root='unused',deadline=time.monotonic()+5,tts_mode=mode)
        for options in (dict(rate=24000), dict(bad=True), dict(frames=800000)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as root:
                with self.assertRaises((ValueError, BufferError)):
                    produce(FakeTTS(**options),'一。二。','t',PCMStreamBinding('s','t',1,'a'*64),
                        root,time.monotonic()+5,convert,complete_pcm,lambda s: None,lambda: True)
        self.assertEqual(sentences('本当！？すごい！！'), ['本当！？', 'すごい！！'])
        for reply in ('a'*161,'一。二。三。四。五。','{}', '！？'):
            with self.assertRaises(ValueError):sentences(reply)
        with tempfile.TemporaryDirectory() as root:
            current=threading.Event();current.set()
            streams=[]
            def second():current.clear()
            with self.assertRaises(InterruptedError):
                produce(FakeTTS(second),'一。二。','t',PCMStreamBinding('s','t',1,'a'*64),
                    root,time.monotonic()+5,convert,complete_pcm,streams.append,current.is_set)
            with self.assertRaises(RuntimeError):streams[0].snapshot()

    def test_cancel_while_second_synthesis_blocked(self):
        started=threading.Event();release=threading.Event();current=threading.Event()
        current.set();streams=[];errors=[]
        def second():
            started.set()
            if not release.wait(2):raise TimeoutError()
        with tempfile.TemporaryDirectory() as root:
            def run():
                try:
                    produce(FakeTTS(second),'一。二。','t',PCMStreamBinding('s','t',1,'a'*64),
                        root,time.monotonic()+5,convert,complete_pcm,streams.append,current.is_set)
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=run)
            thread.start()
            self.assertTrue(started.wait(2))
            self.assertEqual(streams[0].snapshot()['state'],'open')
            current.clear();release.set();thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(errors[0],InterruptedError)
            with self.assertRaises(RuntimeError):streams[0].snapshot()

    def test_real_worker_underpreroll_listening_then_speech(self):
        self._worker_underpreroll(48000)

    def test_real_worker_continuous_non48k_ingress(self):
        self._worker_underpreroll(24000)

    def _worker_underpreroll(self, rate):
        # Real worker + real admission/PCM; fake serial TTS/backend/transport.
        attached=threading.Event(); listened=threading.Event(); roles=[]; streams=[]
        def second():
            if not attached.wait(2) or not listened.wait(2):
                raise AssertionError('first short sentence blocked attach/listening')
        tts=FakeTTS(second, rate=rate, frames=rate)
        row=dict(kind='turn',epoch=1,payload=dict(schema='h3-relay-typed-turn-v1',
            origin='fresh_typed_text',asr_used=False,reply='一。二。',director_model=PI_MODEL,
            reasoning_effort='none',previous_audio_reused=False))
        class Commands:
            def __init__(self,*a):self.n=0
            def receive(self):
                self.n+=1
                return row if self.n==1 else dict(kind='stop')
            def close(self):pass
        class Backend:
            def __init__(self,*a,**kw):pass
            def __call__(self,m):
                c=m['streaming_audio'];roles.append(m['role'] if 'role' in m else
                    ('listening' if c['audio_offer'] is None else 'speech'))
                if attached.is_set() and c['audio_offer'] is None:listened.set()
                time.sleep(.05)
                end=c['committed_before']+(124 if c['sequence']==0 else 102)
                emitted=end-(0 if c['conversation_final'] else 5)
                before=0 if c['sequence']==0 else c['committed_before']-5
                return dict(video=b'video',count=emitted-before,committed_frame=end,
                    emitted_frame=emitted,conditioning_prefix_sha256=None if c['audio_offer'] is None
                    else c['audio_offer']['prefix_sha256'],receipt_sha256='b'*64)
            def retire(self):pass
            def abandon(self,**kw):pass
        class Pump:
            def __init__(self,*a,**kw):pass
            def run(self,packets):
                for packet in packets:pass
                return {}
        from research.h3_personal.live_conversation import LiveConversation
        original=LiveConversation.attach
        def attach(driver,stream):
            streams.append(stream)
            original(driver,stream)
            attached.set()
        with tempfile.TemporaryDirectory() as root, \
             patch('research.h3_personal.live_relay_worker.Commands',Commands), \
             patch('research.h3_personal.live_relay_worker.LiveNativeBackend',Backend), \
             patch('research.h3_personal.live_relay_worker.ProducerPump',Pump), \
             patch('research.h3_personal.live_relay_worker.role_profile_digest',return_value='a'*64), \
             patch.object(LiveConversation,'attach',attach):
            result=run_worker(SimpleNamespace(quality_profile={}),tts,Mock(),
                binding=LiveAVBinding('s','a'*64,0,max_segments=32,max_total_frames=3286),
                context=ResamplingContext(),root=root,deadline=time.monotonic()+10,
                canonical_pcm=convert,tts_mode='sentence_incremental')
            import json
            timing=json.loads((Path(root)/'stage-timing.json').read_text())
            self.assertEqual(timing['component'], 'gpu_worker')
            self.assertEqual(timing['status'], 'completed')
            stages=[event['stage'] for event in timing['events']]
            self.assertIn('speech_pcm_ready', stages)
            self.assertIn('first_speech_packet_ready', stages)
            self.assertNotIn('一。二。', json.dumps(timing, ensure_ascii=False))
        self.assertEqual(result['turns'],1)
        self.assertTrue(listened.is_set())
        self.assertIn('speech',roles)
        self.assertEqual(tts.calls,2)


if __name__=='__main__':unittest.main()
