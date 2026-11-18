"""Resident ownership/buffered playback contracts; no models/network/paid compute."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from research.h3_personal.resident_session import ResidentSession
from research.h3_personal.buffered_stream import BufferedStream, Clip, validate_clip
from research.h3_personal.component_cache import sha256


class ResidentTests(unittest.TestCase):
    def test_model_loaded_once_and_released(self):
        released=[];calls=[];model=object()
        def loader():calls.append(1);return model
        with ResidentSession(lambda:released.append(1)) as s:
            for _ in range(3):
                with s.turn():self.assertIs(s.get({'adapter':'fixed'},loader),model)
            self.assertEqual((s.loads,s.hits),(1,2))
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):s.require_conditioner_available()
        self.assertEqual(released,[1]);self.assertFalse(s.resident)
        with self.assertRaises(RuntimeError):
            with s.turn():pass

    def test_identity_change_rejected_and_failure_releases(self):
        released=[];s=ResidentSession(lambda:released.append(1))
        with s.turn():s.get({'base':1},object)
        with self.assertRaises(ValueError):
            with s.turn():s.get({'base':2},object)
        self.assertFalse(s.resident);self.assertEqual(released,[1])
        with self.assertRaises(ValueError):
            with s.turn():s.get({'base':2},object)
        s.close()

    def test_cancellation_no_reusable_state(self):
        released=[];s=ResidentSession(lambda:released.append(1))
        with self.assertRaises(KeyboardInterrupt):
            with s.turn():
                s.get({'base':1},object);raise KeyboardInterrupt()
        self.assertFalse(s.resident);self.assertEqual(released,[1]);s.close()

    def test_preview_session_wrapper_decode_guard_and_cooperative_interrupt(self):
        from unittest.mock import patch
        from research.hf_identity_preview import generate, decode, TurnInterrupted
        s=ResidentSession(lambda:None)
        def fake_generate(r,cp,case,*,session,is_current):
            return session.get({'fixed':'baseline'},object)
        with patch('research.hf_identity_preview._generate',side_effect=fake_generate):
            first=generate({},None,{},session=s)
            self.assertIs(generate({},None,{},session=s),first)
        self.assertEqual((s.loads,s.hits),(1,1))
        with self.assertRaises(ValueError):decode({}, {},session=s)
        s.close()
        with self.assertRaises(TurnInterrupted):
            generate({},None,{},is_current=lambda:False)

    def test_ordered_owner_and_reentrant_access(self):
        s=ResidentSession(lambda:None)
        with self.assertRaises(RuntimeError):s.get({},object)
        with s.turn():
            with self.assertRaises(RuntimeError):
                with s.turn():pass
            with self.assertRaises(RuntimeError):s.close()
        errors=[]
        def wrong_thread():
            try:s.get({},object)
            except RuntimeError:errors.append(True)
        t=threading.Thread(target=wrong_thread);t.start();t.join()
        self.assertEqual(errors,[True]);s.close()


class BufferedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.clip=Clip('v.mp4','tts.wav','a'*64,'b'*64)
        self.q=BufferedStream(self.root,max_clips=2,max_seconds=2,
                              validator=lambda r,c:Fraction(1,3))

    def test_exact_pts_backpressure_and_original_audio(self):
        turn=self.q.begin();self.q.enqueue(turn,self.clip);self.q.enqueue(turn,self.clip)
        with self.assertRaises(BufferError):self.q.enqueue(turn,self.clip)
        a=self.q.take();self.assertEqual(a.pts,0);self.assertTrue(a.mute_video_audio)
        self.assertEqual(a.clip.original_tts,'tts.wav');self.assertIsNone(self.q.take())
        self.assertTrue(self.q.acknowledge(a));self.q.enqueue(turn,self.clip)
        self.q.finish(turn);b=self.q.take();self.assertEqual(b.pts,Fraction(1,3))
        self.q.acknowledge(b);c=self.q.take();self.assertEqual(c.pts,Fraction(2,3))
        self.q.acknowledge(c);self.assertEqual(self.q.state,'LISTENING')
        self.assertFalse(self.q.current(turn))

    def test_interrupt_discards_pending_inflight_and_late_results(self):
        turn=self.q.begin();self.q.enqueue(turn,self.clip);a=self.q.take()
        self.q.enqueue(turn,self.clip);self.q.interrupt()
        self.assertIsNone(self.q.take());self.assertFalse(self.q.acknowledge(a))
        self.assertFalse(self.q.enqueue(turn,self.clip));self.assertEqual(self.q.state,'INTERRUPTED')
        new=self.q.begin();self.q.enqueue(new,self.clip)
        self.assertNotEqual(new,turn);self.assertEqual(self.q.take().pts,0)

    def test_interruption_during_validation_and_peer_audio_gate(self):
        def validate(r,c):self.q.interrupt();return Fraction(1)
        turn=self.q.begin();self.q.validator=validate
        self.assertFalse(self.q.enqueue(turn,self.clip));self.assertIsNone(self.q.take())
        with self.assertRaises(ValueError):self.q.begin(source='peer')
        self.assertFalse(self.q.current(None))

    def test_producer_done_and_buffer_duration_gate(self):
        self.q.max_seconds=Fraction(1,4);turn=self.q.begin()
        with self.assertRaises(BufferError):self.q.enqueue(turn,self.clip)
        self.q.finish(turn);self.assertFalse(self.q.enqueue(turn,self.clip))
        turn=self.q.begin();self.q.max_seconds=Fraction(2);self.q.enqueue(turn,self.clip)
        self.q.finish(turn)
        with self.assertRaises(RuntimeError):self.q.enqueue(turn,self.clip)
        with self.assertRaises(RuntimeError):self.q.begin()

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_real_synthetic_av_validation_and_tamper(self):
        v=self.root/'v.mp4';a=self.root/'tts.wav'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=32x32:rate=24',
                        '-t','0.25','-c:v','mpeg4',str(v)],check=True)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=16000',
                        '-t','0.2',str(a)],check=True)
        clip=Clip(v.name,a.name,sha256(v),sha256(a))
        self.assertEqual(validate_clip(self.root,clip),Fraction(1,4))
        q=BufferedStream(self.root);turn=q.begin();self.assertTrue(q.enqueue(turn,clip))
        self.assertEqual(q.take().duration,Fraction(1,4))
        a.write_bytes(b'changed')
        with self.assertRaises(ValueError):validate_clip(self.root,clip)
        with self.assertRaises(ValueError):
            validate_clip(self.root,Clip('../v.mp4',a.name,clip.video_sha256,clip.audio_sha256))


if __name__=='__main__':unittest.main()
