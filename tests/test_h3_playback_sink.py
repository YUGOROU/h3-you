"""No actual playback/network/models; real FFmpeg only on tiny synthetic clips."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave
from research.h3_personal.buffered_stream import BufferedStream, Clip
from research.h3_personal.playback_sink import LocalPlaybackSink, PlaybackPump, PlaybackInterrupted, _kill
from research.play_h3_buffered import main, load_manifest
from research.h3_personal.component_cache import sha256


class PumpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.q=BufferedStream(self.tmp.name,validator=lambda r,c:Fraction(1,4))
        self.t=self.q.begin();self.clip=Clip('v','a','a'*64,'b'*64)
        self.q.enqueue(self.t,self.clip);self.q.finish(self.t)
        self.sink=Mock();self.sink.poll.return_value=None
        self.p=PlaybackPump(self.q,self.sink)

    def test_normal_ack_only_on_exit(self):
        self.assertFalse(self.p.tick());self.assertEqual(self.q.state,'SPEAKING')
        self.sink.poll.return_value=0
        self.assertTrue(self.p.tick());self.assertEqual(self.q.state,'LISTENING')
        self.sink.start.assert_called_once()

    def test_interrupt_stops_both_tracks_and_discards(self):
        self.p.tick();self.p.interrupt()
        self.sink.stop.assert_called_once();self.assertFalse(self.q.current(self.t))
        self.assertFalse(self.p.tick());self.assertIsNone(self.q.take())

    def test_external_queue_interrupt_detected(self):
        self.p.tick();self.q.interrupt();self.assertFalse(self.p.tick())
        self.sink.stop.assert_called_once()

    def test_failure_and_late_start_refuse_ack(self):
        self.sink.start.side_effect=PlaybackInterrupted()
        self.assertFalse(self.p.tick())
        # A real interruption has invalidated the queue; no stale acknowledgment.
        self.p.interrupt();self.assertFalse(self.q.current(self.t))

    def test_player_failure_invalidates_remaining_turn(self):
        self.sink.poll.return_value=1
        with self.assertRaises(RuntimeError):self.p.tick()
        self.assertFalse(self.q.current(self.t));self.sink.stop.assert_called_once()

    def test_group_kill_escalates_and_reaps(self):
        child=Mock(pid=123);child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('fake',2),0]
        with patch('research.h3_personal.playback_sink.os.killpg') as kill:
            _kill(child)
        self.assertEqual(kill.call_count,2);self.assertEqual(child.wait.call_count,2)

    def test_opt_in_and_manifest_rejection(self):
        with self.assertRaises(ValueError):LocalPlaybackSink(self.tmp.name)
        p=Path(self.tmp.name)/'manifest.json';p.write_text('{"schema":"wrong","clips":[]}')
        with self.assertRaises(ValueError):load_manifest(p)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class SyntheticSinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        v=self.root/'v.mp4';a=self.root/'tts.wav'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=32x32:rate=24',
                        '-t','0.25','-c:v','mpeg4',str(v)],check=True)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=sample_rate=16000',
                        '-t','0.2',str(a)],check=True)
        self.clip=Clip(v.name,a.name,sha256(v),sha256(a))
        self.q=BufferedStream(self.root);self.turn=self.q.begin();self.q.enqueue(self.turn,self.clip)
        self.item=self.q.take()

    def test_mux_copies_original_pcm_then_player_cleanup(self):
        original_popen=subprocess.Popen;player=Mock();player.poll.return_value=None;commands=[]
        def popen(cmd,*args,**kwargs):
            if cmd[0]=='ffplay':commands.append(cmd);return player
            return original_popen(cmd,*args,**kwargs)
        with patch('research.h3_personal.playback_sink.shutil.which',return_value='/mock'), \
             patch('research.h3_personal.playback_sink.subprocess.Popen',side_effect=popen):
            sink=LocalPlaybackSink(self.root,allow_playback=True)
            sink.start(self.item,is_current=lambda:self.q.current(self.turn))
            mux=Path(commands[0][-1]);self.assertTrue(mux.exists())
            decoded=subprocess.run(['ffmpeg','-v','error','-i',str(mux),'-map','0:a:0',
                '-f','s16le','-'],check=True,capture_output=True).stdout
            with wave.open(str(self.root/'tts.wav'),'rb') as audio:
                self.assertEqual(decoded,audio.readframes(audio.getnframes()))
            self.assertIsNone(sink.poll());player.poll.return_value=0;player.wait.return_value=0
            self.assertEqual(sink.poll(),0);self.assertFalse(mux.exists());sink.close()

    def test_interrupt_during_snapshot_never_opens_player(self):
        from research.h3_personal import playback_sink as module
        snapshot=module._snapshot
        with patch.object(module.shutil,'which',return_value='/mock'):
            sink=LocalPlaybackSink(self.root,allow_playback=True)
            def interrupt(*args):snapshot(*args);sink.stop()
            with patch.object(module,'_snapshot',side_effect=interrupt),patch.object(module.subprocess,'Popen') as spawn:
                with self.assertRaises(PlaybackInterrupted):
                    sink.start(self.item,is_current=lambda:True)
                spawn.assert_not_called()
            self.assertIsNone(sink._temp);sink.close()

    def test_quantized_video_tail_does_not_truncate_tts(self):
        a=self.root/'tail.wav'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=sample_rate=16000',
                        '-t','0.26',str(a)],check=True)
        clip=Clip(self.clip.video,a.name,self.clip.video_sha256,sha256(a))
        from research.h3_personal.buffered_stream import validate_clip
        self.assertEqual(validate_clip(self.root,clip),Fraction(13,50))

    def test_tampering_and_cli_validation_only(self):
        import json
        p=self.root/'manifest.json'
        p.write_text(json.dumps({'schema':'h3-buffered-playback-v1','clips':[self.clip.__dict__]}))
        with patch('research.play_h3_buffered.LocalPlaybackSink') as sink:
            self.assertEqual(main(['--root',str(self.root),'--manifest',str(p)]),0)
            sink.assert_not_called()
        (self.root/'tts.wav').write_bytes(b'changed')
        with patch('research.h3_personal.playback_sink.shutil.which',return_value='/mock'):
            with LocalPlaybackSink(self.root,allow_playback=True) as sink:
                with self.assertRaises(ValueError):sink.start(self.item,is_current=lambda:True)
                self.assertIsNone(sink._temp)


if __name__=='__main__':unittest.main()
