"""Synthetic file/pipe FFmpeg proof; no inference, network or devices."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import wave

from research.h3_personal.continuous_pcm import ContinuousPCM, native_wav
from research.h3_personal.relay_incremental_tts import produce
from research.h3_personal.streaming_pcm import PCMStreamBinding


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class ContinuousTests(unittest.TestCase):
    def convert(self, rate, width, raw):
        return subprocess.run(['ffmpeg', '-nostdin', '-v', 'error',
            '-f', f's{width*8}le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
            '-map', '0:a:0', '-ac', '1', '-ar', '48000', '-c:a', 'pcm_s16le',
            '-f', 's16le', 'pipe:1'], input=raw, capture_output=True,
            check=True, timeout=10).stdout

    def test_exact_raw_and_wav_all_rates_widths_short_tail_odd_splits(self):
        for rate in (44100, 24000, 48000):
            for width in (2, 3, 4):
                for frames in (1, 17, rate//3+7):
                    with self.subTest(rate=rate, width=width, frames=frames), tempfile.TemporaryDirectory() as root:
                        raw = bytes((i*37+11)%256 for i in range(frames*width))
                        expected = self.convert(rate, width, raw)
                        output = []
                        c = ContinuousPCM((rate,width,1), root, time.monotonic()+10, lambda: None, output.append)
                        try:
                            for start in range(0,len(raw),997):
                                c.feed((rate,width,1), raw[start:start+997])
                            if expected:
                                c.finish()
                            else:
                                with self.assertRaises(ValueError):
                                    c.finish()
                            self.assertEqual(b''.join(output), expected)
                        finally:
                            c.close()
                        self.assertFalse(c.reader.is_alive())
                        self.assertIsNotNone(c.process.poll())
                        wav = Path(root)/'input.wav'
                        with wave.open(str(wav),'wb') as w:
                            w.setparams((1,width,rate,0,'NONE','not compressed'))
                            w.writeframes(raw)
                        self.assertEqual(native_wav(wav), ((rate,width,1),raw))
                        wav_pcm = subprocess.run(['ffmpeg','-v','error','-i',str(wav),
                            '-map','0:a:0','-ac','1','-ar','48000','-c:a','pcm_s16le',
                            '-f','s16le','pipe:1'],capture_output=True,check=True,timeout=10).stdout
                        self.assertEqual(expected,wav_pcm)

    def test_cancel_blocked_pipes_and_deadline(self):
        for mode in ('write','read','deadline'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                active = threading.Event(); active.set()
                def check():
                    if not active.is_set():
                        raise InterruptedError('cancelled')
                c = ContinuousPCM((24000,4,1),root,time.monotonic()+(.15 if mode=='deadline' else 5),check,lambda b:None)
                try:
                    if mode in ('write', 'read', 'deadline'):
                        # Stop the child to force blocked input/output polling.
                        import signal
                        c.process.send_signal(signal.SIGSTOP)
                    timer = threading.Timer(.1, active.clear)
                    timer.start()
                    try:
                        with self.assertRaises((InterruptedError,TimeoutError)):
                            if mode == 'write':
                                c.feed((24000,4,1),bytes(24000*4*20))
                            else:
                                if mode == 'deadline':
                                    timer.cancel()
                                c.finish()
                    finally:
                        timer.cancel(); timer.join()
                finally:
                    c.close()
                self.assertFalse(c.reader.is_alive())
                self.assertIsNotNone(c.process.poll())

    def test_drift_and_excess(self):
        with tempfile.TemporaryDirectory() as root:
            c = ContinuousPCM((24000,2,1),root,time.monotonic()+5,lambda:None,lambda b:None)
            try:
                for fmt in ((44100,2,1),(24000,3,1)):
                    with self.assertRaises(ValueError): c.feed(fmt,b'12')
                with self.assertRaises(BufferError): c.feed((24000,2,1),bytes(1440002))
            finally: c.close()

    def test_first_pcm_while_second_tts_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            published = threading.Event()
            validated = []
            class TTS:
                def synthesize(self, text, turn, directory, deadline):
                    if turn.endswith('-2'):
                        if not published.wait(3): raise AssertionError('no incremental PCM')
                    directory.mkdir()
                    path = directory/'reply.wav'
                    with wave.open(str(path),'wb') as w:
                        w.setparams((1,3,44100,0,'NONE','not compressed'))
                        w.writeframes(bytes((i*13)%256 for i in range(44100*3)))
                    return {'wav':path}
            def validate(result, text, turn, directory):
                validated.append(turn)
                return result['wav']
            binding = PCMStreamBinding('c','s',0,'0'*64)
            def forbidden(*a): raise AssertionError('non48 must not use converter')
            stream = produce(TTS(),'a!b!','t',binding,root,time.monotonic()+10,
                forbidden,forbidden,lambda s:published.set(),lambda:True,validate_wav=validate)
            self.assertTrue(published.is_set())
            self.assertEqual(len(validated),2)
            self.assertEqual(len(stream.complete_source().pcm),192000)
            self.assertFalse(any(t.name=='h3-continuous-pcm' for t in threading.enumerate()))

    def test_partial_constructor_failure_reaps_child(self):
        from unittest.mock import patch
        spawned=[]
        original=subprocess.Popen
        def spawn(*args, **kwargs):
            process=original(*args, **kwargs)
            spawned.append(process)
            return process
        with tempfile.TemporaryDirectory() as root, \
             patch('research.h3_personal.continuous_pcm.subprocess.Popen', side_effect=spawn), \
             patch('research.h3_personal.continuous_pcm.os.set_blocking', side_effect=OSError):
            with self.assertRaises(OSError):
                ContinuousPCM((24000,2,1),root,time.monotonic()+5,lambda:None,lambda b:None)
            self.assertEqual(len(spawned),1)
            self.assertIsNotNone(spawned[0].poll())
            self.assertTrue(spawned[0].stdin.closed)
            self.assertTrue(spawned[0].stdout.closed)
            self.assertEqual(list(Path(root).iterdir()),[])


if __name__ == '__main__': unittest.main()
