"""CPU-only deployment tests using synthetic speech; no Hub access."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import wave
from research.bootstrap_h3_fast_ref2va import locked_speech_plan,locked_speech_args


def spec(seconds=12):
    return dict(layout='portrait',runtime_policy='resident-visual',locked_speech=dict(
        repo='owner/private',revision='a'*40,path='reply.wav',sha256='b'*64,
        sample_rate=32000,samples=32000*seconds))


class LockedSpeechGPU(unittest.TestCase):
    def test_windows(self):
        for count,seconds in ((1,1),(2,6),(3,12),(4,18)):
            self.assertEqual(locked_speech_plan(spec(seconds))['windows_required'],count)
        with self.assertRaises(ValueError):locked_speech_plan(spec(22))

    def test_isolation(self):
        for key,value in [('layout','landscape'),('runtime_policy','baseline'),
                          ('regional_compile',True),('face_crop_box',[0,0,1,1])]:
            s=spec();s[key]=value
            with self.assertRaises(ValueError):locked_speech_plan(s)
        for key,value in [('samples',True),('sample_rate',0),('revision','main'),
                          ('sha256','z'*64),('path','../speech.wav'),('repo','bad')]:
            s=spec();s['locked_speech'][key]=value
            with self.assertRaises(ValueError):locked_speech_plan(s)

    def test_wav_and_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'reply.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,32000,0,'NONE','not compressed'))
                wav.writeframes(b'\0\0'*32000*12)
            s=spec();s['locked_speech']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            args,count=locked_speech_args(s,{'speech':root})
            self.assertEqual(count,3)
            self.assertEqual(args,['--locked-speech',str(path),'--locked-speech-sha256',s['locked_speech']['sha256']])
            s['locked_speech']['samples']+=1
            with self.assertRaises(ValueError):locked_speech_args(s,{'speech':root})
            s['locked_speech']['samples']-=1;s['locked_speech']['sha256']='0'*64
            with self.assertRaises(ValueError):locked_speech_args(s,{'speech':root})

    def test_unlocked(self):
        self.assertIsNone(locked_speech_plan({}))
        self.assertEqual(locked_speech_args({},{}),([],4))
