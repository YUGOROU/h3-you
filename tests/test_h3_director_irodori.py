import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace as N
from research.h3_personal.director_irodori import IrodoriResident,SOURCE,MODEL,CODEC,sha

class Tensor:
    def isfinite(self):return self
    def all(self):return self
    def item(self):return True

class ResidentTests(unittest.TestCase):
    def build(self,folder,mark=True):
        ref=Path(folder)/'ref.wav';ref.write_bytes(b'private reference fixture')
        model=N(encode_wav=lambda *a,**k:([1],None));watermarker=N(ready=True,model=model)
        def synth(req,log_fn=None):
            if mark:model.encode_wav(None,24000,[73,82,68,84,83],calc_sdr=False)
            self.assertEqual(req.num_steps,40);self.assertEqual(req.seed,42)
            return N(audio=Tensor(),sample_rate=24000)
        runtime=N(watermarker=watermarker,synthesize=synth,key=N(model_precision='fp32',codec_precision='fp32',compile_model=False))
        def save(path,audio,rate):
            with wave.open(str(path),'wb') as w:w.setparams((1,2,rate,0,'NONE','not compressed'));w.writeframes(b'\0\0'*2400)
        adapter=IrodoriResident(runtime,lambda **kw:N(**kw),save,ref,sha(ref),source_revision=SOURCE,model_revision=MODEL,codec_revision=CODEC)
        return adapter,runtime

    def test_native_owner_reused_and_no_playback(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter,_=self.build(folder)
            for i in range(2):
                out=adapter.synthesize('こんにちは。',Path(folder)/f'{i}.wav')
                self.assertTrue(out['watermark_applied']);self.assertFalse(out['playback_started'])
                self.assertEqual(out['samples'],2400)

    def test_native_silent_watermark_fallback_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter,_=self.build(folder,mark=False);path=Path(folder)/'out.wav'
            with self.assertRaises(RuntimeError):adapter.synthesize('こんにちは。',path)
            self.assertFalse(path.exists())

    def test_stale_output_not_released(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter,_=self.build(folder);calls=iter([True,False]);path=Path(folder)/'out.wav'
            with self.assertRaises(RuntimeError):adapter.synthesize('こんにちは。',path,is_current=lambda:next(calls))
            self.assertFalse(path.exists())

    def test_drift_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter,_=self.build(folder);path=Path(folder)/'existing.wav';path.write_bytes(b'untouched')
            with self.assertRaises(ValueError):adapter.synthesize('こんにちは。',path)
            self.assertEqual(path.read_bytes(),b'untouched')
            adapter.reference.write_bytes(b'changed')
            with self.assertRaises(ValueError):adapter.synthesize('こんにちは。',Path(folder)/'new.wav')

if __name__=='__main__':unittest.main()
