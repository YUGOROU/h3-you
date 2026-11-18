"""Resident native Irodori adapter. No installations, downloads, playback or devices.

Instantiate only inside the separately provisioned, pinned Irodori environment.
request_type/save_wav are native SamplingRequest/save_wav, injected for CPU tests.
"""
import hashlib
import os
from pathlib import Path
import threading
import wave

SOURCE='89f9d8fbd4d51ea019867ee1197725ede1df13c5'
MODEL='2e0c55428ce97268a507f1feeb2478f8d9148e8b'
CODEC='47376ee24834d7a05a48ebabfe3cde29b3c5e214'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IrodoriResident:
    def __init__(self,runtime,request_type,save_wav,reference,reference_sha256,*,source_revision,model_revision,codec_revision):
        if (source_revision,model_revision,codec_revision)!=(SOURCE,MODEL,CODEC):raise ValueError('Irodori binding mismatch')
        if runtime.key.model_precision!='fp32' or runtime.key.codec_precision!='fp32' or runtime.key.compile_model:
            raise ValueError('accepted Irodori FP32/eager policy required')
        self.reference=Path(reference);self.reference_sha256=reference_sha256
        if sha(self.reference)!=reference_sha256:raise ValueError('voice reference drift')
        if not runtime.watermarker.ready:raise RuntimeError('mandatory SilentCipher unavailable')
        self.runtime=runtime;self.request_type=request_type;self.save_wav=save_wav;self.lock=threading.Lock()

    def synthesize(self,text,destination,*,is_current=lambda:True):
        if not isinstance(text,str) or not text.strip() or len(text)>160:raise ValueError('bounded speech text required')
        destination=Path(destination)
        if destination.exists():raise ValueError('immutable output already exists')
        if not self.lock.acquire(blocking=False):raise RuntimeError('TTS owner busy')
        watermarker=self.runtime.watermarker
        model=watermarker.model
        if model is None:self.lock.release();raise RuntimeError('mandatory watermark model unavailable')
        original=model.encode_wav;marked=False;owned=False
        def encode(*args,**kwargs):
            nonlocal marked
            if len(args)<3 or list(args[2])!=[73,82,68,84,83]:raise RuntimeError('watermark payload mismatch')
            result=original(*args,**kwargs)
            if result is None or result[0] is None:raise RuntimeError('empty watermarked audio')
            marked=True;return result
        try:
            if not is_current():raise RuntimeError('stale turn')
            if sha(self.reference)!=self.reference_sha256:raise ValueError('voice reference drift')
            if not watermarker.ready:raise RuntimeError('mandatory SilentCipher unavailable')
            destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            fd=os.open(destination,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd);owned=True
            model.encode_wav=encode
            request=self.request_type(text=text,ref_wav=str(self.reference),num_steps=40,cfg_scale_text=3.0,
                                      cfg_scale_speaker=5.0,seed=42,num_candidates=1)
            result=self.runtime.synthesize(request,log_fn=None)
            if not marked or not watermarker.ready:raise RuntimeError('no successful watermark operation')
            if not is_current():raise RuntimeError('stale TTS result')
            if not result.audio.isfinite().all().item():raise ValueError('non-finite generated audio')
            self.save_wav(destination,result.audio,result.sample_rate)
            with wave.open(str(destination),'rb') as wav:
                samples=wav.getnframes();rate=wav.getframerate();channels=wav.getnchannels()
            if not 0<samples/rate<=30:raise ValueError('invalid generated audio duration; never truncate')
            return dict(path=str(destination),sha256=sha(destination),samples=samples,sample_rate=rate,channels=channels,
                        watermark_applied=True,watermark_detected=None,source_revision=SOURCE,model_revision=MODEL,codec_revision=CODEC,
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),playback_started=False)
        except Exception:
            # Output belongs exclusively to this call; never publish partial or stale speech.
            if owned and destination.exists():destination.unlink()
            raise
        finally:
            model.encode_wav=original;self.lock.release()
