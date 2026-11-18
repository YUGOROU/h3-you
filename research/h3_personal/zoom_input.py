"""Opt-in local Zoom mix→VoxLock→ASR reader. No capture, network or routing.

Caller owns approved capture and ASR lease. Default input eligibility is CLOSED.
Manual human-turn confirmation plus validated echo-tail policy is required for
live use; extraction/nonzero audio never proves human speaker identity.
"""
import hashlib
import json
import math
import time
import threading
import functools


def locked(fn):
    @functools.wraps(fn)
    def call(self,*a,**kw):
        with self._lock:return fn(self,*a,**kw)
    return call

class InputBlocked(RuntimeError):pass


class CaptureInterrupted(InputBlocked):
    """Transient capture-infrastructure loss (process churn, tap abort, queue
    full, premature stream end). Reject this turn without replay, but do NOT
    fault a healthy ASR lease. Data-shape/contract drift stays ValueError."""


class EchoGuard:
    def __init__(self, *, tail_seconds, tail_validated=False, clock=time.monotonic):
        if type(tail_seconds) not in (int,float) or not math.isfinite(tail_seconds) or not 0<tail_seconds<=10:
            raise ValueError('explicit finite echo tail in (0,10] required')
        if type(tail_validated) is not bool:raise ValueError('explicit tail validation bool required')
        self._lock=threading.RLock()
        self.tail=tail_seconds;self.tail_validated=tail_validated;self.clock=clock
        self.epoch=0;self.speaking=False;self.closed=True;self.until=float('inf');self.deadline=0
    @locked
    def output_started(self):
        self.epoch+=1;self.closed=True;self.speaking=True;self.until=float('inf')
    @locked
    def output_stopped(self):
        self.epoch+=1;self.closed=True;self.speaking=False;self.until=self.clock()+self.tail
    @locked
    def arm(self, *, deadline, manual_human_turn, all_participants_consented, offline=False):
        now=self.clock()
        if (type(offline) is not bool or type(deadline) not in (int,float) or
                manual_human_turn is not True or all_participants_consented is not True or
                (not offline and self.tail_validated is not True) or self.speaking or now<self.until or
                not math.isfinite(deadline) or not now<deadline<=now+45):
            raise InputBlocked('input not eligible; no inferred identity or remote echo-tail proof')
        self.epoch+=1;self.closed=False;self.deadline=deadline
        return self.epoch
    @locked
    def initial_idle(self):
        """Operator-established idle baseline; cannot bypass speaking/tail."""
        if self.speaking or (self.until!=float('inf') and self.clock()<self.until):
            raise InputBlocked('cannot override active output/echo tail')
        self.epoch+=1;self.closed=True;self.speaking=False;self.until=self.clock()
    @locked
    def check(self,epoch):
        if self.closed or self.speaking or epoch!=self.epoch or self.clock()>=self.deadline:
            raise InputBlocked('input epoch blocked/expired; discard, never replay')
    @locked
    def dispatch(self,epoch,fn):
        self.check(epoch)
        return fn()
    @locked
    def block(self):
        self.epoch+=1;self.closed=True


def source_contract(*, reference_sha256, capture_source_sha256):
    from research.h3_personal.voxlock_streaming import SOURCE_REVISION,MODEL_REVISION,MODEL_FILES
    for value in (reference_sha256,capture_source_sha256):
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('exact reference/capture fingerprints required')
    value=dict(schema='h3-zoom-input-v1',origin='zoom_process_voxlock_endpoint_asr',
        capture_app='zoom',bundle_candidate='us.zoom.xos',capture_scope='application-mix-not-participant',
        capture_source_sha256=capture_source_sha256,encoding='PCM16LE',sample_rate=16000,channels=1,
        cpu_threads=4,batch_frames=8,capture_quantum_samples=1024,
        extractor='voxlock-v2-onnx-fp32',source_revision=SOURCE_REVISION,model_revision=MODEL_REVISION,
        model_sha256={k:v[1] for k,v in MODEL_FILES.items()},reference_sha256=reference_sha256,
        policy='manual-human-window-halfduplex-no-raw-fallback-v1',
        speaker_identity_verified=False,clone_echo_rejection_by_tse=False)
    value['profile_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


class ZoomPCMReader:
    """Bounded binary reader for existing endpoint_asr.transcribe(pcm=...).

    Capture emits mono16k S16LE. Extractor preserves float32 clock/state; output
    quantization is explicit PCM16, never a raw-input fallback. No capture owns
    or closes here. On any blocked epoch/error discard pending output/state.
    Caller MUST abort/reject an in-flight ASR turn on InputBlocked, including
    any final for already-submitted samples; receipts do not prove human ID.
    """
    def __init__(self,raw,extractor,guard,epoch,*,capture_quantum_samples=1024):
        guard.check(epoch)
        if type(capture_quantum_samples) is not int or not 128<=capture_quantum_samples<=4096:
            raise ValueError('bounded capture quantum required')
        self.quantum=capture_quantum_samples
        self.raw=raw;self.extractor=extractor;self.guard=guard;self.epoch=epoch
        self.pending=bytearray();self.eof=False;self.failed=False
        self.stats=dict(captured_samples=0,extracted_samples=0,submitted_samples=0,discarded_pending_samples=0)
        extractor.reset()
    def _check(self):
        if self.failed:raise InputBlocked('reader terminal')
        self.guard.check(self.epoch)
    def _encode(self,y):
        import numpy as np
        y=np.asarray(y)
        if y.dtype!=np.float32 or y.ndim!=1 or not np.isfinite(y).all() or len(y)>8192:
            raise ValueError('invalid bounded extractor output')
        if len(y) and (float(y.min()) < -1 or float(y.max()) > 32767/32768):
            raise ValueError('extraction would clip PCM16; no silent clipping')
        self.stats['extracted_samples']+=len(y)
        self.pending.extend(np.rint(y*32768).astype('<i2').tobytes())
        if len(self.pending)>16384:raise ValueError('extracted pending bound')
    def read(self,size):
        if type(size) is not int or not 0<size<=8192 or size%2:raise ValueError('bounded aligned read required')
        try:
            self._check()
            while not self.pending and not self.eof:
                data=self.raw.read(self.quantum*2)  # explicit64ms selected profile
                self._check()
                if not isinstance(data,bytes) or len(data)>self.quantum*2 or len(data)%2:
                    raise ValueError('capture PCM framing/bounds')
                if not data:
                    self.eof=True;self._encode(self.extractor.flush())
                    if self.stats['extracted_samples']!=self.stats['captured_samples']:
                        raise ValueError('extraction sample clock/tail mismatch')
                else:
                    import numpy as np
                    self.stats['captured_samples']+=len(data)//2
                    if self.stats['captured_samples']>720000:raise ValueError('capture exceeds45s')
                    y=self.extractor.push(np.frombuffer(data,dtype='<i2').astype(np.float32)/32768)
                    self._check();self._encode(y)
            def release():
                value=bytes(self.pending[:size]);del self.pending[:size]
                self.stats['submitted_samples']+=len(value)//2
                return value
            return self.guard.dispatch(self.epoch,release)
        except BaseException:
            self.failed=True;self.stats['discarded_pending_samples']+=len(self.pending)//2;self.pending.clear()
            try:self.extractor.reset()
            except BaseException:pass
            raise
    def receipt(self):
        return dict(schema='h3-zoom-pcm-receipt-v1',input_epoch=self.epoch,**self.stats,
                    eof=self.eof,failed=self.failed,pending_samples=len(self.pending)//2,
                    raw_audio_fallback=False,speaker_identity_verified=False)
    def close(self):
        self.failed=True;self.stats['discarded_pending_samples']+=len(self.pending)//2;self.pending.clear()
        self.extractor.reset()


def transcribe_extracted(*,reader,source,asr,**asr_kwargs):
    """Explicit injection seam; NEVER opens capture/resumes Endpoint itself."""
    # Rebuild strict versioned contract, do not accept arbitrary origin labels.
    expected=source_contract(reference_sha256=source['reference_sha256'],capture_source_sha256=source['capture_source_sha256'])
    if source!=expected or reader.quantum!=source['capture_quantum_samples']:
        raise ValueError('source contract/reader profile drift')
    try:
        reader._check()
        result=asr(pcm=lambda:reader,completion_policy='input-finalized',**asr_kwargs)
        reader._check()
        if not reader.eof or reader.pending:raise ValueError('ASR returned before extraction EOF/drain')
        from research.h3_personal.endpoint_asr import finalized_covers
        if (result.get('final') is not True or result.get('errors')!=[] or result.get('cleanup_complete') is not True or
            result.get('input_committed') is not True or result.get('completion_policy')!='input-finalized' or
            not finalized_covers(result.get('finalized_audio_seconds'),result.get('audio_sent_seconds')) or
            abs(result['audio_sent_seconds']-reader.stats['submitted_samples']/16000)>1/16000):
            raise ValueError('native finalized coverage of extracted PCM required')
        return dict(result,origin=source['origin'],source_contract=source,extraction_receipt=reader.receipt())
    except BaseException:
        try:reader.close()
        except BaseException:pass
        raise
