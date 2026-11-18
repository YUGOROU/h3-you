"""Local-only VoxLock v2 FP32 stream; no downloads, devices, or routing.

Evidence (Apache-2.0, retain upstream NOTICE when redistributing weights):
https://github.com/arbazsiddiqui/voxlock/blob/07ffc2db7b21ebe46b1a75ef37df44aa4a4f4336/voxlock/core.py
https://github.com/arbazsiddiqui/voxlock/blob/07ffc2db7b21ebe46b1a75ef37df44aa4a4f4336/export_onnx.py
https://github.com/arbazsiddiqui/voxlock/blob/07ffc2db7b21ebe46b1a75ef37df44aa4a4f4336/NOTICE
HF revision a69f1607601789bf102202a1d7497b67c7cf7fb8, onnx/v2 only.
Upstream NOTICE attributes REAL-TSE weights/inference to Ke Zhang under
Apache-2.0. Enrollment includes ECAPA frame features, NOT a standalone vector:
48000 samples -> enroll_n, k, v. Chunk carries six LSTM h/c states.

Input/output: mono float32 16kHz, no implicit resampling/downmixing/AGC.
512-point periodic Hann, hop128 (125 frames/s), reflect256, squared-Hann
OLA normalization, upstream gain -0.5. First128 output at input512 (32ms);
thereafter lag384..511 samples (24..31.94ms), plus inference/caller batching.
No Mac realtime/quality acceptance implied. TSE cannot reliably distinguish
human speech from a clone of that same voice; use a separate echo/turn gate.
Actual graph shapes: enroll_n[257,376], k[32,2,64,298], v[32,2,298,64];
h/c[6,32,256], all float32. FP32 model files total98,812,749 bytes.
Optional runtime dependencies: numpy, onnxruntime (CPU). No torch/voxlock/HF
package needed. Callers must continuously resample their source before push.
One owner per stream, not thread-safe. reset discards pending audio explicitly;
flush is terminal/idempotent and preserves exact input length, including tiny
streams (upstream Stream.flush loses streams <=256 samples).
"""
from pathlib import Path
import hashlib

SOURCE_REVISION = '07ffc2db7b21ebe46b1a75ef37df44aa4a4f4336'
MODEL_REVISION = 'a69f1607601789bf102202a1d7497b67c7cf7fb8'
MODEL_FILES = {
    'enroll.onnx': (21023799, '81f305c9b7a32f13d1401f53bb4c6301826a607c9e807e60912f36ef1b9fae29'),
    'chunk.onnx': (77788950, '11093eda6001794ffa03cd018dc343b12848fc739a2f151e26530e0a66f9659a'),
}
SAMPLE_RATE = 16000
HOP = 128
MAX_PUSH = 4096


def load_cpu(model_dir, threads=1):
    """Hash-verified local FP32 v2 files; explicit CPU threads (integer1..4)."""
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('threads must be an integer in 1..4')
    import onnxruntime as ort
    paths = []
    for name, (size, sha) in MODEL_FILES.items():
        path = Path(model_dir) / name
        if path.stat().st_size != size:
            raise ValueError('wrong model size: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
        if h.hexdigest() != sha:
            raise ValueError('wrong model hash: ' + name)
        paths.append(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return tuple(ort.InferenceSession(str(p), options, providers=['CPUExecutionProvider']) for p in paths)


class VoxLockStream:
    """Inject real or mocked enrollment/chunk ONNX sessions. Reference exactly3s.

    Longer-reference selection is deliberately caller-owned (upstream chooses
    loudest3s within first30s); never silently select a speaker from a mixture.
    No reference is persisted. Failures poison stream until explicit reset.
    batch_frames=1 preserves the baseline; 1..16 bounds native temporal runs.
    Only already available frames are grouped (no extra waiting). Native FP32
    kernels can round differently by temporal length; not bitwise equivalence.
    Upstream export audit: input/mask norms reduce channels, TF-map/attention
    normalize frequencies/reference positions, time LSTMs carry h/c. Band
    communication reshapes time into B*K independent examples before its
    normalization/bidirectional band LSTM; it does not mix temporal frames.
    Actual FP32 graph has dynamic wav[1,S], frames[1,K,512]; no temporal
    normalization change is intended. Caller push quantum still controls the
    number of immediately available frames and adds its own acquisition delay.
    """
    def __init__(self, enroll, chunk, reference, *, sample_rate=16000, batch_frames=1):
        import numpy as np
        self.np = np
        if sample_rate != SAMPLE_RATE:
            raise ValueError('requires 16000Hz mono')
        if type(batch_frames) is not int or not 1 <= batch_frames <= 16:
            raise ValueError('batch_frames must be an integer in 1..16')
        self.batch_frames = batch_frames
        self.chunk = chunk
        if {i.name for i in chunk.get_inputs()} != {'wav', 'enroll_n', 'k', 'v', 'h', 'c'}:
            raise ValueError('requires v2 chunk interface')
        shapes = {i.name: tuple(i.shape) for i in chunk.get_inputs()}
        self.shape = shapes['h']
        if self.shape != (6, 32, 256) or shapes['c'] != self.shape:
            raise ValueError('unexpected LSTM state')
        ref = self._audio(reference, 48000)
        if len(ref) != 48000:
            raise ValueError('reference must be exactly 48000 samples')
        voice = enroll.run(['enroll_n', 'k', 'v'], {'enroll': ref[None]})
        expected = ((257, 376), (32, 2, 64, 298), (32, 2, 298, 64))
        if len(voice) != 3 or any(v.shape != sh or v.dtype != np.float32 or
                                       not np.isfinite(v).all() for v, sh in zip(voice, expected)):
            raise ValueError('invalid enrollment')
        self.voice = dict(zip(('enroll_n', 'k', 'v'), voice))
        self.win2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)).astype(np.float32) ** 2
        self.reset()

    def _audio(self, x, bound):
        np = self.np
        x = np.asarray(x)
        if x.ndim != 1 or len(x) > bound or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('requires bounded finite mono float32 samples')
        return x

    def reset(self):
        """New clock/state using same enrollment; explicitly abandon pending tail."""
        np = self.np
        self.h = np.zeros(self.shape, np.float32)
        self.c = np.zeros_like(self.h)
        self.buf = np.zeros(0, np.float32)
        self.ola = np.zeros(512, np.float32)
        self.env = np.zeros(512, np.float32)
        self.started = self.closed = self.failed = False
        self.drop = 256
        self.n_in = self.n_out = 0

    def _run(self):
        np = self.np
        out = []
        while len(self.buf) >= 512:
            count = min(self.batch_frames, 1 + (len(self.buf) - 512) // HOP)
            # Native graph performs STFT on this contiguous overlapping segment;
            # time is NOT an independent batch axis (LSTM state stays causal).
            segment = self.buf[:512 + (count - 1) * HOP]
            fr, h, c = self.chunk.run(['frames', 'h_out', 'c_out'],
                {'wav': segment[None], **self.voice, 'h': self.h, 'c': self.c})
            if fr.shape != (1, count, 512) or h.shape != self.shape or c.shape != self.shape or any(
                    v.dtype != np.float32 or not np.isfinite(v).all() for v in (fr, h, c)):
                raise ValueError('invalid chunk output')
            self.h, self.c = h, c
            for frame in fr[0]:
                self.ola += frame
                self.env += self.win2
                y = self.ola[:128] / np.where(self.env[:128] > 1e-11, self.env[:128], 1.0)
                cut = min(self.drop, 128)
                self.drop -= cut
                out.append(y[cut:] * -0.5)
                self.ola = np.concatenate((self.ola[128:], np.zeros(128, np.float32)))
                self.env = np.concatenate((self.env[128:], np.zeros(128, np.float32)))
            self.buf = self.buf[count * HOP:].copy()
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def push(self, audio):
        if self.closed or self.failed:
            raise RuntimeError('closed/failed stream; reset required')
        x = self._audio(audio, MAX_PUSH)
        np = self.np
        self.n_in += len(x)
        self.buf = np.concatenate((self.buf, x))
        if not self.started and len(self.buf) > 256:
            self.buf = np.concatenate((self.buf[1:257][::-1], self.buf))
            self.started = True
        try:
            y = self._run() if self.started else np.zeros(0, np.float32)
        except Exception:
            self.failed = True
            raise
        self.n_out += len(y)
        return y

    def flush(self):
        if self.failed:
            raise RuntimeError('failed stream; reset required')
        np = self.np
        if self.closed:
            return np.zeros(0, np.float32)
        self.closed = True
        if not self.n_in:
            return np.zeros(0, np.float32)
        # Tiny streams need both end reflections, including repeated reflection.
        if not self.started:
            self.buf = np.pad(self.buf, (256, 256), mode='reflect')
            self.started = True
        else:
            self.buf = np.concatenate((self.buf, self.buf[-257:-1][::-1]))
        try:
            y = self._run()
            tail = self.ola / np.where(self.env > 1e-11, self.env, 1.0) * -0.5
            y = np.concatenate((y, tail))[self.drop:][:self.n_in - self.n_out]
            if len(y) != self.n_in - self.n_out:
                raise ValueError('tail length mismatch')
        except Exception:
            self.failed = True
            raise
        self.drop = 0
        self.n_out += len(y)
        self.buf = np.zeros(0, np.float32)
        return y
