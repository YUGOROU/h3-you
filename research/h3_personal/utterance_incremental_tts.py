"""Experimental pull-based sentence production; no provider activation.

Single owner serializes feed/produce/ack. Cancellation may also be signalled by
is_current during a blocking callback. Native calls cannot be preempted here;
late results are discarded. No audio transformation is performed.
"""
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import tempfile
import time
import wave

from .director_irodori import SOURCE, MODEL, CODEC
from .incremental_speech import IncrementalText, text_hash

RUNTIME_SHA256 = 'c07cb7d083000c2251ccaa5074d84232d2ac629742b1a9d725ae7a2c9c6f65e0'
WATERMARK_SHA256 = '08c168d596656a66bda985b49b7b1b294f29cf03f57e7e286d263be3fd1327bc'
MAX_BYTES = 30 * 48000 * 2  # finite envelope, not an instruction to resample


@dataclass(frozen=True)
class SentenceAudio:
    pcm: bytes = field(repr=False)
    sample_rate: int
    channels: int
    sample_width: int
    watermark_applied: bool
    source_revision: str = SOURCE
    model_revision: str = MODEL
    codec_revision: str = CODEC
    watermark_detected: bool | None = None


@dataclass(frozen=True)
class SentencePCM:
    epoch: int
    index: int
    text_sha256: str
    audio: SentenceAudio = field(repr=False)
    production: str = 'sentence_incremental'


class ResidentSentenceAdapter:
    """Audit bridge to existing IrodoriResident.synthesize, not a new provider.

    That adapter observes successful encode_wav with IRDTS payload, rejects
    unavailable watermark, and binds the accepted three revisions. Detection
    is intentionally NOT required. Caller injects the already provisioned
    resident method; fake methods are suitable only for protocol tests.
    Temporary complete-sentence WAV is private and removed on every exit.
    """
    def __init__(self, synthesize):
        self.synthesize = synthesize

    def __call__(self, sentence, *, is_current, max_bytes):
        with tempfile.TemporaryDirectory(prefix='h3-sentence-') as directory:
            path = Path(directory) / 'sentence.wav'
            receipt = self.synthesize(sentence, path, is_current=is_current)
            if (receipt.get('watermark_applied') is not True
                    or tuple(receipt.get(k) for k in ('source_revision', 'model_revision', 'codec_revision'))
                    != (SOURCE, MODEL, CODEC)
                    or receipt.get('text_sha256') != text_hash(sentence)
                    or receipt.get('path') != str(path)):
                raise ValueError('unwatermarked or unbound native sentence')
            # Bound file read too, including a finite WAV header allowance.
            if path.stat().st_size > max_bytes + 65536:
                raise BufferError('sentence WAV capacity')
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get('sha256'):
                raise ValueError('sentence WAV digest mismatch')
            with wave.open(str(path), 'rb') as wav:
                rate, channels, width, frames = (wav.getframerate(), wav.getnchannels(),
                                                wav.getsampwidth(), wav.getnframes())
                if (channels != 1 or width != 2 or not 0 < rate <= 48000
                        or not 0 < frames <= 30 * rate or frames * width > max_bytes
                        or (receipt.get('samples'), receipt.get('sample_rate'), receipt.get('channels'))
                        != (frames, rate, channels)):
                    raise ValueError('unsupported or oversized native PCM; never convert')
                pcm = wav.readframes(frames)
                if len(pcm) != frames * width:
                    raise ValueError('truncated native WAV')
            if not is_current():
                raise RuntimeError('stale native sentence')
            return SentenceAudio(pcm, rate, channels, width, True,
                                 watermark_detected=receipt.get('watermark_detected'))


class UtteranceIncrementalTTS:
    """Stable human reply -> bounded Japanese sentence -> complete marked PCM.

    Feed TextRevision events (origin human_reply) through the existing immutable
    prefix validator. Never feed tool arguments/JSON or partial ASR. produce()
    performs exactly one callback, returns before any later sentence synthesis,
    and requires explicit ack before accepting more text or producing again.
    One outstanding output; no internal PCM queue/history. Aggregate counters
    persist after ack. Errors terminally cancel; already published audio cannot
    be recalled if a later reply exceeds the aggregate contract.
    """
    def __init__(self, epoch, synthesize, *, deadline, is_current=lambda: True,
                 clock=time.monotonic, max_bytes=MAX_BYTES):
        if not math.isfinite(deadline) or deadline <= clock():
            raise ValueError('finite future deadline required')
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError('finite PCM byte capacity required')
        self.reply = IncrementalText(epoch, channel='reply', max_chars=160)
        self.synthesize, self.deadline = synthesize, deadline
        self.is_current, self.clock, self.max_bytes = is_current, clock, max_bytes
        self.offset = self.count = self.total_bytes = self.total_frames = 0
        self.format = None
        self.pending = None
        self.busy = self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self.pending = None
        self.reply.cancel()

    def _current(self):
        return not self.cancelled and self.clock() < self.deadline and self.is_current()

    def _check(self):
        if not self._current():
            self.cancel()
            raise RuntimeError('cancelled, stale epoch or deadline')

    def feed(self, event):
        self._check()
        if self.busy or self.pending is not None:
            raise BufferError('ack outstanding sentence before feed')
        try:
            stable = self.reply.accept(event)
            # Fail closed for structured/tool syntax even if mislabelled human_reply.
            if any(c in self.reply.text for c in '{}[]`'):
                raise ValueError('structured/tool text is not speech')
            # Count all committed sentence boundaries before permitting production.
            if sum(c in '。！？!?' for c in stable) > 4:
                raise ValueError('at most four sentences')
        except Exception:
            self.cancel()
            raise

    def _sentence(self):
        stable = self.reply.text[:self.reply.stable_stop]
        for i in range(self.offset, len(stable)):
            if stable[i] in '。！？!?':
                return stable[self.offset:i+1]
        if self.reply.closed:
            return stable[self.offset:]
        return ''

    @property
    def done(self):
        return (not self.cancelled and self.reply.closed and self.pending is None
                and self.offset == len(self.reply.text))

    def produce(self):
        self._check()
        if self.busy or self.pending is not None:
            raise BufferError('one outstanding sentence; explicit ack required')
        sentence = self._sentence()
        if not sentence:
            return None
        try:
            if self.count >= 4 or not sentence.strip():
                raise ValueError('sentence count/empty boundary')
            self.busy = True
            audio = self.synthesize(sentence, is_current=self._current,
                                    max_bytes=self.max_bytes-self.total_bytes)
            self._check()
            if (type(audio) is not SentenceAudio or audio.watermark_applied is not True
                    or (audio.source_revision, audio.model_revision, audio.codec_revision)
                    != (SOURCE, MODEL, CODEC)):
                raise ValueError('mandatory audited watermark and pins')
            if (type(audio.pcm) is not bytes or audio.channels != 1 or audio.sample_width != 2
                    or type(audio.sample_rate) is not int or not 0 < audio.sample_rate <= 48000
                    or not audio.pcm or len(audio.pcm) % 2):
                raise ValueError('invalid original PCM')
            fmt = (audio.sample_rate, audio.channels, audio.sample_width)
            frames = len(audio.pcm)//2
            if self.format is not None and fmt != self.format:
                raise ValueError('format drift; no resampling')
            if (self.total_bytes + len(audio.pcm) > self.max_bytes
                    or self.total_frames + frames > 30 * audio.sample_rate):
                raise BufferError('aggregate PCM capacity; never truncate')
            self.format = fmt
            self.total_bytes += len(audio.pcm)
            self.total_frames += frames
            self.offset += len(sentence)
            self.count += 1
            self.pending = SentencePCM(self.reply.epoch, self.count, text_hash(sentence), audio)
            return self.pending
        except Exception:
            self.cancel()
            raise
        finally:
            self.busy = False

    def ack(self, output):
        self._check()
        if self.pending is None or output is not self.pending:
            raise ValueError('exact outstanding output required')
        self.pending = None
