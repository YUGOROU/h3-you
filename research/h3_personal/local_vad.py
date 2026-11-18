"""Dependency-free acoustic VAD candidate for approved 16 kHz PCM16 windows.

The score is a heuristic speech-likelihood, NOT a calibrated model posterior or
speaker/human identity proof. Energy, zero crossings and periodicity distinguish
this from the legacy energy-only boundary. Live speech/echo quality is unaccepted.
No model download, capture, network, resampling or synthesized silence occurs.
"""
from array import array
from collections import deque
from dataclasses import dataclass
import math
import sys
import time


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 20
    onset_ms: int = 60
    minimum_speech_ms: int = 120
    quiet_ms: int = 500
    pre_roll_ms: int = 200
    no_speech_ms: int = 8000
    maximum_ms: int = 45000
    onset_score: float = .65
    offset_score: float = .35

    def __post_init__(self):
        durations = (self.frame_ms, self.onset_ms, self.minimum_speech_ms,
                     self.quiet_ms, self.pre_roll_ms, self.no_speech_ms, self.maximum_ms)
        if any(type(v) is not int or v <= 0 for v in durations):
            raise ValueError('positive integer VAD durations required')
        if (self.frame_ms != 20 or not 300 <= self.quiet_ms <= 800
                or not self.onset_ms <= self.minimum_speech_ms <= self.pre_roll_ms <= 500
                or not self.pre_roll_ms < self.no_speech_ms <= self.maximum_ms <= 45000
                or any(v % self.frame_ms for v in durations)):
            raise ValueError('bounded 20ms VAD profile required')
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                for v in (self.onset_score, self.offset_score))
                or not 0 < self.offset_score < self.onset_score < 1):
            raise ValueError('finite score hysteresis required')


def speech_score(pcm):
    """20ms (or an unpadded EOF remainder) little-endian signed PCM16."""
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 or len(pcm) > 640:
        raise ValueError('bounded aligned PCM16 frame required')
    samples = array('h'); samples.frombytes(pcm)
    if sys.byteorder != 'little':
        samples.byteswap()
    power = sum(x*x for x in samples)
    rms = math.sqrt(power / len(samples))
    energy = min(1., max(0., (rms - 80.) / 220.))
    if not energy or len(samples) < 2:
        return 0.
    crossings = sum((a < 0) != (b < 0) for a, b in zip(samples, samples[1:])) / (len(samples)-1)
    # Broad voiced/unvoiced range, rejecting DC and high-crossing broadband noise.
    crossing_score = min(1., crossings / .01) * min(1., max(0., (.48-crossings)/.20))
    periodicity = 0.
    for lag in (40, 64, 80, 100, 128, 160, 200):
        if lag >= len(samples):
            break
        a, b = samples[:-lag], samples[lag:]
        norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
        if norm:
            periodicity = max(periodicity, sum(x*y for x, y in zip(a, b)) / norm)
    return energy * crossing_score * (.30 + .70*periodicity)


class VADBoundary:
    """Single-use bounded stream, trimmed only before confirmed speech onset.

    Retains at most pre_roll_ms before speech; never replays blocked input.
    Every returned byte is an unchanged, ordered slice of captured PCM. EOF is
    an application boundary only: native ASR finalized coverage is still required.
    The caller must stop/close the underlying capture on return or cancellation.
    """
    policy = 'application-local-acoustic-vad-v1'

    def __init__(self, stream, *, config=None, scorer=speech_score, clock=time.monotonic):
        self.stream = stream
        self.config = VADConfig() if config is None else config
        if not isinstance(self.config, VADConfig) or not callable(scorer) or not callable(clock):
            raise ValueError('explicit VAD profile/scorer/clock required')
        self.scorer, self.clock = scorer, clock
        self.done = False; self.reason = None
        self.samples = self.sent_samples = 0
        self.speech_start = self.last_speech = self.vad_end = None
        self.speech_start_monotonic = self.last_speech_monotonic = self.vad_end_monotonic = None
        self._pre = deque(); self._pre_bytes = 0; self._output = bytearray()
        self._candidate_samples = self._quiet_samples = 0
        self._onset_sample = None; self._onset_time = None

    def _finish(self, reason):
        self.done = True; self.reason = reason
        self.vad_end = self.samples / 16000
        self.vad_end_monotonic = self.clock()
        self._pre.clear(); self._pre_bytes = 0

    def _next(self):
        c = self.config
        block = bytearray()
        while len(block) < 640:
            part = self.stream.read(640-len(block))
            if not isinstance(part, bytes) or len(part) > 640-len(block):
                raise ValueError('bounded binary capture reads required')
            if not part:
                break
            block.extend(part)
        if len(block) % 2:
            raise ValueError('PCM16 alignment required')
        if not block:
            self._finish('source_eof'); return
        pcm = bytes(block); count = len(pcm)//2
        start = self.samples; self.samples += count
        score = self.scorer(pcm)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('finite speech-likelihood score required')
        now = self.clock()
        if self.speech_start is None:
            self._pre.append(pcm); self._pre_bytes += len(pcm)
            # Confirmation consumes minimum_speech_ms; keep that interval in
            # addition to the promised pre-onset history, never an unbounded queue.
            while self._pre_bytes > (c.pre_roll_ms+c.minimum_speech_ms)*32:
                self._pre_bytes -= len(self._pre.popleft())
            threshold = (c.offset_score if self._candidate_samples >= c.onset_ms*16
                         else c.onset_score)
            if score >= threshold:
                if not self._candidate_samples:
                    self._onset_sample, self._onset_time = start, now
                self._candidate_samples += count
            else:
                self._candidate_samples = 0
            # The minimum-speech check rejects brief noises before any publication.
            if self._candidate_samples >= c.minimum_speech_ms*16:
                self.speech_start = self._onset_sample / 16000
                self.speech_start_monotonic = self._onset_time
                self.last_speech = self.samples / 16000
                self.last_speech_monotonic = now
                self._output.extend(b''.join(self._pre))
                self._pre.clear(); self._pre_bytes = 0
        else:
            self._output.extend(pcm)
            if score >= c.offset_score:
                self._quiet_samples = 0
                self.last_speech = self.samples / 16000
                self.last_speech_monotonic = now
            else:
                self._quiet_samples += count
            if self._quiet_samples >= c.quiet_ms*16:
                self._finish('application_vad_quiet')
        if not self.done and self.samples >= c.maximum_ms*16:
            self._finish('maximum_utterance')
        elif not self.done and self.speech_start is None and self.samples >= c.no_speech_ms*16:
            self._finish('no_speech_window')
        elif not self.done and len(pcm) < 640:
            self._finish('source_eof')

    def read(self, size):
        if type(size) is not int or not 2 <= size <= 65536 or size % 2:
            raise ValueError('bounded aligned read size required')
        while len(self._output) < size and not self.done:
            self._next()
        result = bytes(self._output[:size]); del self._output[:size]
        self.sent_samples += len(result)//2
        return result

    def evidence(self):
        return dict(policy=self.policy, reason=self.reason, sample_rate=16000,
                    captured_samples=self.samples, sent_samples=self.sent_samples,
                    speech_start=self.speech_start, last_speech=self.last_speech,
                    vad_end=self.vad_end, speech_start_monotonic=self.speech_start_monotonic,
                    last_speech_monotonic=self.last_speech_monotonic,
                    vad_end_monotonic=self.vad_end_monotonic,
                    clock='local_monotonic', speaker_identity_verified=False)
