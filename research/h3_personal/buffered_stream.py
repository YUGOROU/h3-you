"""Buffered playback A: bounded, original-TTS A/V queue, NOT causal streaming.

No server/GPU/OBS/Discord is started. Consumer must honor epoch invalidation at
frame/audio boundaries: data already emitted cannot be recalled by this queue.
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import threading
from .component_cache import sha256


@dataclass(frozen=True)
class Turn:
    epoch: int


@dataclass(frozen=True)
class Clip:
    video: str
    original_tts: str
    video_sha256: str
    audio_sha256: str


@dataclass(frozen=True)
class Playback:
    turn: Turn
    sequence: int
    clip: Clip
    pts: Fraction
    duration: Fraction
    mute_video_audio: bool = True


def _asset(root, name, expected):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts:
        raise ValueError('relative contained media path required')
    root = Path(root).resolve(strict=True)
    path = root / p
    if any((root / Path(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts)+1)):
        raise ValueError('media symlinks refused')
    if not path.resolve().is_relative_to(root) or not path.is_file() or sha256(path) != expected:
        raise ValueError('media containment/bytes differ')
    return path


def validate_clip(root, clip):
    from .e2e import probe
    video = probe(_asset(root, clip.video, clip.video_sha256))
    audio = probe(_asset(root, clip.original_tts, clip.audio_sha256))
    vs = [s for s in video['streams'] if s['codec_type'] == 'video']
    aus = [s for s in audio['streams'] if s['codec_type'] == 'audio']
    if len(vs) != 1 or len(aus) != 1:
        raise ValueError('exactly one video/original TTS stream required')
    # Play original TTS separately, not H3-generated audio embedded in the video.
    for s in (vs[0], aus[0]):
        if abs(Fraction(s.get('start_time', '0'))) > Fraction(1, 24):
            raise ValueError('normalize A/V PTS before buffering')
    v, a = Fraction(vs[0]['duration']), Fraction(aus[0]['duration'])
    if v <= 0 or a <= 0 or a > v + Fraction(1, 24):
        raise ValueError('video must cover original TTS; never truncate speech')
    # Preserve the complete TTS tail when frame quantization differs by <=1frame.
    # Consumer holds the final video frame; schedule the next clip after BOTH EOFs.
    return max(v, a)


class BufferedStream:
    def __init__(self, root, *, max_clips=2, max_seconds=15, validator=validate_clip):
        if type(max_clips) is not int or max_clips < 1 or Fraction(max_seconds) <= 0:
            raise ValueError('positive buffer limits required')
        self.root = Path(root).resolve(strict=True)
        self.max_clips = max_clips; self.max_seconds = Fraction(max_seconds)
        self.validator = validator; self._lock = threading.RLock()
        self._epoch = 0; self._active = None; self._pending = deque()
        self._inflight = None; self._pts = Fraction(0); self._sequence = 0
        self._done = False; self.state = 'LISTENING'

    def current(self, turn):
        with self._lock:
            return isinstance(turn, Turn) and self._active is not None and self._active == turn

    def begin(self, *, source='assistant'):
        with self._lock:
            if source != 'assistant':
                raise ValueError('peer audio must never drive avatar lip sync')
            if self._active is not None:
                raise RuntimeError('finish or interrupt existing turn first')
            self._epoch += 1; self._active = Turn(self._epoch)
            self._pts = Fraction(0); self._sequence = 0; self._done = False
            self.state = 'GENERATING'
            return self._active

    def has_capacity(self, duration):
        with self._lock:
            items = list(self._pending) + ([self._inflight] if self._inflight else [])
            return (len(items) < self.max_clips and
                    sum((p.duration for p in items), Fraction(0)) + Fraction(duration) <= self.max_seconds)

    def enqueue(self, turn, clip):
        # Expensive probe/hash outside lock; interruption invalidates the result.
        if not self.current(turn):
            return False
        duration = self.validator(self.root, clip)
        if not isinstance(duration, Fraction) or duration <= 0:
            raise ValueError('validated positive rational duration required')
        with self._lock:
            if not self.current(turn):
                return False
            if self._done:
                raise RuntimeError('producer already completed')
            items = list(self._pending) + ([self._inflight] if self._inflight else [])
            if len(items) >= self.max_clips or sum((p.duration for p in items), Fraction(0)) + duration > self.max_seconds:
                raise BufferError('playback backpressure: wait for acknowledgment')
            self._sequence += 1
            self._pending.append(Playback(turn, self._sequence, clip, self._pts, duration))
            self._pts += duration
            return True

    def take(self):
        with self._lock:
            if self._inflight is not None or not self._pending:
                return None
            self._inflight = self._pending.popleft(); self.state = 'SPEAKING'
            return self._inflight

    def acknowledge(self, item):
        with self._lock:
            if not self.current(item.turn) or self._inflight != item:
                return False
            self._inflight = None
            self._settle()
            return True

    def finish(self, turn):
        with self._lock:
            if not self.current(turn):
                return False
            self._done = True; self._settle(); return True

    def _settle(self):
        if self._inflight is None and not self._pending:
            self.state = 'LISTENING' if self._done else 'GENERATING'
            if self._done:
                self._active = None

    def interrupt(self):
        with self._lock:
            self._epoch += 1; self._active = None
            self._pending.clear(); self._inflight = None
            self.state = 'INTERRUPTED'
