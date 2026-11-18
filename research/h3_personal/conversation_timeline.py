"""Finite CPU conversation PCM timeline; no device, decoder or model access.

Frame boundaries use the GLOBAL rational clock, never a per-source rounded
clock. Conditioning preroll is deliberately absent from this transport model.
Create one producer per binding; segments must cover the declared window once.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import re


MAX_PCM_BYTES = 64 * 1024 * 1024
MAX_SOURCES = 4
MAX_FRAMES = 100000


def _int(value, minimum=0, maximum=MAX_FRAMES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('bounded strict integer required')


def _id(value):
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError('bounded explicit ID required')


@dataclass(frozen=True)
class FrozenPCMSource:
    source_id: str
    pcm: bytes = field(repr=False)
    sample_rate: int
    channels: int
    sha256: str
    sample_width: int = 2
    format: str = 'S16LE'

    def __post_init__(self):
        _id(self.source_id)
        _int(self.sample_rate, 1, 192000)
        _int(self.channels, 1, 2)
        if self.sample_width != 2 or type(self.sample_width) is not int or self.format != 'S16LE':
            raise ValueError('only original S16LE PCM accepted')
        if (type(self.pcm) is not bytes or not 0 < len(self.pcm) <= MAX_PCM_BYTES
                or len(self.pcm) % (2 * self.channels)):
            raise ValueError('bounded complete immutable sample frames required')
        if (type(self.sha256) is not str or not re.fullmatch('[0-9a-f]{64}', self.sha256)
                or sha256(self.pcm).hexdigest() != self.sha256):
            raise ValueError('original source bytes hash mismatch')

    @property
    def samples(self):
        return len(self.pcm) // (2 * self.channels)


@dataclass(frozen=True)
class TimelineSpan:
    frame_start: int
    frame_stop: int
    kind: str
    source_id: str | None = None

    def __post_init__(self):
        _int(self.frame_start)
        _int(self.frame_stop, self.frame_start + 1)
        if self.kind not in ('listening', 'speech', 'tailpadding'):
            raise ValueError('explicit listening/speech/tailpadding required')
        if self.kind == 'speech':
            _id(self.source_id)
        elif self.source_id is not None:
            raise ValueError('silence has no original source')


@dataclass(frozen=True)
class AudioSlice:
    kind: str
    sample_start: int
    sample_stop: int
    pcm: bytes = field(repr=False)
    source_id: str | None = None
    source_start: int | None = None
    source_stop: int | None = None


@dataclass(frozen=True)
class ConversationBinding:
    conversation_id: str
    sources: tuple[FrozenPCMSource, ...]
    spans: tuple[TimelineSpan, ...]
    sample_rate: int
    channels: int
    fps: Fraction = Fraction(24)
    max_sources: int = MAX_SOURCES
    _claimed: list = field(default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self):
        _id(self.conversation_id)
        _int(self.sample_rate, 1, 192000)
        _int(self.channels, 1, 2)
        _int(self.max_sources, 1, MAX_SOURCES)
        if type(self.fps) is not Fraction or self.fps <= 0:
            raise ValueError('positive rational fps required')
        if (type(self.sources) is not tuple or len(self.sources) > self.max_sources
                or any(type(s) is not FrozenPCMSource for s in self.sources)):
            raise ValueError('finite immutable sources required')
        if (type(self.spans) is not tuple or not self.spans or len(self.spans) > 128
                or any(type(s) is not TimelineSpan for s in self.spans)):
            raise ValueError('finite immutable spans required')
        ids = [s.source_id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate source ID')
        if any((s.sample_rate, s.channels) != (self.sample_rate, self.channels) for s in self.sources):
            raise ValueError('format drift: no fallback or resampling')
        if sum(len(s.pcm) for s in self.sources) > MAX_PCM_BYTES:
            raise ValueError('aggregate PCM bound exceeded')
        cursor = self.spans[0].frame_start
        used = []
        for span in self.spans:
            if span.frame_start != cursor:
                raise ValueError('timeline must be contiguous and ordered')
            cursor = span.frame_stop
            if span.kind == 'speech':
                if span.source_id not in ids or span.source_id in used:
                    raise ValueError('each original source must occur exactly once')
                used.append(span.source_id)
                source = self.sources[ids.index(span.source_id)]
                if source.samples > self.boundary(span.frame_stop) - self.boundary(span.frame_start):
                    raise ValueError('full original source must fit declared window; no truncation')
        if set(used) != set(ids):
            raise ValueError('all original sources require full coverage')
        if self.boundary(cursor) - self.boundary(self.spans[0].frame_start) > MAX_PCM_BYTES // (2*self.channels):
            raise ValueError('finite output byte bound exceeded')

    def boundary(self, frame):
        _int(frame)
        return (frame * self.sample_rate * self.fps.denominator) // self.fps.numerator

    def producer(self):
        if self._claimed:
            raise RuntimeError('one original PCM producer per binding')
        self._claimed.append(True)
        return _PCMProducer(self)


@dataclass(frozen=True)
class ConversationSegment:
    binding: ConversationBinding
    index: int
    frame_start: int
    frame_stop: int

    def __post_init__(self):
        if type(self.binding) is not ConversationBinding:
            raise ValueError('conversation binding required, not AV delivery Binding')
        _int(self.index)
        _int(self.frame_start, self.binding.spans[0].frame_start)
        _int(self.frame_stop, self.frame_start + 1, self.binding.spans[-1].frame_stop)
        if self.frame_stop - self.frame_start > 119:
            raise ValueError('bounded decoder segment required')


class _PCMProducer:
    """Serialized calls only; failed admission does not advance the cursor."""
    def __init__(self, binding):
        self.binding = binding
        self.frame = binding.spans[0].frame_start
        self.index = 0
        self.closed = False

    def emit(self, segment):
        b = self.binding
        if (type(segment) is not ConversationSegment or segment.binding is not b
                or self.closed or segment.index != self.index or segment.frame_start != self.frame):
            raise ValueError('foreign, replayed or out-of-order segment')
        slices = []
        sources = {s.source_id: s for s in b.sources}
        for span in b.spans:
            lo, hi = max(segment.frame_start, span.frame_start), min(segment.frame_stop, span.frame_stop)
            if lo >= hi:
                continue
            start, stop = b.boundary(lo), b.boundary(hi)
            if span.kind == 'speech':
                source = sources[span.source_id]
                onset = b.boundary(span.frame_start)
                a, z = min(start-onset, source.samples), min(stop-onset, source.samples)
                if z > a:
                    slices.append(AudioSlice('speech', start, start+z-a,
                        source.pcm[a*2*b.channels:z*2*b.channels], source.source_id, a, z))
                start += z-a
                kind = 'tailpadding'
            else:
                kind = span.kind
            if start < stop:
                slices.append(AudioSlice(kind, start, stop, bytes((stop-start)*2*b.channels)))
        self.frame = segment.frame_stop
        self.index += 1
        return tuple(slices)

    def finish(self):
        if self.closed or self.frame != self.binding.spans[-1].frame_stop:
            raise ValueError('full conversation coverage required')
        self.closed = True
