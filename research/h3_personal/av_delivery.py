"""Finite, single-producer CPU AV delivery contract (no actual playback).

Serialized calls only: parent owns synchronization and the actual atomic AV sink.
Hashes bind caller-supplied video/profile evidence, not decoded video validation.
ACK is logical in-process ownership, never external crash-safe exactly-once.
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import re

from .conversation_state import SpeechSource


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("strict integer out of range")


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA256 required")


@dataclass(frozen=True)
class Binding:
    session_id: str
    source_id: str
    source: SpeechSource
    profile_sha256: str
    fps: Fraction = Fraction(24)
    source_frame: int = 0

    def __post_init__(self):
        if not all(type(s) is str and s for s in (self.session_id, self.source_id)):
            raise ValueError("explicit session/source IDs required")
        if not isinstance(self.source, SpeechSource):
            raise ValueError("original PCM source required")
        _digest(self.profile_sha256)
        if type(self.fps) is not Fraction or self.fps <= 0:
            raise ValueError("positive rational fps required")
        _integer(self.source_frame)

    def boundary(self, frame):
        _integer(frame, self.source_frame)
        return (Fraction(frame - self.source_frame) * self.source.sample_rate / self.fps).__floor__()


@dataclass(frozen=True)
class Segment:
    binding: Binding
    index: int
    frame_start: int
    frame_stop: int
    video_sha256: str
    committed: bool = True

    def __post_init__(self):
        if not isinstance(self.binding, Binding):
            raise ValueError("binding required")
        _integer(self.index)
        _integer(self.frame_start, self.binding.source_frame)
        _integer(self.frame_stop, self.frame_start + 1)
        _digest(self.video_sha256)
        if self.committed is not True:
            raise ValueError("only committed video can be delivered")

    @property
    def time_base(self):
        return 1 / self.binding.fps

    @property
    def video_pts(self):
        return self.frame_start * self.time_base

    @property
    def end_pts(self):
        return self.frame_stop * self.time_base

    @property
    def sample_start(self):
        return min(self.binding.source.samples, self.binding.boundary(self.frame_start))

    @property
    def sample_stop(self):
        return min(self.binding.source.samples, self.binding.boundary(self.frame_stop))

    @property
    def padding_samples(self):
        """Explicit transport silence, not original or conditioning PCM."""
        return self.binding.boundary(self.frame_stop) - self.binding.boundary(self.frame_start) - (self.sample_stop - self.sample_start)

    @property
    def audio_pts(self):
        # Quantized original PCM clock; differs from video by <1 sample.
        return self.binding.source_frame / self.binding.fps + Fraction(
            self.binding.boundary(self.frame_start), self.binding.source.sample_rate)


@dataclass(frozen=True)
class Lease:
    token: int
    segment: Segment
    original_pcm: bytes


class Backpressure(RuntimeError):
    pass


class Delivery:
    """One original PCM producer, <=2 outstanding segments INCLUDING leased one.

    enqueue is nonblocking; Backpressure changes nothing. close_producer freezes
    the final boundary and requires complete source coverage. final_flush returns
    False until every AV lease is ACKed. cancel/failure/uncertain ACK are terminal:
    neither stream's acknowledged cursor advances and nothing can be replayed.
    """
    def __init__(self, binding, original_pcm, *, max_segments=3, max_frames=119):
        if not isinstance(binding, Binding):
            raise ValueError("binding required")
        _integer(max_segments, 1)
        _integer(max_frames, 1)
        source = binding.source
        if type(original_pcm) is not bytes or len(original_pcm) != source.samples * source.channels * source.sample_width:
            raise ValueError("original PCM byte length mismatch")
        if sha256(original_pcm).hexdigest() != source.digest:
            raise ValueError("original PCM digest mismatch")
        self.binding = binding
        self._pcm = original_pcm
        self._max_segments = max_segments
        self._max_frames = max_frames
        self._queue = deque()
        self._lease = None
        self._next = 0
        self._tail = binding.source_frame
        self.acked_frame = binding.source_frame
        self.acked_sample = 0
        self._token = 0
        self.closed = False
        self.terminal = None

    @property
    def outstanding(self):
        return len(self._queue) + (self._lease is not None)

    def _active(self):
        if self.terminal is not None:
            raise RuntimeError("terminal delivery refuses work: " + self.terminal)

    def enqueue(self, segment):
        self._active()
        if self.closed:
            raise RuntimeError("producer closed")
        if not isinstance(segment, Segment) or segment.binding != self.binding:
            raise ValueError("session/source/profile binding mismatch")
        if segment.index != self._next or segment.frame_start != self._tail:
            raise ValueError("noncontiguous index/frame")
        if self._next >= self._max_segments or segment.frame_stop - segment.frame_start > self._max_frames:
            raise ValueError("finite producer bounds exceeded")
        if self.outstanding == 2:
            raise Backpressure("two AV segments already owned")
        self._queue.append(segment)
        self._next += 1
        self._tail = segment.frame_stop

    def lease(self):
        self._active()
        if self._lease is not None:
            raise RuntimeError("sink already owns an AV lease")
        if not self._queue:
            return None
        segment = self._queue.popleft()
        stride = self.binding.source.channels * self.binding.source.sample_width
        self._token += 1
        self._lease = Lease(self._token, segment, self._pcm[segment.sample_start * stride:segment.sample_stop * stride])
        return self._lease

    def ack(self, lease, *, certain=True):
        self._active()
        if type(certain) is not bool:
            raise ValueError("strict boolean certainty required")
        if lease is not self._lease or lease is None:
            raise ValueError("stale/foreign ACK")
        if not certain:
            self.cancel("ACK uncertain; replay refused")
            return
        self.acked_frame = lease.segment.frame_stop
        self.acked_sample = lease.segment.sample_stop
        self._lease = None

    def cancel(self, reason="cancelled"):
        self._active()
        if type(reason) is not str or not reason:
            raise ValueError("failure reason required")
        self.terminal = reason
        self._queue.clear()
        self._lease = None
        self._pcm = b""

    def sink_failed(self):
        self.cancel("sink failed; AV outcome may be uncertain")

    def close_producer(self):
        self._active()
        if self.closed:
            raise RuntimeError("already closed")
        if not self._next or self.binding.boundary(self._tail) < self.binding.source.samples:
            raise ValueError("final boundary does not cover complete original PCM")
        self.closed = True

    def final_flush(self):
        self._active()
        if not self.closed:
            raise RuntimeError("close producer before final flush")
        if self.outstanding:
            return False
        self.terminal = "flushed"
        self._pcm = b""
        return True


class FakeClock:
    def __init__(self):
        self.now = Fraction(0)

    def advance(self, seconds):
        if type(seconds) is not Fraction or seconds < 0:
            raise ValueError("nonnegative rational clock delta required")
        self.now += seconds


class HeadlessConsumer:
    """One shared AV clock, startup after first committed segment (target=1).

    poll starts a lease or ACKs it only after its entire rational AV duration.
    No audio/video independent progress. Underrun freezes media time until next
    segment; stall duration is counted, never hidden via skips/loops/rate changes.
    No sleeps, devices, payload decoding, or unbounded event trace.
    """
    def __init__(self, delivery, clock):
        self.delivery = delivery
        self.clock = clock
        self._last = clock.now
        self._lease = None
        self._deadline = None
        self._idle_since = None
        self.stall = Fraction(0)
        self.first_av_at = None

    def poll(self):
        now = self.clock.now
        if type(now) is not Fraction or now < self._last:
            raise ValueError("monotonic rational clock required")
        self._last = now
        self.delivery._active()
        if self._lease is not None:
            if now < self._deadline:
                return False
            self.delivery.ack(self._lease)
            self._lease = None
            self._idle_since = self._deadline
        lease = self.delivery.lease()
        if lease is None:
            return False
        if self.first_av_at is None:
            self.first_av_at = now
        elif self._idle_since is not None:
            self.stall += now - self._idle_since
        self._lease = lease
        self._deadline = now + lease.segment.end_pts - lease.segment.video_pts
        self._idle_since = None
        return True
