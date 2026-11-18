"""Conversation-span AV delivery: multi-source PCM timeline into ONE sink clock.

Single original-PCM producer, <=2 outstanding AV segments INCLUDING the leased
one, and one continuous FFmpeg encoder clock across every source boundary. The
audio bytes for a segment are original source samples plus explicitly typed
listening/tailpadding silence; conditioning preroll and model-generated audio
are never sent. ACK stays a logical in-process pipe acceptance, never external
crash-safe delivery or device playback.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import threading
import time

from .conversation_timeline import (
    AudioSlice, ConversationBinding, ConversationSegment, MAX_FRAMES,
)
from .headless_av_sink import HeadlessAVSink


def _integer(value, minimum=0, maximum=MAX_FRAMES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('strict bounded integer required')


def _digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('SHA256 required')


@dataclass(frozen=True, eq=False)
class ConversationAVSegment:
    """Committed conversation frames plus their immutable video evidence."""
    timeline: ConversationSegment
    video_sha256: str
    conversation_final: bool = False

    def __post_init__(self):
        if type(self.timeline) is not ConversationSegment:
            raise ValueError('timeline conversation segment required')
        _digest(self.video_sha256)
        if type(self.conversation_final) is not bool:
            raise ValueError('explicit boolean final flag required')

    @property
    def binding(self):
        return self.timeline.binding

    @property
    def index(self):
        return self.timeline.index

    @property
    def frame_start(self):
        return self.timeline.frame_start

    @property
    def frame_stop(self):
        return self.timeline.frame_stop

    @property
    def sample_start(self):
        return self.binding.boundary(self.frame_start)

    @property
    def sample_stop(self):
        return self.binding.boundary(self.frame_stop)

    @property
    def padding_samples(self):
        """Always zero: the timeline already declares explicit silence spans."""
        return 0

    @property
    def time_base(self):
        return 1 / self.binding.fps

    @property
    def video_pts(self):
        return self.frame_start * self.time_base

    @property
    def end_pts(self):
        return self.frame_stop * self.time_base


@dataclass(frozen=True)
class ConversationLease:
    token: int
    segment: ConversationAVSegment
    audio_pcm: bytes = field(repr=False)
    slices: tuple = field(repr=False)


class Backpressure(RuntimeError):
    pass


class ConversationDelivery:
    """One original-PCM producer over every conversation span.

    enqueue is nonblocking and derives the segment audio exactly once; a failed
    admission never advances the PCM cursor. cancel/failure/uncertain ACK are
    terminal: acknowledged cursors never advance and nothing is replayed.
    """
    def __init__(self, binding, *, max_segments=240, max_frames=119):
        if type(binding) is not ConversationBinding:
            raise ValueError('ConversationBinding required')
        _integer(max_segments, 1)
        _integer(max_frames, 1)
        if max_frames > 119:
            raise ValueError('decoder segment bound is 119 frames')
        self.binding = binding
        self._stride = binding.channels * 2
        self._producer = binding.producer()
        self._max_segments = max_segments
        self._max_frames = max_frames
        self._pending = []
        self._lease = None
        self._next = 0
        self._tail = binding.spans[0].frame_start
        self._token = 0
        self.acked_frame = binding.spans[0].frame_start
        self.acked_sample = 0
        self.closed = False
        self.terminal = None

    @property
    def outstanding(self):
        return len(self._pending) + (self._lease is not None)

    @property
    def tail_frame(self):
        """Last admitted conversation frame; coverage proof for finish()."""
        return self._tail

    def _active(self):
        if self.terminal is not None:
            raise RuntimeError('terminal delivery refuses work: ' + self.terminal)

    def enqueue(self, segment):
        self._active()
        if self.closed:
            raise RuntimeError('producer closed')
        if type(segment) is not ConversationAVSegment or segment.binding is not self.binding:
            raise ValueError('conversation binding mismatch')
        if segment.index != self._next or segment.frame_start != self._tail:
            raise ValueError('noncontiguous index/frame')
        if (self._next >= self._max_segments
                or segment.frame_stop - segment.frame_start > self._max_frames):
            raise ValueError('finite producer bounds exceeded')
        if (segment.conversation_final
                and segment.frame_stop != self.binding.spans[-1].frame_stop):
            raise ValueError('conversation final must end the conversation')
        if self.outstanding == 2:
            raise Backpressure('two AV segments already owned')
        slices = self._producer.emit(segment.timeline)
        audio = b''.join(s.pcm for s in slices)
        expected = (segment.sample_stop - segment.sample_start) * self._stride
        if len(audio) != expected:
            raise ValueError('derived conversation PCM range mismatch')
        self._pending.append((segment, audio, slices))
        self._next += 1
        self._tail = segment.frame_stop

    def lease(self):
        self._active()
        if self._lease is not None:
            raise RuntimeError('sink already owns an AV lease')
        if not self._pending:
            return None
        segment, audio, slices = self._pending.pop(0)
        self._token += 1
        self._lease = ConversationLease(self._token, segment, audio, slices)
        return self._lease

    def ack(self, lease, *, certain=True):
        self._active()
        if type(certain) is not bool:
            raise ValueError('strict boolean certainty required')
        if lease is None or lease is not self._lease:
            raise ValueError('stale/foreign ACK')
        if not certain:
            self.cancel('ACK uncertain; replay refused')
            return
        self.acked_frame = lease.segment.frame_stop
        self.acked_sample = lease.segment.sample_stop
        self._lease = None

    def cancel(self, reason='cancelled'):
        self._active()
        if type(reason) is not str or not reason:
            raise ValueError('failure reason required')
        self.terminal = reason
        self._pending.clear()
        self._lease = None

    def sink_failed(self):
        self.cancel('sink failed; AV outcome may be uncertain')

    def close_producer(self):
        self._active()
        if self.closed:
            raise RuntimeError('already closed')
        if not self._next or self._tail != self.binding.spans[-1].frame_stop:
            raise ValueError('complete conversation coverage required')
        self._producer.finish()
        self.closed = True

    def final_flush(self):
        self._active()
        if not self.closed:
            raise RuntimeError('close producer before final flush')
        if self.outstanding:
            return False
        self.terminal = 'flushed'
        return True


class ConversationAVStream:
    """Producer-side callbacks into one continuous headless sink, two-deep.

    accept() returns only after both pipes accepted the segment. The consumer
    thread owns leasing/ACK. Failure is terminal for both streams and never
    partially published.
    """
    def __init__(self, root, binding, *, width, height, timeout=60, paced=False,
                 sink_factory=HeadlessAVSink):
        if type(paced) is not bool:
            raise ValueError('explicit boolean pacing required')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('finite timeout <=300 required')
        if type(binding) is not ConversationBinding:
            raise ValueError('ConversationBinding required')
        self.root = Path(root).resolve(strict=True)
        self.binding = binding
        self.width, self.height = width, height
        self.paced = paced
        self.deadline = time.monotonic() + timeout
        self.timeout = timeout
        self.delivery = ConversationDelivery(binding)
        self.sink = sink_factory(self.root, binding, width=width, height=height,
                                 max_frames=119, max_total_frames=binding.spans[-1].frame_stop,
                                 timeout=timeout, paced=paced)
        self._condition = threading.Condition()
        self._paths = {}
        self._failure = []
        self._stopped = False
        self._first = None
        self._submitted = False
        self.highwater = 0
        self.ranges = []
        self.timing = []
        self._worker = threading.Thread(target=self._consume, name='h3-conversation-av')
        self._worker.start()

    def remaining(self):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('conversation AV deadline')
        return value

    def _check(self):
        self.remaining()
        if self._failure:
            raise RuntimeError('conversation AV consumer failed') from self._failure[0]
        if self._stopped:
            raise RuntimeError('conversation AV cancelled')

    def _consume(self):
        try:
            while True:
                with self._condition:
                    self._check()
                    lease = self.delivery.lease()
                    while lease is None:
                        if self.delivery.closed or self.delivery.terminal is not None:
                            return
                        self._condition.wait(self.remaining())
                        self._check()
                        lease = self.delivery.lease()
                    relative = self._paths[lease.segment.index]
                self.sink.timeout = self.remaining()
                submission = self.sink.accept(relative, lease)
                with self._condition:
                    self._check()
                    self.delivery.ack(lease)  # never retry an uncertain ACK
                    self._paths.pop(lease.segment.index)
                    segment = lease.segment
                    self.ranges.append(dict(index=segment.index,
                        frame_start=segment.frame_start, frame_stop=segment.frame_stop,
                        sample_start=segment.sample_start, sample_stop=segment.sample_stop,
                        slices=[dict(kind=s.kind, sample_start=s.sample_start,
                                     sample_stop=s.sample_stop, source_id=s.source_id,
                                     source_start=s.source_start, source_stop=s.source_stop)
                                for s in lease.slices]))
                    self.timing.append(dict(accepted_monotonic=submission.accepted_monotonic,
                                            frame_stop=submission.frame_stop))
                    if self._first is None:
                        self._first = submission.first_av_submission_monotonic
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._failure.append(exc)
                if self.delivery.terminal is None:
                    self.delivery.sink_failed()
                self._condition.notify_all()

    def accept(self, event):
        """Committed-segment callback; blocks only until the 2-deep bound frees."""
        with self._condition:
            self._check()
            if type(event) is not dict:
                raise ValueError('typed committed callback required')
            for key in ('index', 'frame_count', 'frame_start', 'frame_stop',
                        'time_base_numerator', 'time_base_denominator', 'sha256',
                        'conversation_final'):
                if key not in event:
                    raise ValueError('complete callback fields required')
            start, stop = event['frame_start'], event['frame_stop']
            _integer(start)
            _integer(stop, start + 1)
            if (event['frame_count'] != stop - start or event['time_base_numerator'] != 1
                    or event['time_base_denominator'] != 24
                    or event.get('video_only') is not True
                    or event.get('committed', True) is not True):
                raise ValueError('invalid committed segment boundary')
            video = Path(event['path'])
            if video.is_symlink():
                raise ValueError('callback video symlink')
            try:
                relative = video.resolve(strict=True).relative_to(self.root)
            except ValueError as exc:
                raise ValueError('callback must be contained in the private root') from exc
            if not relative.parts or len(relative.parts) > 2 or '..' in relative.parts:
                raise ValueError('callback must be directly contained in the private root')
            if video.stat().st_size > 128 * 1024 * 1024:
                raise ValueError('video byte bound exceeded')
            self.remaining()
            digest = sha256()
            with video.open('rb') as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    self.remaining()
            if digest.hexdigest() != event['sha256']:
                raise ValueError('callback SHA mismatch')
            segment = ConversationAVSegment(
                ConversationSegment(self.binding, event['index'], start, stop),
                event['sha256'], event['conversation_final'])
            while self.delivery.outstanding >= 2:
                self._condition.wait(self.remaining())
                self._check()
            self.delivery.enqueue(segment)
            self._paths[segment.index] = relative
            self.highwater = max(self.highwater, self.delivery.outstanding)
            self._condition.notify_all()
            self._submitted = True

    def finish(self, *, conversation_final):
        """Require an explicit final callback, then flush both streams once."""
        if type(conversation_final) is not bool or conversation_final is not True:
            raise ValueError('explicit conversation final required')
        with self._condition:
            self._check()
            if not self._submitted:
                raise ValueError('no committed conversation segment')
            if self.delivery.tail_frame != self.binding.spans[-1].frame_stop:
                raise ValueError('conversation not fully delivered')
            self.delivery.close_producer()
            self._condition.notify_all()
        self._worker.join(self.remaining())
        with self._condition:
            self._check()
            if self._worker.is_alive() or not self.delivery.final_flush():
                raise TimeoutError('consumer did not flush')
        self.sink.timeout = self.remaining()
        completed = self.sink.finish()
        self.remaining()
        return completed

    @property
    def first_av_submission_monotonic(self):
        return self._first

    def receipt(self):
        return dict(
            conversation_id=self.binding.conversation_id,
            source_ids=[s.source_id for s in self.binding.sources],
            spans=[dict(frame_start=s.frame_start, frame_stop=s.frame_stop, kind=s.kind,
                        source_id=s.source_id) for s in self.binding.spans],
            frames=self.binding.spans[-1].frame_stop,
            sample_rate=self.binding.sample_rate, channels=self.binding.channels,
            sample_clock='original PCM floor(frame*rate/fps)',
            ranges=self.ranges, timing=self.timing,
            bounded_queue_highwater=self.highwater, audio_producers=1,
            classification='headless-read-paced' if self.paced else 'headless-only-unpaced',
            externalDurability=False)

    def cancel(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        try:
            self.delivery.cancel('cancelled by owner')
        except RuntimeError:
            pass
        self.sink.cancel()
        if self._worker is not threading.current_thread():
            self._worker.join(timeout=max(1, min(self.timeout, 5)))
