"""Finite, ordered CPU encode/publication worker; never runs producer/GPU code.

Call submit on the producing thread *after* producing a segment. Its backpressure
is real elapsed time, not subtracted from latency. Callbacks must cooperate with
cancel_event (including terminating subprocesses); Python cannot kill threads.
A join timeout is an explicit failure, never a successful flush. No retries.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, get_ident
from time import monotonic
from typing import Any, Callable
import math


class PipelineCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentTiming:
    segment_id: int
    head_trim_frames: int
    newly_emitted_frames: int
    pts_start: int
    pts_end: int
    time_base: tuple[int, int] = (1, 24)


def segment_timing(segment_id: int) -> SegmentTiming:
    """Half-open integer PTS, after removal of the continuation's shared head."""
    if type(segment_id) is not int or segment_id < 0:
        raise ValueError("segment_id must be a nonnegative integer")
    start = 0 if segment_id == 0 else 124 + (segment_id - 1) * 123
    count = 124 if segment_id == 0 else 123
    return SegmentTiming(segment_id, int(segment_id != 0), count, start, start + count)


class SegmentPipeline:
    """One active job plus at most one waiting job, with ordered reports.

    encode(job, cancel_event) returns a dict. Optional publish(dict, event)
    returns a dict after publication is complete. Without publish, encode must
    perform publication too. Timestamps are monotonic seconds on the supplied
    clock, including queue waits. Only the constructing thread may submit/close;
    cancel may be called from another thread. Reports are snapshots.
    """

    def __init__(self, encode: Callable[[Any, Event], dict], *,
                 publish: Callable[[dict, Event], dict] | None = None,
                 clock: Callable[[], float] = monotonic,
                 join_timeout: float = 30.0,
                 timing_factory: Callable[[int], SegmentTiming] = segment_timing):
        if not math.isfinite(join_timeout) or join_timeout <= 0:
            raise ValueError("join_timeout must be finite and positive")
        self.cancel_event = Event()
        self._queue: Queue = Queue(maxsize=1)
        self._done = Event()
        self._lock = Lock()
        self._owner = get_ident()
        self._encode, self._publish, self._clock = encode, publish, clock
        self._timeout = join_timeout
        self._timing_factory = timing_factory
        self.started_at = clock()
        self._reports: list[dict] = []
        self._failure: BaseException | None = None
        self._closed = False
        self._next = 0
        self._blocked = 0.0
        self._thread = Thread(target=self._run, name="h3-segment-publication", daemon=True)
        self._thread.start()

    def _check_owner(self):
        if get_ident() != self._owner:
            raise RuntimeError("submit/close must run on the producer thread")

    def _check(self):
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure
        if self.cancel_event.is_set():
            raise PipelineCancelled("segment pipeline cancelled")

    @property
    def reports(self) -> list[dict]:
        with self._lock:
            return [dict(report) for report in self._reports]

    @property
    def producer_blocked_seconds(self) -> float:
        return self._blocked

    @property
    def first_publish_at(self) -> float | None:
        reports = self.reports
        return reports[0]["publication_finished_at"] if reports else None

    @property
    def first_publish_seconds(self) -> float | None:
        timestamp = self.first_publish_at
        return None if timestamp is None else timestamp - self.started_at

    def submit(self, job: Any, *, segment_id: int | None = None) -> SegmentTiming:
        self._check_owner()
        self._check()
        if self._closed:
            raise RuntimeError("pipeline closed")
        index = self._next if segment_id is None else segment_id
        timing = self._timing_factory(index)
        if not isinstance(timing, SegmentTiming) or timing.segment_id != index:
            raise ValueError('segment timing factory returned wrong identity')
        if index != self._next:
            raise ValueError("segment IDs must be sequential starting at zero")
        submitted = self._clock()
        try:
            while True:
                self._check()
                try:
                    self._queue.put((job, timing, submitted), timeout=0.02)
                    break
                except Full:
                    pass
        finally:
            self._blocked += self._clock() - submitted
        self._next += 1
        self._check()
        return timing

    def cancel(self):
        self.cancel_event.set()
        self._done.set()

    def close(self) -> list[dict]:
        self._check_owner()
        self._closed = True
        self._done.set()
        self._thread.join(self._timeout)
        if self._thread.is_alive():
            self.cancel()
            raise TimeoutError("segment callback did not finish within bounded join")
        self._check()
        return self.reports

    def _run(self):
        try:
            while not self.cancel_event.is_set():
                try:
                    job, timing, submitted = self._queue.get(timeout=0.02)
                except Empty:
                    if self._done.is_set():
                        return
                    continue
                self._check()
                encode_started = self._clock()
                result = self._encode(job, self.cancel_event)
                encode_finished = self._clock()
                self._check()
                if not isinstance(result, dict):
                    raise TypeError("encode callback must return dict")
                if self._publish is not None:
                    result = self._publish(result, self.cancel_event)
                    if not isinstance(result, dict):
                        raise TypeError("publish callback must return dict")
                published = self._clock()
                self._check()
                report = dict(result)
                report.update(segment_id=timing.segment_id, timing=timing,
                              submitted_at=submitted, encode_started_at=encode_started,
                              encode_finished_at=encode_finished,
                              publication_finished_at=published,
                              publication_elapsed_seconds=published - self.started_at)
                with self._lock:
                    self._reports.append(report)
        except BaseException as exc:
            with self._lock:
                self._failure = exc
            self.cancel_event.set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.cancel()
            try:
                self.close()
            except PipelineCancelled:
                pass
        else:
            self.close()
        return False
