"""CPU-only resident lifecycle seam; actual LiveVoiceRuntime wiring is PENDING.

Adapters must enforce Deadline.check() at native blocking/cancellation boundaries.
This synchronous coordinator cannot interrupt a stuck callable: tick() fences new
admission, not native work. Its deadlines never replace native AV clocks, native
release acknowledgements, or provider watchdogs. No retries or replay occur.
"""
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Protocol


class State(str, Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    QUARANTINED = "quarantined"
    DRAINING = "draining"
    CLOSED = "closed"


class AdmissionError(RuntimeError):
    pass


class DeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class Deadline:
    expires_at: float
    clock: Callable[[], float]

    def check(self) -> None:
        if self.clock() >= self.expires_at:
            raise DeadlineExceeded("resident deadline expired")


@dataclass(frozen=True)
class CleanupAck:
    """Positive native evidence, not merely a successful Python return."""
    cleanup_complete: bool
    tts_reusable: bool


@dataclass(frozen=True)
class ReleaseAck:
    released: bool


class ResourceAdapter(Protocol):
    """start owns rollback on failure; cleanup/release must report native truth.

    Session work must not release resident models. cleanup must finish native
    session release (including decoder/PCM ownership) before acknowledging reuse.
    """
    def start(self, deadline: Deadline) -> Any: ...
    def cleanup(self, resources: Any, deadline: Deadline) -> CleanupAck: ...
    def release(self, resources: Any, deadline: Deadline) -> ReleaseAck: ...


class WarmConversationService:
    """One startup, serial bounded sessions, retained process-local resources.

    Construction starts fixed startup/service deadlines but performs no work.
    Idle is an absolute service-age cap (not a sliding last-request timer), so
    admissions cannot keep the resident alive indefinitely. Call tick from an
    external scheduler for idle expiry; no background threads are created here.
    close is explicit and never repeats an uncertain release. Quarantine is
    terminal for admission, even if a late native callback eventually succeeds.
    """
    def __init__(self, adapter: ResourceAdapter, *, startup_seconds: float,
                 session_seconds: float, service_seconds: float,
                 idle_seconds: float, clock: Callable[[], float] = time.monotonic):
        values = (startup_seconds, session_seconds, service_seconds, idle_seconds)
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("deadlines must be finite and positive")
        self._clock = clock
        self._adapter = adapter
        self._lock = threading.RLock()
        now = clock()
        self.service_deadline = now + service_seconds
        self.startup_deadline = min(now + startup_seconds, self.service_deadline)
        self.idle_deadline = min(now + idle_seconds, self.service_deadline)
        if not all(math.isfinite(v) for v in
                   (self.service_deadline, self.startup_deadline, self.idle_deadline)):
            raise ValueError("invalid clock/deadline")
        self._session_seconds = session_seconds
        self._state = State.STARTING
        self._started = False
        self._active = False
        self._resources = None
        self._has_resources = False
        self._release_attempted = False
        self._session_deadline = None

    @property
    def state(self) -> State:
        self.tick()
        with self._lock:
            return self._state

    def tick(self) -> State:
        with self._lock:
            now = self._clock()
            expired = now >= self.service_deadline or now >= self.idle_deadline
            expired |= self._state == State.STARTING and now >= self.startup_deadline
            expired |= (self._session_deadline is not None and
                        now >= self._session_deadline)
            if expired and self._state not in (State.CLOSED, State.QUARANTINED):
                self._state = State.QUARANTINED if self._active else State.DRAINING
            return self._state

    def _deadline(self, end: float) -> Deadline:
        return Deadline(min(end, self.service_deadline, self.idle_deadline), self._clock)

    def start(self) -> None:
        with self._lock:
            self.tick()
            if self._state != State.STARTING or self._started:
                raise AdmissionError("startup unavailable")
            self._started = self._active = True
            deadline = self._deadline(self.startup_deadline)
        try:
            deadline.check()
            resources = self._adapter.start(deadline)
            with self._lock:
                self._resources, self._has_resources = resources, True
            deadline.check()
            with self._lock:
                if self._state != State.STARTING:
                    raise AdmissionError("startup fenced")
                self._state = State.READY
        except BaseException:
            with self._lock:
                self._state = State.QUARANTINED
            raise
        finally:
            with self._lock:
                self._active = False

    def run_session(self, session: Callable[[Any, Deadline], Any]) -> Any:
        with self._lock:
            self.tick()
            if self._state != State.READY or self._active:
                raise AdmissionError("resident is not ready")
            self._state, self._active = State.BUSY, True
            deadline = self._deadline(self._clock() + self._session_seconds)
            self._session_deadline = deadline.expires_at
        try:
            deadline.check()
            result = session(self._resources, deadline)
            deadline.check()
            ack = self._adapter.cleanup(self._resources, deadline)
            deadline.check()
            if not (isinstance(ack, CleanupAck) and
                    ack.cleanup_complete is True and ack.tts_reusable is True):
                raise RuntimeError("native cleanup/reusable TTS not acknowledged")
            with self._lock:
                if self._state != State.BUSY:
                    raise AdmissionError("session fenced")
                self._state = State.READY
            return result
        except BaseException:
            with self._lock:
                self._state = State.QUARANTINED
            raise
        finally:
            with self._lock:
                self._active = False
                self._session_deadline = None

    def close(self) -> None:
        """Drain without interrupting active native work; call again once it exits.

        Release must fit the original service/idle lifetime, never a fresh grace
        period. Expired/uncertain release requires external native teardown; it
        cannot be represented as CLOSED by this protocol.
        """
        with self._lock:
            if self._state == State.CLOSED:
                return
            if self._active:
                if self._state != State.QUARANTINED:
                    self._state = State.DRAINING
                return
            if self._release_attempted:
                return
            self._release_attempted = True
            if not self._has_resources:
                self._state = State.QUARANTINED if self._started else State.CLOSED
                return
            self._active = True
            self._state = State.DRAINING
            deadline = self._deadline(self.service_deadline)
        try:
            deadline.check()
            ack = self._adapter.release(self._resources, deadline)
            deadline.check()
            if not (isinstance(ack, ReleaseAck) and ack.released is True):
                raise RuntimeError("native release not acknowledged")
            with self._lock:
                self._resources = None
                self._has_resources = False
                self._state = State.CLOSED
        except BaseException:
            with self._lock:
                self._state = State.QUARANTINED
            raise
        finally:
            with self._lock:
                self._active = False
