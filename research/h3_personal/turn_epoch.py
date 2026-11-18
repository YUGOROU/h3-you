"""Bounded CPU-only turn admission guard; not a transport or native cancel API.

Call speech_onset() *before* caller-side cancellation/presentation effects. Its
returned epoch is already current; all previous epochs are rejected. Complete
those effects outside this object, then call finish_interrupt(). Every async
result must carry its epoch. is_current() is a point-in-time predicate, not a
lease permitting later unchecked IO: sinks must check at their own admission
boundary. commit() reserves a turn once, but does not execute an external side
effect or promise exactly-once delivery across processes.

No callbacks, IO, threads, queues, or per-turn history are owned here. Deadline
expiry is enforced on access (or explicit expire()), not by a background timer.
"""
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time


class TurnState(str, Enum):
    LISTENING = "LISTENING"
    CAPTURING = "CAPTURING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"
    DRAINING = "DRAINING"
    CLOSED = "CLOSED"


class TurnReason(str, Enum):
    EOS = "eos"
    INTERRUPT = "interrupt"
    FAILURE = "failure"
    STOP = "stop"
    DEADLINE = "deadline"


@dataclass(frozen=True)
class TurnSnapshot:
    epoch: int
    state: TurnState
    deadline: float
    committed: bool
    reason: TurnReason | None


class TurnEpoch:
    """Serialized state transitions with constant storage and finite admission.

    deadline is an absolute time.monotonic() deadline shared by all turns.
    max_turns bounds onset admission; reaching it rejects further onsets without
    invalidating the current turn. stop/expiry enter DRAINING; close acknowledges
    caller-owned cleanup. Neither operation stops any model or resource itself.
    """

    def __init__(self, *, deadline: float, max_turns: int = 10000):
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError("deadline must be finite and in the monotonic future")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        self._deadline = deadline
        self._max_turns = max_turns
        self._lock = threading.Lock()
        self._epoch = 0
        self._state = TurnState.LISTENING
        self._committed = False
        self._reason = None

    def _expire(self):
        # Only called under the lock. No user-supplied clock/callback is invoked.
        if (time.monotonic() >= self._deadline
                and self._state not in (TurnState.DRAINING, TurnState.CLOSED)):
            self._state = TurnState.DRAINING
            self._reason = TurnReason.DEADLINE

    def _matches(self, epoch):
        return (type(epoch) is int and epoch > 0 and epoch == self._epoch
                and self._state in (TurnState.CAPTURING, TurnState.THINKING,
                                    TurnState.SPEAKING))

    def snapshot(self) -> TurnSnapshot:
        with self._lock:
            self._expire()
            return TurnSnapshot(self._epoch, self._state, self._deadline,
                                self._committed, self._reason)

    def expire(self) -> bool:
        """Poll expiry; true iff the session ended because of its deadline."""
        return self.snapshot().reason is TurnReason.DEADLINE

    def is_current(self, epoch: int) -> bool:
        """Whether results for this epoch may currently be admitted."""
        with self._lock:
            self._expire()
            return self._matches(epoch)

    def speech_onset(self) -> int:
        """Synchronously invalidate old work, returning a new interrupt token.

        Concurrent calls are serialized as distinct onsets; only the latest token
        can finish interruption. Detector event deduplication belongs to caller.
        """
        with self._lock:
            self._expire()
            if self._state in (TurnState.DRAINING, TurnState.CLOSED):
                raise RuntimeError("session is stopping")
            if self._epoch >= self._max_turns:
                raise RuntimeError("turn limit reached")
            self._epoch += 1
            self._state = TurnState.INTERRUPTING
            self._committed = False
            self._reason = TurnReason.INTERRUPT
            return self._epoch

    def finish_interrupt(self, epoch: int, *, capture: bool = True) -> bool:
        """Acknowledge caller's invalidation; false onset can return to listening."""
        with self._lock:
            self._expire()
            if (type(epoch) is not int or epoch != self._epoch
                    or self._state is not TurnState.INTERRUPTING):
                return False
            self._state = TurnState.CAPTURING if capture else TurnState.LISTENING
            return True

    def commit(self, epoch: int) -> bool:
        """Reserve finalized input exactly once, transitioning to THINKING."""
        with self._lock:
            self._expire()
            if not self._matches(epoch) or self._state is not TurnState.CAPTURING:
                return False
            if self._committed:
                return False
            self._committed = True
            self._state = TurnState.THINKING
            return True

    def speaking(self, epoch: int) -> bool:
        with self._lock:
            self._expire()
            if not self._matches(epoch) or self._state is not TurnState.THINKING:
                return False
            self._state = TurnState.SPEAKING
            return True

    def finish(self, epoch: int, reason: TurnReason) -> bool:
        """EOS is speech completion; FAILURE abandons a turn, not the session."""
        if reason not in (TurnReason.EOS, TurnReason.FAILURE) or not isinstance(reason, TurnReason):
            raise ValueError("finish requires EOS or FAILURE; use onset/stop separately")
        with self._lock:
            self._expire()
            if not self._matches(epoch):
                return False
            if reason is TurnReason.EOS and self._state is not TurnState.SPEAKING:
                return False
            self._state = TurnState.LISTENING
            self._reason = reason
            return True

    def stop(self) -> bool:
        """Invalidate all turn results; request caller-owned cleanup once."""
        with self._lock:
            self._expire()
            if self._state in (TurnState.DRAINING, TurnState.CLOSED):
                return False
            self._state = TurnState.DRAINING
            self._reason = TurnReason.STOP
            return True

    def close(self) -> bool:
        """Acknowledge cleanup, without claiming native/transport termination."""
        with self._lock:
            self._expire()
            if self._state is not TurnState.DRAINING:
                return False
            self._state = TurnState.CLOSED
            return True
