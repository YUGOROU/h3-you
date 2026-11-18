"""Bounded half-duplex capture eligibility; no device, browser or network ownership.

Attended half-duplex means the next input capture may start only after the
*previous* AI speech was actually presented by the caller's sink. Pi publication
success, relay ACK, transport receipt, worker completion or a synthetic VAD
boundary are NOT presentation evidence and never enable capture here.

Sink-level evidence must be an already-produced, same-process monotonic record
(versioned dict, see ``EVIDENCE_SCHEMA``). This module does not sample the sink,
poll a browser playhead, start a timer thread, queue rejected evidence or replay
a blocked/backlogged capture. Expiry is enforced on access, fail closed.

Constructing a gate is not capture permission: the caller must already hold the
explicit per-operation input approval. Publication for turn N authorises capture
for turn N+1 only after presentation-complete evidence for turn N is accepted.
"""
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time

EVIDENCE_SCHEMA = 'h3-sink-presentation-complete-v1'
EVIDENCE_FIELDS = frozenset(('schema', 'session', 'epoch', 'completed_at_monotonic',
                             'output_seconds'))


class GateState(str, Enum):
    CAPTURE_ELIGIBLE = 'CAPTURE_ELIGIBLE'
    CAPTURE_ACTIVE = 'CAPTURE_ACTIVE'
    WAITING_OUTPUT = 'WAITING_OUTPUT'
    OUTPUT_PRESENTED = 'OUTPUT_PRESENTED'
    BLOCKED = 'BLOCKED'
    STOPPED = 'STOPPED'


@dataclass(frozen=True)
class GateSnapshot:
    state: GateState
    capture_epoch: int
    output_epoch: int
    reason: str | None
    deadlines_left: float
    rejections: int


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class HalfDuplexCaptureGate:
    """Strict, fail-closed turn gate for one attended conversation.

    ``session`` is the opaque session identifier the sink evidence must carry;
    it is never logged or interpolated. ``deadline`` is an absolute
    ``time.monotonic()`` value shared by the whole conversation and must be
    finite and in the future. ``max_turns`` bounds the number of capture epochs
    that may ever be started.
    """

    def __init__(self, *, session, deadline, max_turns=2):
        if type(session) is not str or not session or len(session) > 128:
            raise ValueError('bounded session identifier required')
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline <= time.monotonic()):
            raise ValueError('finite monotonic deadline in the future required')
        if type(max_turns) is not int or isinstance(max_turns, bool) or not 1 <= max_turns <= 8:
            raise ValueError('bounded finite turn limit required')
        self._session = session
        self._deadline = float(deadline)
        self._max_turns = max_turns
        self._lock = threading.RLock()
        self._state = GateState.CAPTURE_ELIGIBLE
        self._capture_epoch = 0
        self._output_epoch = 0
        self._reason = None
        self._rejections = 0

    # -- internal helpers (call with the lock held) ------------------------
    def _expire(self):
        if (time.monotonic() >= self._deadline
                and self._state not in (GateState.BLOCKED, GateState.STOPPED)):
            self._state, self._reason = GateState.BLOCKED, 'deadline'

    def _block(self, reason):
        self._state, self._reason = GateState.BLOCKED, reason

    def _fail_closed(self, reason):
        # Ordering/authenticity violation: terminal, never queued or replayed.
        self._block(reason)
        raise ValueError('capture gate blocked: ' + reason)

    def _reject(self, message):
        # Structural violation only: no state change, reusable after caller fix.
        with self._lock:
            self._rejections += 1
        raise ValueError(message)

    # -- public API -------------------------------------------------------
    def note_publication(self, epoch):
        """Record that a Pi reply for ``epoch`` was published to the sink.

        Emission alone never makes a later capture eligible; presentation
        evidence for this same epoch is still required.
        """
        with self._lock:
            self._expire()
            if type(epoch) is not int or isinstance(epoch, bool) or epoch < 1:
                self._reject('bounded publication epoch required')
            if self._state in (GateState.BLOCKED, GateState.STOPPED):
                self._reject('publication after terminal state')
            if self._state is not GateState.CAPTURE_ACTIVE or epoch != self._capture_epoch:
                self._fail_closed('publication_order')
            self._output_epoch = epoch
            self._state = GateState.WAITING_OUTPUT

    def note_presentation_complete(self, evidence):
        """Accept caller-supplied sink evidence that output ``epoch`` finished."""
        if type(evidence) is not dict:
            self._reject('versioned sink presentation evidence required')
        if set(evidence) != EVIDENCE_FIELDS:
            self._reject('exact sink presentation evidence fields required')
        if evidence.get('schema') != EVIDENCE_SCHEMA:
            self._reject('versioned sink presentation evidence required')
        if evidence.get('session') != self._session:
            self._reject('sink evidence session mismatch')
        epoch = evidence.get('epoch')
        if type(epoch) is not int or isinstance(epoch, bool) or epoch < 1:
            self._reject('bounded sink evidence epoch required')
        completed = evidence.get('completed_at_monotonic')
        if not _finite(completed) or completed <= 0:
            self._reject('finite monotonic completion timestamp required')
        output_seconds = evidence.get('output_seconds')
        if not _finite(output_seconds) or output_seconds <= 0:
            self._reject('finite presented output duration required')
        with self._lock:
            self._expire()
            if self._state is GateState.STOPPED:
                self._fail_closed('post_cancel_completion')
            if self._state is GateState.BLOCKED:
                self._reject('completion after terminal state')
            if self._state is not GateState.WAITING_OUTPUT:
                self._fail_closed('unexpected_completion')
            if epoch != self._output_epoch:
                self._fail_closed('epoch_mismatch')
            if completed > time.monotonic():
                self._reject('completion timestamp in the future')
            if output_seconds > completed:
                self._reject('presented duration exceeds elapsed process time')
            self._state = GateState.OUTPUT_PRESENTED
            return True

    def request_capture(self, epoch):
        """Reserve the next capture epoch exactly once; False means not eligible.

        Eligible states are the initial constructor state (turn 1) and
        OUTPUT_PRESENTED (after accepted presentation evidence). Anything else,
        including BLOCKED/STOPPED/expired, refuses without queueing or replaying.
        """
        with self._lock:
            self._expire()
            if type(epoch) is not int or isinstance(epoch, bool) or epoch < 1:
                self._reject('bounded capture epoch required')
            if self._state not in (GateState.CAPTURE_ELIGIBLE, GateState.OUTPUT_PRESENTED):
                return False
            if epoch != self._capture_epoch + 1 or epoch > self._max_turns:
                return False
            self._capture_epoch = epoch
            self._state = GateState.CAPTURE_ACTIVE
            return True

    def cancel(self):
        """Terminal attended stop; no capture and no late completion accepted."""
        with self._lock:
            if self._state is GateState.STOPPED:
                return False
            self._state, self._reason = GateState.STOPPED, 'canceled'
            return True

    def snapshot(self):
        with self._lock:
            self._expire()
            return GateSnapshot(self._state, self._capture_epoch, self._output_epoch,
                                self._reason,
                                max(0.0, self._deadline - time.monotonic()),
                                self._rejections)
