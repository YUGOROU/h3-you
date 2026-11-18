"""CPU-only serialized contract, not a model or transport implementation.

Call reduce once per event, retaining its returned state before delivering effects.
Exactly-once means logical suffix/sample assignment, not crash-safe sink delivery.
Receipts are caller attestations; this module cannot verify model continuity.
"""
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
import re


class Phase(str, Enum):
    READY = "ready"
    LISTENING = "listening"
    PREPARING_SPEECH = "preparing_speech"
    SPEAKING = "speaking"
    FAULTED = "faulted"
    CLOSED = "closed"


class ListeningPolicy(str, Enum):
    OMIT_CONDITION = "omit_condition"
    ENCODED_SILENCE = "encoded_silence"
    ZERO_LATENT = "zero_latent"
    JOINT_GENERATED_DISCARDED = "joint_generated_discarded"


class Action(str, Enum):
    LISTEN = "listen"
    PREPARE = "prepare"
    PREPARED = "prepared"
    PREPARATION_FAILED = "preparation_failed"
    RESERVE = "reserve"
    COMMIT = "commit"
    CANCEL = "cancel"
    INFERENCE_FAILED = "inference_failed"
    CLOSE = "close"


def _positive(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class SpeechSource:
    digest: str
    sample_rate: int
    samples: int
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self):
        if not isinstance(self.digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ValueError("source requires SHA256 of original PCM")
        if not all(_positive(v) for v in (
            self.sample_rate, self.samples, self.channels, self.sample_width
        )):
            raise ValueError("invalid PCM metadata")
        if self.channels not in (1, 2) or self.sample_width not in (1, 2, 3, 4):
            raise ValueError("unsupported PCM format")

    @property
    def owner(self):
        return "pcm:" + self.digest


@dataclass(frozen=True)
class Handoff:
    previous_owner: str
    next_owner: str
    frame: int
    visual_receipt: str
    audio_guide_reset_receipt: str
    audio_guide_reset_supported: bool = False

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (
            self.previous_owner, self.next_owner
        )):
            raise ValueError("nonempty handoff owners required")
        if type(self.frame) is not int or self.frame < 0:
            raise ValueError("nonnegative handoff frame required")
        if not all(isinstance(v, str) for v in (
            self.visual_receipt, self.audio_guide_reset_receipt
        )):
            raise ValueError("handoff receipts must be strings")
        if type(self.audio_guide_reset_supported) is not bool:
            raise ValueError("audio guide reset support must be boolean")


@dataclass(frozen=True)
class Suffix:
    token: str
    owner: str
    start: int
    frames: int


@dataclass(frozen=True)
class Event:
    action: Action
    source: SpeechSource | None = None
    handoff: Handoff | None = None
    token: str = ""
    frames: int = 0

    def __post_init__(self):
        if not isinstance(self.action, Action):
            raise ValueError("explicit action required")
        if not isinstance(self.token, str):
            raise ValueError("token must be a string")
        if type(self.frames) is not int or self.frames < 0:
            raise ValueError("nonnegative event frames required")
        if self.source is not None and not isinstance(self.source, SpeechSource):
            raise ValueError("invalid speech source")
        if self.handoff is not None and not isinstance(self.handoff, Handoff):
            raise ValueError("invalid handoff")


@dataclass(frozen=True)
class Output:
    suffix: Suffix
    video_pts: Fraction
    audio_pts: Fraction | None
    source: SpeechSource | None
    sample_start: int
    sample_stop: int
    producer: str = "original-source-pcm"


@dataclass(frozen=True)
class State:
    # Listening policy is deliberately mandatory; no equivalence is assumed.
    policy: ListeningPolicy
    fps: Fraction = Fraction(24)
    max_pending: int = 2
    max_window_frames: int = 119
    phase: Phase = Phase.READY
    frame: int = 0
    next_suffix: int = 0
    owner: str = ""
    source: SpeechSource | None = None
    source_frame: int = 0
    sample_cursor: int = 0
    pending: tuple[Suffix, ...] = ()
    retired_sources: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.policy, ListeningPolicy):
            raise ValueError("explicit listening policy required")
        if not isinstance(self.fps, Fraction) or self.fps <= 0:
            raise ValueError("positive rational fps required")
        if type(self.max_pending) is not int or not 1 <= self.max_pending <= 2:
            raise ValueError("max_pending must be 1 or 2")
        if not _positive(self.max_window_frames):
            raise ValueError("finite positive window required")
        if not isinstance(self.phase, Phase):
            raise ValueError("explicit phase required")
        if any(type(v) is not int or v < 0 for v in (
            self.frame, self.next_suffix, self.source_frame, self.sample_cursor
        )):
            raise ValueError("nonnegative integer clocks required")
        if not isinstance(self.owner, str):
            raise ValueError("owner must be a string")
        if self.source is not None and not isinstance(self.source, SpeechSource):
            raise ValueError("invalid speech source")
        if (not isinstance(self.pending, tuple)
                or len(self.pending) > self.max_pending
                or any(not isinstance(s, Suffix) for s in self.pending)):
            raise ValueError("bounded tuple of pending suffixes required")
        if (not isinstance(self.retired_sources, tuple)
                or any(not isinstance(s, str)
                       or not re.fullmatch(r"pcm:[0-9a-f]{64}", s)
                       for s in self.retired_sources)):
            raise ValueError("tuple of retired PCM owners required")

    @property
    def listening_owner(self):
        return "listening:" + self.policy.value


def _switch(state, owner, receipt):
    if state.pending:
        raise ValueError("drain or cancel pending suffixes before handoff")
    if state.owner and owner != state.owner:
        if (receipt is None or receipt.previous_owner != state.owner
                or receipt.next_owner != owner or receipt.frame != state.frame
                or not receipt.visual_receipt
                or not receipt.audio_guide_reset_receipt
                or receipt.audio_guide_reset_supported is not True):
            raise ValueError("new visual continuity and supported audio guide reset required")


def reduce(state: State, event: Event) -> tuple[State, tuple[Output, ...]]:
    """Pure, ordered reducer. Rejected events leave the input unchanged.

    PREPARE preempts further listening reservations; already reserved work must
    drain or be explicitly cancelled. Cancellation drops only uncommitted work.
    No event resets a faulted model. No generated or padding audio is emitted.
    """
    a = event.action
    if a == Action.CLOSE:
        return replace(state, phase=Phase.CLOSED, pending=()), ()
    if state.phase in (Phase.FAULTED, Phase.CLOSED):
        raise ValueError("terminal state refuses work")
    if a == Action.INFERENCE_FAILED:
        return replace(state, phase=Phase.FAULTED, pending=()), ()
    if a == Action.LISTEN:
        if state.phase not in (Phase.READY, Phase.SPEAKING):
            raise ValueError("cannot enter listening")
        if state.source and state.sample_cursor != state.source.samples:
            raise ValueError("speech incomplete; cancel explicitly")
        _switch(state, state.listening_owner, event.handoff)
        return replace(state, phase=Phase.LISTENING, owner=state.listening_owner,
                       source=None, sample_cursor=0), ()
    if a == Action.PREPARE and state.phase == Phase.LISTENING:
        return replace(state, phase=Phase.PREPARING_SPEECH), ()
    if a == Action.PREPARATION_FAILED and state.phase == Phase.PREPARING_SPEECH:
        return replace(state, phase=Phase.LISTENING), ()
    if a == Action.PREPARED and state.phase == Phase.PREPARING_SPEECH:
        source = event.source
        if source is None or source.owner in state.retired_sources:
            raise ValueError("new immutable source required; replay refused")
        _switch(state, source.owner, event.handoff)
        return replace(state, phase=Phase.SPEAKING, owner=source.owner,
                       source=source, source_frame=state.frame, sample_cursor=0,
                       retired_sources=state.retired_sources + (source.owner,)), ()
    if a == Action.CANCEL:
        if state.phase == Phase.SPEAKING:
            clean = replace(state, pending=())
            _switch(clean, state.listening_owner, event.handoff)
            return replace(clean, phase=Phase.LISTENING, owner=state.listening_owner,
                           source=None, sample_cursor=0), ()
        if state.phase in (Phase.LISTENING, Phase.PREPARING_SPEECH):
            return replace(state, phase=Phase.LISTENING, pending=()), ()
    if a == Action.RESERVE and state.phase in (Phase.LISTENING, Phase.SPEAKING):
        if len(state.pending) >= state.max_pending:
            raise ValueError("backpressure")
        start = state.frame + sum(s.frames for s in state.pending)
        # Monotonic reservation tokens also reject late cancelled completions.
        if event.token != str(state.next_suffix) or not _positive(event.frames):
            raise ValueError("invalid suffix token/length")
        if event.frames > state.max_window_frames:
            raise ValueError("window exceeds finite bound")
        suffix = Suffix(event.token, state.owner, start, event.frames)
        return replace(state, pending=state.pending + (suffix,),
                       next_suffix=state.next_suffix + 1), ()
    if a == Action.COMMIT and state.pending:
        suffix = state.pending[0]
        if event.token != suffix.token:
            raise ValueError("commit must be ordered and exactly once")
        end = state.frame + suffix.frames
        stop = state.sample_cursor
        audio_pts = None
        if state.source:
            rate = state.source.sample_rate
            stop = min(state.source.samples,
                       int(Fraction(end - state.source_frame) * rate / state.fps))
            audio_pts = Fraction(state.source_frame, 1) / state.fps + Fraction(state.sample_cursor, rate)
        output = Output(suffix, Fraction(state.frame, 1) / state.fps,
                        audio_pts, state.source, state.sample_cursor, stop)
        return replace(state, frame=end, sample_cursor=stop,
                       pending=state.pending[1:]), (output,)
    raise ValueError("invalid transition")
