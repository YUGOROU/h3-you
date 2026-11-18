"""Ordered listening→speech→listening transitions for one finite scenario.

Driven by ACTUAL emitted segments and their rank receipts, never by intent.
The state machine refuses replayed sources, unsupported audio-guide resets and
handoffs without visual continuity evidence. Times are monotonic host clocks
and remain measurements of this process, not device playback timing.
"""
from dataclasses import dataclass
import time

from .conversation_state import (
    Action, Event, Handoff, ListeningPolicy, Phase, SpeechSource, State, Suffix, reduce,
)


class SequenceError(ValueError):
    pass


@dataclass(frozen=True)
class Transition:
    action: str
    phase: str
    frame: int
    emitted_end: int
    monotonic: float
    detail: dict


class ConversationSequence:
    """One finite sequence; call in the exact emitted-segment order."""
    def __init__(self, *, speech_digest, sample_rate, samples, channels=1,
                 sample_width=2, policy=ListeningPolicy.ENCODED_SILENCE,
                 fps=None, clock=time.monotonic):
        from fractions import Fraction
        if not callable(clock):
            raise ValueError('monotonic clock callable required')
        self._clock = clock
        self.source = SpeechSource(speech_digest, sample_rate, samples, channels, sample_width)
        self.state = State(policy, fps=Fraction(24) if fps is None else fps)
        self.transitions = []
        self._speech_requested = False
        self._speech_prepared = False

    @property
    def phase(self):
        return self.state.phase

    def _apply(self, event, detail=None, emitted_end=None):
        self.state, outputs = reduce(self.state, event)
        self.transitions.append(Transition(event.action.value, self.state.phase.value,
            self.state.frame, self.state.frame if emitted_end is None else emitted_end,
            self._clock(), dict(detail or {})))
        return outputs

    def listen(self, *, handoff_frame, visual_receipt, audio_guide_reset_receipt):
        """Enter listening with explicit previous-owner handoff evidence.

        handoff_frame is the EMITTED global boundary (F-5 where applicable), which
        is what this state machine's frame clock tracks.
        """
        if self.state.phase is Phase.READY:
            return self._apply(Event(Action.LISTEN), dict(initial=True), 0)
        handoff = Handoff(self.state.owner, self.state.listening_owner, handoff_frame,
                          visual_receipt, audio_guide_reset_receipt, True)
        return self._apply(Event(Action.LISTEN, handoff=handoff),
                           dict(handoff_frame=handoff_frame), handoff_frame)

    def request_speech(self):
        """PREPARING_SPEECH preempts further listening reservations."""
        if self._speech_requested:
            raise SequenceError('speech already requested')
        self._speech_requested = True
        return self._apply(Event(Action.PREPARE))

    def speech_prepared(self, *, handoff_frame, visual_receipt, audio_guide_reset_receipt):
        if not self._speech_requested or self._speech_prepared:
            raise SequenceError('preparation not requested')
        if (not isinstance(audio_guide_reset_receipt, str)
                or len(audio_guide_reset_receipt) != 64
                or any(c not in '0123456789abcdef' for c in audio_guide_reset_receipt)):
            raise SequenceError('new source audio reset receipt required')
        handoff = Handoff(self.state.owner, self.source.owner, handoff_frame,
                          visual_receipt, audio_guide_reset_receipt, True)
        self._speech_prepared = True
        return self._apply(Event(Action.PREPARED, source=self.source, handoff=handoff),
                           dict(handoff_frame=handoff_frame), handoff_frame)

    def emit(self, *, index, frames, emitted_end):
        """Reserve then commit exactly one actual emitted segment."""
        if frames <= 0:
            raise SequenceError('positive emitted frames required')
        token = str(index)
        if token != str(self.state.next_suffix):
            raise SequenceError('emitted segment order differs from reservation token')
        self._apply(Event(Action.RESERVE, token=token, frames=frames))
        outputs = self._apply(Event(Action.COMMIT, token=token), emitted_end=emitted_end)
        return outputs[0]

    def close(self):
        if self.state.phase is not Phase.LISTENING:
            raise SequenceError('final phase must be listening')
        return self._apply(Event(Action.CLOSE))

    def cancel(self, *, handoff_frame=None, visual_receipt=None,
               audio_guide_reset_receipt=None):
        """Drop only uncommitted suffixes; already emitted frames stay delivered."""
        if self.state.phase is Phase.SPEAKING:
            handoff = Handoff(self.state.owner, self.state.listening_owner, handoff_frame,
                              visual_receipt, audio_guide_reset_receipt, True)
            return self._apply(Event(Action.CANCEL, handoff=handoff))
        return self._apply(Event(Action.CANCEL))

    def receipt(self):
        return dict(schema='h3-conversation-sequence-v1',
                    policy=self.state.policy.value,
                    transitions=[dict(action=t.action, phase=t.phase, frame=t.frame,
                                      emitted_end=t.emitted_end,
                                      monotonic=t.monotonic, **t.detail)
                                 for t in self.transitions],
                    retired_sources=list(self.state.retired_sources),
                    speech_requested=self._speech_requested,
                    speech_prepared=self._speech_prepared,
                    final_phase=self.state.phase.value,
                    exactly_once='logical suffix assignment only; not crash-safe delivery')
