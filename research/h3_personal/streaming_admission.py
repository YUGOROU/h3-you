"""Finite CPU scheduling only; not wired to the native owner/network/devices.

Drain one reserved idle window if PCM arrives during it. Continue just-in-time
idle while a speech window is not ready; never insert silence inside a speaking
source on starvation. Native commit and decoder ACK are separate boundaries.
"""
from dataclasses import dataclass
import math
import threading
import time

from .continuation_plan import Window
from .streaming_pcm import PCMIngress, _int, _id, _sha
from .streaming_windows import ResamplingContext, inspect_audio_window, WindowAudioOffer


@dataclass(frozen=True)
class AdmissionDecision:
    status: str
    role: str
    window: Window | None
    audio: WindowAudioOffer | None = None
    conversation_final: bool = False
    reason: str = ''


@dataclass(frozen=True)
class AdmissionTicket:
    sequence: int
    decision: AdmissionDecision


@dataclass(frozen=True)
class NativeCommitEvidence:
    committed_frame: int
    conditioning_prefix_sha256: str | None
    receipt_sha256: str


class StreamingAdmission:
    def __init__(self, conversation_id, *, context, max_windows=12,
                 max_idle_windows=6, max_turns=2, timeout=120,
                 clock=time.monotonic, extended_endurance=False, long_session=False):
        _id(conversation_id)
        if type(context) is not ResamplingContext:
            raise ValueError('explicit resampling context required')
        if type(long_session) is not bool or (long_session and extended_endurance):
            raise ValueError('explicit mutually exclusive long session required')
        _int(max_windows,3,2542 if long_session else 144)
        _int(max_idle_windows,1,max_windows-1)
        _int(max_turns,1,1000 if long_session else 16)
        if type(extended_endurance) is not bool:
            raise ValueError('explicit boolean endurance flag required')
        if (type(timeout) not in (int,float) or not math.isfinite(timeout)
                or not 0 < timeout <= (10800 if long_session else 720 if extended_endurance else 300) or not callable(clock)):
            raise ValueError('finite admission deadline required')
        now=clock()
        if type(now) not in (int,float) or not math.isfinite(now):
            raise ValueError('finite clock required')
        self.conversation_id=conversation_id; self.context=context
        self._clock=clock; self._last_clock=now; self._deadline=now+timeout
        self._max_windows=max_windows; self._max_idle=max_idle_windows; self._max_turns=max_turns
        self._long_session=long_session
        self._lock=threading.RLock(); self._frame=0; self._emitted=0
        self._sequence=0; self._idle=0; self._turns=0
        self._stream=None; self._onset=None; self._speech_index=0; self._speech_final=False
        self._active=None; self._committed=False; self._closing=False
        self._state='listening'; self._epochs=set(); self._sources=set()
        self._voice=None; self._format=None
        self._interrupt_epoch=None; self._interrupted_turns=0

    def _live(self):
        if self._state in ('failed','closed'):
            raise RuntimeError('terminal admission controller')
        now=self._clock()
        if (type(now) not in (int,float) or not math.isfinite(now)
                or now < self._last_clock or now >= self._deadline):
            self._state='failed'
            raise TimeoutError('admission clock/deadline rejected')
        self._last_clock=now

    def attach(self, stream):
        with self._lock:
            self._live()
            if (type(stream) is not PCMIngress or self._stream is not None
                    or self._closing or self._state != 'listening'):
                raise RuntimeError('one waiting speech source in listening only')
            b=stream.binding
            fmt=(b.sample_rate,b.channels,b.sample_width,b.format)
            if (b.conversation_id != self.conversation_id
                    or (self._epochs and b.epoch <= max(self._epochs))
                    or b.source_id in self._sources or len(self._epochs) >= self._max_turns
                    or (self._voice is not None and self._voice != b.voice_binding_sha256)
                    or (self._format is not None and self._format != fmt)):
                raise ValueError('stale/foreign/over-budget speech source')
            stream.snapshot()  # refuse cancelled/expired ingress before binding
            self.context.context(b.sample_rate)  # no unsupported-rate admission
            self._stream=stream; self._epochs.add(b.epoch); self._sources.add(b.source_id)
            self._voice=b.voice_binding_sha256; self._format=fmt

    def inspect(self):
        with self._lock:
            self._live()
            if self._active is not None:
                return AdmissionDecision('busy',self._active.decision.role,None,
                                         reason='reserved-window-must-drain')
            if self._interrupt_epoch is not None:
                return AdmissionDecision('interrupt','speech',None,
                                         reason='speech-interruption-awaiting-retirement')
            if self._sequence >= self._max_windows:
                return AdmissionDecision('limit','listening',None,reason='window-limit')
            window=Window(0,124,0) if not self._frame else Window(self._frame-22,self._frame+102,22)
            if self._closing:
                return AdmissionDecision('ready','listening',window,conversation_final=True,
                                         reason='explicit-final-listening')
            if self._stream is not None and self._frame:
                onset=self._frame if self._onset is None else self._onset
                audio=inspect_audio_window(self._stream,onset_frame=onset,
                                          index=self._speech_index,context=self.context,
                                          long_session=self._long_session)
                if audio.status=='end':
                    return AdmissionDecision('end','speech',None,audio,reason='explicit-source-end')
                if audio.status=='ready':
                    if self._sequence >= self._max_windows-1:
                        return AdmissionDecision('limit','speech',None,audio,reason='final-listening-slot-reserved')
                    return AdmissionDecision('ready','speech',audio.window,audio)
                if self._onset is not None:
                    return AdmissionDecision('wait','speech',None,audio,reason='speech-starvation-not-silence')
            if self._idle >= self._max_idle or self._sequence >= self._max_windows-1:
                return AdmissionDecision('limit','listening',None,reason='finite-idle-limit')
            return AdmissionDecision('ready','listening',window,reason='intentional-idle-silence')

    def reserve(self, decision):
        with self._lock:
            current=self.inspect()
            if type(decision) is not AdmissionDecision or decision.status!='ready' or current!=decision:
                raise ValueError('stale or non-ready admission')
            ticket=AdmissionTicket(self._sequence,current)
            self._active=ticket; self._committed=False
            return ticket

    def commit(self, ticket, evidence):
        """Publish only after real backend vote/commit; cannot undo publication."""
        with self._lock:
            self._live()
            if ticket is not self._active or self._committed:
                raise ValueError('stale/duplicate admission commit')
            d=ticket.decision
            try:
                if type(evidence) is not NativeCommitEvidence:
                    raise ValueError('native commit evidence required')
                _sha(evidence.receipt_sha256)
                expected=None if d.audio is None else d.audio.prefix_sha256
                if (type(evidence.committed_frame) is not int
                        or evidence.committed_frame != d.window.end
                        or evidence.conditioning_prefix_sha256 != expected):
                    raise ValueError('native output/window/conditioning binding drift')
                if self._stream is not None and d.audio is not None:
                    if self._stream.prefix_digest(d.audio.required_source_samples)!=expected:
                        raise ValueError('speech prefix changed/cancelled before publication')
            except Exception:
                # The native backend may already have published. Never repair
                # metadata by replaying a forward or accepting a second guess.
                self._state='failed'
                raise
            self._frame=d.window.end; self._sequence+=1; self._committed=True
            if d.role=='speech':
                if self._onset is None:self._onset=d.audio.onset_frame
                self._speech_index+=1; self._state='speaking'
                self._speech_final=d.audio.final_source_window
            elif not d.conversation_final:
                self._idle+=1

    def ack_decoded(self, ticket, emitted_frame):
        with self._lock:
            self._live()
            if ticket is not self._active or not self._committed:
                raise ValueError('decode ACK requires current published native commit')
            final=ticket.decision.conversation_final
            if type(emitted_frame) is not int or emitted_frame != self._frame-(0 if final else 5):
                self._state='failed'
                raise ValueError('native lookahead/final flush ACK mismatch')
            self._emitted=emitted_frame; self._active=None; self._committed=False
            if final:self._state='closed'

    def finish_speech(self):
        """Retire a sealed, fully covered source; do NOT flush the decoder."""
        with self._lock:
            decision=self.inspect()
            if decision.status!='end' or self._onset is None:
                raise RuntimeError('source not sealed/covered/drained')
            self._turns+=1; self._stream=None; self._onset=None; self._speech_index=0
            self._speech_final=False
            self._state='listening'

    def request_speech_interrupt(self, epoch):
        """Thread-safe request, NOT preemption: retain PCM through native ACK.

        Caller invalidates presentation independently before this remote action.
        No next window can be reserved until explicit retirement below.
        """
        with self._lock:
            self._live()
            if (type(epoch) is not int or self._stream is None
                    or self._stream.binding.epoch != epoch or self._closing):
                raise ValueError('current speech epoch required')
            if self._interrupt_epoch is not None:
                return False
            self._interrupt_epoch=epoch
            return True

    def retire_interrupted_speech(self):
        """Only after native commit/decoded ACK; no EOS or ingress cancellation.

        Caller retains the old ingress for the persistent decoder's lookahead.
        Return an explicit native transition marker, to accompany the next window.
        """
        with self._lock:
            self._live()
            if self._interrupt_epoch is None or self._active is not None:
                raise RuntimeError('interrupt requires drained native window')
            stream=self._stream
            marker=(dict(schema='h3-speech-interruption-v1',
                         source_id=stream.binding.source_id, committed_frame=self._frame)
                    if self._onset is not None and not self._speech_final else None)
            self._interrupted_turns+=1
            self._stream=None; self._onset=None; self._speech_index=0
            self._interrupt_epoch=None; self._state='listening'; self._speech_final=False
            return marker

    def cancel_pending_speech(self):
        """Cancel unstarted speech only; never abort a native speech forward."""
        with self._lock:
            self._live()
            if (self._interrupt_epoch is not None or self._stream is None or self._onset is not None
                    or (self._active is not None and self._active.decision.role=='speech')):
                raise RuntimeError('only unreserved pending speech can be cancelled')
            self._stream.cancel(); self._stream=None

    def request_stop(self):
        with self._lock:
            self._live()
            if self._active is not None or self._onset is not None or self._stream is not None:
                raise RuntimeError('drain/retire speech before graceful final listening')
            self._closing=True

    def fail(self):
        """Stop admission. Caller separately cleans up native ranks and ingress."""
        with self._lock:self._state='failed'

    def snapshot(self):
        with self._lock:
            return dict(phase=self._state,committed_frame=self._frame,emitted_frame=self._emitted,
                        reserved=self._active is not None,windows=self._sequence,
                        idle_windows=self._idle,completed_turns=self._turns,
                        interrupted_turns=self._interrupted_turns,
                        interruption_pending=self._interrupt_epoch is not None,
                        speech_onset_frame=self._onset,native_runtime_connected=False)
