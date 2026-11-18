"""Exclusive, finite Mac-side turn control; device-free epoch sink only.

Caller supplies already-owned Commands/EpochAVSink, genuine finalized ASR input
and validated Pi payloads. No capture, playback, provider or model is started.
VAD is not connected automatically: input echo separation still needs approval
and evidence. Native safe-boundary control is distinct from local invalidation.
"""
import threading

from .epoch_av_sink import EpochAVSink
from .live_relay_commands import Commands
from .turn_epoch import TurnEpoch


class TurnController:
    def __init__(self, commands, sink, *, deadline, max_turns=2):
        if (type(commands) is not Commands or not commands._turn_interruptions
                or not isinstance(sink, EpochAVSink)
                or commands.session != sink.binding.session):
            raise ValueError('exclusive interrupt-enabled Commands and matching epoch sink required')
        if type(max_turns) is not int or not 1 <= max_turns <= 2:
            raise ValueError('existing finite two-turn scope required')
        self.commands, self.sink = commands, sink
        self._guard = TurnEpoch(deadline=deadline, max_turns=max_turns)
        self._state = threading.Lock(); self._io = threading.Lock()
        self._published = self._publishing = self._presentation = self._interrupted = 0
        self._failed = False

    def current(self, epoch):
        with self._state:
            return not self._failed and self._guard.is_current(epoch)

    def begin_input(self):
        """On a confirmed human onset: invalidate BEFORE waiting for control IO.

        Returns the new input token after sending any necessary old-turn control.
        An in-flight publication may finish once, but its result stays stale and
        is followed by ordered interruption, never a transport retry.
        """
        with self._state:
            if self._failed:raise RuntimeError('terminal turn controller')
            epoch = self._guard.speech_onset()
            old = self._publishing or self._published
            should_interrupt = old > self._interrupted
            if should_interrupt:
                self._presentation += 1
                self._interrupted = old
                presentation = self._presentation
        try:
            if should_interrupt:
                self.sink.invalidate(presentation, interrupted_source_epoch=old)
                with self._io:
                    self.commands.send('interrupt_turn', old,
                        dict(schema='h3-turn-interruption-v1', presentation_epoch=presentation))
            with self._state:
                if not self._guard.finish_interrupt(epoch):
                    raise RuntimeError('stale input onset')
            return epoch
        except BaseException:
            self._fault()
            raise

    def commit_input(self, epoch):
        """Reserve a genuine finalized input once; does not attest ASR evidence."""
        with self._state:
            return not self._failed and self._guard.commit(epoch)

    def publish_reply(self, epoch, payload):
        """Reject stale Pi callbacks BEFORE relay use. False means quarantined.

        If onset races an already-admitted send, the request may reach the worker
        once; begin_input orders its interrupt afterward and no replay is used.
        """
        with self._io:
            with self._state:
                if self._failed or not self._guard.is_current(epoch):
                    return False
                if self._guard.snapshot().state.value != 'THINKING':
                    return False
                self._publishing = epoch
            try:
                self.commands.send('turn', epoch, payload)
            except BaseException:
                self._fault()
                raise
            with self._state:
                # Publish bookkeeping atomically: onset must never see a gap
                # between the in-flight and successfully sent old turn.
                self._published = epoch
                self._publishing = 0
                return self._guard.speaking(epoch)

    def _fault(self):
        with self._state:
            self._failed = True
            self._publishing = 0
            self._guard.stop()
        self.sink.cancel()

    def stop(self):
        """Session stop, not turn interruption; no cloud resource operations."""
        with self._state:
            changed = self._guard.stop()
        self.sink.cancel()
        if changed:
            with self._io:
                with self._state:
                    epoch = self._published
                self.commands.send('stop', epoch, {})
        return changed

    def snapshot(self):
        with self._state:
            state = self._guard.snapshot()
            return dict(epoch=state.epoch, state=state.state.value,
                        presentation_epoch=self._presentation,
                        published_turn_epoch=self._published,
                        reason=None if state.reason is None else state.reason.value,
                        failed=self._failed)
