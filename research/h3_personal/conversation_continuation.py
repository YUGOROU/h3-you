"""Explicit CPU tensor handoff adapter. No decoder, model or legacy Session changes.

Canonical audio is NEW-source audio, including VAE-encoded preroll for guided
sources. Encoding/provenance and original PCM placement belong to the caller;
no receipt is accepted in lieu of tensor validation here. Serialize all calls.
"""
from copy import copy
from dataclasses import dataclass
from .fast_continuation import Session, Suffix, WindowInput, TorchOps
from .continuation_plan import Window, video_t, integer


@dataclass(frozen=True)
class GlobalNativeToken:
    model: WindowInput
    local: WindowInput
    profile: str
    canvas: tuple
    epoch: int
    source_id: str

    @property
    def window(self):
        return self.model.window


@dataclass(frozen=True)
class GlobalSuffix:
    suffix: Suffix
    source_id: str
    source_audio_start: int
    source_audio_end: int
    committed_frames: int
    conversation_final: bool


@dataclass(frozen=True, eq=False)
class _PreparedCommit:
    owner: object
    epoch: int
    token: GlobalNativeToken
    session: Session
    result: GlobalSuffix
    tail: object
    tail_digest: str
    sequence: int
    index: int
    complete: bool


class ConversationOwner:
    """Finite opt-in adapter; model health is an external explicit admission gate.

    ack_decoded_count takes the actual cumulative decoded frame count, not a
    receipt or a requested count. Source completion never requests final flush.
    The external persistent decoder retains ownership of its lookahead buffer.
    """
    MAX_IMMUTABLE_UNITS = 4  # ordinary source Sessions; streaming subclass opts in

    def __init__(self, conversation_id, *, profile, height, width,
                 model_healthy, enabled=False, max_sessions=4, ops=None):
        if enabled is not True or model_healthy is not True:
            raise ValueError('explicit opt-in and known healthy external model required')
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError('conversation id required')
        if not isinstance(profile, str) or not profile:
            raise ValueError('fixed profile binding required')
        integer(max_sessions, 'max sessions', 1)
        if max_sessions > self.MAX_IMMUTABLE_UNITS:
            raise ValueError('immutable source/window limit exceeded')
        for n in (height, width):
            integer(n, 'canvas', 2)
            if n % 2:
                raise ValueError('even canvas required')
        self.conversation_id = conversation_id
        self._binding = (profile, (height, width))
        self.ops = ops or TorchOps()
        self._per = height//2 * (width//2)
        self._limit = max_sessions
        self.committedF = self.emittedF = 0
        self._epoch = 0
        self._sequence = 0
        self._final_acked = False
        self._ids = set()
        self._session = self._active = self._tail = None
        self._terminal = self._final = False
        self._complete = True
        self._prepared = None
        self._awaiting_ack = False

    @property
    def source_session(self):
        """Borrowed read-only Session; callers must never operate on it."""
        return self._session

    @property
    def active_native(self):
        return self._active

    @property
    def pending_window(self):
        return self._active.model if self._active is not None else None

    @property
    def per_frame(self):
        return self._per

    @property
    def canonical_digest(self):
        return self._canonical_digest

    @property
    def faulted(self):
        return self._session is not None and self._session.faulted

    def _check(self, tensor, shape):
        self._session._check(tensor, shape)

    def _source_ok(self):
        self._session._source_ok()
        if self.ops.digest(self._canonical) != self._canonical_digest:
            raise ValueError('new canonical mutated')

    def _live(self):
        if self._terminal or self._final:
            raise RuntimeError('terminal conversation')

    def start_source(self, source_id, *, canonical_audio, total_frames,
                     profile, height, width, audio_origin=None):
        self._live()
        if (profile, (height, width)) != self._binding:
            raise ValueError('profile/canvas drift')
        if self._active is not None or not self._complete:
            raise RuntimeError('previous source incomplete')
        if self.committedF and self.emittedF != self.committedF-5:
            raise RuntimeError('previous source decode acknowledgement required')
        if not isinstance(source_id, str) or not source_id or source_id in self._ids:
            raise ValueError('new unique source id required')
        if self._epoch >= self._limit:
            raise ValueError('finite source limit')
        from .conversation_audio import audio_origin as expected_origin
        if audio_origin != expected_origin(self.committedF, total_frames):
            raise ValueError('verified source audio origin/preroll required')
        # Own canonical storage: never replace/mutate an existing Session source.
        canonical = self.ops.copy(canonical_audio)
        session = Session(source_id, source_audio=canonical, total_frames=total_frames,
                          height=height, width=width, window_frames=124, ops=self.ops)
        if len(session.windows) > 4:
            raise ValueError('at most four local windows')
        canonical_digest = self.ops.digest(canonical)
        base = self.committedF-22 if self.committedF else 0
        if self._tail is not None and self.ops.digest(self._tail) != self._tail_digest:
            raise ValueError('owned tail mutated')
        if self._session is not None:
            self._session.close()
        self._session = session
        self._canonical = canonical
        self._canonical_digest = canonical_digest
        self._base = base
        self._guided = bool(self.committedF)
        self._epoch += 1
        self._ids.add(source_id)
        self._index = 0
        self._complete = False

    def _visual_guide(self):
        """Copy the prior generated tail; subclasses may opt into diagnostics."""
        return self.ops.copy(self._tail)

    def begin_window(self):
        self._live()
        if self._session is None or self._complete or self._active is not None:
            raise RuntimeError('no available source window')
        if self._awaiting_ack:
            raise RuntimeError('previous commit decode acknowledgement required')
        actual = self._session.windows[self._index]
        candidate = Window(self._base+actual.start, self._base+actual.end,
                           22 if self._guided and self._index == 0 else actual.overlap)
        if candidate.audio_end-candidate.audio_start != actual.audio_end-actual.audio_start:
            raise ValueError('unsupported global/local target audio count')
        local = self._session.begin_window(self._index)
        # Retain local token even if subsequent allocation fails: abandon can abort.
        self._pending_local = local
        w = local.window
        global_w = Window(self._base+w.start, self._base+w.end,
                          22 if self._guided and self._index == 0 else w.overlap)
        first = self._guided and self._index == 0
        if first and self.ops.digest(self._tail) != self._tail_digest:
            raise ValueError('owned tail mutated')
        gv = self._visual_guide() if first else (
            self.ops.copy(local.guide_video) if local.guide_video is not None else None)
        ga = (self.ops.copy(self.ops.slice(self._canonical, 1, w.audio_start,
                                                      w.audio_start+global_w.guide_audio_t))
              if first else (self.ops.copy(local.guide_audio)
                             if local.guide_audio is not None else None))
        locked = self.ops.copy(local.locked_audio)
        model = WindowInput(self.conversation_id, self._index, local.attempt,
                            global_w, global_w.temporal_offset,
                            video_t(self.committedF)*self._per if self.committedF else 0,
                            global_w.audio_start, gv, ga, locked, local.sampler)
        token = GlobalNativeToken(model, local, *self._binding, self._epoch,
                            self._session.session_id)
        self._digests = tuple(None if t is None else self.ops.digest(t)
                              for t in (gv, ga, locked))
        self._active = token
        return token

    def prepare_commit(self, token, *, clean_video, clean_audio, observed_sampler,
                       conversation_final=False):
        """Allocate and validate privately, before the caller's common MIN vote."""
        self._live()
        if token is not self._active or not isinstance(token, GlobalNativeToken):
            raise ValueError('stale or foreign native token')
        if self._prepared is not None:
            raise RuntimeError('discard existing prepared ticket first')
        if type(conversation_final) is not bool:
            raise ValueError('explicit final flag must be bool')
        s = self._session
        last = self._index+1 == len(s.windows)
        if conversation_final and not last:
            raise ValueError('conversation final requires source completion')
        m = token.model
        expected_w = Window(self._base+token.local.window.start,
                            self._base+token.local.window.end,
                            22 if self._guided and self._index == 0 else token.local.window.overlap)
        if (token.profile, token.canvas) != self._binding or token.epoch != self._epoch or token.source_id != s.session_id:
            raise ValueError('binding drift')
        if m.window != expected_w or m.temporal_offset != expected_w.temporal_offset or m.video_row_offset != (video_t(self.committedF)*self._per if self.committedF else 0) or m.audio_time_offset != expected_w.audio_start:
            raise ValueError('global offset drift')
        if self.ops.digest(self._canonical) != self._canonical_digest:
            raise ValueError('new canonical mutated')
        for t, digest in zip((m.guide_video, m.guide_audio, m.locked_audio), self._digests):
            if (None if t is None else self.ops.digest(t)) != digest:
                raise ValueError('model conditioning mutated')
        vt = video_t(m.window.frames)
        at = token.local.window.audio_end-token.local.window.audio_start
        s._check(clean_video, (vt*self._per, 96))
        s._check(clean_audio, (2, at, 32))
        if not self.ops.equal(clean_audio, m.locked_audio):
            raise ValueError('new source audio changed')
        first = self._guided and self._index == 0
        trim = 7 if first else token.local.window.guide_video_t
        atrim = m.window.guide_audio_t if first else token.local.window.guide_audio_t
        # All external allocations BEFORE Session's transactional commit.
        video = self.ops.copy(self.ops.slice(clean_video, 0, trim*self._per, vt*self._per))
        audio = self.ops.copy(self.ops.slice(clean_audio, 1, atrim, at))
        tail = self.ops.copy(self.ops.slice(clean_video, 0, (vt-7)*self._per, vt*self._per))
        tail_digest = self.ops.digest(tail)
        rows = video_t(m.window.end)*self._per
        offset = m.video_row_offset
        if offset+self.ops.shape(video)[0] != rows:
            raise ValueError('global suffix gap')
        audio_start = token.local.window.audio_start+atrim
        result = GlobalSuffix(Suffix(self.conversation_id, self._sequence,
                                     video, audio, offset, m.window.audio_start+atrim, rows,
                                     m.window.audio_end),
                              token.source_id, audio_start, token.local.window.audio_end,
                              m.window.end, conversation_final)
        staged = copy(s)  # canonical source and immutable conditioning are read-only shared
        staged.commit_window(token.local, clean_video=clean_video, clean_audio=clean_audio,
                             observed_sampler=observed_sampler)
        ticket = _PreparedCommit(self, self._epoch, token, staged, result, tail,
                                 tail_digest, result.suffix.index+1, self._index+1, last)
        self._prepared = ticket
        return ticket

    def discard_prepared(self, ticket):
        if ticket is not self._prepared or ticket.owner is not self:
            raise ValueError('stale or foreign prepared ticket')
        self._prepared = None

    def apply_prepared(self, ticket):
        """Post-vote publication: identity checks and field swaps only, no tensor ops.

        Retired Session disposal is deliberately inlined: Session.abort/close's
        known field clearing, without any possibly throwing method calls. The
        original borrowed session must never be reused by the caller.
        """
        if (not isinstance(ticket, _PreparedCommit) or ticket is not self._prepared
                or ticket.owner is not self or ticket.epoch != self._epoch
                or ticket.token is not self._active or self._terminal or self._final):
            raise ValueError('stale or foreign prepared ticket')
        old = self._session
        self._session = ticket.session
        self._tail = ticket.tail
        self._tail_digest = ticket.tail_digest
        self.committedF = ticket.result.committed_frames
        self._sequence = ticket.sequence
        self._index = ticket.index
        self._complete = ticket.complete
        self._final = ticket.result.conversation_final
        self._active = self._pending_local = self._prepared = None
        self._awaiting_ack = True
        old._active = None
        old._fault = False
        old._closed = True
        old._video_tail = old._audio_tail = old._source = None
        return ticket.result

    def commit_window(self, token, **kwargs):
        """Single-owner CPU convenience; distributed callers must vote in between."""
        return self.apply_prepared(self.prepare_commit(token, **kwargs))

    def ack_decoded_count(self, count, *, final_flush=False):
        if self._terminal:
            raise RuntimeError('abandoned conversation')
        integer(count, 'actual decoded count')
        if self._final_acked:
            raise RuntimeError('final decode already acknowledged')
        if type(final_flush) is not bool or final_flush != self._final:
            raise ValueError('final flush only on explicit conversation final')
        if not self._awaiting_ack:
            raise RuntimeError('no unacknowledged commit')
        expected = self.committedF if final_flush else max(0, self.committedF-5)
        if count != expected or count < self.emittedF:
            raise ValueError('actual decoded count mismatch')
        self._awaiting_ack = False
        self.emittedF = count
        self._final_acked = final_flush

    def abandon(self):
        if self._terminal:
            raise RuntimeError('already abandoned')
        if self._session is not None:
            pending = self._session.pending_window
            if pending is not None:
                self._session.abort_window(pending)
            elif self._session.faulted:
                self._session.abort_window()
            self._session.close()
        self._prepared = None
        self._tail = self._canonical = self._active = None
        self._terminal = True
