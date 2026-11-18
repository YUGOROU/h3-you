"""Explicit immutable-window Sessions; never grow an existing source Session.

The model/decoder keep ONE conversation owner/tail. Native Session IDs identify
fixed124frame conditioning windows, separate from logical speech source IDs.
Earlier encoded overlap wins within a logical source; switches encode real
preceding waveform history. Finite per-owner window budget, no restart/reset.
"""
from .conversation_continuation import ConversationOwner
from .conversation_audio import audio_origin
from .streaming_pcm import _id, _int


class StreamingWindowOwner(ConversationOwner):
    MAX_IMMUTABLE_UNITS = 144

    def __init__(self, conversation_id, *, max_windows, idle_anchor_interval=None,
                 visual_ancestry_policy=None, long_session=False, **kwargs):
        # Diagnostic only: no automatic fallback or production default change.
        if idle_anchor_interval is not None and (
                type(idle_anchor_interval) is not int or idle_anchor_interval != 3):
            raise ValueError('diagnostic idle anchor interval must be None or exactly 3')
        if visual_ancestry_policy is not None and (
                type(visual_ancestry_policy) is not str or
                visual_ancestry_policy != 'conversation-tail-every3-v1'):
            raise ValueError('unknown visual ancestry policy')
        if visual_ancestry_policy is not None and idle_anchor_interval is not None:
            raise ValueError('visual ancestry and idle anchor are mutually exclusive')
        self._visual_ancestry_policy=visual_ancestry_policy
        self._ancestry_anchor=None
        self._ancestry_anchor_digest=None
        self._ancestry_evidence=dict(policy=visual_ancestry_policy, sequence=0,
            kind='unconditioned', anchor_sha256=None, selected_sha256=None,
            anchor_sequence=None)
        self._idle_anchor_interval=idle_anchor_interval
        self._idle_anchor=None
        self._idle_anchor_digest=None
        if type(long_session) is not bool:raise ValueError('explicit long_session selection required')
        bound = 2542 if long_session else 144
        _int(max_windows,1,bound)
        self.MAX_IMMUTABLE_UNITS = bound
        self._max_windows=max_windows
        super().__init__(conversation_id,max_sessions=max_windows,**kwargs)
        self._logical=None
        self._logical_index=-1
        self._logical_final=True
        self._logical_interrupted=False
        self._interrupted_sources=set()
        self._logical_role=None
        self._audio_tail=None
        self._audio_tail_digest=None

    def start_source(self, *args, **kwargs):
        raise RuntimeError('streaming owner accepts explicit immutable windows only')

    def interrupt_source(self, source_id, committed_frame):
        """Mark a decoded speech boundary, not EOS; leave all native state intact."""
        _id(source_id)
        if type(committed_frame) is not int:
            raise ValueError('integer committed frame required')
        self._live()
        if (self._session is None or self.faulted or self._logical_role!='speech'
                or self._logical_final or self._logical_interrupted
                or self.active_native is not None or self._awaiting_ack
                or not self._complete or self.emittedF!=self.committedF-5
                or source_id!=self._logical or committed_frame!=self.committedF):
            raise ValueError('decoded active speech boundary required')
        self._source_ok()
        self._logical_interrupted=True
        self._interrupted_sources.add(source_id)

    def start_window(self, window_id, *, canonical_audio, logical_source_id,
                     logical_index, role, source_final, profile, height, width):
        _id(logical_source_id);_int(logical_index,0,self._max_windows-1)
        if role not in ('listening','speech') or type(source_final) is not bool:
            raise ValueError('explicit role/source finality required')
        if logical_source_id in self._interrupted_sources:
            raise ValueError('interrupted logical source cannot be reused')
        same=logical_source_id==self._logical
        if same:
            if (role!=self._logical_role or logical_index!=self._logical_index+1
                    or self._logical_final):
                raise ValueError('logical source window drift')
        elif logical_index!=0 or (self._logical_role=='speech'
                                  and not (self._logical_final or self._logical_interrupted)):
            raise ValueError('new source requires explicit previous speech EOS')
        if self.ops.shape(canonical_audio)!=(2,207,32) or not self.ops.finite_cpu(canonical_audio):
            raise ValueError('fixed native encoded window required')
        canonical=self.ops.copy(canonical_audio)
        if same:
            if self.ops.digest(self._audio_tail)!=self._audio_tail_digest:
                raise ValueError('prior canonical overlap mutated')
            # Never recompute/replace the historical canonical overlap. This is
            # exactly window-context-canonical-v1 prior-canonical-wins policy.
            canonical=self._join(self._audio_tail,self.ops.slice(canonical,1,37,207))
        tail=self.ops.copy(self.ops.slice(canonical,1,170,207))
        digest=self.ops.digest(tail)
        super().start_source(window_id,canonical_audio=canonical,total_frames=124,
            profile=profile,height=height,width=width,
            audio_origin=audio_origin(self.committedF,124))
        if not same or role!=self._logical_role:
            self._idle_anchor=None
            self._idle_anchor_digest=None
        self._logical=logical_source_id;self._logical_index=logical_index
        self._logical_role=role;self._logical_final=source_final
        self._logical_interrupted=False
        self._audio_tail=tail;self._audio_tail_digest=digest

    @property
    def visual_ancestry_evidence(self):
        """Defensive scalar snapshot of the most recently selected guide."""
        return dict(self._ancestry_evidence)

    def begin_window(self):
        # Capture the first generated tail before any conditioning selection.
        # Never reset this conversation-level anchor on role/source transitions.
        if self._visual_ancestry_policy is not None:
            if self._sequence == 1 and self._ancestry_anchor is None:
                self._check(self._tail, (7*self._per,96))
                if self.ops.digest(self._tail) != self._tail_digest:
                    raise ValueError('owned tail mutated')
                anchor=self.ops.copy(self._tail)
                self._check(anchor, (7*self._per,96))
                self._ancestry_anchor=anchor
                self._ancestry_anchor_digest=self.ops.digest(anchor)
            if self._ancestry_anchor is not None:
                self._check(self._ancestry_anchor, (7*self._per,96))
                if self.ops.digest(self._ancestry_anchor) != self._ancestry_anchor_digest:
                    raise ValueError('conversation visual anchor mutated')
        token=super().begin_window()
        selected=token.model.guide_video
        anchored=(self._visual_ancestry_policy is not None and
                  self._sequence > 0 and self._sequence % 3 == 0)
        self._ancestry_evidence=dict(policy=self._visual_ancestry_policy,
            sequence=self._sequence,
            kind='unconditioned' if selected is None else ('anchor' if anchored else 'latest'),
            anchor_sha256=self._ancestry_anchor_digest,
            selected_sha256=None if selected is None else self.ops.digest(selected),
            anchor_sequence=1 if self._ancestry_anchor is not None else None)
        return token

    def _visual_guide(self):
        if self._visual_ancestry_policy is not None:
            if self._sequence > 0 and self._sequence % 3 == 0:
                if self._ancestry_anchor is None:
                    raise ValueError('missing conversation visual anchor')
                return self.ops.copy(self._ancestry_anchor)
            return super()._visual_guide()
        if self._idle_anchor_interval is None or self._logical_role!='listening':
            return super()._visual_guide()
        if self._logical_index==1 and self._idle_anchor is None:
            anchor=self.ops.copy(self._tail)
            self._check(anchor,(7*self._per,96))
            digest=self.ops.digest(anchor)
            self._idle_anchor=anchor
            self._idle_anchor_digest=digest
        if self._idle_anchor is not None:
            self._check(self._idle_anchor,(7*self._per,96))
            if self.ops.digest(self._idle_anchor)!=self._idle_anchor_digest:
                raise ValueError('idle visual anchor mutated')
        if self._logical_index>=3 and self._logical_index%3==0:
            if self._idle_anchor is None:
                raise ValueError('missing idle visual anchor')
            return self.ops.copy(self._idle_anchor)
        return super()._visual_guide()

    def _join(self, guide, suffix):
        # TorchOps used by production; injected CPU ops expose cat_time for
        # value-level tests without adding numpy/torch dependencies locally.
        if hasattr(self.ops,'cat_time'):
            return self.ops.cat_time(guide,suffix)
        import torch
        return torch.cat((guide,suffix),dim=1).clone().contiguous()
