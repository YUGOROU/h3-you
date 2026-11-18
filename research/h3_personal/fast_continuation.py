"""Host-side native latent-tail bridge; FastVideo integration is NOT activated.

Reuses the GPU-established native continuation planner, not its Turbo sampler.
Canonical private tensors: video [T*H/2*W/2,96], audio [2,T,32], clean,
normalized, finite CPU values. Ops.copy MUST detach, clone and compact storage;
Ops.slice MUST slice the given axis, never flatten channel-major audio.
Ops.digest and Ops.equal validate actual values (not shapes or object ids).
Injected ops are trusted, synchronous; serialize access to a session.

Backend binding still required (these are NOT generate() keyword arguments):
* latentprep: retain ordered original image/text caches; append clean AV guide
  rows without RGB/Qwen re-encoding, apply native .999 visual RF augmentation,
  use build_layout below and its GLOBAL target audio length (not local rounding).
* denoise: hold guide AND target source audio at timestep1, skip audio integration;
  keep existing SMD4 schedule/weights/CP, check cancellation collectively before
  forwards/commit, and return CLEAN target-only rows plus locked audio.
* commit_window emits only the new suffix; caller owns incremental decode/output.
  No cumulative GPU state, decoder, model loading/shutdown or sampler is here.
Dynamic guide is a native reference block, never FL2VA/last-frame conditioning.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Protocol, Any

from .continuation_plan import Window, plan, video_t, audio_boundary, integer

INTEGRATION_ACTIVATED = False
PACKING_SHA = '852649a9b1a539a76cbb3fb481b2f33c5176b6ed74dea17d41057194aaada0d6'
REFERENCE_SHA = '103c3e31011b943fbeb42470bbd3b798e71016cf911fe7dbf0eb93de05b85a08'


class TensorOps(Protocol):
    def shape(self, tensor): ...
    def copy(self, tensor): ...
    def slice(self, tensor, axis, start, end): ...
    def digest(self, tensor) -> str: ...
    def equal(self, left, right) -> bool: ...
    def finite_cpu(self, tensor) -> bool: ...


class TorchOps:
    """Lazy optional implementation; importing this module never imports torch."""
    def shape(self, tensor):
        return tuple(tensor.shape)

    def copy(self, tensor):
        return tensor.detach().cpu().clone().contiguous()

    def slice(self, tensor, axis, start, end):
        indices = [slice(None)] * tensor.ndim
        indices[axis] = slice(start, end)
        return tensor[tuple(indices)]

    def digest(self, tensor):
        import torch
        value = tensor.detach().cpu().contiguous()
        header = repr((tuple(value.shape), str(value.dtype))).encode()
        return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()

    def equal(self, left, right):
        import torch
        return left.dtype == right.dtype and torch.equal(left, right)

    def finite_cpu(self, tensor):
        import torch
        return tensor.device.type == 'cpu' and bool(torch.isfinite(tensor).all())


@dataclass(frozen=True)
class SamplerContract:
    """Receipt of the existing backend, not a scheduler configuration request."""
    name: str = 'SMD4'
    forwards: int = 4

    def validate(self):
        if self.name != 'SMD4' or type(self.forwards) is not int or self.forwards != 4:
            raise ValueError('only existing SMD4/four forwards; no Turbo8/shift12 substitution')


@dataclass(frozen=True)
class WindowInput:
    session_id: str
    index: int
    attempt: int
    window: Window
    temporal_offset: Fraction
    video_row_offset: int
    audio_time_offset: int
    guide_video: Any
    guide_audio: Any
    locked_audio: Any
    sampler: SamplerContract


@dataclass(frozen=True)
class CleanTarget:
    """Backend result, target-only clean rows; audio remains channel-major."""
    video: Any
    audio: Any
    observed_sampler: SamplerContract


class BackendHooks(Protocol):
    """OUR binding interface, not an existing FastVideo public API.

    prepare must bind latentprep/layout/RF using WindowInput's actual latents.
    denoise must use the existing SMD4 backend and return clean target-only rows.
    check_current must coordinate CP ranks at safe boundaries; the backend also
    checks before each forward. No hook may commit, close, or replace a model.
    Implementations are not provided/activated by this host-only module.
    """
    def check_current(self) -> bool: ...
    def prepare(self, window: WindowInput) -> Any: ...
    def denoise(self, prepared: Any) -> CleanTarget: ...


class WindowCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Suffix:
    session_id: str
    index: int
    video: Any
    audio: Any
    video_row_offset: int
    audio_time_offset: int
    cumulative_video_rows: int
    cumulative_audio_t: int


class Session:
    """One serialized owner. Frozen envelopes do not make tensors immutable:
    exposed tensors are isolated copies and digest-checked at commit. Tokens are
    identity-bound, single-use, and cannot cross sessions. A fault poisons the
    in-flight attempt until abort_window; committed tails/counters stay intact.
    Aborting never shuts down the underlying warm model (we do not own one).
    Source is borrowed read-only; caller must retain it unchanged for the session.
    """
    def __init__(self, session_id, *, source_audio, total_frames, height, width,
                 window_frames, ops=None, overlap_frames=22,
                 sampler=SamplerContract()):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('nonempty session id required')
        for n in (height, width):
            integer(n, 'latent canvas', 2)
            if n % 2:
                raise ValueError('even latent canvas required')
        sampler.validate()
        self.windows = plan(total_frames, window_frames, overlap_frames)
        self.session_id = session_id
        self.ops = ops or TorchOps()
        self._source = source_audio
        self._total = total_frames
        self._per_frame = (height // 2) * (width // 2)
        self._sampler = sampler
        self._check(source_audio, (2, audio_boundary(total_frames), 32))
        self._source_digest = self.ops.digest(source_audio)
        self._index = self._attempt = 0
        self._video_rows = self._audio_t = 0
        self._video_tail = self._audio_tail = None
        self._active = None
        self._fault = False
        self._closed = False

    @property
    def committed_shape(self):
        return self._video_rows, self._audio_t

    @property
    def faulted(self):
        return self._fault

    def _check(self, tensor, shape):
        if tuple(self.ops.shape(tensor)) != shape or not self.ops.finite_cpu(tensor):
            raise ValueError('finite CPU latent shape mismatch: expected ' + repr(shape))

    def _source_ok(self):
        if self.ops.digest(self._source) != self._source_digest:
            raise ValueError('source audio digest changed')

    def _copy_slice(self, tensor, axis, start, end):
        return self.ops.copy(self.ops.slice(tensor, axis, start, end))

    def begin_window(self, index):
        if self._closed or self._active is not None or self._fault:
            raise RuntimeError('closed or in-flight; explicit abort required after fault')
        if type(index) is not int or index != self._index or index >= len(self.windows):
            raise ValueError('out-of-order window')
        # Reserve attempt BEFORE any ops: even preparation failure needs abort.
        self._attempt += 1
        self._active = self._attempt
        try:
            self._source_ok()
            w = self.windows[index]
            locked = self._copy_slice(self._source, 1, w.audio_start, w.audio_end)
            if w.overlap:
                gv = self.ops.copy(self._video_tail)
                ga = self.ops.copy(self._audio_tail)
            else:
                gv = ga = None
            token = WindowInput(self.session_id, index, self._attempt, w,
                                w.temporal_offset, self._video_rows, self._audio_t,
                                gv, ga, locked, self._sampler)
            self._digests = tuple(None if t is None else self.ops.digest(t)
                                  for t in (gv, ga, locked))
            self._active = token
            return token
        except BaseException:
            self._fault = True
            raise

    def _token(self, token):
        if self._closed or token is not self._active or not isinstance(token, WindowInput):
            raise ValueError('stale, duplicate, or foreign window token')

    def commit_window(self, token, *, clean_video, clean_audio, observed_sampler):
        self._token(token)
        if self._fault:
            raise RuntimeError('faulted window requires explicit abort')
        try:
            observed_sampler.validate()
            if observed_sampler != self._sampler:
                raise ValueError('sampler drift')
            self._source_ok()
            for t, digest in zip((token.guide_video, token.guide_audio, token.locked_audio), self._digests):
                if t is not None and self.ops.digest(t) != digest:
                    raise ValueError('backend mutated borrowed condition')
            w = token.window
            vt, at = video_t(w.frames), w.audio_end - w.audio_start
            self._check(clean_video, (vt * self._per_frame, 96))
            self._check(clean_audio, (2, at, 32))
            expected = self.ops.slice(self._source, 1, w.audio_start, w.audio_end)
            if not self.ops.equal(clean_audio, expected):
                raise ValueError('target audio differs from global source')
            if w.overlap:
                guide = self.ops.slice(expected, 1, 0, w.guide_audio_t)
                if not self.ops.equal(token.guide_audio, guide):
                    raise ValueError('previous audio tail differs from source guide')
            sv = self._copy_slice(clean_video, 0, w.guide_video_t * self._per_frame, vt * self._per_frame)
            sa = self._copy_slice(clean_audio, 1, w.guide_audio_t, at)
            vr = self._video_rows + self.ops.shape(sv)[0]
            ar = self._audio_t + self.ops.shape(sa)[1]
            if vr != video_t(w.end) * self._per_frame or ar != w.audio_end:
                raise ValueError('suffix append gap/overlap')
            # Keep only the exact next guide, not the full current/cumulative clip.
            next_w = self.windows[self._index + 1] if self._index + 1 < len(self.windows) else None
            tv = self._copy_slice(clean_video, 0, (vt-next_w.guide_video_t)*self._per_frame,
                                  vt*self._per_frame) if next_w else None
            ta = self._copy_slice(clean_audio, 1, at-next_w.guide_audio_t, at) if next_w else None
            result = Suffix(self.session_id, self._index, sv, sa, self._video_rows,
                            self._audio_t, vr, ar)
            # No state mutation until every validation/allocation has succeeded.
            self._video_tail, self._audio_tail = tv, ta
            self._video_rows, self._audio_t = vr, ar
            self._index += 1
            self._active = None
            return result
        except BaseException:
            self._fault = True
            raise

    def fault_window(self, token):
        """Low-level backend must report any failed/cancelled in-flight work."""
        self._token(token)
        self._fault = True

    def run_window(self, index, backend: BackendHooks):
        """Optional transactional driver for the explicit backend hook contract.

        Any hook error/cancellation requires abort_window(token). On driver
        failure the active token is available via pending_window; no retry,
        implicit abort, schedule substitution, or model shutdown occurs here.
        """
        token = self.begin_window(index)
        try:
            if not backend.check_current():
                raise WindowCancelled('stale continuation window')
            prepared = backend.prepare(token)
            if not backend.check_current():
                raise WindowCancelled('stale continuation window')
            clean = backend.denoise(prepared)
            if not backend.check_current():
                raise WindowCancelled('stale continuation window')
            return self.commit_window(token, clean_video=clean.video,
                                      clean_audio=clean.audio,
                                      observed_sampler=clean.observed_sampler)
        except BaseException:
            self.fault_window(token)
            raise

    @property
    def pending_window(self):
        return self._active if isinstance(self._active, WindowInput) else None

    def abort_window(self, token=None):
        """Discard attempt only. None allowed solely for failed begin_window."""
        if self._closed or self._active is None:
            raise ValueError('no in-flight window to abort')
        if isinstance(self._active, WindowInput):
            self._token(token)
        elif token is not None:
            raise ValueError('failed begin has no token')
        self._active = None
        self._fault = False

    def close(self):
        """Explicit local disposal, never model shutdown; no pending attempt allowed."""
        if self._closed or self._active is not None:
            raise RuntimeError('already closed or pending window; abort first')
        self._closed = True
        self._video_tail = self._audio_tail = self._source = None


def verify_packing(packing, references):
    for module, sha in ((packing, PACKING_SHA), (references, REFERENCE_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != sha:
            raise ValueError('requires pinned ba371e33 FastVideo packing/reference')
    if packing.temporal_position_grid.__module__ != packing.__name__:
        raise ValueError('mixed packing helper roots')


# FastVideo's model shards with transport padding, trims QKV BEFORE attention,
# and gathers/unpads BEFORE selecting modality outputs. Unlike the older native
# CP path, an odd logical sequence is valid. Do not pad text/latents/layout here.
SP_TRANSPORT_SHAS = {
    'models/dits/minimax_h3.py': 'b7d470816af7b18620710e3b40c7625f2658d1950d7859f018ecc5130d379753',
    'distributed/communication_op.py': '075f5ea6aa3fbc27abb06cc335a62b5e7651b76f381c0842e36abb1c87a995b6',
    'distributed/utils.py': 'b3744b9fdd5a8ca916c3c6e3161c53d3cd3721605f1b4a69b51e132a3be9cdaf',
    'attention/layer.py': '7545abede1fa105a24c23f7f2e713e6120abd0f325073ec5dddb7f88660071f8',
}


def verify_sequence_transport(packing, sequence_length, ranks):
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError('positive logical sequence length required')
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError('only 1/2 ranks supported')
    if ranks == 2:
        root = Path(packing.__file__).resolve().parents[3]
        for relative, expected in SP_TRANSPORT_SHAS.items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('pinned FastVideo SP transport differs: ' + relative)


def build_layout(packing, references, *, text_tags, images, window, height, width, ranks=2,
                 temporal_policy='global-v1'):
    """Concrete pinned FastVideo packing port; NOT installed in latentprep.

    Original ordered IMAGE references stay fixed. Final synthetic prepared video
    describes clean latent guide geometry ONLY; never send it to media encoders.
    FastVideo uses _reference_temporal_span (sequential sum), NOT the Diffusers
    _temporal_position_span (NumPy pairwise sum). Preserve that source contract.
    """
    if temporal_policy not in ('global-v1', 'window-local-v1'):
        raise ValueError('unknown temporal policy')
    verify_packing(packing, references)
    import torch
    if not isinstance(window, Window):
        raise TypeError('planner Window required')
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError('only 1/2 ranks supported')
    for n in (height, width):
        integer(n, 'latent canvas', 2)
        if n % 2:
            raise ValueError('even latent canvas required')
    if text_tags.ndim != 1 or text_tags.dtype != torch.long or not text_tags.numel():
        raise ValueError('nonempty int64 text tags required')
    if not bool(((text_tags == 0) | (text_tags == 1)).all()):
        raise ValueError('invalid text modality tags')
    refs = list(images)
    if not refs or any(r.media_type != 'image' or r.num_latent_frames != 1 or
                       r.has_audio or r.num_audio_latents for r in refs):
        raise ValueError('ordered original image references only; no FL2VA substitution')
    fixed_rows = sum((r.latent_height//2)*(r.latent_width//2) for r in refs)
    if window.overlap:
        refs.append(references.MiniMaxH3PreparedReference(
            media_type='video', has_audio=True, num_latent_frames=window.guide_video_t,
            latent_height=height, latent_width=width, num_audio_latents=window.guide_audio_t))
    layout = packing.build_ref2va_packed_sequence(
        text_tags.cpu(), refs, video_t(window.frames), height, width,
        window.audio_end-window.audio_start, (1, 2, 2))
    verify_sequence_transport(packing, layout.sequence_length, ranks)
    pos = layout.position_ids.clone()
    if window.overlap:
        advance = max(float(window.guide_audio_t), packing._reference_temporal_span(window.guide_video_t))
        pos[layout.video_indices[layout.num_condition_video_rows:], 0] -= advance
        pos[layout.audio_indices[layout.num_condition_audio_rows:], 0] -= advance
    if temporal_policy == 'global-v1':
        pos[layout.video_indices[fixed_rows:], 0] += float(window.temporal_offset)
        pos[layout.audio_indices, 0] += float(window.temporal_offset)
    if not bool(torch.isfinite(pos).all()):
        raise ValueError('nonfinite rotary positions')
    return replace(layout, position_ids=pos)
