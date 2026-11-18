"""Opt-in persistent native conversation decoder; not yet a stage hook.

Serialize decode/abandon at host boundaries. MIN gates cover host preparation
and output failures, not process death: the external finite PG timeout remains
mandatory. Native wrappers synchronize their own clip failures.
"""
from dataclasses import dataclass

from .conversation_continuation import ConversationOwner, GlobalNativeToken, GlobalSuffix
from .continuation_plan import video_t
from .fast_continuation_decode import FastVideoDecodeStream, _unpack


class _TensorOps:
    def __init__(self, vae):
        import torch
        self.t = torch
        self.device = vae.latents_mean.device
        self.d = torch.distributed
        if not self.d.is_initialized() or self.d.get_world_size() != 2:
            raise ValueError('initialized two-rank default process group required')
        self.rank = self.d.get_rank()
        # Allocate control storage before any recoverable work.
        self.status = torch.ones((), dtype=torch.int64, device=self.device)

    def common(self, error):
        self.status.fill_(0 if error else 1)
        self.d.all_reduce(self.status, op=self.d.ReduceOp.MIN)
        if not self.status.item():
            raise RuntimeError('conversation decode common failure') from error

    def prepare(self, vae, rows, count, geometry):
        c, h, w = geometry
        z = _unpack(rows, count, h, w, c)
        if tuple(z.shape) != (1, c, count, h, w):
            raise ValueError('native unpack shape drift')
        return vae.denormalize_latents(z.to(device=self.device, dtype=self.t.float32))

    def inference(self):
        return self.t.inference_mode()

    def autocast(self):
        return self.t.autocast(device_type=self.device.type, dtype=self.t.float16,
                               enabled=self.device.type == 'cuda')

    def pixels(self, vae, chunk):
        return vae.denormalize_pixels(chunk.float()).clamp_(0, 1).mul_(255).clamp_(0, 255).to(self.t.uint8).cpu()

    def output(self, chunks):
        return (self.t.cat(chunks, dim=2) if self.rank == 0 else
                self.t.empty((0, 3, 0, 0, 0), device='cpu', dtype=self.t.float32))


@dataclass
class _Decoder:
    owner: ConversationOwner
    binding: tuple
    stream: object
    ops: object
    sequence: int = 0
    failed: bool = False


def abandon(stage):
    """Explicit safe-host-boundary cleanup only; never call inside forward.

    Does not close the model or infer its health, nor abandon the owner itself.
    The poisoned slot is retained to refuse accidental stale reuse.
    """
    state = getattr(stage, '_h3_conversation_decoder', None)
    if state is not None:
        state.failed = True
        state.stream.closed = True
        state.stream.clear()


def _validate(state, batch, args, context):
    owner = context['conversation_owner']
    token = context['global_token']
    result = context['global_suffix']
    manifest = context['manifest']
    if not isinstance(owner, ConversationOwner) or not isinstance(token, GlobalNativeToken) or not isinstance(result, GlobalSuffix):
        raise ValueError('committed conversation context required')
    m, local, s = token.model, token.local, result.suffix
    if (owner._terminal or owner._active is not None or owner._session is None or
            owner._session.faulted or owner._session._closed or
            owner._session.pending_window is not None or owner._final_acked):
        raise ValueError('owner is not committed and live')
    binding = (owner.conversation_id, *owner._binding)
    if (token.profile, token.canvas) != owner._binding or token.epoch != owner._epoch or token.source_id != owner._session.session_id:
        raise ValueError('token binding drift')
    w = local.window
    from .continuation_plan import Window
    expected = Window(owner._base+w.start, owner._base+w.end,
                      22 if owner._guided and local.index == 0 else w.overlap)
    if (m.window != expected or m.session_id != owner.conversation_id or
            m.index != local.index or m.attempt != local.attempt or
            local.session_id != token.source_id or local.index != owner._index-1 or
            local.attempt != owner._session._attempt or
            local.window != owner._session.windows[local.index] or
            m.temporal_offset != expected.temporal_offset or m.audio_time_offset != expected.audio_start or
            m.sampler != local.sampler or
            manifest.get('source_index') != local.index):
        raise ValueError('immutable native token/source index drift')
    if (s.session_id != owner.conversation_id or result.source_id != token.source_id or
            s.index != owner._sequence-1 or s.index != (state.sequence if state else 0) or
            result.committed_frames != owner.committedF or m.window.end != owner.committedF or
            type(result.conversation_final) is not bool or result.conversation_final != owner._final):
        raise ValueError('stale/uncommitted global suffix')
    c, h, width = (24, *token.canvas)
    raw = tuple(batch.raw_latent_shape)
    if raw != (1, c, video_t(m.window.frames), h, width):
        raise ValueError('native raw geometry drift')
    count = video_t(owner.committedF) - (state.stream.received if state else 0)
    offset = (state.stream.received if state else 0)*owner._per
    if (count <= 0 or s.video_row_offset != offset or m.video_row_offset != offset or
            s.cumulative_video_rows != video_t(owner.committedF)*owner._per):
        raise ValueError('global row offset gap/overlap')
    owner._session._check(s.video, (count*owner._per, 96))
    owner._session._check(m.locked_audio, (2, local.window.audio_end-local.window.audio_start, 32))
    if m.guide_video is not None:
        owner._session._check(m.guide_video, (m.window.guide_video_t*owner._per, 96))
    if m.guide_audio is not None:
        owner._session._check(m.guide_audio, (2, m.window.guide_audio_t, 32))
    if (s.audio_time_offset != m.window.audio_start+m.window.guide_audio_t or
            s.cumulative_audio_t != m.window.audio_end or
            result.source_audio_end != local.window.audio_end or
            result.source_audio_start != local.window.audio_start+m.window.guide_audio_t):
        raise ValueError('global/source audio offset drift')
    if state is None and owner.emittedF != 0:
        raise ValueError('new decoder cannot resume acknowledged state')
    for tensor, digest in zip((m.guide_video, m.guide_audio, m.locked_audio), owner._digests):
        if (None if tensor is None else owner.ops.digest(tensor)) != digest:
            raise ValueError('native token tensor mutated')
    schedule = (manifest.get('efficiency') or {}).get('decode_schedule', 'alternating-v1')
    if schedule not in ('alternating-v1', 'two-clip-waves-v1'):
        raise ValueError('invalid native schedule')
    waves = schedule == 'two-clip-waves-v1'
    if (getattr(args, 'vae_parallel_decode', False) is not True or
            getattr(args, 'vae_cpu_offload', False) or
            getattr(args, 'video_decode_backend', 'h3-vae') != 'h3-vae' or
            getattr(args, 'output_type', 'pil') == 'latent'):
        raise ValueError('resident parallel native pixel decode required')
    if state and (state.failed or state.owner is not owner or state.binding != binding or
                  state.stream.closed or state.stream.waves != waves or
                  owner.emittedF != state.stream.emitted_frames):
        raise ValueError('persistent decoder binding/state drift')
    return owner, binding, s, count, (c, h, width), waves, result.conversation_final


def decode(stage, batch, args, context):
    """Decode committed global suffix, ACK actual counts only after common success.

    Private factories permit CPU protocol testing without replacing native math
    in production. Source-terminal flags deliberately have no flush semantics.
    """
    state = getattr(stage, '_h3_conversation_decoder', None)
    ops = state.ops if state else getattr(stage, '_h3_conversation_tensor_ops', _TensorOps)(stage.vae)
    error = None
    try:
        owner, binding, suffix, count, geometry, waves, final = _validate(state, batch, args, context)
        if state and state.stream.native is not stage.vae:
            raise ValueError('resident VAE changed')
        if state is None:
            factory = getattr(stage, '_h3_conversation_stream_factory', FastVideoDecodeStream)
            state = _Decoder(owner, binding, factory(stage.vae, parallel=True, waves=waves), ops)
            stage._h3_conversation_decoder = state
        z = ops.prepare(stage.vae, suffix.video, count, geometry)
        chunks = []
        offset = state.stream.emitted_frames
    except Exception as exc:
        error = exc
    try:
        ops.common(error)  # unpack/copy/shape/init errors before native collectives
        stream = state.stream
        with ops.inference(), ops.autocast():
            start = 0
            while start < count:
                error = None
                try:
                    retained = 0 if stream.buffer is None else stream.buffer.shape[2]
                    end = min(count, start+(12 if waves else 7)-retained)
                    piece = z[:, :, start:end]
                except Exception as exc:
                    error = exc
                ops.common(error)
                decoded = stream.push(piece, final=final and end == count)
                error = None
                try:
                    for chunk in decoded:
                        if ops.rank == 0:
                            chunks.append(ops.pixels(stage.vae, chunk))
                except Exception as exc:
                    error = exc
                ops.common(error)  # no next wave after peer pixel/copy failure
                start = end
        error = None
        try:
            expected = owner.committedF if final else owner.committedF-5
            if stream.received != video_t(owner.committedF) or stream.emitted_frames != expected or stream.closed != final:
                raise ValueError('actual cumulative native decode mismatch')
            if not final and (stream.buffer is None or stream.buffer.shape[2] != 2):
                raise ValueError('native lookahead lost')
            output = ops.output(chunks)
            if ops.rank == 0 and output.shape[2] != stream.emitted_frames-offset:
                raise ValueError('actual output count mismatch')
            from .late_listening_diagnostic import capture
            capture(stage, batch, context, z, output, final, ops)
        except Exception as exc:
            error = exc
        ops.common(error)  # includes output allocation, before either rank ACKs
        owner.ack_decoded_count(stream.emitted_frames, final_flush=final)
        state.sequence += 1
        batch.output = output
        # The shared post-decode hook persists this scalar receipt; a conversation
        # decode still emits the F/F-5 count and the persistent lookahead state.
        from .fast_continuation_decode import RECEIPT_KEY
        batch.extra[RECEIPT_KEY] = dict(
            frame_offset=offset, frame_count=stream.emitted_frames-offset,
            latent_received=stream.received,
            retained_latents=0 if stream.buffer is None else int(stream.buffer.shape[2]),
            retained_overlap_frames=0 if stream.overlap is None else int(stream.overlap.shape[2]),
            max_clip_latents=7, two_latent_lookahead=2, bounded_buffers=True,
            final=final, causal=False,
            execution_policy=('two-rank-two-clip-waves' if waves else
                              'two-rank-alternating-native-clips'),
            parallel_preserved=True, participation=True, output_worker=False,
            original_gather_throughput_verified=False, speed_verified=False,
            parity_verified=False, conversation_owner=True,
            committed_frames=owner.committedF, emitted_frames=stream.emitted_frames)
        return batch
    except BaseException:
        abandon(stage)
        raise
