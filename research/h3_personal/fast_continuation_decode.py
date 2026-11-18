# Temporal protocol derived from the Apache-2.0 MiniMax / HuggingFace decoder.
"""Opt-in suffix-only FastVideo decode, no model lifecycle or audio operations.

The video stage's early return calls decode_suffix only when _h3_continuation
exists. It returns the ForwardBatch (None when inactive). The trusted shared
context contains manifest, owner Session, token WindowInput and committed suffix
Suffix. Keep this context through video decode; audio bypass is the caller's
responsibility. Output is CPU uint8 NCTHW, never cumulative video.

Parallel decode preserves two-rank participation via alternating native clips,
not the original gather scheduling/throughput. Both ranks retain temporal state.
The default process group and its finite timeout must be configured externally.
The existing warm VAE must already be resident; CPU offload is unsupported.
"""
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path

from .incremental_decode import _DecodeStream
from .fast_continuation import Session, WindowInput, Suffix, PACKING_SHA
from .continuation_plan import video_t, integer

FAST_VAE_SHA = 'e3cec57d51623c8343c3bc25160e863d253997fb4158aa283bba874154f70fa1'
KEY = '_h3_continuation'
RECEIPT_KEY = '_h3_continuation_decode'


def verify_fast_source(vae):
    path = Path(inspect.getfile(type(vae))).resolve()
    if (tuple(path.parts[-4:]) != ('fastvideo', 'models', 'vaes', 'minimax_h3_video.py') or
            hashlib.sha256(path.read_bytes()).hexdigest() != FAST_VAE_SHA):
        raise ValueError('pinned ba371e33 FastVideo video VAE source required')
    # Reject instance overrides / mixed decoder implementations, not just a
    # class whose original file happens to have the expected SHA.
    for name in ('_decode_clip', '_blend', 'denormalize_latents', 'denormalize_pixels'):
        method = getattr(vae, name)
        original = getattr(type(vae), name)
        if (getattr(method, '__func__', method) is not original or
                Path(inspect.getfile(original)).resolve() != path):
            raise ValueError('mixed or overridden native VAE method')


class _TwoRankVAE:
    """Delegate native math; exchange bounded tensor headers, never objects.

    Exactly one original _decode_clip call per clip. Owner errors are published
    before either rank raises. Process death/collective failure still relies on
    the caller-configured distributed timeout; no retry or group lifecycle here.
    """
    def __init__(self, native):
        import torch
        self.native = native
        self.torch = torch
        self.dist = torch.distributed
        self.index = 0
        self._check_group()
        ratio = native.spatial_compression_ratio
        if type(ratio) is not int or ratio <= 0:
            raise ValueError('positive native spatial ratio required')
        self.ratio = ratio

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _check_group(self):
        d = self.dist
        if not d.is_available() or not d.is_initialized() or d.get_world_size() != 2:
            raise ValueError('initialized default two-rank process group required')
        rank = d.get_rank()
        if rank not in (0, 1):
            raise ValueError('global rank 0/1 required')
        return rank

    def _agree(self, values, device):
        t = self.torch.tensor(values, dtype=self.torch.int64, device=device)
        peers = [self.torch.empty_like(t) for _ in range(2)]
        self.dist.all_gather(peers, t)
        if any(p.tolist() != values for p in peers):
            raise ValueError('two-rank decode request/order/geometry drift')

    def _decode_clip(self, z):
        torch, d = self.torch, self.dist
        rank = self._check_group()
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        if (z.ndim != 5 or tuple(z.shape[:3]) != (1, 24, 7) or
                any(type(n) is not int or n <= 0 for n in z.shape) or
                z.dtype not in dtypes or z.device != self.native.latents_mean.device):
            raise ValueError('native single-batch 24-channel seven-latent clip required')
        backend = str(d.get_backend()).lower()
        if ((backend == 'nccl' and z.device.type != 'cuda') or
                backend not in ('nccl', 'gloo')):
            raise ValueError('supported NCCL/CUDA or Gloo tensor group required')
        # Same fixed-size collective on both ranks even when clip indices differ.
        self._agree([self.index, *z.shape, dtypes.index(z.dtype), self.ratio], z.device)
        owner = self.index % 2
        shape = (1, 3, 28, z.shape[3]*self.ratio, z.shape[4]*self.ratio)
        clip = None
        failure = None
        header = [0, self.index, 0, *shape]
        if rank == owner:
            try:
                clip = self.native._decode_clip(z)
                if (not isinstance(clip, torch.Tensor) or tuple(clip.shape) != shape or
                        clip.dtype not in dtypes or clip.device != z.device):
                    raise ValueError('native decoded clip shape/dtype/device mismatch')
                clip = clip.contiguous()
                header = [1, self.index, dtypes.index(clip.dtype), *shape]
            except BaseException as exc:
                failure = exc
        wire = torch.tensor(header, dtype=torch.int64, device=z.device)
        d.broadcast(wire, src=owner)
        header = wire.tolist()
        if (len(header) != 8 or header[1] != self.index or tuple(header[3:]) != shape or
                header[0] not in (0, 1) or header[2] not in range(3)):
            raise ValueError('invalid bounded native clip header')
        if not header[0]:
            raise RuntimeError(f'native clip {self.index} owner {owner} failed') from failure
        # Coordinate recoverable receiver allocation errors before payload broadcast.
        ready = 1
        try:
            if rank != owner:
                clip = torch.empty(shape, dtype=dtypes[header[2]], device=z.device)
        except BaseException as exc:
            ready, failure = 0, exc
        status = torch.tensor([ready], dtype=torch.int64, device=z.device)
        statuses = [torch.empty_like(status) for _ in range(2)]
        d.all_gather(statuses, status)
        if any(s.tolist() != [1] for s in statuses):
            raise RuntimeError('two-rank native clip payload allocation failed') from failure
        d.broadcast(clip, src=owner)
        self.index += 1
        return clip


class _TwoRankWaveVAE(_TwoRankVAE):
    """Two native forwards in flight, then ordered bounded tensor exchange."""
    def __init__(self, native):
        super().__init__(native)
        self.current = None
        self.queued = []

    def clear_decode_wave(self):
        self.current = None
        self.queued.clear()

    def prepare_decode_wave(self, zbuffer):
        torch, d = self.torch, self.dist
        rank = self._check_group()
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        count = 2 if zbuffer.ndim == 5 and zbuffer.shape[2] >= 12 else 1
        local = None
        failure = None
        valid = (zbuffer.ndim == 5 and tuple(zbuffer.shape[:2]) == (1, 24) and
                 zbuffer.shape[2] >= 7 and zbuffer.dtype in dtypes and
                 zbuffer.device == self.native.latents_mean.device and
                 all(type(n) is int and n > 0 for n in zbuffer.shape) and
                 str(d.get_backend()).lower() in ('gloo', 'nccl') and
                 (str(d.get_backend()).lower() != 'nccl' or zbuffer.device.type == 'cuda'))
        shape = ((1, 3, 28, zbuffer.shape[3]*self.ratio, zbuffer.shape[4]*self.ratio)
                 if zbuffer.ndim == 5 else (1, 3, 28, 1, 1))
        # Every prepare, including cached clips, uses the same bounded rendezvous
        # before choosing a collective path. Native forwards remain independent.
        descriptor = [int(valid), self.index, int(self.current is not None),
                      len(self.queued), count, zbuffer.ndim,
                      *(list(zbuffer.shape) if zbuffer.ndim == 5 else [0]*5),
                      dtypes.index(zbuffer.dtype) if zbuffer.dtype in dtypes else -1,
                      self.ratio]
        try:
            self._agree(descriptor, self.native.latents_mean.device)
            if not valid or self.current is not None or len(self.queued) > 1:
                raise ValueError('invalid wave latent descriptor/state')
            if self.queued:
                self.current = self.queued.pop(0)
                return
        except BaseException:
            self.clear_decode_wave()
            raise
        try:
            for j in range(count):
                if (self.index+j) % 2 == rank:
                    local = self.native._decode_clip(zbuffer[:, :, j*5:j*5+7])
                    if (not isinstance(local, torch.Tensor) or tuple(local.shape) != shape or
                            local.dtype not in dtypes or local.device != zbuffer.device):
                        raise ValueError('native decoded clip shape/dtype/device mismatch')
                    local = local.contiguous().clone()
        except BaseException as exc:
            failure = exc
        try:
            status = torch.tensor([int(failure is None)], dtype=torch.int64, device=zbuffer.device)
            statuses = [torch.empty_like(status) for _ in range(2)]
            d.all_gather(statuses, status)
            if any(s.tolist() != [1] for s in statuses):
                raise RuntimeError('two-rank native wave forward/descriptor failed') from failure
            self._agree([self.index, count, *zbuffer.shape, dtypes.index(zbuffer.dtype), self.ratio],
                        zbuffer.device)
            for j in range(count):
                index = self.index+j
                owner = index % 2
                clip = local if rank == owner else None
                wire = torch.tensor([1, index, dtypes.index(local.dtype) if rank == owner else 0,
                                     *shape], dtype=torch.int64, device=zbuffer.device)
                d.broadcast(wire, src=owner)
                header = wire.tolist()
                ready = int(len(header) == 8 and header[:2] == [1, index] and
                            header[2] in range(3) and tuple(header[3:]) == shape)
                try:
                    if ready and rank != owner:
                        clip = torch.empty(shape, dtype=dtypes[header[2]], device=zbuffer.device)
                except BaseException as exc:
                    ready, failure = 0, exc
                status = torch.tensor([ready], dtype=torch.int64, device=zbuffer.device)
                statuses = [torch.empty_like(status) for _ in range(2)]
                d.all_gather(statuses, status)
                if any(s.tolist() != [1] for s in statuses):
                    raise RuntimeError('two-rank native wave header/allocation failed') from failure
                d.broadcast(clip, src=owner)
                if j == 0:
                    self.current = clip
                else:
                    self.queued.append(clip)
        except BaseException:
            self.clear_decode_wave()
            raise

    def _decode_clip(self, z):
        if self.current is None:
            raise ValueError('prepare decode wave required')
        clip, self.current = self.current, None
        self.index += 1
        return clip


class FastVideoDecodeStream(_DecodeStream):
    def __init__(self, vae, *, parallel=False, waves=False):
        if waves and not parallel:
            raise ValueError('wave decode requires parallel decode')
        verify_fast_source(vae)
        # Validate native geometry before constructing the distributed delegate.
        super().__init__(vae)
        self.native = vae
        self.parallel = parallel
        self.waves = waves
        if parallel:
            self.vae = (_TwoRankWaveVAE if waves else _TwoRankVAE)(vae)


def _unpack(rows, count, height, width, channels):
    # Use the existing channel-major packing implementation, never a new
    # sampler or an unverified replacement for the native helper.
    from fastvideo.pipelines.basic.minimax_h3 import packing
    if hashlib.sha256(Path(packing.__file__).read_bytes()).hexdigest() != PACKING_SHA:
        raise ValueError('pinned FastVideo packing source required')
    return packing.unpatchify_video_tokens(rows, count, height, width, channels, (1, 2, 2))


@dataclass
class _Turn:
    owner: Session
    total: int
    geometry: tuple
    stream: FastVideoDecodeStream
    index: int = 0
    failed: bool = False


def _validate(context, batch):
    if not isinstance(context, dict):
        raise ValueError('continuation dict required')
    manifest = context.get('manifest')
    owner, token, suffix = (context.get(k) for k in ('owner', 'token', 'suffix'))
    if (not isinstance(manifest, dict) or not isinstance(owner, Session) or
            not isinstance(token, WindowInput) or not isinstance(suffix, Suffix)):
        raise ValueError('committed Session/WindowInput/Suffix context required')
    index = integer(manifest.get('index'), 'index')
    total = integer(manifest.get('total_frames'), 'total_frames', 5)
    video_t(total)
    final = manifest.get('final')
    if type(final) is not bool or index >= len(owner.windows):
        raise ValueError('invalid final/index')
    if (manifest.get('session_id') != owner.session_id or
            token.session_id != owner.session_id or suffix.session_id != owner.session_id or
            token.index != index or suffix.index != index or token.window != owner.windows[index] or
            total != owner._total or final != (index == len(owner.windows)-1) or
            owner._index != index+1 or owner.pending_window is not None or
            owner.faulted or owner._closed or token.attempt != owner._attempt or
            owner.committed_shape != (suffix.cumulative_video_rows, suffix.cumulative_audio_t)):
        raise ValueError('stale, foreign or uncommitted continuation context')
    raw = batch.raw_latent_shape
    if (raw is None or len(raw) != 5 or any(type(n) is not int or n <= 0 for n in raw)):
        raise ValueError('positive native NCTHW raw geometry required')
    n, channels, local_frames, height, width = raw
    if local_frames != video_t(token.window.frames):
        raise ValueError('raw local latent time disagrees with committed window')
    if n != 1 or channels != 24 or height % 2 or width % 2:
        raise ValueError('native single-batch 24-channel even canvas required')
    per_frame = height//2 * (width//2)
    if per_frame != owner._per_frame:
        raise ValueError('session canvas drift')
    rows = suffix.cumulative_video_rows-suffix.video_row_offset
    count = video_t(token.window.frames)-token.window.guide_video_t
    if (rows != count*per_frame or suffix.video_row_offset != token.video_row_offset or
            suffix.cumulative_video_rows != video_t(token.window.end)*per_frame):
        raise ValueError('suffix append gap/overlap')
    owner._check(suffix.video, (rows, 96))
    return owner, index, total, final, suffix, count, (channels, height, width)


def decode_suffix(stage, batch, args):
    """Decode ONLY a committed clean suffix; serialize calls on the stage.

    Any decode failure poisons this turn (no overlapping replay/retry). Only a
    completed final grid permits replacing the stage's bounded per-turn state.
    Receipt contains private scalar metadata only, no media or audio state.
    """
    context = batch.extra.get(KEY)
    if context is None:
        return None
    if 'conversation_owner' in context:
        from .conversation_decode import decode
        return decode(stage, batch, args, context)
    parallel = getattr(args, 'vae_parallel_decode', False)
    manifest = context.get('manifest', {})
    efficiency = manifest.get('efficiency') or {}
    schedule = efficiency.get('decode_schedule')
    if schedule is not None and schedule not in ('alternating-v1', 'two-clip-waves-v1'):
        raise ValueError('invalid manifest decode schedule')
    waves = (schedule == 'two-clip-waves-v1' if schedule is not None else
             os.environ.get('H3_SMD_PARALLEL_DECODE_WAVES') == '1')
    capture = getattr(stage, '_h3_decode_parity_capture', None)
    parallel_acceptance = manifest.get('parallel_acceptance') is True
    if parallel_acceptance and capture is not None and capture.session != manifest.get('session_id'):
        raise ValueError('parallel acceptance is single-turn; never silently reuse capture')
    if parallel_acceptance:
        if not parallel:
            raise ValueError('parallel acceptance requires parallel decode')
        waves = False  # First acceptance turn is the alternating baseline.
    if waves and not parallel:
        raise ValueError('wave decode requires parallel decode')
    if type(parallel) is not bool:
        raise ValueError('explicit boolean parallel decode policy required')
    if getattr(args, 'vae_cpu_offload', False):
        raise ValueError('continuation requires already-resident VAE; no model load/offload')
    if (getattr(args, 'video_decode_backend', 'h3-vae') != 'h3-vae' or
            getattr(args, 'output_type', 'pil') == 'latent'):
        raise ValueError('native pixel decode required')
    owner, index, total, final, suffix, count, geometry = _validate(context, batch)
    import torch
    from fastvideo.distributed import model_parallel_is_initialized
    from fastvideo.pipelines.basic.minimax_h3.encoder_split import h3_is_output_worker
    # Parallel mode MUST enter on both ranks, regardless of encoder ownership.
    if parallel:
        d = torch.distributed
        if not d.is_available() or not d.is_initialized() or d.get_world_size() != 2:
            raise ValueError('initialized default two-rank process group required')
        output_worker = d.get_rank() == 0
    else:
        output_worker = not model_parallel_is_initialized() or h3_is_output_worker(args)
    if not parallel and not output_worker:
        batch.output = torch.empty((0, 3, 0, 0, 0), device='cpu', dtype=torch.float32)
        return batch
    turn = getattr(stage, '_h3_continuation_decoder', None)
    if turn is None or turn.owner is not owner:
        if index != 0 or (turn is not None and (not turn.stream.closed or turn.failed or
                                               turn.owner.session_id == owner.session_id)):
            raise ValueError('new turn requires previous complete final grid and fresh session')
        if stage.vae is None:
            raise ValueError('already-resident native video VAE required')
        turn = _Turn(owner, total, geometry, FastVideoDecodeStream(stage.vae, parallel=parallel, waves=waves))
        stage._h3_continuation_decoder = turn
    if (turn.failed or turn.index != index or turn.total != total or turn.geometry != geometry or
            turn.stream.native is not stage.vae or turn.stream.parallel != parallel or turn.stream.waves != waves or turn.stream.closed):
        if turn.stream.waves:
            turn.failed = True
            turn.stream.closed = True
            turn.stream.clear()
        raise ValueError('decoder turn/index/geometry drift or closed stream')
    per_frame = geometry[1]//2 * (geometry[2]//2)
    stream = turn.stream
    if stream.received*per_frame != suffix.video_row_offset:
        raise ValueError('decoder suffix offset drift')
    offset = stream.emitted_frames
    try:
        with torch.inference_mode():
            z = _unpack(suffix.video, count, geometry[1], geometry[2], geometry[0])
            parity_capture = None
            if parallel_acceptance or os.environ.get('H3_SMD_DECODE_PARITY') == '1':
                from .decode_parity_capture import record_suffix
                parity_capture = record_suffix(stage, context, z)
            device = stage.vae.latents_mean.device
            z = stage.vae.denormalize_latents(z.to(device=device, dtype=torch.float32))
            chunks = []
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                start = 0
                while start < count:
                    retained = 0 if stream.buffer is None else stream.buffer.shape[2]
                    end = min(count, start+(12 if waves else 7)-retained)
                    for chunk in stream.push(z[:, :, start:end], final=final and end == count):
                        if output_worker:
                            pixels = stage.vae.denormalize_pixels(chunk.float()).clamp_(0, 1)
                            chunks.append(pixels.mul_(255).clamp_(0, 255).to(torch.uint8).cpu())
                    start = end
            expected = 17*((stream.received-2)//5)+(5 if final else 0)
            if (stream.received != video_t(owner.windows[index].end) or
                    stream.emitted_frames != expected or
                    (final and (stream.received != video_t(total) or expected != total or not stream.closed)) or
                    (not final and (stream.buffer is None or stream.buffer.shape[2] != 2))):
                raise ValueError('cumulative emitted frames/final native grid mismatch')
            output = (torch.cat(chunks, dim=2) if output_worker else
                      torch.empty((0, 3, 0, 0, 0), device='cpu', dtype=torch.float32))
            if output_worker and output.shape[2] != stream.emitted_frames-offset:
                raise ValueError('suffix frame count mismatch')
        if final and parity_capture is not None:
            from .decode_parity_capture import finish_parity
            finish_parity(stage, parity_capture)
        turn.index += 1
        batch.output = output
        batch.extra[RECEIPT_KEY] = dict(
            frame_offset=offset, frame_count=stream.emitted_frames-offset, latent_received=stream.received,
            retained_latents=0 if stream.buffer is None else stream.buffer.shape[2],
            retained_overlap_frames=0 if stream.overlap is None else stream.overlap.shape[2],
            max_clip_latents=7, two_latent_lookahead=2, bounded_buffers=True,
            final=final, causal=False,
            execution_policy=('two-rank-two-clip-waves' if waves else
                              'two-rank-alternating-native-clips' if parallel else 'explicit-serial-output-worker'),
            parallel_preserved=parallel, participation=parallel, output_worker=output_worker,
            original_gather_throughput_verified=False, speed_verified=False, parity_verified=False)
        if waves:
            batch.extra[RECEIPT_KEY].update(decode_schedule='two-clip-waves-v1',
                                           max_buffer_latents=12, max_queued_clips=1)
        return batch
    except BaseException:
        capture = getattr(stage, '_h3_decode_parity_capture', None)
        if capture is not None:
            capture.fault()
        turn.failed = True
        stream.closed = True
        stream.clear()
        raise
