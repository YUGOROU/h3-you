"""Offline serial decode diagnostic; allocates full native RGB, NOT production.

No loading, launch, media export, quality or realtime acceptance is performed.
The caller owns the already-resident pinned VAE and exclusive serial access.
"""
import inspect
import math
from pathlib import Path

from .fast_continuation_decode import (
    FAST_VAE_SHA, FastVideoDecodeStream, verify_fast_source,
)


def _verify_native(vae):
    verify_fast_source(vae)
    path = Path(inspect.getfile(type(vae))).resolve()
    for name in ('_decode', '_decode_chunks', '_temporal_decode_plan'):
        method = getattr(vae, name, None)
        original = vars(type(vae)).get(name)
        if (original is None or getattr(method, '__self__', None) is not vae or
                getattr(method, '__func__', None) is not original or
                Path(inspect.getfile(original)).resolve() != path):
            raise ValueError('pinned bound native serial methods required')


def _validate(shape, arrivals, atol):
    if (len(shape) != 5 or any(type(n) is not int or n <= 0 for n in shape) or
            shape[:2] != (1, 24) or shape[3] % 2 or shape[4] % 2 or
            not 7 <= shape[2] <= 127 or shape[2] % 5 != 2 or
            shape[3] * shape[4] > 56 * 32):
        raise ValueError('bounded single batch 24-channel even canvas, T=5n+2 <=127 required')
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError('finite nonnegative atol required')
    total = 0
    for n in arrivals:
        if type(n) is not int or not 1 <= n <= 37:
            raise ValueError('positive arrivals <=37 required')
        total += n
        if total < 7 or total % 5 != 2:
            raise ValueError('each cumulative arrival must be 5n+2')
    if not arrivals or total != shape[2]:
        raise ValueError('arrivals must sum to T')


def compare_decode(vae, normalized_latents, *, arrivals=(37, 30, 30), atol=0.0):
    """Return scalar-only pre-codec parity evidence, never tensors or media."""
    arrivals = tuple(arrivals)
    _validate(tuple(normalized_latents.shape), arrivals, atol)
    _verify_native(vae)
    import torch

    def finite(x):
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError('nonfinite diagnostic tensor')

    def unchanged(a, b):
        if not torch.equal(a, b):
            raise ValueError('diagnostic input mutated')

    def metrics(a, b):
        d = (a.float() - b.float()).abs()
        finite(d)
        frames = d.permute(2, 0, 1, 3, 4).reshape(d.shape[2], -1)
        maximum = float(d.max().item())
        return dict(max=maximum, mean=float(d.mean().item()),
                    exact=bool(torch.equal(a, b)), within_atol=maximum <= atol,
                    per_frame_max=frames.max(dim=1).values.tolist(),
                    per_frame_mean=frames.mean(dim=1).tolist())

    device = normalized_latents.device
    ratio = vae.spatial_compression_ratio
    if type(ratio) is not int or ratio <= 0:
        raise ValueError('positive spatial compression ratio required')
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        finite(normalized_latents)
        snapshot = normalized_latents.clone()
        source = vae.denormalize_latents(normalized_latents.to(dtype=torch.float32).clone())
        if source.dtype != torch.float32 or source.shape != snapshot.shape:
            raise ValueError('float32 denormalized NCTHW required')
        finite(source)
        saved = source.clone()
        native_input, candidate_input = source.clone(), source.clone()
        native = vae._decode(native_input).clone()
        unchanged(native_input, saved)
        stream = FastVideoDecodeStream(vae, parallel=False)
        if stream.closed or stream.received or stream.emitted_frames:
            raise ValueError('fresh serial stream required')
        chunks, emitted, offset = [], [], 0
        for i, count in enumerate(arrivals):
            before = stream.emitted_frames
            for chunk in stream.push(candidate_input[:, :, offset:offset+count],
                                     final=i == len(arrivals)-1):
                chunks.append(chunk.clone())
            offset += count
            emitted.append(stream.emitted_frames-before)
            if stream.closed != (i == len(arrivals)-1) or stream.received != offset:
                raise ValueError('stream lifecycle mismatch')
        frames = 17*((source.shape[2]-2)//5)+5
        expected = (1, 3, frames, source.shape[3]*ratio, source.shape[4]*ratio)
        candidate = torch.cat(chunks, dim=2)
        if (tuple(native.shape) != expected or tuple(candidate.shape) != expected or
                stream.emitted_frames != frames or sum(emitted) != frames or
                stream.buffer is not None or stream.overlap is not None):
            raise ValueError('decoded geometry or exactly-once frame count mismatch')
        finite(native)
        finite(candidate)
        raw = metrics(native, candidate)
        def pixels(x):
            p = vae.denormalize_pixels(x.float().clone())
            finite(p)
            return p.clamp(0, 1)
        native_pixels, candidate_pixels = pixels(native), pixels(candidate)
        pixel_float = metrics(native_pixels, candidate_pixels)
        pixel = metrics(native_pixels.mul(255).to(torch.uint8),
                        candidate_pixels.mul(255).to(torch.uint8))
        unchanged(normalized_latents, snapshot)
        unchanged(source, saved)
        unchanged(candidate_input, saved)
    return dict(schema='h3-decode-parity-v1', diagnostic_only=True,
                full_native_allocation=True, quality_verified=False, realtime_verified=False,
                source_sha256=FAST_VAE_SHA, source_revision='ba371e33ed080e51f1884ae166baf7c4447a66fe', serial=True,
                comparison_scope='same cumulative committed timeline; no independent-window resets',
                native_parallel_verified=False, learned_vae_execution_asserted=False,
                normalized_shape=list(snapshot.shape), decoded_shape=list(expected),
                arrivals=list(arrivals), emitted_counts=emitted, closed=True,
                frames_exactly_once=True, inputs_unchanged=True, atol=float(atol),
                input_dtype=str(snapshot.dtype), denormalized_dtype='torch.float32',
                cuda_fp16_autocast=device.type == 'cuda', raw_float=raw,
                denormalized_pixels=pixel_float, uint8=pixel,
                parity=raw['within_atol'] and pixel_float['within_atol'] and pixel['within_atol'])
