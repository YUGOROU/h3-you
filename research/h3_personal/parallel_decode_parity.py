"""In-memory alternating versus waves diagnostic, not native gather acceptance.

Torch is lazy. Caller owns an initialized two-rank group AND a finite external
process/provider deadline: phase agreements cannot rescue a failed VAE collective.
No tensors or media are saved.
"""
from .decode_parity import _validate
from .fast_continuation_decode import FAST_VAE_SHA, FastVideoDecodeStream, verify_fast_source


def compare_decode(vae, normalized_latents, *, arrivals=(37, 30, 30), atol=0.0):
    """Compare identical denormalized clones under the same inference/autocast."""
    import torch
    d = torch.distributed
    if not d.is_available() or not d.is_initialized() or d.get_world_size() != 2:
        raise ValueError('initialized default two-rank process group required; finite external deadline required')
    device = vae.latents_mean.device

    def phase(name, operation):
        result, error = None, None
        try:
            result = operation()
        except Exception as exc:
            error = exc
        ok = torch.tensor(int(error is None), dtype=torch.int32, device=device)
        d.all_reduce(ok, op=d.ReduceOp.MIN)
        if not bool(ok.item()):
            raise RuntimeError(f'parallel parity {name} rejected') from error
        return result

    def finite(x):
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError('nonfinite diagnostic tensor')

    def unchanged(a, b):
        if not torch.equal(a, b):
            raise ValueError('diagnostic input mutated')

    def metrics(a, b):
        delta = (a.float() - b.float()).abs()
        finite(delta)
        frames = delta.permute(2, 0, 1, 3, 4).reshape(delta.shape[2], -1)
        maximum = float(delta.max().item())
        return dict(max=maximum, mean=float(delta.mean().item()),
                    exact=bool(torch.equal(a, b)), within_atol=maximum <= atol,
                    per_frame_max=frames.max(dim=1).values.tolist(),
                    per_frame_mean=frames.mean(dim=1).tolist())

    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        def prepare():
            counts = tuple(arrivals)
            _validate(tuple(normalized_latents.shape), counts, atol)
            verify_fast_source(vae)
            if normalized_latents.device != device:
                raise ValueError('input must reside on VAE device')
            ratio = vae.spatial_compression_ratio
            if type(ratio) is not int or ratio <= 0:
                raise ValueError('positive spatial compression ratio required')
            finite(normalized_latents)
            snapshot = normalized_latents.clone()
            source = vae.denormalize_latents(normalized_latents.to(dtype=torch.float32).clone())
            if source.dtype != torch.float32 or source.shape != snapshot.shape:
                raise ValueError('float32 denormalized NCTHW required')
            finite(source)
            saved = source.clone()
            inputs = (source.clone(), source.clone())
            streams = tuple(FastVideoDecodeStream(vae, parallel=True, waves=w) for w in (False, True))
            frames = 17*((source.shape[2]-2)//5)+5
            expected = (1, 3, frames, source.shape[3]*ratio, source.shape[4]*ratio)
            return counts, snapshot, source, saved, inputs, streams, expected
        counts, snapshot, source, saved, inputs, streams, expected = phase('input preparation', prepare)

        def run(stream, value):
            if stream.closed or stream.received or stream.emitted_frames:
                raise ValueError('fresh parallel stream required')
            chunks, emitted, offset = [], [], 0
            for i, count in enumerate(counts):
                before = stream.emitted_frames
                for chunk in stream.push(value[:, :, offset:offset+count], final=i == len(counts)-1):
                    chunks.append(chunk.clone())
                offset += count
                emitted.append(stream.emitted_frames-before)
                if stream.closed != (i == len(counts)-1) or stream.received != offset:
                    raise ValueError('stream lifecycle mismatch')
            output = torch.cat(chunks, dim=2)
            expected_emitted = [17*((n-2)//5) for n in (sum(counts[:i+1]) for i in range(len(counts)))]
            expected_emitted[-1] += 5
            expected_emitted = [n-(expected_emitted[i-1] if i else 0) for i, n in enumerate(expected_emitted)]
            if (tuple(output.shape) != expected or emitted != expected_emitted or
                    stream.emitted_frames != expected[2] or stream.buffer is not None or stream.overlap is not None):
                raise ValueError('decoded geometry or exactly-once frame count mismatch')
            finite(output)
            unchanged(value, saved)
            return output, emitted
        reference, reference_emitted = phase('alternating completion', lambda: run(streams[0], inputs[0]))
        candidate, emitted = phase('waves completion', lambda: run(streams[1], inputs[1]))

        def measure():
            raw = metrics(reference, candidate)
            def pixels(x):
                p = vae.denormalize_pixels(x.float().clone())
                finite(p)
                return p.clamp(0, 1)
            a, b = pixels(reference), pixels(candidate)
            pixel_float = metrics(a, b)
            pixel = metrics(a.mul(255).to(torch.uint8), b.mul(255).to(torch.uint8))
            unchanged(normalized_latents, snapshot)
            unchanged(source, saved)
            for value in inputs:
                unchanged(value, saved)
            return dict(schema='h3-parallel-decode-parity-v1', diagnostic_only=True,
                        full_native_allocation=True, quality_verified=False, realtime_verified=False,
                        source_sha256=FAST_VAE_SHA, source_revision='ba371e33ed080e51f1884ae166baf7c4447a66fe',
                        serial=False, reference_schedule='alternating-v1', candidate_schedule='two-clip-waves-v1',
                        comparison_scope='same cumulative committed timeline; no independent-window resets',
                        native_parallel_verified=False, learned_vae_execution_asserted=False,
                        normalized_shape=list(snapshot.shape), decoded_shape=list(expected),
                        arrivals=list(counts), emitted_counts=emitted, reference_emitted_counts=reference_emitted,
                        closed=True, frames_exactly_once=True, inputs_unchanged=True, atol=float(atol),
                        input_dtype=str(snapshot.dtype), denormalized_dtype='torch.float32',
                        cuda_fp16_autocast=device.type == 'cuda', raw_float=raw,
                        denormalized_pixels=pixel_float, uint8=pixel,
                        parity=raw['within_atol'] and pixel_float['within_atol'] and pixel['within_atol'])
        return phase('metrics', measure)
