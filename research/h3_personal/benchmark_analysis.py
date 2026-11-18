"""Strict, CPU-only analysis of resident batch reports; never launches compute.

Phase subtotals are nested, not additive with generation/decode totals.
Observed speed is not causal streaming or subjective identity acceptance.
"""
import math
import re


def seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'invalid timing: {label}')
    return float(value)


def analyze_batch(report):
    if report.get('schema') != 'h3-same-condition-resident-batch-v1':
        raise ValueError('unsupported batch schema')
    results = report.get('results')
    if not isinstance(results, list) or len(results) != 3:
        raise ValueError('three fixed-condition cases required')
    rows = []
    hashes = []
    durations = []
    for index, result in enumerate(results):
        if result.get('case') != index:
            raise ValueError('case order differs')
        digest = result.get('latent_file_sha256', '')
        if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
            raise ValueError('invalid latent digest')
        hashes.append(digest)
        gen, dec = result['generation'], result['decode']
        if gen.get('schema') != 'h3-inference-profile-v1' or gen.get('phase') != 'generate' or dec.get('schema') != 'h3-inference-profile-v1' or dec.get('phase') != 'decode':
            raise ValueError('profile schema/phase differs')
        if gen.get('precision') != 'bf16' or gen.get('quantization') != 'none' or gen.get('denoiser_forwards') != 49:
            raise ValueError('baseline BF16/49 forwards required')
        duration = seconds(gen['video_seconds'], 'video_seconds')
        if duration <= 0:
            raise ValueError('positive video duration required')
        durations.append(duration)
        generation = seconds(gen['generation_seconds'], 'generation')
        decode = seconds(dec['decode_seconds'], 'decode')
        denoise = seconds(gen['denoise_seconds'], 'denoise')
        phases = {k: seconds(v, k) for k, v in gen['phases'].items()}
        vae_load = seconds(dec['vae_load_seconds'], 'vae_load')
        vae_decode = seconds(dec['vae_decode_seconds'], 'vae_decode')
        gen_remaining = generation - denoise - sum(phases.values())
        dec_remaining = decode - vae_load - vae_decode
        if min(gen_remaining, dec_remaining) < -0.001:
            raise ValueError('nested phases exceed total')
        for key in ('condition_cache_hit', 'model_resident_hit'):
            if type(gen.get(key)) is not bool:
                raise ValueError('boolean cache/residency evidence required')
        rows.append(dict(case=index, generation_seconds=generation, decode_seconds=decode,
                         render_seconds=generation + decode, denoise_seconds=denoise,
                         generation_phases=phases, generation_other_seconds=max(0, gen_remaining),
                         vae_load_seconds=vae_load, vae_decode_seconds=vae_decode,
                         decode_other_seconds=max(0, dec_remaining),
                         condition_cache_hit=gen['condition_cache_hit'],
                         model_resident_hit=gen['model_resident_hit'],
                         render_realtime_factor=(generation + decode) / duration,
                         denoise_realtime_factor=denoise / duration))
    if len(set(durations)) != 1:
        raise ValueError('geometry duration differs')
    equal = len(set(hashes)) == 1
    if type(report.get('exact_latent_file_bytes_equal')) is not bool or report['exact_latent_file_bytes_equal'] != equal:
        raise ValueError('parity assertion contradicts hashes')
    if rows[-1]['render_seconds'] <= 0:
        raise ValueError('positive render timing required')
    return dict(schema='h3-resident-batch-analysis-v1', results=rows,
                exact_latent_file_bytes_equal=equal,
                cold_to_last_render_speedup=rows[0]['render_seconds'] / rows[-1]['render_seconds'],
                last_denoise_fraction=rows[-1]['denoise_seconds'] / rows[-1]['render_seconds'],
                subjective_identity_verified=False, causal_streaming_verified=False,
                scope='generation plus decode only; excludes preparation, upload, transfer, playback and chat',
                memory_note='Recorded peaks may include resident DiT/prior phases; not additive isolated VAE sizes')
