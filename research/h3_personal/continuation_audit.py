"""Offline, stdlib-only comparison of private independent/continuation evidence.

Returns field names and digests, never configuration values (including prompts).
Does not infer quality, tensor equality, or controlled benchmark equivalence.
"""
import hashlib
import json
from pathlib import Path


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            name = f'{prefix}.{key}' if prefix else key
            if key not in a or key not in b:
                result.append(name)
            else:
                result.extend(_differences(a[key], b[key], name))
        return result
    return [] if type(a) is type(b) and a == b else [prefix]


def _seconds(intervals):
    total = 0.
    for start, end in intervals:
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not 0 <= start <= end < float('inf')):
            raise ValueError('invalid generation interval')
        total += end - start
    if total <= 0 or total == float('inf'):
        raise ValueError('positive finite generation duration required')
    return total


def compare(baseline, continuation):
    """Hash actual local evidence and compare complete specs without leaking values.

    Baseline warm rate excludes its cold first request. Continuation rate uses
    every generated window but only unique emitted frames; never overlap frames.
    Generation intervals include decode, not isolated CUDA denoising.
    """
    roots = [Path(baseline), Path(continuation)]
    profiles, specs, evidence = [], [], []
    for root in roots:
        paths = ['deployment/spec.json', 'deployment/model-manifest.json',
                 'deployment/asset-manifest.json', 'verified-results/profile.json']
        records = {}
        documents = {}
        for relative in paths:
            path = root / relative
            before = _digest(path)
            documents[relative] = json.loads(path.read_text())
            if before != _digest(path):
                raise ValueError('evidence changed while reading')
            records[relative] = before
        evidence.append(records)
        profiles.append(documents['verified-results/profile.json'])
        specs.append(documents['deployment/spec.json'])
    b, c = profiles
    if b.get('independent_ref2va_clips') is not True or c.get('mode') != 'latent-tail-v1':
        raise ValueError('independent baseline and latent-tail continuation required')
    if (b.get('external_audio_conditioning') is not False
            or c.get('speech_pcm_exact') is not True):
        raise ValueError('expected baseline/continuation audio evidence missing')
    intervals = b['generation_intervals']
    if len(intervals) < 2 or b['unique_frames'] != len(intervals) * 124:
        raise ValueError('baseline full independent frame grid differs')
    # Both immutable deployment configs must explicitly agree on the clock.
    fps = specs[0]['fast_config']['fps']
    if type(fps) not in (int, float) or fps != 24 or specs[1]['fast_config']['fps'] != fps:
        raise ValueError('expected explicit shared 24fps clock')
    warm_duration = (len(intervals) - 1) * 124 / fps
    unique_frames = c['unique_frames']
    if type(unique_frames) is not int or unique_frames <= 0:
        raise ValueError('invalid continuation unique frame count')
    seconds = _seconds(c['generation_intervals'])
    return dict(schema='h3-continuation-source-audit-v1', evidence_sha256=evidence,
        spec_difference_fields=_differences(*specs),
        baseline_warm_generation=dict(video_seconds=warm_duration,
            generation_seconds=_seconds(intervals[1:]),
            throughput=warm_duration / _seconds(intervals[1:]), cold_first_excluded=True),
        continuation_generation=dict(unique_video_seconds=unique_frames / fps,
            generation_seconds=seconds, throughput=unique_frames / fps / seconds),
        audio_conditioning_matches=False, controlled_comparison=False,
        generation_includes_decode=True, quality_cause_verified=False,
        derived_adapter_file_hash_matches=(b['derived_adapter_receipt']['derived_sha256'] ==
                                          c['derived_adapter_receipt']['derived_sha256']),
        derived_adapter_tensor_equality='unverified')
