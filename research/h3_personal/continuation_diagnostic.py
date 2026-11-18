"""Offline preparation only for one fixed, reused-generated-audio diagnostic.

No launch, download, write, playback, permission grant or GPU acceptance. Metadata
validation is not source verification; preflight reads a caller-supplied local WAV
through the existing full-header/PCM/hash planner. No private source is bundled.
"""
from .continuation_plan import plan as window_plan
from .fast_continuation_render import plan as wav_plan
from .live_voice_runtime import POLICY

SOURCE_REPO = 'YUGOROU/h3-fast-ref2va-smd4-gpu-live-4182459cb5'
SOURCE_REVISION = 'ed5df76d9efac41f53cf4ee7dc879b71af66324b'
SOURCE_PATH = 'live/turn-001/reply.wav'
SOURCE_SHA256 = '7fd491b50ac16f7f9c0644795632101613a5b576fbd1e8551f9c80a564f6a585'


def source_contract():
    """Return a fresh fixed provenance declaration, not a download receipt."""
    return dict(repo=SOURCE_REPO, revision=SOURCE_REVISION, path=SOURCE_PATH,
                sha256=SOURCE_SHA256, source_samples=577920,
                source_sample_rate=48000, source_channels=1,
                input_kind='reused generated diagnostic audio', new_speech=False)


def validate_metadata(source, planned, *, timeout_seconds):
    """Validate declarations only; even matching synthetic metadata is unverified.

    Timeout is caller-owned, integer seconds, bounded by one hour. No monetary
    gate is introduced. Provider/license/location/concurrency/budget checks remain
    external prerequisites; this function cannot satisfy any of them.
    """
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise ValueError('caller timeout must be integer seconds in 1..3600')
    expected_source = source_contract()
    if not isinstance(source, dict) or source != expected_source or any(
            type(source[k]) is not type(v) for k, v in expected_source.items()):
        raise ValueError('fixed diagnostic source provenance/metadata mismatch')
    expected = dict(schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
                    sha256=SOURCE_SHA256, source_samples=577920,
                    source_sample_rate=48000, source_channels=1,
                    total_frames=328, window_frames=124, overlap_frames=22,
                    expected_windows=3, sampler='SMD4', fps=24, seed=314159,
                    windows=[w.record() for w in window_plan(328, 124, 22)])
    if not isinstance(planned, dict) or any(
            type(planned.get(k)) is not type(v) or planned[k] != v
            for k, v in expected.items()):
        raise ValueError('fixed three-window diagnostic plan mismatch')
    return dict(schema='h3-continuation-diagnostic-v1', status='planned-only',
                source=expected_source, local_source_verified=False,
                execution_authorized=False, gpu_ready=False,
                quality_verified=False, realtime_verified=False,
                job_count=1, gpu='H200', gpu_count=2,
                timeout_seconds=timeout_seconds, total_frames=328,
                expected_windows=3, suffix_frames=[119, 102, 107],
                sampler='SMD4', forwards_per_window=4, precision='BF16',
                policy=POLICY, fallback_allowed=False,
                planned_tests=['three-window latent-tail continuation',
                               'suffix coverage and original PCM preservation',
                               'artifact continuity/identity/lip-sync review',
                               'unique-suffix generation/decode throughput'])


def preflight(wav, *, timeout_seconds):
    """Read only local WAV; fail closed on actual hash/header/PCM/plan mismatch.

    Verification applies at planning time only, not to a future uploaded file.
    Fixed repository provenance is a declaration, not checked over the network.
    """
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise ValueError('caller timeout must be integer seconds in 1..3600')
    planned = wav_plan(wav, SOURCE_SHA256)
    result = validate_metadata(source_contract(), planned,
                               timeout_seconds=timeout_seconds)
    result['local_source_verified'] = True
    return result
