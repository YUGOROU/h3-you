"""Pure CPU analysis; no IO, launch, or measured-wall claims from stage events.

Workloads are mappings with frames, fps, and wall_seconds. Decimal spellings of
finite int/float inputs are interpreted exactly, with Fraction results retained.
"""
from fractions import Fraction
import math


STAGES = ('latent_preparation_stage', 'denoising_stage',
          'video_decoding_stage', 'audio_decoding_stage')
SCHEMA = 'h3-ref2va-warm-stage-diagnostic-v1'


def _number(value, label, *, positive=False):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f'{label}: finite int/float required')
    result = Fraction(str(value))
    if result < 0 or (positive and result == 0):
        raise ValueError(f'{label}: invalid sign')
    return result


def _workload(value):
    if type(value) is not dict or set(value) != {'frames', 'fps', 'wall_seconds'}:
        raise ValueError('workload requires frames, fps, wall_seconds only')
    if type(value['frames']) is not int or value['frames'] <= 0:
        raise ValueError('frames: positive integer required')
    fps = _number(value['fps'], 'fps', positive=True)
    wall = _number(value['wall_seconds'], 'wall_seconds', positive=True)
    duration = Fraction(value['frames']) / fps
    return dict(video_seconds=duration, wall_seconds=wall,
                video_seconds_per_wall_second=duration / wall)


def compare_workloads(baseline, candidate, *, relative_tolerance=0, absolute_tolerance=0):
    """Compare normalized throughput, not raw frames or unequal-work wall times.

    Tolerances apply to throughput: candidate + absolute + baseline*relative
    must reach baseline. Both must be nonnegative; relative must be below one.
    """
    relative = _number(relative_tolerance, 'relative_tolerance')
    absolute = _number(absolute_tolerance, 'absolute_tolerance')
    if relative >= 1:
        raise ValueError('relative_tolerance must be below one')
    old, new = _workload(baseline), _workload(candidate)
    a, b = (row['video_seconds_per_wall_second'] for row in (old, new))
    time_ratio = new['wall_seconds'] / old['wall_seconds']
    output_ratio = new['video_seconds'] / old['video_seconds']
    return dict(baseline=old, candidate=new, throughput_ratio=b / a,
                throughput_change_fraction=b / a - 1,
                time_increase_ratio=time_ratio, useful_output_ratio=output_ratio,
                decomposition=dict(output_ratio=output_ratio, inverse_time_ratio=1 / time_ratio,
                                   product=output_ratio / time_ratio),
                counterfactual=dict(label='Arithmetic only; not a performance prediction',
                    candidate_output_at_baseline_time=new['video_seconds'] / old['wall_seconds']),
                regression=b + absolute + a * relative < a)


def summarize_stage_profiles(records, *, run_id):
    """Validate actual warm diagnostic records and group ordered hit/rank pairs.

    Input record order is immaterial; stage order is contractual. Maxima are
    descriptive cross-rank event/host maxima, never a wall-time reconstruction.
    run_id is caller provenance, not independently verified evidence.
    """
    if type(run_id) is not str or not run_id.strip():
        raise ValueError('nonempty caller-supplied run_id required')
    if type(records) is not list or not records:
        raise ValueError('nonempty record list required')
    grouped = {}
    keys = {'schema', 'worker_rank', 'cache_hit', 'synchronization_added',
            'benchmark_comparable', 'stages'}
    for record in records:
        if type(record) is not dict or set(record) != keys or record['schema'] != SCHEMA:
            raise ValueError('unsupported stage profile schema')
        rank, hit = record['worker_rank'], record['cache_hit']
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError('rank must be 0 or 1')
        if type(hit) is not int or not 1 <= hit <= 6:
            raise ValueError('cache_hit must be an integer from 1 through 6')
        if record['synchronization_added'] is not True or record['benchmark_comparable'] is not False:
            raise ValueError('diagnostic synchronization flags required')
        stages = record['stages']
        if type(stages) is not list or len(stages) != len(STAGES):
            raise ValueError('four ordered stages required')
        timings = []
        for name, stage in zip(STAGES, stages):
            if (type(stage) is not dict or
                    set(stage) != {'name', 'host_seconds', 'default_stream_elapsed_ms'} or
                    stage['name'] != name):
                raise ValueError('stage schema/order differs')
            timings.append(dict(name=name,
                host_seconds=_number(stage['host_seconds'], 'host_seconds'),
                default_stream_elapsed_ms=_number(stage['default_stream_elapsed_ms'], 'default_stream_elapsed_ms')))
        pair = grouped.setdefault(hit, {})
        if rank in pair:
            raise ValueError('duplicate hit/rank pair')
        pair[rank] = timings
    hits = sorted(grouped)
    if hits != list(range(1, hits[-1] + 1)):
        raise ValueError('cache hits must be contiguous from one')
    output = []
    for hit in hits:
        pair = grouped[hit]
        if set(pair) != {0, 1}:
            raise ValueError('both ranks required for every hit')
        maxima = [dict(name=name, **{metric: max(pair[r][i][metric] for r in (0, 1))
                   for metric in ('host_seconds', 'default_stream_elapsed_ms')})
                  for i, name in enumerate(STAGES)]
        output.append(dict(cache_hit=hit,
            ranks=[dict(worker_rank=r, stages=pair[r]) for r in (0, 1)],
            descriptive_stage_maxima=maxima))
    return dict(run_id=run_id, hits=output, synchronization_added=True,
                benchmark_comparable=False, wall_attribution_proven=False,
                scope='Descriptive per-stage timings only; not additive wall time or realtime acceptance')
