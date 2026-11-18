"""Bounded diagnostic helpers, not identity scores or demonstrated acceleration."""
from contextlib import contextmanager
import json
from pathlib import Path
import time


def review_groups():
    """Separate within-clip late drift from independent-clip boundary changes."""
    groups={f'clip-{i}':[i*124+n for n in (0,31,62,93,123)] for i in range(4)}
    groups.update({f'boundary-{i}':[i*124+n for n in (-2,-1,0,1)] for i in range(1,4)})
    return groups


def aspect_candidate(reference_width,reference_height,target_width,target_height):
    """Plan only: swap target axes at identical pixel/token area, never warp input."""
    from fractions import Fraction
    values=(reference_width,reference_height,target_width,target_height)
    if any(type(v) is not int or v<=0 for v in values):raise ValueError('positive integer geometry required')
    if target_width%16 or target_height%16:raise ValueError('target geometry must be divisible by16')
    ref=Fraction(reference_width,reference_height)
    def mismatch(w,h):
        ratio=Fraction(w,h)/ref
        return float(max(ratio,1/ratio))
    return dict(current_aspect_mismatch=mismatch(target_width,target_height),
        swapped_aspect_mismatch=mismatch(target_height,target_width),
        candidate_width=target_height,candidate_height=target_width,
        same_pixel_area=True,reference_modified=False,automatic_application=False,
        speedup_demonstrated=False,gpu_and_human_validation_required=True)


class WarmStageProfile:
    """Opt-in CUDA event+host timings. Synchronizes at finish; diagnostic only.

    Events measure default-stream elapsed intervals, not summed kernel execution.
    Neither cross-rank summation nor measured speedup is meaningful here.
    """
    def __init__(self,folder=None,*,cuda=None,max_hits=3):
        if type(max_hits) is not int or max_hits not in (3,6):
            raise ValueError('bounded three-warm or six-arm profile required')
        self.max_hits=max_hits
        self.folder=Path(folder) if folder else None
        self.cuda=cuda;self.records=[]

    @contextmanager
    def stage(self,name):
        if self.folder is None:
            yield
            return
        if self.cuda is None:
            import torch
            self.cuda=torch.cuda
        start=self.cuda.Event(enable_timing=True);end=self.cuda.Event(enable_timing=True)
        start.record();begin=time.perf_counter()
        yield
        host=time.perf_counter()-begin;end.record()
        self.records.append((name,start,end,host))

    def finish(self,rank,hit):
        if self.folder is None:return None
        if type(rank) is not int or rank not in (0,1) or type(hit) is not int or not 1<=hit<=self.max_hits:
            raise ValueError('bounded two-rank warm profile only')
        if len(self.records)!=4:raise ValueError('all four warm stages required')
        self.records[-1][2].synchronize()
        report=dict(schema='h3-ref2va-warm-stage-diagnostic-v1',worker_rank=rank,cache_hit=hit,
            synchronization_added=True,benchmark_comparable=False,
            stages=[dict(name=n,host_seconds=h,default_stream_elapsed_ms=a.elapsed_time(b))
                    for n,a,b,h in self.records])
        self.folder.mkdir(mode=0o700,parents=True,exist_ok=True)
        path=self.folder/f'rank-{rank}-warm-{hit}.json'
        with path.open('x') as f:json.dump(report,f,indent=2)
        return report
