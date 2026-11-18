"""Finite offline H3 continuation contracts; no model, network or GPU imports.

Algorithm reference: vLLM-Omni 139a47a578e01eb7cd4a0c3d35a02eba4b4ded6a
continuation.py (Apache-2.0). Independent integer/rational contract implementation.
"""
from dataclasses import asdict,dataclass
from fractions import Fraction
import math


def integer(n,name,minimum=0):
    if type(n) is not int or n<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return n


def video_t(frames):
    integer(frames,'frames',5)
    if frames%17!=5:raise ValueError('frames must satisfy17n+5')
    return (frames-5)//17*5+2


def audio_boundary(frame):
    return round(Fraction(integer(frame,'frame')*5,3))


def covering_frames(samples,sample_rate):
    integer(samples,'samples',1);integer(sample_rate,'sample_rate',1)
    frames=math.ceil(Fraction(samples*24,sample_rate))
    return max(124,5+17*max(0,math.ceil(Fraction(frames-5,17))))


@dataclass(frozen=True)
class Window:
    start:int
    end:int
    overlap:int

    def __post_init__(self):
        integer(self.start,'start');integer(self.end,'end',5);integer(self.overlap,'overlap')
        video_t(self.frames)
        if self.start%17 or self.overlap>=self.frames or ((self.start==0)!=(self.overlap==0)):
            raise ValueError('invalid global window/overlap contract')
        if self.overlap:video_t(self.overlap)

    @property
    def frames(self):return self.end-self.start
    @property
    def audio_start(self):return audio_boundary(self.start)
    @property
    def audio_end(self):return audio_boundary(self.end)
    @property
    def guide_audio_t(self):return audio_boundary(self.start+self.overlap)-self.audio_start
    @property
    def guide_video_t(self):return video_t(self.overlap) if self.overlap else 0
    @property
    def temporal_offset(self):return Fraction(self.start*5,3)

    def record(self):
        return dict(**asdict(self),frames=self.frames,video_t=video_t(self.frames),
                    audio_start=self.audio_start,audio_end=self.audio_end,
                    guide_audio_t=self.guide_audio_t,guide_video_t=self.guide_video_t,
                    temporal_offset=str(self.temporal_offset))


def plan(total_frames,window_frames=277,overlap_frames=22):
    video_t(total_frames);video_t(window_frames);video_t(overlap_frames)
    if total_frames>7196:raise ValueError('finite offline total must not exceed300seconds')
    if not 124<=window_frames<=345 or overlap_frames>=window_frames:
        raise ValueError('window must be124..345 and exceed overlap')
    end=min(total_frames,window_frames);windows=[Window(0,end,0)]
    while end<total_frames:
        start=end-overlap_frames;end=min(total_frames,start+window_frames)
        windows.append(Window(start,end,overlap_frames))
    # Do not silently bypass the current target5..15s guard on a short tail.
    if any(not 124<=w.frames<=345 or w.guide_audio_t>=w.audio_end-w.audio_start for w in windows):
        raise ValueError('short final target or oversized guide: explicitly choose another grid plan')
    if sum(video_t(w.frames)-w.guide_video_t for w in windows)!=video_t(total_frames):
        raise ValueError('inconsistent cumulative video lattice')
    if sum(w.audio_end-w.audio_start-w.guide_audio_t for w in windows)!=audio_boundary(total_frames):
        raise ValueError('inconsistent cumulative audio lattice')
    return tuple(windows)


def request(samples,sample_rate,*,window_frames=277,overlap_frames=22,video_shift=12):
    if type(video_shift) not in (int,float) or video_shift!=12:
        raise ValueError('this experiment preserves video shift12; shift6 needs a separately bound experiment')
    total=covering_frames(samples,sample_rate);windows=plan(total,window_frames,overlap_frames)
    return dict(schema='h3-latent-tail-plan-v1',reference_policy='correct-hw-v2',
        samples=samples,sample_rate=sample_rate,total_frames=total,fps=24,
        original_audio_seconds=str(Fraction(samples,sample_rate)),
        model_video_seconds=str(Fraction(total,24)),video_shift=12,schedule_points=9,forwards=8,
        windows=[w.record() for w in windows],offline=True,causal_verified=False,
        gpu_verified=False,raw_conditioning_do_not_upload=True)
