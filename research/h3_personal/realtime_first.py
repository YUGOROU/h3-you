"""Offline contract for the isolated FastH3 realtime-first experiment.

No model imports, submission, ledger mutation, media reads or pacing. The runner
must resolve pinned assets locally and supply actual monotonic server events.
CPU validation is not GPU compatibility, live playback or causal-audio proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import re
from typing import Sequence

FASTVIDEO_SHA = "ba371e33ed080e51f1884ae166baf7c4447a66fe"
DUMMY_PROMPT = "A colorful geometric toy rotates slowly on a plain studio table."
REJECTED_METHODS = {
    "fp8": "FastH3 full-rank delta fusion plus generic FP8 is unverified; no silent BF16 fallback.",
    "ref2va": "FastH3 preview is T2VA; external audio conditioning is not supported here.",
    "personal_adapter": "Identity adapter composition with FastH3 delta fusion is unverified.",
    "compile": "Compilation disabled to bound cold startup within the single Job timeout.",
}


def _integer(value: object, name: str, minimum: int = 0) -> int:
    # Deliberately reject bool, numpy scalars and coercible strings.
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a native integer >= {minimum}")
    return value


def _time(value: object, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative server timestamp")
    return float(value)


def validate_pin(value: str, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be an immutable lowercase SHA40")
    return value


def _money(value: Decimal, name: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative Decimal")
    return value


def budget_bound(*, ledger_bound: Decimal = Decimal("161.37"),
                 hourly_rate: Decimal = Decimal("10.00002"),
                 timeout_seconds: int = 3000,
                 cap: Decimal = Decimal("170")) -> dict[str, Decimal]:
    """Pure bound, not a reservation or invoice; rate covers both allocated GPUs."""
    for name, value in (("ledger_bound", ledger_bound), ("hourly_rate", hourly_rate), ("cap", cap)):
        _money(value, name)
    _integer(timeout_seconds, "timeout_seconds", 1)
    if timeout_seconds > 3000 or hourly_rate > Decimal("10.00002") or cap > Decimal("170"):
        raise ValueError("outside isolated single-Job authorization")
    cost = hourly_rate * Decimal(timeout_seconds) / Decimal(3600)
    if cost > Decimal("8.33335") or ledger_bound + cost > cap:
        raise ValueError("cumulative budget exceeded")
    return {"job_bound": cost, "cumulative_bound": ledger_bound + cost,
            "remaining": cap - ledger_bound - cost}


@dataclass(frozen=True)
class RealtimeFirstConfig:
    # Audited FastVideo source SHA, not our runner/deployment revision.
    runtime_revision: str
    base_revision: str
    adapter_revision: str
    height: int = 256
    width: int = 448
    frames: int = 124
    fps: int = 24
    segments: int = 4
    overlap_frames: int = 1
    precision: str = "bf16"
    transformer_forwards: int = 4
    sigma_grid_points: int = 5
    vsa_sparsity: float = 0.9
    sp_size: int = 2
    gpu_count: int = 2
    gpu: str = "H200"
    compile_enabled: bool = False
    prompt: str = DUMMY_PROMPT
    personalized: bool = False
    personal_inputs: tuple = ()
    timeout_seconds: int = 3000

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict:raise ValueError('configuration object required')
        value=dict(value)
        if type(value.get('personal_inputs')) is list and not value['personal_inputs']:
            value['personal_inputs']=()
        result=cls(**value);result.validate();return result

    def validate(self) -> None:
        for name in ("runtime_revision", "base_revision", "adapter_revision"):
            validate_pin(getattr(self, name), name)
        if self.runtime_revision != FASTVIDEO_SHA:
            raise ValueError("runtime must match audited FastVideo revision")
        if self.precision != "bf16":
            raise ValueError(REJECTED_METHODS["fp8"])
        expected = {"height": 256, "width": 448, "frames": 124, "fps": 24,
                    "segments": 4, "overlap_frames": 1, "transformer_forwards": 4,
                    "sigma_grid_points": 5, "sp_size": 2, "gpu_count": 2,
                    }
        for name, value in expected.items():
            actual = _integer(getattr(self, name), name)
            if actual != value:
                raise ValueError(f"{name} must be {value} for this isolated experiment")
        if not 1<=_integer(self.timeout_seconds,'timeout_seconds',1)<=3000:
            raise ValueError('timeout must be positive and <=3000seconds')
        if self.height % 32 or self.width % 32:
            raise ValueError("resolution must use multiples of 32")
        if type(self.vsa_sparsity) is not float or self.vsa_sparsity != 0.9:
            raise ValueError("VSA sparsity must be 0.9")
        if self.gpu != "H200" or self.compile_enabled is not False:
            raise ValueError("requires H200x2 with compilation disabled")
        if type(self.personalized) is not bool:raise ValueError('personalized must be explicit bool')
        expected_prompt=DUMMY_PROMPT
        if self.personalized:
            from research.build_h3_personal_fast_adapter import PERSONAL_PROMPT
            expected_prompt=PERSONAL_PROMPT
        if self.prompt != expected_prompt or type(self.personal_inputs) is not tuple or self.personal_inputs:
            raise ValueError('fixed approved prompt; no reference/audio inputs')

    def runner_contract(self) -> dict:
        """Declarative data; parent runner translates this to pinned FastVideo API."""
        self.validate()
        return {
            "runtime_revision": self.runtime_revision,
            "base": {"repo": "MiniMaxAI/MiniMax-H3", "revision": self.base_revision},
            "adapter": {"repo": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
                        "revision": self.adapter_revision,
                        "filename": "vsa-datafree/adapter_model.safetensors",
                        "loading": "full_rank_delta_fusion", "strength": 1.0},
            "height": self.height, "width": self.width, "num_frames": self.frames,
            "fps": self.fps, "segments": self.segments, "warmup_segments": 0,
            "overlap_frames": 1, "new_frames_after_first": 123,
            "num_inference_steps": 5, "transformer_forwards": 4,
            "attention_backend": "VIDEO_SPARSE_ATTN_H3", "VSA_sparsity": 0.9,
            "VSA_tile_size": 64, "sp_size": 2, "gpu_count": 2, "gpu": "H200",
            "precision": "bf16", "inference_torch_compile": False,
            "vae_parallel_decode": True, "vae_parallel_decode_strategy": "gather",
            "prompt": self.prompt, "timeout_seconds": self.timeout_seconds,
            "rejected_methods": {k:v for k,v in REJECTED_METHODS.items() if not (self.personalized and k=='personal_adapter')},
            "personalized": self.personalized,
            "external_audio_conditioning": False, "future_audio_causality_verified": False,
        }


@dataclass(frozen=True)
class SegmentMeasurement:
    """Times share one monotonic server clock; PTS are integer ticks at 24Hz.

    publication_finished is after encode AND actual publication, not scheduled
    playback. pts_start_frame refers to the trimmed, published segment.
    """
    index: int
    generation_started: float
    generation_finished: float
    encode_finished: float
    publication_finished: float
    pts_start_frame: int
    published_frames: int
    reused_frames_trimmed: int


def evaluate_segments(config: RealtimeFirstConfig,
                      measurements: Sequence[SegmentMeasurement], *,
                      job_started: float, evidence: str = "server_incremental") -> dict:
    """Evaluate all four real segments, retaining cold first in output.

    Warm denominator spans first publication to last publication, thereby
    including generation, encode, publication and intervening stalls. No paced
    prerecorded-file timing may be substituted for these observations.
    GPU generations are serial but may overlap the prior encode/publication.
    Encode/publication is one ordered consumer; timestamps alone prove neither
    client playback nor future-audio causality.
    """
    config.validate()
    if evidence != "server_incremental":
        raise ValueError("prerecorded pacing is not generation latency evidence")
    started = _time(job_started, "job_started")
    if len(measurements) != config.segments:
        raise ValueError("requires four actual segments, no throwaway warmup")
    end_frame = 0
    previous_publication = started
    previous_generation_finish = started
    publications = []
    for index, item in enumerate(measurements):
        if not isinstance(item, SegmentMeasurement):
            raise ValueError("requires SegmentMeasurement records")
        if _integer(item.index, "index") != index:
            raise ValueError("segments must be ordered without gaps")
        times = [_time(getattr(item, name), name) for name in
                 ("generation_started", "generation_finished", "encode_finished", "publication_finished")]
        if (times[0] < previous_generation_finish or
                times[2] < previous_publication or
                any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError("serial GPU or ordered encode/publication event ordering violated")
        if times[0] == times[-1]:
            raise ValueError("zero-duration synthetic segment")
        trim = 0 if index == 0 else config.overlap_frames
        count = config.frames - trim
        if (_integer(item.reused_frames_trimmed, "trim") != trim or
                _integer(item.published_frames, "published_frames", 1) != count or
                _integer(item.pts_start_frame, "PTS") != end_frame):
            raise ValueError("PTS discontinuity or missing reused-frame trim")
        end_frame += count
        previous_generation_finish = times[1]
        previous_publication = times[-1]
        publications.append(times[-1])
    warm_wall = publications[-1] - publications[0]
    if warm_wall <= 0:
        raise ValueError("positive warm measurement interval required")
    warm_seconds = (config.segments - 1) * (config.frames - config.overlap_frames) / config.fps
    # Lateness against a producer serving playback from first publication.
    # Segment i must be available when all preceding unique frames are consumed.
    lateness = [0.0] + [max(0.0, publications[i] - publications[0] -
                           (config.frames + (i - 1) * 123) / config.fps)
                          for i in range(1, config.segments)]
    growing = any(b > a + 1e-9 for a, b in zip(lateness, lateness[1:]))
    throughput = warm_seconds / warm_wall
    generation_wall=measurements[-1].generation_finished-measurements[1].generation_started
    if generation_wall<=0:raise ValueError('positive warm generation interval required')
    generation_throughput=warm_seconds/generation_wall
    # A slow first publication can prebuffer later GPU results and falsely inflate
    # publication throughput. Both lanes must independently sustain new video.
    consumer_wall=sum(item.publication_finished-max(item.generation_finished,measurements[i-1].publication_finished)
                      for i,item in enumerate(measurements) if i>0)
    consumer_throughput=warm_seconds/consumer_wall if consumer_wall>0 else None
    consumer_ok=consumer_throughput is not None and consumer_throughput>=1
    return {"segment_count": len(measurements), "unique_frames": end_frame,
            "cold_first_publication_seconds": publications[0] - started,
            "warm_wall_seconds": warm_wall, "warm_video_seconds": warm_seconds,
            "server_incremental_throughput": throughput,
            "publication_lateness_seconds": lateness, "growing_backlog": growing,
            "warm_generation_throughput": generation_throughput,
            "warm_consumer_throughput": consumer_throughput,
            "server_throughput_gate_passed": throughput >= 1 and generation_throughput>=1 and consumer_ok and not growing,
            "realtime_complete": False, "future_audio_causality_verified": False,
            "client_playback_verified": False,
            "latency_scope": "server publication, not client display or prerecorded pacing"}
