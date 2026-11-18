"""Offline target-only TREAD routing contract; not wired into the GPU trainer.

Sequence indices must come from the actual packed layout, never modality tags
alone: reference pictures are video tokens too. This plan does not change cache
bytes, timestep tables, or losses. A runtime implementation must slice/rejoin
activations and rotary/AdaLN metadata together outside checkpoint recomputation.
"""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class RoutingPlan:
    keep_indices: tuple[int, ...]
    bypass_indices: tuple[int, ...]
    start_block: int
    end_block: int  # exclusive


def target_video_routing_plan(*, sequence_length, video_indices,
                              video_loss_mask, bypass_ratio, seed,
                              start_block=2, end_block=47, block_count=50,
                              training=False, latent_frames=1):
    """Build a seeded plan retaining every non-target row in original order.

    Inference and one-latent-frame samples are identity plans. Inputs are strict
    Python values for source-only tests; tensor conversion belongs at runtime.
    A valid plan is NOT evidence of GPU throughput or numerical compatibility.
    """
    for value in (sequence_length, seed, start_block, end_block, block_count, latent_frames):
        if type(value) is not int:
            raise ValueError("integer routing geometry and seed required")
    if sequence_length <= 0 or latent_frames <= 0 or not 0 <= start_block < end_block <= block_count:
        raise ValueError("invalid routing geometry/window")
    if type(training) is not bool:
        raise ValueError("explicit boolean training flag required")
    if (isinstance(bypass_ratio, bool) or not isinstance(bypass_ratio, (int, float))
            or not math.isfinite(bypass_ratio) or not 0 <= bypass_ratio < 1):
        raise ValueError("bypass ratio must be finite in [0, 1)")
    indices, mask = tuple(video_indices), tuple(video_loss_mask)
    if (not indices or len(indices) != len(mask)
            or any(type(i) is not int or not 0 <= i < sequence_length for i in indices)
            or len(set(indices)) != len(indices)
            or any(type(m) is not bool for m in mask)):
        raise ValueError("invalid packed video indices/loss mask")
    targets = tuple(i for i, generated in zip(indices, mask) if generated)
    if not targets:
        raise ValueError("at least one target video row required")
    bypass = ()
    if training and latent_frames > 1 and bypass_ratio and len(targets) > 1:
        keep_count = max(1, round(len(targets) * (1 - bypass_ratio)))
        retained_targets = set(random.Random(seed).sample(targets, keep_count))
        bypass = tuple(sorted(set(targets) - retained_targets))
    removed = set(bypass)
    keep = tuple(i for i in range(sequence_length) if i not in removed)
    return RoutingPlan(keep, bypass, start_block, end_block)
