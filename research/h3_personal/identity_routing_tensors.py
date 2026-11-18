"""Experimental packed TREAD tensor boundary; not installed into production.

Call slicing/rejoin outside the per-block checkpoint closure. Original sequence
metadata must be restored after rejoin. No packing/cache/timestep changes here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutedState:
    hidden_states: object
    rotary_emb: tuple
    adaln_indices: object
    attention_mask: object
    original_hidden_states: object
    keep_indices: object

    def rejoin(self, hidden_states):
        if (hidden_states.shape != self.hidden_states.shape
                or hidden_states.dtype != self.original_hidden_states.dtype
                or hidden_states.device != self.original_hidden_states.device):
            raise ValueError("routed activation shape/dtype/device changed")
        return self.original_hidden_states.index_copy(1, self.keep_indices, hidden_states)


def route_packed_activations(hidden_states, rotary_emb, adaln_indices, attention_mask, plan):
    import torch
    from .identity_routing import RoutingPlan
    if not isinstance(plan, RoutingPlan) or hidden_states.ndim != 3:
        raise ValueError("packed batch/sequence/channel tensor and routing plan required")
    n = hidden_states.shape[1]
    keep, bypass = plan.keep_indices, plan.bypass_indices
    if (not keep or any(type(i) is not int for i in (*keep, *bypass))
            or tuple(sorted(set(keep))) != keep or tuple(sorted(set(bypass))) != bypass
            or set(keep) & set(bypass) or set(keep) | set(bypass) != set(range(n))):
        raise ValueError("routing indices must exactly partition the original sequence")
    if (len(rotary_emb) != 2 or any(t.ndim != 2 or t.shape[0] != n for t in rotary_emb)
            or rotary_emb[0].shape != rotary_emb[1].shape
            or adaln_indices.shape != (n,) or adaln_indices.dtype != torch.long):
        raise ValueError("rotary/AdaLN metadata must match the full sequence")
    if any(t.device != hidden_states.device for t in (*rotary_emb, adaln_indices)):
        raise ValueError("activation/metadata device mismatch")
    idx = torch.tensor(keep, dtype=torch.long, device=hidden_states.device)
    mask = None
    if attention_mask is not None:
        if (attention_mask.shape != (n, n) or attention_mask.dtype != torch.bool
                or attention_mask.device != hidden_states.device):
            raise ValueError("expected full boolean sequence attention mask")
        mask = attention_mask.index_select(0, idx).index_select(1, idx)
    return RoutedState(hidden_states.index_select(1, idx),
                       tuple(t.index_select(0, idx) for t in rotary_emb),
                       adaln_indices.index_select(0, idx), mask, hidden_states, idx)
