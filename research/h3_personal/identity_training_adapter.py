"""Header-only audit of the pinned Ostris Ref2VA v1 candidate, NOT a loader.

No tensor bytes, merge, PEFT registration or production adoption happens here.
Shapes follow pinned Diffusers H3 geometry. Scaling follows the explicitly
selected Fizgig consumer convention, not alpha inferred from rank metadata.
"""
from dataclasses import dataclass
import re

REPO = "ostris/minimax_h3_training_adapter"
REVISION = "2d793bfb8f215e68f0e2251623c262c744991e2a"
FILENAME = "minimax_h3_ref2va_training_adapter_v1.safetensors"
SCALE_RULE = "fizgig-7ee4986-missing-alpha-means-rank"


@dataclass(frozen=True)
class ProjectionPlan:
    source_a: str
    source_b: str
    target: str
    b_rows: tuple[int, int]
    input_features: int
    output_features: int
    rank: int = 16
    scale: float = 1.0


def audit_ref2va_v1_header(header, *, scale_rule):
    """Require all 416 BF16 tensors and plan 312 exact linear projections.

    Returns immutable mapping records. At runtime B must be sliced on output
    rows, A shared unchanged, and every live Linear shape checked separately.
    This cannot verify payload finiteness/hash, base equivalence or rights.
    """
    if scale_rule != SCALE_RULE:
        raise ValueError("explicit pinned consumer scaling convention required")
    if not isinstance(header, dict):
        raise ValueError("header must be a dictionary")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or metadata.get("ss_base_model_version") != "minimax_h3_ref2va":
        raise ValueError("Ref2VA candidate metadata required")
    expected, plans = {}, []
    # Attention width = 56*128, distinct from model hidden width5376.
    groups = [(f"blocks.{i}", f"transformer_blocks.{i}") for i in range(50)]
    groups += [(f"token_refiner.blocks.{i}", f"token_refiner.refiner_blocks.{i}") for i in range(2)]
    for source, target in groups:
        for suffix, in_dim, out_dim, mappings in (
            ("attn.qkv_proj", 5376, 21504, [(f"attn.to_{q}", j*7168, (j+1)*7168) for j, q in enumerate("qkv")]),
            ("attn.out_proj", 7168, 5376, [("attn.to_out.0", 0, 5376)]),
            ("mlp.fc1", 5376, 28672, [("ff.net.0.proj", 0, 28672)]),
            ("mlp.fc2", 14336, 5376, [("ff.net.2", 0, 5376)]),
        ):
            a = f"diffusion_model.{source}.{suffix}.lora_A.weight"
            b = f"diffusion_model.{source}.{suffix}.lora_B.weight"
            expected[a], expected[b] = [16, in_dim], [out_dim, 16]
            for module, start, stop in mappings:
                plans.append(ProjectionPlan(a, b, f"{target}.{module}", (start, stop), in_dim, stop-start))
    if set(header) - {"__metadata__"} != set(expected):
        raise ValueError("candidate must contain exactly the audited tensor keys")
    intervals = []
    for key, shape in expected.items():
        spec = header[key]
        if (not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}
                or spec["dtype"] != "BF16" or spec["shape"] != shape
                or any(type(d) is not int for d in spec["shape"])):
            raise ValueError("candidate tensor dtype/shape/schema mismatch")
        offsets = spec["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int or x < 0 for x in offsets)
                or offsets[1] - offsets[0] != 2*shape[0]*shape[1]):
            raise ValueError("candidate tensor byte range mismatch")
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("candidate byte ranges must be contiguous and disjoint")
        end = stop
    return tuple(plans)


@dataclass(frozen=True)
class SubjectScope:
    trainable: tuple[str, ...]
    frozen: tuple[str, ...]


def subject_parameter_scope(parameter_names, *, start_block=20, end_block=50):
    """Offline freeze plan retaining the existing full default-PEFT contract.

    Only pass subject adapter parameter names, not base parameters. Runtime must
    verify excluded B tensors remain exactly zero (zero effective deltas), apply
    requires_grad before optimizer/DDP, and still export ALL default tensors.
    No production modules are changed by this function.
    """
    if (type(start_block) is not int or type(end_block) is not int
            or not 0 <= start_block < end_block <= 50):
        raise ValueError("invalid subject block window")
    names = tuple(parameter_names)
    expected = set()
    for stem in ([f"transformer_blocks.{i}" for i in range(50)]
                 + [f"token_refiner.refiner_blocks.{i}" for i in range(2)]):
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            for ab in "AB":
                expected.add(f"{stem}.attn.{projection}.lora_{ab}.default.weight")
    normalized = {}
    for name in names:
        if not isinstance(name, str):
            raise ValueError("parameter names must be strings")
        clean = name.removeprefix("base_model.model.")
        if clean not in expected or clean in normalized:
            raise ValueError("unexpected or duplicate subject adapter parameter")
        normalized[clean] = name
    if set(normalized) != expected:
        raise ValueError("full default subject adapter parameter set required")
    trainable, frozen = [], []
    for clean, original in normalized.items():
        match = re.match(r"transformer_blocks\.(\d+)\.", clean)
        active = match is not None and start_block <= int(match[1]) < end_block
        (trainable if active else frozen).append(original)
    return SubjectScope(tuple(sorted(trainable)), tuple(sorted(frozen)))
