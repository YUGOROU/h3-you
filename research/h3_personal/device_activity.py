"""Opt-in, bounded profiler primitive; no Torch import or runtime wiring.

Assumes Torch FunctionEvent fields time_range.start/end (microseconds),
 device_type, name and input_shapes. GPU behavior is UNVERIFIED here. Stream
and correlation IDs, stacks, arguments, tensor values and session IDs are never
exported. All clocks are rank-local; reports cannot align different ranks.
No trace exporter is called. Factories must themselves honor these privacy
settings: injected third-party code is outside this primitive's trust boundary.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import re


CATEGORIES = ("compute", "collective", "copy", "unknown")
# Exact semantic whitelist, deliberately not substring matching arbitrary names.
_NAMES = {
    "aten::mm": "compute", "aten::bmm": "compute",
    "aten::matmul": "compute", "aten::addmm": "compute",
    "aten::convolution": "compute", "aten::scaled_dot_product_attention": "compute",
    "c10d::allreduce_": "collective", "c10d::allgather_": "collective",
    "c10d::alltoall_": "collective", "c10d::alltoall_base_": "collective",
    "c10d::all_to_all": "collective", "c10d::all_to_all_single": "collective",
    "aten::copy_": "copy", "aten::to": "copy", "aten::_to_copy": "copy",
}


@dataclass(frozen=True)
class Policy:
    enabled: bool = False
    selected_window: int = 1
    max_events: int = 4096
    max_bytes: int = 65536

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("invalid policy")
        for value in (self.selected_window, self.max_events, self.max_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("invalid policy")


def _get(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def capability(profiler=None):
    """Probe only a supplied profiler namespace, without importing Torch.

    supported_activities() and ProfilerActivity.CPU/CUDA are expected Torch APIs.
    Failures expose only fixed booleans, never exception messages.
    """
    result = {"cpu": False, "cuda": False}
    try:
        activities = profiler.supported_activities()
        enum = profiler.ProfilerActivity
        result = {"cpu": enum.CPU in activities, "cuda": enum.CUDA in activities}
    except Exception:
        pass
    return result


def _device(value, device_types):
    # Only compare against trusted, explicitly supplied enum members. Never str().
    for label in ("cpu", "cuda"):
        expected = device_types.get(label)
        if expected is not None:
            try:
                if value is expected or value == expected:
                    return label
            except Exception:
                pass
    # FunctionEvent supplies a PyTorch enum. Avoid relying on a private
    # torch.autograd.DeviceType namespace or serializing arbitrary reprs.
    enum_name = _get(value, 'name')
    if type(enum_name) is str and enum_name in ('CPU', 'CUDA'):
        return enum_name.lower()
    if type(value) is str and value in ("cpu", "cuda", "CPU", "CUDA"):
        return value.lower()
    return "unknown"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


# Reviewed CUDA families only. Labels, never matched suffixes, leave this module.
_CUDA_FAMILIES = (
    ("collective", "nccl", ("ncclDevKernel", "ncclKernel")),
    ("copy", "memcpy", ("Memcpy HtoD", "Memcpy DtoH", "Memcpy DtoD",
                         "Memcpy HtoH", "Memcpy PtoP")),
    ("compute", "gemm", ("ampere_sgemm_", "ampere_bf16_s16816gemm_",
                         "ampere_fp16_s16816gemm_", "sm80_xmma_gemm_",
                         "sm90_xmma_gemm_", "void cutlass::gemm::")),
    ("compute", "flash_attention", ("void flash::flash_fwd_kernel", 
                                     "void flash::flash_bwd_kernel",
                                     "flash_fwd_kernel", "flash_bwd_kernel")),
    ("compute", "norm", ("void at::native::layer_norm", 
                          "void at::native::vectorized_layer_norm_kernel",
                          "void at::native::rms_norm")),
    ("compute", "triton", ("triton_",)),
)


def _classify(name, device):
    if type(name) is not str:
        return "unknown", "unknown"
    if device == "cpu":
        category = _NAMES.get(name, "unknown")
        return category, "operator" if category != "unknown" else "unknown"
    if device == "cuda":
        for category, family, prefixes in _CUDA_FAMILIES:
            if name.startswith(prefixes):
                return category, family
    return "unknown", "unknown"


_SEMANTIC = re.compile(r"h3\.(?:(?:block|attention|mlp)\.(?:[0-9]|[1-4][0-9])|forward[0-3]|validation\.(?:before|hold)|warm_forward|prepare|denoise|video_decode|audio_decode)\Z")


@contextmanager
def semantic_forwards(model, record_function):
    """Selected-capture-only instrumentation; restore instance dictionary exactly."""
    saved = []
    def wrap(module, label):
        if module is None:
            return
        original = module.forward
        present = 'forward' in vars(module)
        previous = vars(module).get('forward')
        def forward(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)
        saved.append((module, present, previous))
        module.forward = forward
    try:
        for i, block in enumerate(getattr(model, 'transformer_blocks', ())):
            if i >= 50:
                break
            wrap(block, f'h3.block.{i}')
            wrap(getattr(block, 'attn', None), f'h3.attention.{i}')
            wrap(getattr(block, 'ff', None), f'h3.mlp.{i}')
        yield
    finally:
        for module, present, previous in reversed(saved):
            if present:
                module.forward = previous
            else:
                del module.forward


def _partition(intervals):
    points = {}
    for start, end, category in intervals:
        points.setdefault(start, []).append((category, 1))
        points.setdefault(end, []).append((category, -1))
    active = {c: 0 for c in CATEGORIES}
    durations = {}
    previous = None
    for point in sorted(points):
        if previous is not None and point > previous:
            key = "+".join(c for c in CATEGORIES if active[c])
            if key:
                durations[key] = durations.get(key, 0.0) + point - previous
        for category, delta in points[point]:
            active[category] += delta
        previous = point
    return durations


def _gap(intervals, window):
    """Measure the complement of the interval union, not span minus busy."""
    cursor, end = window
    gaps = []
    for start, stop, _ in sorted(intervals):
        if start > cursor:
            gaps.append(start - cursor)
        cursor = max(cursor, stop)
    if cursor < end:
        gaps.append(end - cursor)
    return math.fsum(gaps)


def analyze(events, *, rank=0, policy=None, device_types=None,
            capture_window_start_us=None, capture_window_end_us=None):
    """Analyze event-like iterable, bounded by max_events, returning JSON-safe data.

    Endpoints use time_range, not cpu_time_total (nested totals double count).
    CPU and CUDA have separate unions, across ALL streams within each device.
    Optional trusted rank-local capture endpoints must use the event clock.
    Gaps are observed gaps, not proof of GPU idle or workload coverage.
    No independent CPU/GPU clock origins are aligned.
    Raises fixed ValueError if even the bounded report exceeds max_bytes.
    """
    policy = policy or Policy()
    if type(rank) is not int or rank < 0:
        raise ValueError("invalid rank")
    records = []
    groups = {d: [] for d in ("cpu", "cuda", "unknown")}
    window = None
    if capture_window_start_us is not None or capture_window_end_us is not None:
        start, end = capture_window_start_us, capture_window_end_us
        if not (_number(start) and _number(end)) or end <= start or not math.isfinite(end - start):
            raise ValueError("invalid capture window")
        window = (start, end)
    families = {d: {} for d in groups}
    streams = set()
    semantic = []
    gpu_rows = []
    warm = []
    native_collectives = 0
    count = invalid = 0
    excluded_gpu_annotations = 0
    truncated = False
    if policy.enabled:
        for event in events:
            if count >= policy.max_events:
                truncated = True
                break
            count += 1
            timer = _get(event, "time_range")
            start, end = _get(timer, "start"), _get(timer, "end")
            if not (_number(start) and _number(end)) or end <= start:
                invalid += 1
                continue
            # Overflowing finite endpoints must not create infinite durations.
            if not math.isfinite(end - start):
                invalid += 1
                continue
            if window is not None and (start < window[0] or end > window[1]):
                invalid += 1
                continue
            device = _device(_get(event, "device_type"), device_types or {})
            name = _get(event, "name")
            # CUDA FunctionEvents can mirror host/user ranges. These reviewed
            # annotation kinds are not kernels; arbitrary unknown symbols remain.
            if device == "cuda" and (
                _get(event, "is_user_annotation") is True or
                (type(name) is str and
                 (_SEMANTIC.fullmatch(name) or name.startswith("minimax_h3.")))
            ):
                excluded_gpu_annotations += 1
                continue
            records.append((device, name, start, end,
                            _get(event, "stream", _get(event, "thread"))))
            if device == "cpu" and name == "h3.warm_forward":
                warm.append((start, end))

    auto_window = window is None and bool(warm)
    if auto_window:
        window = (min(s for s, _ in warm), max(e for _, e in warm))
    excluded = clipped = 0
    for device, name, start, end, stream in records:
        if auto_window:
            left, right = max(start, window[0]), min(end, window[1])
            if right <= left:
                excluded += 1
                continue
            if (left, right) != (start, end):
                clipped += 1
            start, end = left, right
        category, family = _classify(name, device)
        if device == "cpu" and type(name) is str and _SEMANTIC.fullmatch(name):
            semantic.append((name, start, end))
        if device == "cpu" and category == "collective":
            native_collectives += 1
        if device == "cuda":
            gpu_rows.append((category, family, start, end))
            if type(stream) is int and 0 <= stream <= 2**31 - 1:
                streams.add(stream)
        groups[device].append((start, end, category))
        families[device][family] = families[device].get(family, 0) + 1
    devices = {}
    for device, intervals in groups.items():
        origin = window[0] if window else min((s for s, _, _ in intervals), default=0)
        normalized = [(s - origin, e - origin, c) for s, e, c in intervals]
        if any(not math.isfinite(e) for _, e, _ in normalized):
            raise ValueError("invalid interval span")
        partition = _partition(normalized)
        busy = sum(partition.values())
        span = window[1] - window[0] if window else None
        devices[device] = {"events": len(intervals), "busy_us": busy,
                           "disjoint_us": partition,
                           "category_union_us": {c: sum(v for k, v in partition.items()
                                                        if c in k.split("+"))
                                                 for c in CATEGORIES},
                           "family_event_counts": families[device],
                           "event_semantics": "operator" if device == "cpu" else
                                              "device_activity" if device == "cuda" else "unknown",
                           "capture_span_us": span,
                           "observed_gap_us": _gap(intervals, window) if window else None}
    devices["cuda"]["stream_groups"] = len(streams)
    report = {"schema": 1, "enabled": policy.enabled, "rank": rank,
              "clock": "rank_local_us_no_cross_device_alignment", "events_seen": count,
              "invalid_events": invalid, "truncated": truncated,
              "coverage_proven": False,
              "annotation_filter_applied": True,
              "excluded_gpu_annotations": excluded_gpu_annotations,
              "supplied_cuda_events_classified": bool(groups["cuda"]) and not (
                  truncated or invalid or groups["unknown"] or
                  any(c == "unknown" for _, _, c in groups["cuda"])),
              "devices": devices, "shapes": []}
    if auto_window:
        report.update(scope="warm-window-intersection",
                      excluded_events=excluded, clipped_events=clipped)
    origin = window[0] if window else min((s for _, s, _ in semantic), default=0)
    report.update(
        semantic_intervals=[dict(label=n, start_us=s-origin, duration_us=e-s)
                            for n, s, e in semantic[:512]],
        semantic_intervals_truncated=len(semantic) > 512,
        gpu_category_intervals=[dict(category=c, family=f, start_us=s-(window[0] if window else min(x[2] for x in gpu_rows)), duration_us=e-s)
                                for c, f, s, e in gpu_rows[:2048]],
        gpu_intervals_truncated=len(gpu_rows) > 2048,
        cuda_timeline_available=bool(gpu_rows),
        native_cpu_collective_events=native_collectives,
        collective_source_equivalence="unknown",
        gpu_block_attribution="unavailable_without_correlation")
    # Shape telemetry needs a separate operator/runtime mechanism in the future:
    # record_shapes=True can retain input tensors, even with bounded output.
    return bound_report(report, min(policy.max_bytes, 262144))


def bound_report(report, max_bytes=262144):
    """Bound canonical metadata, trimming only timeline tails, never aggregates/shapes.

    Geometric chunks keep repeated serialization linear in timeline size.
    Mutates the supplied report; an oversized non-timeline base fails closed.
    """
    if type(max_bytes) is not int or not 0 < max_bytes <= 262144:
        raise ValueError("invalid report byte limit")
    for key, flag in (('gpu_category_intervals', 'gpu_intervals_truncated'),
                      ('semantic_intervals', 'semantic_intervals_truncated')):
        while len(serialize(report)) > max_bytes and report.get(key):
            rows = report[key]
            del rows[-max(1, (len(rows) + 1) // 2):]
            report[flag] = True
    if len(serialize(report)) > max_bytes:
        raise ValueError("report byte limit exceeded")
    return report


def serialize(report):
    return json.dumps(report, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class Capture:
    """One selected subsequent window per helper (one rank/session owner).

    No caller/session identifier is retained or exported. OFF never calls the
    factory. Re-entry/retries cannot capture a second window, even after failure.
    Construct a separate owner for each rank/session; do not share across ranks.
    """
    def __init__(self, policy=None):
        self.policy = policy or Policy()
        self._claimed = False

    @contextmanager
    def window(self, index, *, factory=None, capabilities=None):
        if type(index) is not int or index < 0:
            raise ValueError("invalid window")
        if not self.policy.enabled or index != self.policy.selected_window or self._claimed:
            yield None
            return
        self._claimed = True
        caps = capabilities or {}
        activities = tuple(d for d in ("cpu", "cuda") if caps.get(d) is True)
        if factory is None or not activities:
            yield None
            return
        with factory(activities=activities, record_shapes=False,
                     with_stack=False, profile_memory=False,
                     with_flops=False, with_modules=False) as profiler:
            yield profiler
