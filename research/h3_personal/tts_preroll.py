"""Bounded, sample-count-only PCM admission and offline arrival-trace evaluation.

Trace schema: ``sample_rate``, ``source_samples`` (final total),
``duration_seconds`` (source_samples / sample_rate), and ``events``. Each event
has ``now`` and ``samples`` (a positive incremental chunk), or ``now`` and
``sealed=True`` (one final completion event, with no samples). Times are seconds
on one monotonic clock; the trace span is at most 120 seconds (source audio remains at most 30 seconds). Completion means
protocol completion, not independently verified model EOS or audible playback.
No waveform data is stored, copied, synthesized, or queued here.
"""

from dataclasses import dataclass
import math
from collections.abc import Mapping


_RATE = 48000
_MAX_SAMPLES = 30 * _RATE


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric seconds")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite numeric seconds")
    return value


@dataclass(frozen=True)
class PCMPrerollPolicy:
    sample_rate: int = 48000
    reserve_samples: int = 48000
    max_reserve_samples: int = 96000
    max_wait_seconds: float = 2.0

    def __post_init__(self):
        if _integer(self.sample_rate, "sample_rate") != _RATE:
            raise ValueError("only 48000 Hz mono PCM is supported")
        reserve = _integer(self.reserve_samples, "reserve_samples")
        maximum = _integer(self.max_reserve_samples, "max_reserve_samples")
        if not 1 <= reserve <= maximum <= 96000:
            raise ValueError("require 1 <= reserve <= max_reserve <= 96000")
        if not 0 <= _finite(self.max_wait_seconds, "max_wait_seconds") <= 5:
            raise ValueError("max_wait_seconds must be in [0, 5]")


@dataclass(frozen=True)
class PCMPrerollDecision:
    status: str
    reason: str
    available_samples: int
    reserve_samples: int
    first_pcm_monotonic: float | None
    start_monotonic: float | None
    additional_wait_seconds: float
    short_sealed: bool


class PCMPrerollGate:
    """Observation-driven gate: no timer, queue, or background deadline enforcement."""

    def __init__(self, policy: PCMPrerollPolicy):
        if not isinstance(policy, PCMPrerollPolicy):
            raise TypeError("policy must be PCMPrerollPolicy")
        self.policy = policy
        self._now = None
        self._available = 0
        self._first = None
        self._start = None
        self._sealed = False
        self._short = False
        self._failure = None

    def observe(self, available_samples, now, sealed=False):
        available = _integer(available_samples, "available_samples")
        now = _finite(now, "now")
        if type(sealed) is not bool:
            raise ValueError("sealed must be boolean")
        if not self._available <= available <= _MAX_SAMPLES:
            raise ValueError("available_samples must be nondecreasing and <= 30 seconds")
        if self._now is not None and now < self._now:
            raise ValueError("clock moved backwards")
        if self._sealed and (not sealed or available != self._available):
            raise ValueError("sealed source cannot change or reopen")
        self._now, self._available = now, available
        self._sealed = sealed
        if self._first is None and available:
            self._first = now
        if self._failure is not None:
            raise self._failure
        if self._start is None:
            if self._first is not None and now - self._first > self.policy.max_wait_seconds:
                self._failure = TimeoutError("PCM reserve wait exceeded at observation")
                raise self._failure
            if sealed and not available:
                raise ValueError("cannot admit empty sealed PCM")
            if available >= self.policy.reserve_samples or (sealed and available):
                self._start = now
                self._short = available < self.policy.reserve_samples
        ready = self._start is not None
        wait = 0.0 if self._first is None else (
            (self._start if ready else now) - self._first
        )
        return PCMPrerollDecision(
            "ready" if ready else "waiting",
            ("short_sealed" if self._short else "reserve_reached") if ready else "awaiting_reserve",
            available, self.policy.reserve_samples, self._first, self._start,
            wait, self._short,
        )


def evaluate_trace(measurement, policy):
    """Evaluate a complete trace without using future arrivals to choose admission.

    Deficit is max(0, arrival - (start + preceding_samples / rate)) over
    chunks. It measures cumulative supply lateness, not sink underruns, decoder
    behavior, or playback acceptance. An oversized late chunk cannot erase an
    earlier deficit. Events already buffered at start have zero deficit.
    """
    if not isinstance(measurement, Mapping):
        raise ValueError("measurement must be a mapping")
    if _integer(measurement.get("sample_rate"), "sample_rate") != _RATE:
        raise ValueError("trace rate must be 48000")
    total = _integer(measurement.get("source_samples"), "source_samples")
    if not 1 <= total <= _MAX_SAMPLES:
        raise ValueError("source_samples must be positive and <= 30 seconds")
    duration = _finite(measurement.get("duration_seconds"), "duration_seconds")
    if not math.isclose(duration, total / _RATE, rel_tol=0, abs_tol=1e-9):
        raise ValueError("duration does not match source samples")
    events = measurement.get("events")
    if not isinstance(events, (list, tuple)) or not 2 <= len(events) <= 1025:
        raise ValueError("require up to 1024 chunks plus one completion event")
    # Validate the entire input even if admission would fail early.
    chunks = []
    available = 0
    previous = origin = None
    normalized = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError("event must be a mapping")
        now = _finite(event.get("now"), "event now")
        if origin is None:
            origin = now
        if (previous is not None and now < previous) or now - origin > 120:
            raise ValueError("events must be monotonic within 120 seconds")
        previous = now
        sealed = event.get("sealed", False)
        if type(sealed) is not bool:
            raise ValueError("sealed must be boolean")
        if sealed:
            if index != len(events) - 1 or "samples" in event:
                raise ValueError("completion must be a separate final event")
        else:
            count = _integer(event.get("samples"), "chunk samples")
            if count <= 0:
                raise ValueError("chunk samples must be positive")
            chunks.append((now, available))
            available += count
            if available > _MAX_SAMPLES:
                raise ValueError("trace exceeds 30 seconds of PCM")
        normalized.append((available, now, sealed))
    if not normalized[-1][2] or available != total:
        raise ValueError("require sealed completion and exact source count")
    gate = PCMPrerollGate(policy)
    decision = None
    rejected = False
    for count, now, sealed in normalized:
        try:
            decision = gate.observe(count, now, sealed)
        except TimeoutError:
            rejected = True
            break
    first = chunks[0][0]
    start = None if rejected else decision.start_monotonic

    def deficit(start_time):
        return max(0.0, *(now - start_time - preceding / _RATE for now, preceding in chunks))

    return {
        "status": "rejected" if rejected else "ready",
        "reason": "reserve_wait_exceeded" if rejected else decision.reason,
        "first_pcm_monotonic": first,
        "start_monotonic": start,
        "additional_wait_seconds": None if rejected else start - first,
        "short_sealed": False if rejected else decision.short_sealed,
        "source_samples": total,
        "reserve_samples": policy.reserve_samples,
        "no_reserve_max_timing_deficit_seconds": deficit(first),
        "chosen_start_max_timing_deficit_seconds": None if rejected else deficit(start),
        "protocol_completion_observed": True,
        "model_eos_verified": False,
        "playback_verified": False,
        "trace_only": True,
        "caveats": "Protocol completion is not independent model EOS; timing deficits are not verified playback. Deadlines are checked only at observations.",
    }
