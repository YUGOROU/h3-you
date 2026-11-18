"""Forward-credit allocation proposal; no billing reconciliation or job launch.

Amounts are integer cents. Provider pools are not transferable. Training caps
include startup/final verification and same-Job previews. Future inference has
its own protected allocation. Proposals are not evidence of convergence.
"""
from decimal import Decimal

PROVIDER_LIMITS = {"modal": 5541, "huggingface": 6000}
CATEGORIES = {"pipeline", "training", "future_inference", "contingency"}


def proposed_allocation():
    return {
        "modal": {"pipeline": 0, "training": 4500, "future_inference": 500, "contingency": 541},
        "huggingface": {"pipeline": 1000, "training": 2500, "future_inference": 2000, "contingency": 500},
    }


def validate_allocation(allocation):
    if not isinstance(allocation, dict) or set(allocation) != set(PROVIDER_LIMITS):
        raise ValueError("both independent provider pools required")
    for provider, entries in allocation.items():
        if (not isinstance(entries, dict) or set(entries) != CATEGORIES
                or any(type(v) is not int or v < 0 for v in entries.values())
                or sum(entries.values()) > PROVIDER_LIMITS[provider]):
            raise ValueError("invalid or oversubscribed provider allocation")
    if sum(v["future_inference"] for v in allocation.values()) <= 0:
        raise ValueError("protect a future-inference allocation")
    return allocation


def exposure_outlook(allocation, *, provider, hourly_rate, seconds_per_update,
                     setup_seconds, final_seconds, preview_seconds, samples=715):
    """Conditional cost/coverage arithmetic, NOT measured speed or bill bounds.

    Rate must cover CPU/RAM and applicable surcharges. Inputs describe one run
    in ONE provider pool; HF contingency cannot automatically extend Modal.
    """
    validate_allocation(allocation)
    if provider not in PROVIDER_LIMITS:
        raise ValueError("unknown provider")
    rate, speed = Decimal(str(hourly_rate)), Decimal(str(seconds_per_update))
    if not rate.is_finite() or rate <= 0 or not speed.is_finite() or speed <= 0:
        raise ValueError("finite positive rate/speed required")
    if any(type(x) is not int or x < 0 for x in (setup_seconds, final_seconds, preview_seconds)):
        raise ValueError("nonnegative integer phase reserves required")
    if type(samples) is not int or samples <= 0:
        raise ValueError("positive sample count required")
    dollars = Decimal(allocation[provider]["training"]) / 100
    total_seconds = dollars * 3600 / rate
    reserved = setup_seconds + final_seconds + preview_seconds
    training_seconds = max(Decimal(0), total_seconds - reserved)
    updates = int(training_seconds / speed)
    target_seconds = Decimal(samples) * speed + reserved
    return {
        "training_cap_usd": str(dollars), "total_seconds_at_rate": float(total_seconds),
        "conditional_updates": updates, "one_coverage_epoch_fits": updates >= samples,
        "required_seconds_per_update_for_coverage": float(training_seconds / samples),
        "target_run_cost_usd": float(target_seconds * rate / 3600),
        "advance_approval_required": rate > 10 or target_seconds >= 1800,
        "quality_verified": False, "throughput_measured": False,
    }
