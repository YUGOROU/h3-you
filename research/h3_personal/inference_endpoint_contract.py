"""Offline HF custom-container migration admission; never a deployment API.

Evidence is supplied by a separate trusted, redacting collector. These records
contain only identifiers, hashes, observations and clocks, never credentials,
URLs, headers or raw catalog/HTTP responses. Validation cannot authenticate the
collector or prove that a remote service exists. A passing report is not launch
permission, GPU quality acceptance, or an implementation of a deadline watchdog.
Jobs and Modal observations are deliberately not transferable to this contract.
"""
from dataclasses import dataclass, fields
import math
import re
from typing import get_type_hints

ORIGINAL_REFERENCE_SHA256 = '6789db42c765d2611a640bc9b47934b4912584e45c800d2700b1fddf9902faeb'
REGION_COUNTRIES = {
    'aws': {'ap-northeast-1': 'JP', 'ap-northeast-3': 'JP',
            'ap-southeast-2': 'AU', 'ap-southeast-4': 'AU'},
    'gcp': {'asia-northeast1': 'JP', 'asia-northeast2': 'JP',
            'australia-southeast1': 'AU', 'australia-southeast2': 'AU'},
    'azure': {'japaneast': 'JP', 'japanwest': 'JP', 'australiaeast': 'AU',
              'australiasoutheast': 'AU', 'australiacentral': 'AU',
              'australiacentral2': 'AU'},
}


class _Strict:
    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            expected = get_type_hints(type(self))[field.name]
            if type(value) is not expected:
                raise TypeError(f"{field.name}: invalid type")
            if expected is float and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{field.name}: invalid clock")
            if expected is str and not re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,256}", value):
                raise ValueError(f"{field.name}: invalid identifier")


@dataclass(frozen=True)
class EndpointContract(_Strict):
    namespace: str
    provider: str
    region: str
    instance_type: str
    image_digest: str
    runtime_revision: str
    config_sha256: str
    source_sha256: str
    reference_sha256: str
    service_deadline: float
    session_deadline: float
    idle_deadline: float
    platform: str = "hf-inference-endpoints"
    engine: str = "custom-container"
    reference: str = "original-reference1024"
    precision: str = "BF16-native-FP32"
    adapter: str = "SMD4"
    pcm: str = "original"
    watermark: str = "preserved"
    frames: int = 124
    overlap: int = 22
    width: int = 252
    height: int = 448
    # One HTTP/controller worker owns a native TWO-rank model, not one GPU rank.
    serving_workers: int = 1
    instance_size: str = 'x2'
    min_replicas: int = 1
    max_replicas: int = 1
    access: str = "protected"


@dataclass(frozen=True)
class EndpointEvidence(_Strict):
    """Sanitized observations bound to one immutable candidate and namespace.

    All times use the caller's single monotonic clock domain. country must be
    provider-confirmed for this exact Endpoint region, not a geographic guess.
    live_namespace_catalog means an actual namespace-scoped Endpoint hardware
    catalog observation, not a public SKU list or a Jobs capability statement.
    """
    namespace: str
    provider: str
    region: str
    instance_type: str
    image_digest: str
    runtime_revision: str
    config_sha256: str
    source_sha256: str
    reference_sha256: str
    observed_at: float
    expires_at: float
    country: str
    origin: str = "hf-inference-endpoints"
    live_namespace_catalog: bool = False
    eligible: bool = False
    country_specific_region_confirmed: bool = False
    gpu_model: str = "unknown"
    actual_gpu_count: int = 0
    distinct_devices_verified: bool = False
    actual_serving_workers: int = 0
    instance_size: str = 'x2'
    actual_replicas: int = 0
    rank0_ready: bool = False
    rank1_ready: bool = False
    tts_ready: bool = False
    health_gated_on_ranks_and_tts: bool = False
    owner_scope_auth_succeeded: bool = False
    anonymous_denied: bool = False
    nonowner_denied: bool = False
    deadline_shutdown_verified: bool = False
    gateway_incremental_av_verified: bool = False
    gateway_session_affinity_verified: bool = False
    gateway_timeout_verified: bool = False


@dataclass(frozen=True)
class MigrationReadiness:
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def deployment_authorized(self) -> bool:
        return False


def validate_migration(contract: EndpointContract, evidence: EndpointEvidence | None,
                       *, now: float) -> MigrationReadiness:
    """Return stable, value-free blockers. No I/O, discovery, fallback or secrets.

    Missing/expired catalog or region evidence is unsupported, not provisionally
    eligible. Gateway streaming, affinity and timeout checks must be exercised
    through the actual Endpoint gateway; generic HTTP compatibility is no proof.
    """
    if type(contract) is not EndpointContract or (
            evidence is not None and type(evidence) is not EndpointEvidence):
        raise TypeError("invalid contract/evidence type")
    if type(now) is not float or not math.isfinite(now) or now < 0:
        raise ValueError("now: finite nonnegative float required")
    blockers = []

    def require(condition, reason):
        if not condition:
            blockers.append(reason)

    for name, expected in (
        ("platform", "hf-inference-endpoints"), ("engine", "custom-container"),
        ("reference", "original-reference1024"), ("precision", "BF16-native-FP32"),
        ("adapter", "SMD4"), ("pcm", "original"), ("watermark", "preserved"),
        ("frames", 124), ("overlap", 22),
        ("width", 252), ("height", 448), ("serving_workers", 1),
        ("instance_type", "nvidia-h200"), ("instance_size", "x2"),
        ("min_replicas", 1), ("max_replicas", 1), ("access", "protected"),
    ):
        require(getattr(contract, name) == expected, f"contract.{name}: unsupported")
    require(contract.provider in ("aws", "azure", "gcp"), "provider: unsupported")
    for name, pattern in (
        ("image_digest", r"sha256:[0-9a-f]{64}"),
        ("runtime_revision", r"[0-9a-f]{40}"),
        *((name, r"[0-9a-f]{64}") for name in
          ("config_sha256", "source_sha256", "reference_sha256")),
    ):
        require(re.fullmatch(pattern, getattr(contract, name)) is not None,
                f"pins.{name}: immutable pin required")
    require(contract.reference_sha256 == ORIGINAL_REFERENCE_SHA256,
            'pins.reference_sha256: original reference required')
    country = REGION_COUNTRIES.get(contract.provider, {}).get(contract.region)
    require(country in ('JP', 'AU'), 'region: unknown or disallowed provider region')
    require(now < contract.idle_deadline <= contract.session_deadline
            <= contract.service_deadline, "deadlines: expired or unordered")
    if evidence is None:
        blockers.append("evidence: absent; catalog/region/auth/runtime/gateway unverified")
        return MigrationReadiness(tuple(blockers))
    for name in ("namespace", "provider", "region", "instance_type", "instance_size", "image_digest",
                 "runtime_revision", "config_sha256", "source_sha256", "reference_sha256"):
        require(getattr(contract, name) == getattr(evidence, name),
                f"evidence.{name}: candidate binding mismatch")
    require(evidence.origin == "hf-inference-endpoints", "evidence.origin: Jobs/Modal not transferable")
    require(evidence.observed_at <= now < evidence.expires_at,
            "evidence.clock: stale or future observation")
    require(evidence.live_namespace_catalog and evidence.eligible,
            "catalog: unsupported without live namespace eligibility")
    require(country is not None and evidence.country == country
            and evidence.country_specific_region_confirmed,
            "region: unsupported without country-specific JP/AU provider evidence")
    require(evidence.gpu_model == "H200" and evidence.actual_gpu_count == 2
            and evidence.distinct_devices_verified, "runtime: actual two distinct H200 required")
    require(evidence.actual_serving_workers == 1 and evidence.actual_replicas == 1,
            "runtime: exactly one serving worker and replica required")
    require(evidence.rank0_ready and evidence.rank1_ready and evidence.tts_ready
            and evidence.health_gated_on_ranks_and_tts,
            "health: both ranks and TTS must gate readiness")
    require(evidence.owner_scope_auth_succeeded and evidence.anonymous_denied
            and evidence.nonowner_denied, "access: owner-scope success and anonymous/nonowner denial required")
    require(evidence.deadline_shutdown_verified, "deadlines: finite shutdown enforcement unverified")
    require(evidence.gateway_incremental_av_verified, "gateway: real incremental AV unverified")
    require(evidence.gateway_session_affinity_verified, "gateway: session affinity unverified")
    require(evidence.gateway_timeout_verified, "gateway: finite stream timeout compatibility unverified")
    return MigrationReadiness(tuple(blockers))
