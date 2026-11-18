"""Synthetic offline evidence only; these tests do not accept a live Endpoint."""
from dataclasses import FrozenInstanceError, replace
import unittest

from research.h3_personal.inference_endpoint_contract import (
    EndpointContract, EndpointEvidence, validate_migration, ORIGINAL_REFERENCE_SHA256,
)


class EndpointContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = EndpointContract(
            namespace="owner", provider="aws", region="ap-northeast-1",
            instance_type="nvidia-h200", image_digest="sha256:" + "a" * 64,
            runtime_revision="b" * 40, config_sha256="c" * 64,
            source_sha256="d" * 64, reference_sha256=ORIGINAL_REFERENCE_SHA256,
            service_deadline=1000.0, session_deadline=500.0, idle_deadline=200.0,
        )
        self.evidence = EndpointEvidence(
            **{name: getattr(self.contract, name) for name in (
                "namespace", "provider", "region", "instance_type", "image_digest",
                "runtime_revision", "config_sha256", "source_sha256", "reference_sha256")},
            observed_at=10.0, expires_at=150.0, country="JP",
            live_namespace_catalog=True, eligible=True,
            country_specific_region_confirmed=True, gpu_model="H200",
            actual_gpu_count=2, distinct_devices_verified=True,
            actual_serving_workers=1, actual_replicas=1,
            rank0_ready=True, rank1_ready=True, tts_ready=True,
            health_gated_on_ranks_and_tts=True, owner_scope_auth_succeeded=True,
            anonymous_denied=True, nonowner_denied=True,
            deadline_shutdown_verified=True, gateway_incremental_av_verified=True,
            gateway_session_affinity_verified=True, gateway_timeout_verified=True,
        )

    def report(self, contract=None, evidence=None):
        return validate_migration(contract or self.contract,
                                  evidence or self.evidence, now=20.0)

    def test_complete_synthetic_evidence_is_not_deployment_permission(self):
        report = self.report()
        self.assertTrue(report.ready)
        self.assertFalse(report.deployment_authorized)
        with self.assertRaises(FrozenInstanceError):
            self.contract.precision = "FP8"

    def test_no_evidence_is_not_ready(self):
        report = validate_migration(self.contract, None, now=20.0)
        self.assertFalse(report.ready)
        self.assertIn("catalog/region/auth/runtime/gateway", report.blockers[-1])

    def test_strict_types_and_no_secret_fields(self):
        for field, value in (("min_replicas", True), ("frames", 124.0),
                             ("service_deadline", 1000), ("namespace", {})):
            with self.subTest(field=field), self.assertRaises(TypeError):
                replace(self.contract, **{field: value})
        with self.assertRaises(TypeError):
            replace(self.evidence, eligible=1)
        with self.assertRaises(TypeError):
            replace(self.contract, token="secret")
        with self.assertRaises(ValueError):
            replace(self.contract, namespace="Bearer secret")
        with self.assertRaises(TypeError):
            validate_migration({}, None, now=20.0)

    def test_finite_clocks(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(self.contract, service_deadline=value)
                with self.assertRaises(ValueError):
                    replace(self.evidence, observed_at=value)
                with self.assertRaises(ValueError):
                    validate_migration(self.contract, self.evidence, now=value)
        for value in (True, 20, "20"):
            with self.assertRaises(ValueError):
                validate_migration(self.contract, self.evidence, now=value)
        for change in ({"idle_deadline": 20.0}, {"session_deadline": 1001.0}):
            self.assertFalse(self.report(contract=replace(self.contract, **change)).ready)
        for change in ({"observed_at": 21.0}, {"expires_at": 20.0}):
            self.assertFalse(self.report(evidence=replace(self.evidence, **change)).ready)

    def test_no_quality_provider_or_replica_fallback(self):
        for field, value in (
            ("platform", "hf-jobs"), ("engine", "tgi"), ("provider", "modal"),
            ("reference", "reference512"), ("precision", "FP8"), ("adapter", "other"),
            ("frames", 123), ("overlap", 21), ("width", 512), ("height", 512),
            ("serving_workers", 2), ("min_replicas", 0), ("max_replicas", 2),
            ("instance_type", "nvidia-h100"), ("instance_size", "x4"),
            ("access", "public"), ("image_digest", "latest"),
            ("runtime_revision", "main"), ("config_sha256", "unknown"),
            ("source_sha256", "unknown"), ("reference_sha256", "unknown"),
        ):
            with self.subTest(field=field):
                self.assertFalse(self.report(contract=replace(self.contract, **{field: value})).ready)

    def test_every_observation_is_required(self):
        for field in (
            "live_namespace_catalog", "eligible", "country_specific_region_confirmed",
            "distinct_devices_verified", "rank0_ready", "rank1_ready", "tts_ready",
            "health_gated_on_ranks_and_tts", "owner_scope_auth_succeeded",
            "anonymous_denied", "nonowner_denied", "deadline_shutdown_verified",
            "gateway_incremental_av_verified", "gateway_session_affinity_verified",
            "gateway_timeout_verified",
        ):
            with self.subTest(field=field):
                report = self.report(evidence=replace(self.evidence, **{field: False}))
                self.assertFalse(report.ready)
                self.assertEqual(len(report.blockers), 1)

    def test_no_region_or_hardware_inference(self):
        for field, value in (
            ("country", "unknown"), ("country", "AP"), ("country", "US"),
            ("origin", "hf-jobs"), ("origin", "modal"), ("gpu_model", "H100"),
            ("actual_gpu_count", 1), ("actual_gpu_count", 4),
            ("actual_serving_workers", 2), ("actual_replicas", 2),
        ):
            with self.subTest(field=field, value=value):
                self.assertFalse(self.report(evidence=replace(self.evidence, **{field: value})).ready)

    def test_country_claim_cannot_override_region(self):
        for region in ('us-east-1', 'ap-northeast-2', 'ap', 'unknown'):
            with self.subTest(region=region):
                self.assertFalse(self.report(
                    contract=replace(self.contract, region=region),
                    evidence=replace(self.evidence, region=region, country='JP')).ready)
        self.assertFalse(self.report(evidence=replace(self.evidence,country='AU')).ready)
        self.assertFalse(self.report(contract=replace(self.contract,
            reference_sha256='f'*64)).ready)

    def test_evidence_is_candidate_bound(self):
        for field in ("namespace", "provider", "region", "instance_type", "image_digest",
                      "runtime_revision", "config_sha256", "source_sha256", "reference_sha256"):
            with self.subTest(field=field):
                report = self.report(evidence=replace(self.evidence, **{field: "different"}))
                self.assertIn(f"evidence.{field}: candidate binding mismatch", report.blockers)


if __name__ == "__main__":
    unittest.main()
