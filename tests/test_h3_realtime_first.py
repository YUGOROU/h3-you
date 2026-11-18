"""Stdlib-only tests: no GPU, network, private media or ledger access."""
from dataclasses import replace
from decimal import Decimal
import unittest

from research.h3_personal.realtime_first import (
    FASTVIDEO_SHA, RealtimeFirstConfig, SegmentMeasurement,
    budget_bound, evaluate_segments, validate_pin,
)


class RealtimeFirstTests(unittest.TestCase):
    def setUp(self):
        self.config = RealtimeFirstConfig(FASTVIDEO_SHA, "a" * 40, "b" * 40)

    def measurements(self, interval=4.0):
        result = []
        previous = 100.0
        start_frame = 0
        for i in range(4):
            end = previous + (20.0 if i == 0 else interval)
            result.append(SegmentMeasurement(i, previous, end - 1, end - .5,
                                             end, start_frame, 124 if i == 0 else 123,
                                             0 if i == 0 else 1))
            start_frame += 124 if i == 0 else 123
            previous = end
        return result

    def evaluate(self, rows=None, **kwargs):
        return evaluate_segments(self.config, rows or self.measurements(),
                                 job_started=100.0, **kwargs)

    def test_runner_contract(self):
        contract = self.config.runner_contract()
        self.assertEqual(contract["num_inference_steps"], 5)
        self.assertEqual(contract["transformer_forwards"], 4)
        self.assertEqual(contract["warmup_segments"], 0)
        self.assertEqual(contract["new_frames_after_first"], 123)
        self.assertFalse(contract["inference_torch_compile"])
        self.assertIn("fp8", contract["rejected_methods"])
        self.assertFalse(contract["future_audio_causality_verified"])
        contract["rejected_methods"].clear()
        self.assertIn("fp8", self.config.runner_contract()["rejected_methods"])

    def test_pins_all_three(self):
        for field in ("runtime_revision", "base_revision", "adapter_revision"):
            for value in ("main", "a" * 39, "a" * 41, "A" * 40, None, 123):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{field: value}).validate()
        with self.assertRaises(ValueError):
            replace(self.config, runtime_revision="c" * 40).validate()
        self.assertEqual(validate_pin("a" * 40, "test"), "a" * 40)

    def test_fixed_scope_and_resolution(self):
        for field, value in (("height", 240), ("height", 512), ("width", 449),
                             ("frames", 125), ("segments", 3), ("fps", 30),
                             ("overlap_frames", 0), ("sp_size", 1), ("gpu_count", 1),
                             ("sigma_grid_points", 4), ("transformer_forwards", 5),
                             ("timeout_seconds", 3001), ("compile_enabled", True),
                             ("compile_enabled", 0), ("gpu", "H100"),
                             ("vsa_sparsity", float("nan")), ("prompt", "private"),
                             ("personal_inputs", ("photo.png",))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.config, **{field: value}).validate()
        self.assertEqual(self.config.height % 32, 0)
        self.assertEqual(self.config.width % 32, 0)

    def test_fp8_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no silent BF16 fallback"):
            replace(self.config, precision="fp8").runner_contract()
        self.assertEqual(self.config.precision, "bf16")

    def test_cost_bound(self):
        result = budget_bound()
        self.assertEqual(result["job_bound"], Decimal("8.33335"))
        self.assertEqual(result["cumulative_bound"], Decimal("169.70335"))
        self.assertEqual(result["remaining"], Decimal("0.29665"))
        self.assertEqual(budget_bound(ledger_bound=Decimal("161.66665"))["remaining"], 0)

    def test_corrective_timeout_respects_remaining_budget(self):
        config=replace(self.config,timeout_seconds=2400);config.validate()
        bound=budget_bound(ledger_bound=Decimal('162.88'),timeout_seconds=config.timeout_seconds)
        self.assertEqual(bound['job_bound'],Decimal('6.66668'))
        self.assertEqual(bound['cumulative_bound'],Decimal('169.54668'))
        with self.assertRaises(ValueError):budget_bound(ledger_bound=Decimal('162.88'),timeout_seconds=3000)

    def test_budget_rejections(self):
        for kwargs in ({"ledger_bound": Decimal("161.66666")},
                       {"hourly_rate": Decimal("10.00003")}, {"timeout_seconds": 3001},
                       {"timeout_seconds": 0}, {"timeout_seconds": True},
                       {"cap": Decimal("171")}, {"ledger_bound": 161.37},
                       {"ledger_bound": Decimal("NaN")}, {"hourly_rate": Decimal("Infinity")},
                       {"ledger_bound": Decimal("-1")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                budget_bound(**kwargs)

    def test_warm_throughput_includes_encode_publication(self):
        report = self.evaluate()
        self.assertEqual(report["unique_frames"], 493)
        self.assertEqual(report["cold_first_publication_seconds"], 20)
        self.assertEqual(report["warm_wall_seconds"], 12)
        self.assertEqual(report["warm_video_seconds"], 369 / 24)
        self.assertAlmostEqual(report["server_incremental_throughput"], 369 / 24 / 12)
        self.assertTrue(report["server_throughput_gate_passed"])
        for name in ("realtime_complete", "future_audio_causality_verified", "client_playback_verified"):
            self.assertFalse(report[name])

    def overlapping_measurements(self):
        # GPU 1 starts while segment 0 encodes; GPU 2 starts while 1
        # waits for the single encode/publication consumer to become free.
        return [
            SegmentMeasurement(0, 100, 104, 108, 110, 0, 124, 0),
            SegmentMeasurement(1, 104, 108, 112, 114, 124, 123, 1),
            SegmentMeasurement(2, 108, 112, 116, 118, 247, 123, 1),
            SegmentMeasurement(3, 112, 116, 120, 122, 370, 123, 1),
        ]

    def test_overlapping_pipeline_accepted(self):
        report = self.evaluate(self.overlapping_measurements())
        self.assertEqual(report["cold_first_publication_seconds"], 10)
        self.assertEqual(report["warm_wall_seconds"], 12)
        self.assertTrue(report["server_throughput_gate_passed"])
        self.assertFalse(report["future_audio_causality_verified"])
        self.assertFalse(report["realtime_complete"])

    def test_backward_gpu_and_consumer_rejected(self):
        for changes in ({"generation_started": 103},
                        {"encode_finished": 109},
                        {"encode_finished": 108, "publication_finished": 109},
                        {"generation_finished": 113},
                        {"publication_finished": 111}):
            rows = self.overlapping_measurements()
            rows[1] = replace(rows[1], **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.evaluate(rows)
        rows = self.overlapping_measurements()
        rows[0] = replace(rows[0], generation_started=99)
        with self.assertRaises(ValueError):
            self.evaluate(rows)

    def test_cold_publication_prebuffer_does_not_fake_realtime(self):
        rows=[SegmentMeasurement(0,100,110,120,130,0,124,0),
              SegmentMeasurement(1,110,120,131,133,124,123,1),
              SegmentMeasurement(2,120,130,134,136,247,123,1),
              SegmentMeasurement(3,130,140,140.5,141,370,123,1)]
        report=self.evaluate(rows)
        self.assertGreater(report['server_incremental_throughput'],1)
        self.assertLess(report['warm_generation_throughput'],1)
        self.assertFalse(report['server_throughput_gate_passed'])

    def test_slow_publication_and_growing_backlog(self):
        report = self.evaluate(self.measurements(8))
        self.assertTrue(report["growing_backlog"])
        self.assertFalse(report["server_throughput_gate_passed"])
        self.assertGreater(report["publication_lateness_seconds"][-1], 0)

    def test_no_prerecorded_latency(self):
        with self.assertRaises(ValueError):
            self.evaluate(evidence="prerecorded_pacing")

    def test_missing_duplicate_trim_and_pts(self):
        rows = self.measurements()
        with self.assertRaises(ValueError):
            self.evaluate(rows[:3])
        for field, value in (("index", 0), ("pts_start_frame", 123),
                             ("published_frames", 124), ("reused_frames_trimmed", 0)):
            changed = rows.copy()
            changed[1] = replace(changed[1], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.evaluate(changed)

    def test_nonfinite_nonpositive_and_event_order_failures(self):
        for field, value in (("generation_started", -1), ("generation_started", 118),
                             ("generation_finished", float("nan")),
                             ("encode_finished", float("inf")),
                             ("publication_finished", 120), ("publication_finished", "124")):
            rows = self.measurements()
            rows[1] = replace(rows[1], **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.evaluate(rows)
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                evaluate_segments(self.config, self.measurements(), job_started=value)

    def test_numpy_like_scalars_not_coerced(self):
        # A stdlib stand-in ensures subclasses/coercible foreign scalar types
        # cannot bypass native scalar validation; numpy is not a dependency.
        class ForeignInt(int):
            pass
        class ForeignFloat(float):
            pass
        with self.assertRaises(ValueError):
            replace(self.config, height=ForeignInt(256)).validate()
        rows = self.measurements()
        rows[1] = replace(rows[1], publication_finished=ForeignFloat(124))
        with self.assertRaises(ValueError):
            self.evaluate(rows)


if __name__ == "__main__":
    unittest.main()
