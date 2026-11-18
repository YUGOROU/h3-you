import unittest
from dataclasses import FrozenInstanceError

from research.h3_personal.tts_preroll import (
    PCMPrerollGate, PCMPrerollPolicy, evaluate_trace,
)


class PolicyTests(unittest.TestCase):
    def test_immutable_and_boundaries(self):
        policy = PCMPrerollPolicy(reserve_samples=1, max_wait_seconds=0)
        with self.assertRaises(FrozenInstanceError):
            policy.reserve_samples = 2
        PCMPrerollPolicy(reserve_samples=96000, max_wait_seconds=5)
        for kwargs in (
            {"sample_rate": 48000.0}, {"sample_rate": True},
            {"sample_rate": 44100}, {"reserve_samples": 0},
            {"reserve_samples": True}, {"max_reserve_samples": 96001},
            {"max_wait_seconds": float("nan")}, {"max_wait_seconds": 5.01},
            {"max_wait_seconds": -1}, {"max_wait_seconds": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PCMPrerollPolicy(**kwargs)


class GateTests(unittest.TestCase):
    def test_boundary_and_once_only(self):
        gate = PCMPrerollGate(PCMPrerollPolicy())
        self.assertEqual(gate.observe(0, 10).status, "waiting")
        self.assertEqual(gate.observe(1, 20).first_pcm_monotonic, 20)
        ready = gate.observe(48000, 22)
        self.assertEqual(ready.start_monotonic, 22)
        self.assertEqual(ready.additional_wait_seconds, 2)
        later = gate.observe(96000, 100)
        self.assertEqual(later.start_monotonic, 22)
        self.assertEqual(later.additional_wait_seconds, 2)
        with self.assertRaises(ValueError):
            gate.observe(96000, 99)

    def test_late_threshold_rejected_permanently(self):
        gate = PCMPrerollGate(PCMPrerollPolicy())
        gate.observe(1, 0)
        with self.assertRaises(TimeoutError):
            gate.observe(48000, 2.001)
        with self.assertRaises(TimeoutError):
            gate.observe(48001, 3, sealed=True)

    def test_short_sealed_and_empty(self):
        gate = PCMPrerollGate(PCMPrerollPolicy())
        ready = gate.observe(1, 0, sealed=True)
        self.assertTrue(ready.short_sealed)
        self.assertEqual(ready.start_monotonic, 0)
        self.assertEqual(gate.observe(1, 100, sealed=True).status, "ready")
        with self.assertRaises(ValueError):
            gate.observe(2, 101, sealed=True)
        with self.assertRaises(ValueError):
            PCMPrerollGate(PCMPrerollPolicy()).observe(0, 0, sealed=True)

    def test_input_validation(self):
        for count, now, sealed in (
            (True, 0, False), (-1, 0, False), (1440001, 0, False),
            (1.0, 0, False), (0, float("inf"), False),
            (0, True, False), (0, 0, 1),
        ):
            with self.subTest(count=count, now=now, sealed=sealed):
                with self.assertRaises(ValueError):
                    PCMPrerollGate(PCMPrerollPolicy()).observe(count, now, sealed)
        gate = PCMPrerollGate(PCMPrerollPolicy())
        gate.observe(2, 1)
        with self.assertRaises(ValueError):
            gate.observe(1, 2)
        with self.assertRaises(ValueError):
            gate.observe(2, 0)


def trace(times=(0, 1.5, 3, 4.5), count=48000):
    total = len(times) * count
    return {"sample_rate": 48000, "source_samples": total,
            "duration_seconds": total / 48000,
            "events": [{"now": t, "samples": count} for t in times]
            + [{"now": times[-1], "sealed": True}]}


class TraceTests(unittest.TestCase):
    def test_residual_when_rtf_exceeds_one(self):
        result = evaluate_trace(trace(), PCMPrerollPolicy(reserve_samples=96000))
        self.assertEqual(result["start_monotonic"], 1.5)
        self.assertEqual(result["no_reserve_max_timing_deficit_seconds"], 1.5)
        self.assertEqual(result["chosen_start_max_timing_deficit_seconds"], 0)
        longer = evaluate_trace(trace((0, 1.5, 3, 4.5, 6, 7.5)),
                                PCMPrerollPolicy(reserve_samples=96000))
        self.assertEqual(longer["chosen_start_max_timing_deficit_seconds"], 1)
        self.assertFalse(longer["model_eos_verified"])
        self.assertFalse(longer["playback_verified"])
        self.assertTrue(longer["protocol_completion_observed"])

    def test_no_future_admission_and_rejection(self):
        result = evaluate_trace(trace((0, 2.1)), PCMPrerollPolicy(reserve_samples=96000))
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["start_monotonic"])
        self.assertEqual(result["reserve_samples"], 96000)
        short = trace((0,), 100)
        short["events"][-1]["now"] = 1
        result = evaluate_trace(short, PCMPrerollPolicy())
        self.assertEqual(result["start_monotonic"], 1)
        self.assertTrue(result["short_sealed"])

    def test_invalid_traces(self):
        cases = []
        for field, value in (("sample_rate", 44100), ("source_samples", 0),
                             ("duration_seconds", 99), ("events", []),
                             ("events", [{}] * 1025)):
            item = trace()
            item[field] = value
            cases.append(item)
        for events in (
            [{"now": 0, "samples": 0}, {"now": 0, "sealed": True}],
            [{"now": 1, "samples": 192000}, {"now": 0, "sealed": True}],
            [{"now": 0, "samples": 192000}, {"now": 121, "sealed": True}],
            [{"now": float("nan"), "samples": 192000}],
            [{"now": 0, "samples": 192000}],
            [{"now": 0, "samples": 1}, {"now": 0, "sealed": True}],
        ):
            item = trace()
            item["events"] = events
            cases.append(item)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                evaluate_trace(item, PCMPrerollPolicy())


if __name__ == "__main__":
    unittest.main()
