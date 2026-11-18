"""CPU doubles assume real Torch FunctionEvent.time_range.start/end in us,
name/device_type/input_shapes; no real Torch or GPU acceptance is claimed.
"""
import contextlib
import subprocess
import sys
import unittest
from types import SimpleNamespace as NS

from research.h3_personal.device_activity import (
    Capture, Policy, analyze, capability, serialize,
)


def event(start=0, end=10, name="ampere_sgemm_128x64", device="cuda", **kwargs):
    return NS(time_range=NS(start=start, end=end), name=name,
              device_type=device, **kwargs)


class ActivityTests(unittest.TestCase):
    def report(self, events, **kwargs):
        return analyze(events, policy=Policy(enabled=True, **kwargs))

    def test_cuda_annotations_not_activity(self):
        from research.hf_h3_utilization_comparison import trace_scalars
        report = self.report([
            event(0, 100, 'h3.warm_forward', 'cpu'),
            event(0, 100, 'h3.block.0', 'cpu', is_user_annotation=True),
            event(0, 100, 'private_user_range', is_user_annotation=True),
            event(0, 100, 'h3.warm_forward'),
            event(0, 100, 'minimax_h3.private_native_range'),
            event(10, 30),
            event(20, 40, 'ncclDevKernel_AllReduce'),
            event(25, 45, 'Memcpy DtoD'),
            event(35, 50, 'private_unknown_kernel'),
        ])
        clean = trace_scalars(report)
        self.assertIs(clean['annotation_filter_applied'], True)
        self.assertEqual(clean['excluded_gpu_annotations'], 3)
        gpu = clean['devices']['cuda']
        self.assertEqual(gpu['events'], 4)
        self.assertEqual(gpu['busy_us'], 40)
        self.assertEqual(gpu['observed_gap_us'], 60)
        self.assertEqual(gpu['category_union_us'],
                         dict(compute=20, collective=20, copy=20, unknown=15))
        self.assertEqual(gpu['family_event_counts']['unknown'], 1)
        self.assertFalse(report['supplied_cuda_events_classified'])
        self.assertEqual(clean['devices']['cpu']['busy_us'], 100)
        self.assertEqual(len(clean['semantic_intervals']), 2)
        for output in (report, clean):
            encoded = serialize(output)
            for private in (b'private_user_range', b'private_native_range',
                            b'private_unknown_kernel'):
                self.assertNotIn(private, encoded)
        for value in (False, 1, 'true', None):
            with self.assertRaises(ValueError):
                trace_scalars(dict(report, annotation_filter_applied=value))
        for value in (True, -1, 1.5, 200001):
            with self.assertRaises(ValueError):
                trace_scalars(dict(report, excluded_gpu_annotations=value))

    def test_annotation_flag_exact_true_and_prefix_boundary(self):
        report = self.report([
            event(name='private_kernel', is_user_annotation=1),
            event(name='minimax_h3_kernel'),
            event(name='h3.block.50'),
        ])
        self.assertEqual(report['excluded_gpu_annotations'], 0)
        self.assertEqual(report['devices']['cuda']['events'], 3)
        self.assertEqual(report['devices']['cuda']['family_event_counts'], {'unknown': 3})

    def test_annotation_filter_off_never_accesses_events(self):
        class Forbidden:
            def __iter__(self):
                raise AssertionError('events accessed while disabled')
        report = analyze(Forbidden())
        self.assertIs(report['annotation_filter_applied'], True)
        self.assertEqual(report['excluded_gpu_annotations'], 0)
        self.assertEqual(report['events_seen'], 0)

    def test_warm_intersection_through_worker(self):
        from research.hf_h3_utilization_comparison import trace_scalars
        rows = [event(100, 120, 'h3.warm_forward', 'cpu'),
                event(0, 90, 'aten::mm', 'cpu'),
                event(130, 200, 'aten::mm', 'cpu'),
                event(90, 105, 'h3.block.0', 'cpu'),
                event(0, 80, stream=7),
                event(90, 108, stream=1),
                event(105, 115, stream=2),
                event(118, 130, stream=3),
                event(140, 150, stream=4)]
        # Previous auto-domain [100,150] included all CPU busy (185us):
        # its span-minus-busy was negative. This is a fixture, not the lost
        # EphemeralGPU error's exact measured scalar.
        self.assertLess(50 - (90 + 20 + 70 + 10), 0)
        report = self.report(rows)
        clean = trace_scalars(report)
        self.assertEqual(clean['scope'], 'warm-window-intersection')
        self.assertEqual(clean['excluded_events'], 4)
        self.assertEqual(clean['clipped_events'], 3)
        self.assertEqual(clean['devices']['cpu']['events'], 2)
        self.assertEqual(clean['devices']['cuda']['events'], 3)
        self.assertEqual(clean['devices']['cuda']['busy_us'], 17)
        self.assertEqual(clean['devices']['cuda']['observed_gap_us'], 3)
        self.assertEqual(report['devices']['cuda']['stream_groups'], 3)
        self.assertEqual(clean['devices']['cuda']['family_event_counts'], {'gemm': 3})
        self.assertFalse(clean['coverage_proven'])
        for key in ('semantic_intervals', 'gpu_category_intervals'):
            for row in clean[key]:
                self.assertGreaterEqual(row['start_us'], 0)
                self.assertLessEqual(row['start_us'] + row['duration_us'], 20)
        for value in (-1, 200001, True, 1.5):
            with self.assertRaises(ValueError):
                trace_scalars(dict(report, excluded_events=value))
        with self.assertRaises(ValueError):
            trace_scalars(dict(report, scope='untrusted'))

    def test_intersection_unknown_and_truncation_remain_unproven(self):
        from research.hf_h3_utilization_comparison import trace_scalars
        report = self.report([
            event(10, 20, 'h3.warm_forward', 'cpu'),
            event(5, 15, 'private_kernel'),
            event(12, 18, 'private_device', 'other'),
            event(18, 19)], max_events=3)
        clean = trace_scalars(report)
        self.assertTrue(clean['truncated'])
        self.assertFalse(clean['coverage_proven'])
        self.assertFalse(report['supplied_cuda_events_classified'])
        self.assertEqual(clean['devices']['cuda']['family_event_counts'], {'unknown': 1})
        self.assertEqual(clean['devices']['unknown']['events'], 1)
        self.assertEqual(clean['clipped_events'], 1)
        self.assertEqual(clean['devices']['cuda']['observed_gap_us'], 5)

    def test_explicit_window_still_rejects_outside(self):
        report = analyze([event(0, 15), event(12, 18)],
                         policy=Policy(enabled=True),
                         capture_window_start_us=10, capture_window_end_us=20)
        self.assertEqual(report['invalid_events'], 1)
        self.assertEqual(report['devices']['cuda']['observed_gap_us'], 4)
        self.assertNotIn('scope', report)

    def test_semantic_restore_on_exception(self):
        from research.h3_personal.device_activity import semantic_forwards
        labels = []
        class Module:
            def forward(self):
                raise RuntimeError('test')
        block = Module()
        block.attn = Module()
        block.ff = Module()
        prior = lambda: 3
        block.ff.forward = prior
        @contextlib.contextmanager
        def record(label):
            labels.append(label)
            yield
        with self.assertRaises(RuntimeError):
            with semantic_forwards(NS(transformer_blocks=[block]), record):
                block.attn.forward()
        self.assertNotIn('forward', vars(block))
        self.assertNotIn('forward', vars(block.attn))
        self.assertIs(block.ff.forward, prior)
        self.assertEqual(labels, ['h3.attention.0'])

    def test_semantic_bounds_and_queued_work(self):
        rows = [event(100, 120, 'h3.warm_forward', 'cpu'),
                event(101, 110, 'h3.block.49', 'cpu'),
                event(102, 109, 'h3.block.50', 'cpu'),
                event(103, 108, 'h3.block.0_PRIVATE', 'cpu'),
                event(125, 130)]
        r = self.report(rows)
        self.assertEqual([x['label'] for x in r['semantic_intervals']],
                         ['h3.warm_forward', 'h3.block.49'])
        self.assertEqual(r['devices']['cuda']['capture_span_us'], 20)
        self.assertEqual(r['devices']['cuda']['observed_gap_us'], 20)
        self.assertEqual(r['gpu_category_intervals'], [])
        self.assertEqual(r['excluded_events'], 1)
        self.assertNotIn(b'PRIVATE', serialize(r))
        r = self.report([event(i, i+1, 'h3.block.0', 'cpu') for i in range(600)])
        self.assertEqual(len(r['semantic_intervals']), 512)
        self.assertTrue(r['semantic_intervals_truncated'])
        r = self.report([event(i, i+1) for i in range(2200)], max_bytes=262144)
        self.assertLessEqual(len(r['gpu_category_intervals']), 2048)
        self.assertTrue(r['gpu_intervals_truncated'])
        self.assertLessEqual(len(serialize(r)), 262144)

    def test_nested_and_overlapping_streams(self):
        r = self.report([event(100, 110), event(102, 108),
                         event(105, 115, "Memcpy HtoD (Pinned -> Device)"),
                         event(107, 112, "ncclDevKernel_AllReduce")])
        gpu = r["devices"]["cuda"]
        self.assertEqual(gpu["busy_us"], 15)
        self.assertEqual(sum(gpu["disjoint_us"].values()), 15)
        self.assertEqual(gpu["disjoint_us"]["compute+collective+copy"], 3)
        shifted = self.report([event(0, 10), event(2, 8),
                               event(5, 15, "Memcpy HtoD (Pinned -> Device)"),
                               event(7, 12, "ncclDevKernel_AllReduce")])
        self.assertEqual(r, shifted)

    def test_privacy_canary(self):
        secret = "PRIVATE_ID_PROMPT_PATH_CANARY"
        r = self.report([event(name=secret, input_shapes=[[2, 3]],
                               stack=secret, args=secret, stream=secret)])
        data = serialize(r)
        self.assertNotIn(secret.encode(), data)
        self.assertEqual(r["shapes"], [])
        self.assertEqual(r["devices"]["cuda"]["disjoint_us"], {"unknown": 10})
        r = self.report([event(input_shapes=[[True, secret]])])
        self.assertEqual(r["shapes"], [])

    def test_kernel_families_and_private_suffix(self):
        secret = "PRIVATE_CANARY_PATH_PROMPT"
        for prefix, category in (("ncclDevKernel", "collective"),
                                 ("ncclKernel", "collective"),
                                 ("Memcpy DtoH", "copy"),
                                 ("Memcpy DtoD", "copy"),
                                 ("ampere_sgemm_", "compute"),
                                 ("void flash::flash_fwd_kernel", "compute"),
                                 ("void at::native::vectorized_layer_norm_kernel", "compute"),
                                 ("triton_", "compute")):
            r = self.report([event(name=prefix + secret, thread=1234567)])
            self.assertEqual(r["devices"]["cuda"]["disjoint_us"], {category: 10})
            self.assertTrue(r["supplied_cuda_events_classified"])
            self.assertFalse(r["coverage_proven"])
            self.assertNotIn("complete_gpu_attribution", r)
            self.assertNotIn(secret.encode(), serialize(r))
            self.assertNotIn(b"1234567", serialize(r))
        for name in ("aten::mm", "c10d::allreduce_", "random_gemm", "Memcpy mystery"):
            self.assertFalse(self.report([event(name=name)])["supplied_cuda_events_classified"])

    def test_cpu_semantics_not_cuda_kernels(self):
        for name in ("aten::mm", "c10d::alltoall_", "c10d::alltoall_base_",
                     "c10d::all_to_all", "c10d::all_to_all_single"):
            r = self.report([event(name=name, device="cpu")])
            cpu = r["devices"]["cpu"]
            self.assertEqual(cpu["event_semantics"], "operator")
            self.assertEqual(cpu["family_event_counts"], {"operator": 1})
            self.assertEqual(r["devices"]["cuda"]["events"], 0)

    def test_capture_denominator_overlap_unknown_and_streams(self):
        r = analyze([event(102, 110, stream=7), event(102, 110, stream=7),
                     event(105, 112, name="unknown", stream=8),
                     event(106, 108, name="Memcpy DtoH", stream=True)],
                    policy=Policy(enabled=True), capture_window_start_us=100,
                    capture_window_end_us=120)
        gpu = r["devices"]["cuda"]
        self.assertEqual(gpu["busy_us"], 10)
        self.assertEqual(gpu["capture_span_us"], 20)
        self.assertEqual(gpu["observed_gap_us"], 10)
        self.assertEqual(gpu["stream_groups"], 2)
        self.assertEqual(gpu["category_union_us"],
                         dict(compute=8, collective=0, copy=2, unknown=7))
        self.assertEqual(sum(gpu["disjoint_us"].values()), 10)
        self.assertFalse(r["supplied_cuda_events_classified"])
        self.assertFalse(r["coverage_proven"])
        self.assertIsNone(self.report([event()])["devices"]["cuda"]["observed_gap_us"])
        with self.assertRaisesRegex(ValueError, "invalid capture window"):
            analyze([], capture_window_start_us=1)
        outside = analyze([event(0, 10)], policy=Policy(enabled=True),
                          capture_window_start_us=2, capture_window_end_us=20)
        self.assertEqual(outside["invalid_events"], 1)

    def test_malformed_nonfinite(self):
        items = [NS(), event(float("nan"), 2), event(0, float("inf")),
                 event(3, 2), event(True, 3), event(-1e308, 1e308)]
        r = self.report(items)
        self.assertEqual(r["invalid_events"], len(items))
        self.assertFalse(r["supplied_cuda_events_classified"])
        serialize(r)

    def test_caps(self):
        r = self.report([event(), event(), event()], max_events=2)
        self.assertEqual(r["events_seen"], 2)
        self.assertTrue(r["truncated"])
        self.assertFalse(r["supplied_cuda_events_classified"])
        self.assertFalse(self.report([event(), event()], max_events=2)["truncated"])
        with self.assertRaisesRegex(ValueError, "report byte limit exceeded"):
            self.report([event()], max_bytes=1)

    def test_cpu_only_and_rank_independence(self):
        r = self.report([event(device="cpu")])
        self.assertFalse(r["supplied_cuda_events_classified"])
        self.assertEqual(r["devices"]["cuda"]["busy_us"], 0)
        self.assertEqual(r["devices"]["cpu"]["busy_us"], 10)
        self.assertEqual(self.report([])["devices"]["cuda"]["busy_us"], 0)

    def test_enum_and_no_stringification(self):
        class Device:
            def __str__(self):
                raise AssertionError("must not stringify")
        cuda = Device()
        r = analyze([event(device=cuda)], policy=Policy(enabled=True),
                    device_types={"cuda": cuda})
        self.assertEqual(r["devices"]["cuda"]["events"], 1)
        self.assertFalse(self.report([event(device=Device())])["supplied_cuda_events_classified"])

    def test_capability_fixed_failure(self):
        class Broken:
            def supported_activities(self):
                raise RuntimeError("PRIVATE_CANARY")
        self.assertEqual(capability(Broken()), {"cpu": False, "cuda": False})
        enum = NS(CPU=object(), CUDA=object())
        self.assertEqual(capability(NS(ProfilerActivity=enum,
                         supported_activities=lambda: {enum.CPU})),
                         {"cpu": True, "cuda": False})

    def test_single_selected_window(self):
        calls = []
        @contextlib.contextmanager
        def factory(**kwargs):
            calls.append(kwargs)
            yield "handle"
        capture = Capture(Policy(enabled=True, selected_window=2))
        for index in (0, 1, 2, 2, 3):
            with capture.window(index, factory=factory,
                                capabilities={"cpu": True, "cuda": False}) as handle:
                self.assertEqual(handle, "handle" if index == 2 and len(calls) == 1
                                 and not getattr(self, "seen", False) else None)
                if handle:
                    self.seen = True
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], dict(activities=("cpu",), record_shapes=False,
                                      with_stack=False, profile_memory=False,
                                      with_flops=False, with_modules=False))

    def test_failed_capture_is_consumed(self):
        def fail(**kwargs):
            raise RuntimeError("factory failure")
        capture = Capture(Policy(enabled=True))
        with self.assertRaises(RuntimeError):
            with capture.window(1, factory=fail, capabilities={"cpu": True}):
                pass
        with capture.window(1, factory=fail, capabilities={"cpu": True}) as handle:
            self.assertIsNone(handle)

    def test_off_does_not_iterate_or_import(self):
        def forbidden():
            raise AssertionError("iterated")
            yield
        self.assertEqual(analyze(forbidden())["events_seen"], 0)
        with Capture().window(1, factory=lambda **kw: self.fail("called")) as handle:
            self.assertIsNone(handle)
        code = """
import sys
import research.h3_personal.device_activity
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
