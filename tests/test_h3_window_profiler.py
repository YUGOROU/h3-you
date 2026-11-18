import json
from pathlib import Path
import tempfile
import unittest

from research.h3_personal.window_profiler import WindowProfiler
from research.h3_personal.live_native_backend import LiveNativeBackend
from test_h3_live_native_backend import FakeRuntime, driver, export_stub, two_turn_packets


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, n): self.value += n


class WindowProfilerTests(unittest.TestCase):
    def test_nested_spans_and_tts_share_origin(self):
        clock = Clock(); out = []; p = WindowProfiler(clock=clock, emit=out.append)
        with p.span(0, 'driver_total'):
            clock.advance(1)
            with p.activity('tts_complete_wav'):
                with p.span(0, 'native_call'): clock.advance(2)
        p.emit_window(0); p.emit_remaining()
        self.assertEqual(len(out), 3)
        row = p.snapshot()['windows'][0]
        self.assertEqual(row['stages']['driver_total']['duration_seconds'], 3)
        self.assertEqual(row['stages']['native_call']['start_seconds'], 1)
        self.assertEqual(p.snapshot()['activities'][0]['start_seconds'], 1)
        self.assertEqual(p.snapshot()['dropped'], 0)

    def test_primary_error_not_masked_by_emitter(self):
        def broken(_): raise OSError('PRIVATE EMITTER')
        clock = Clock(); p = WindowProfiler(clock=clock, emit=broken)
        with self.assertRaisesRegex(RuntimeError, 'primary'):
            with p.span(0, 'video_export'):
                clock.advance(1)
                raise RuntimeError('primary')
        p.emit_remaining()
        self.assertEqual(p.snapshot()['windows'][0]['stages']['video_export']['status'], 'failed')
        self.assertGreater(p.dropped, 0)
        self.assertNotIn('PRIVATE', json.dumps(p.snapshot()))

    def test_bad_clock_and_input_drop_without_breaking_body(self):
        clock = Clock(); p = WindowProfiler(clock=clock, emit=lambda _:None)
        with p.span(0, 'driver_total'): clock.value = float('nan')
        p.interval(0, 'unknown-secret', 0, 1)
        p.interval(True, 'driver_total', 0, 1)
        p.interval(144, 'driver_total', 0, 1)
        p.role(0, 'PRIVATE ROLE')
        self.assertGreater(p.dropped, 0)
        self.assertNotIn('PRIVATE', json.dumps(p.snapshot()))

    def test_bounds_and_defensive_snapshot(self):
        p = WindowProfiler(clock=lambda:0, emit=lambda _:None)
        for i in range(200): p.interval(i, 'driver_total', 0, 1)
        self.assertEqual(len(p.snapshot()['windows']), 144)
        r = p.snapshot(); r['windows'][0]['stages']['driver_total']['duration_seconds'] = 99
        self.assertEqual(p.snapshot()['windows'][0]['stages']['driver_total']['duration_seconds'], 1)
        for _ in range(20):
            with p.activity('canonical_pcm'): pass
        self.assertEqual(len(p.snapshot()['activities']), 16)

    def test_incremental_wait_attempts_do_not_replay_native(self):
        p = WindowProfiler(clock=lambda:0, emit=lambda _:None)
        p.interval(0, 'driver_prepare', 0, 1)
        p.interval(0, 'driver_prepare', 2, 3)
        p.interval(0, 'native_call', 3, 4)
        p.interval(0, 'native_call', 4, 5)
        row = p.snapshot()['windows'][0]['stages']
        self.assertEqual(row['driver_prepare']['attempts'], 2)
        self.assertEqual(row['driver_prepare']['prior_attempt_seconds'], 1)
        self.assertEqual(row['native_call']['end_seconds'], 4)
        self.assertEqual(p.dropped, 1)

    def test_driver_backend_integration_preserves_packets_and_retires(self):
        def run(root, profiled):
            clock = Clock(); p = WindowProfiler(clock=clock, emit=lambda _:None)
            runtime = FakeRuntime(root); d = driver(root)
            original_generate = runtime._generate
            def generate(**kw):
                begin = clock(); result, _ = original_generate(**kw); clock.advance(.03)
                return result, (begin, clock())
            runtime._generate = generate
            b = LiveNativeBackend(runtime, Path(root)/'native', exporter=export_stub)
            if profiled: d._window_profiler = b._window_profiler = p
            originals = []; packets = list(two_turn_packets(d, b, originals))
            b.retire(); d.close()
            if profiled:
                rows = p.snapshot()['windows']
                self.assertEqual(len(rows), 6)
                for row in rows:
                    self.assertEqual(set(row['stages']), p.STAGES)
                    self.assertAlmostEqual(row['stages']['native_generate_sync']['duration_seconds'], .03)
                self.assertEqual(p.dropped, 0)
            return [(x.video, x.pcm, x.envelope.frame_start, x.envelope.frame_stop) for x in packets]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(run(a, False), run(b, True))

    def test_export_failure_keeps_quarantine_and_failed_stage(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = FakeRuntime(root); d = driver(root)
            def broken(*args): raise OSError('export primary')
            b = LiveNativeBackend(runtime, Path(root)/'native', exporter=broken)
            p = WindowProfiler(clock=lambda:0, emit=lambda _:None)
            b._window_profiler = d._window_profiler = p
            with self.assertRaisesRegex(OSError, 'export primary'): d.step(b)
            self.assertEqual(runtime._state, 'degraded')
            self.assertEqual(p.snapshot()['windows'][0]['stages']['video_export']['status'], 'failed')
            d.close()


if __name__ == '__main__': unittest.main()
