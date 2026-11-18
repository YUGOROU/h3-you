"""Offline canonical metadata byte-bound regressions; no devices/network."""
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from research.h3_personal.device_activity import Policy, analyze, bound_report, serialize
from research import hf_h3_utilization_comparison as worker


class TraceBoundTests(unittest.TestCase):
    def test_analyzer_shapes_consumer_publication(self):
        events = [NS(name='ampere_sgemm_128x64', device_type='cuda',
                     time_range=NS(start=i * 1.123456789123, end=i * 1.123456789123 + .123456789))
                  for i in range(2048)]
        # Fill semantic metadata too, exercising actual analyzer bounds.
        events += [NS(name='h3.block.0', device_type='cpu',
                      time_range=NS(start=i * 1.123456789123, end=i * 1.123456789123 + .123456789))
                   for i in range(512)]
        report = analyze(events, policy=Policy(enabled=True, max_events=10000, max_bytes=262144))
        aggregate = copy.deepcopy(report['devices'])
        report['shapes'] = [dict(window_index=1, forward_index=i, tokens=4096,
                                 channels=3072, logical_rows=8192, local_rows=4096)
                            for i in range(4)]
        bound_report(report)
        self.assertEqual(report['devices'], aggregate)
        self.assertEqual(len(report['shapes']), 4)
        self.assertLessEqual(len(serialize(report)), 262144)
        # Default whitespace can reject an otherwise valid canonical trace.
        probe = copy.deepcopy(report)
        while len(serialize(probe)) < 250000:
            probe.setdefault('semantic_intervals', []).append(
                dict(label='h3.block.0', start_us=1.123456789123, duration_us=.123456789123))
        self.assertLessEqual(len(serialize(probe)), 262144)
        self.assertGreater(len(json.dumps(probe).encode()), 262144)
        clean = worker.trace_scalars(report)
        self.assertEqual(clean['shapes'], report['shapes'])
        metrics = {k: 1 for k in ('first_segment_ready_seconds', 'total_wall_seconds',
            'generation_total_seconds', 'unique_video_seconds', 'unique_throughput_total',
            'unique_throughput_generation')}
        metrics['segment_ready_elapsed_seconds'] = [1, 2, 3]
        profile = dict(timing_metrics=metrics, raw='RAW_CANARY', utilization=dict(ranks=[
            dict(rank=i, trace_summary=report) for i in (0, 1)]))
        with tempfile.TemporaryDirectory() as tmp:
            path, destination = worker.progress_checkpoint(tmp, 'trace', profile)
            hf = Mock()
            hf.upload_file(path_or_fileobj=str(path), path_in_repo=destination)
            hf.upload_file.assert_called_once()
            data = path.read_bytes()
            self.assertLessEqual(len(data), 1048576)
            self.assertNotIn(b'RAW_CANARY', data)
            self.assertEqual(data, serialize(json.loads(data)))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_oversized_base_refuses_and_preserves_unknown(self):
        report = dict(shapes=[dict(tokens=1)], aggregate='x' * 262144)
        with self.assertRaisesRegex(ValueError, '^report byte limit exceeded$'):
            bound_report(report)
        report = dict(gpu_category_intervals=[dict(category='unknown', family='unknown',
                     start_us=i, duration_us=1) for i in range(2048)],
                     devices={'unknown': {'events': 2048}}, shapes=[{'tokens': 1}])
        base = copy.deepcopy(report['devices'])
        bound_report(report, 4096)
        self.assertTrue(report['gpu_intervals_truncated'])
        self.assertEqual(report['devices'], base)
        self.assertEqual(report['shapes'], [{'tokens': 1}])
        self.assertLessEqual(len(serialize(report)), 4096)
