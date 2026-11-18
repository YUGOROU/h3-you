"""Offline report privacy, clock and RGB regression tests (no Torch)."""
import subprocess
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from research import hf_h3_utilization_comparison as worker


class SanitizedReportTests(unittest.TestCase):
    def test_progress_allowlist_privacy_and_full_report_bound(self):
        metrics = {k: 1 for k in ('first_segment_ready_seconds', 'total_wall_seconds',
            'generation_total_seconds', 'unique_video_seconds', 'unique_throughput_total',
            'unique_throughput_generation')}
        metrics['segment_ready_elapsed_seconds'] = [1, 2, 3]
        report = dict(devices={}, shapes=[dict(tokens=64)], semantic_intervals=[
            dict(label='h3.block.0', start_us=0, duration_us=2)])
        profile = dict(timing_metrics=metrics, input_proof={'secret': '/private/path'},
            derived_adapter_receipt={'prompt': 'PRIVATE'}, utilization=dict(ranks=[
                dict(rank=i, trace_summary=report) for i in (0, 1)]))
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('baseline', 'trace'):
                path, destination = worker.progress_checkpoint(tmp, arm, profile)
                self.assertEqual(destination, f'outputs/utilization/progress-{arm}.json')
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                text = path.read_text()
                self.assertNotIn('PRIVATE', text)
                self.assertNotIn('/private', text)
                data = json.loads(text)
                self.assertEqual(set(data), {'schema', 'arm', 'completed_owner',
                    'candidate_adopted', 'measurement_context', 'timing', 'evidence'})
                self.assertEqual(data['measurement_context'],
                                 'profiled' if arm == 'trace' else 'profileOFF')
                self.assertEqual(data['evidence'][0]['trace_summary']['shapes'], report['shapes'])
            for arm in ('../private', 'comparison-profile', 'baseline.json', None):
                with self.assertRaises(ValueError):
                    worker.progress_checkpoint(tmp, arm, profile)
            for bound in (True, 0, 1048577, 1):
                with self.assertRaises(ValueError):
                    worker.progress_checkpoint(tmp, 'candidate', profile, max_bytes=bound)
            self.assertFalse((Path(tmp)/'progress/progress-candidate.json').exists())

    def test_timelines_both_schema_versions(self):
        for schema in (1, 2):
            report = dict(schema=schema, devices={}, semantic_intervals=[
                dict(label='h3.block.0', start_us=0, duration_us=2)],
                gpu_category_intervals=[dict(category='compute', family='gemm', start_us=1, duration_us=1)],
                shapes=[dict(tokens=64, channels=128)])
            clean = worker.trace_scalars(report)
            self.assertEqual(clean['semantic_intervals'], report['semantic_intervals'])
            self.assertEqual(clean['gpu_category_intervals'], report['gpu_category_intervals'])
            self.assertEqual(clean['shapes'], report['shapes'])
            report['semantic_intervals'][0]['label'] = 'private raw event'
            with self.assertRaises(ValueError):
                worker.trace_scalars(report)

    def test_invalid_metadata(self):
        for report in (dict(shapes=[dict(secret=1)]),
                       dict(shapes=[dict(tokens=True)]),
                       dict(semantic_intervals=[dict(label='h3.denoise', start_us=0, duration_us=float('nan'))]),
                       dict(gpu_category_intervals=[dict(category='compute', family='private', start_us=0, duration_us=1)])):
            with self.assertRaises(ValueError):
                worker.trace_scalars(report)

    def test_parent_clock(self):
        metrics = {k: 1 for k in ('first_segment_ready_seconds', 'total_wall_seconds',
            'generation_total_seconds', 'unique_video_seconds', 'unique_throughput_total',
            'unique_throughput_generation')}
        metrics['segment_ready_elapsed_seconds'] = [1, 2, 3]
        keys = ('request_start', 'request_end', 'encoding_start', 'encoding_end', 'validation_start', 'validation_end')
        rows = [dict(zip(keys, range(i*10, i*10+6)), window_index=i, clock='parent_monotonic') for i in range(3)]
        profile = dict(timing_metrics=metrics, pipeline_timing=rows)
        result = worker.timing_scalars(profile)['pipeline_timing']
        self.assertEqual(result[0]['gap_before_next_request_seconds'], 5)
        self.assertEqual(result[0]['request_seconds'], 1)
        self.assertIsNone(result[-1]['gap_before_next_request_seconds'])
        rows[0]['clock'] = 'worker_monotonic'
        with self.assertRaises(ValueError):
            worker.timing_scalars(profile)

    def test_rgb_exact_and_different(self):
        for right, maximum in ((b'\x00\x01\xff', 0), (b'\xff\x01\x00', 255)):
            with patch.object(worker.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, stdout=b'\x00\x01\xff'),
                    subprocess.CompletedProcess([], 0, stdout=right)]):
                result = worker.compare_rgb('a', 'b', 1e20, width=1, height=1, frames=1)
            self.assertEqual(result['max'], maximum)
            self.assertEqual(result['exact'], maximum == 0)
