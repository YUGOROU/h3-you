"""No CUDA/network: own-rank post-request telemetry."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from research.h3_personal.fast_ref2va import _write_utilization


class MetadataTests(unittest.TestCase):
    def receipt(self, cuda):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, H3_REF2VA_COVERAGE_DIR=folder), patch.dict(sys.modules, torch=NS(cuda=cuda, distributed=NS(get_rank=lambda: 1))):
            _write_utilization(NS(), dict(session_id='test', index=0, measure_memory=True), None, None, None, 1., 2., True)
            return json.loads(next(Path(folder).glob('*.json')).read_text())

    def test_missing_is_unavailable(self):
        row = self.receipt(NS())
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'post_request_free_bytes', 'post_request_total_bytes'):
            self.assertIsNone(row[key])

    def test_fresh_own_device(self):
        calls = []
        def counter(value):
            def read(device):
                calls.append(device)
                return value
            return read
        row = self.receipt(NS(current_device=lambda: 1, max_memory_allocated=counter(123), max_memory_reserved=counter(456), mem_get_info=counter((789, 1000))))
        self.assertEqual(calls, [1, 1, 1])
        self.assertEqual(row['post_request_free_bytes'], 789)
        self.assertEqual(row['peak_allocated_bytes'], 123)
        self.assertEqual(row['rank'], 1)
