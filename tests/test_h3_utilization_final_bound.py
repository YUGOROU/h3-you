"""Publication byte boundary; CPU-only, no Hub or Torch access."""
import json
from pathlib import Path
import tempfile
import unittest

from research.hf_h3_utilization_comparison import FINAL_MAX_BYTES, write_final_report


class FinalBoundTests(unittest.TestCase):
    def test_exact_boundary_and_canonical_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final.json'
            row = {'padding': 'x' * (FINAL_MAX_BYTES - len(b'{"padding":""}'))}
            write_final_report(path, row)
            self.assertEqual(path.stat().st_size, FINAL_MAX_BYTES)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_bytes()), row)
            with self.assertRaises(FileExistsError):
                write_final_report(path, row)

    def test_total_limit_not_per_trace_and_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final.json'
            # Each arm is below the trace bound; their aggregate exceeds final cap.
            row = {'arms': [{'data': 'x' * 240000} for _ in range(5)]}
            with self.assertRaisesRegex(ValueError, 'final report byte bound'):
                write_final_report(path, row)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                write_final_report(path, {'invalid': float('nan')})
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
