"""Offline final result persistence, including Path-valued output."""
import json
from pathlib import Path
import tempfile
import unittest
from research.launch_h3_direct_conversation import durable_json


class Tests(unittest.TestCase):
    def test_path_output_serializes_and_receipt_is_exclusive(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'result.json'
            durable_json(p, dict(output=Path(root)/'receiver', complete=True))
            self.assertEqual(json.loads(p.read_text())['output'], str(Path(root)/'receiver'))
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                durable_json(p, {})

    def test_unsupported_value_does_not_leave_partial_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'result.json'
            for value in (object(), float('nan')):
                with self.assertRaises((TypeError, ValueError)):
                    durable_json(p, dict(bad=value))
                self.assertFalse(p.exists())


if __name__ == '__main__':
    unittest.main()
