import contextlib
import io
import json
import unittest
from unittest.mock import patch
from research.run_h3_zoom_preflight import main

class ZoomPreflightTests(unittest.TestCase):
    def test_default_description_no_process_model_network(self):
        with patch('subprocess.run',side_effect=AssertionError('no processes')):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):self.assertEqual(main([]),0)
        value=json.loads(out.getvalue())
        self.assertFalse(value['devices_accessed']);self.assertFalse(value['runtime_verified'])
        self.assertFalse(value['asr_started']);self.assertIn('HTTP',value['transport'])
    def test_explicit_finite_paths_required(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['--run-offline'])
