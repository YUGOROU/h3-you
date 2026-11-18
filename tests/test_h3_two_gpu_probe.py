import contextlib
import io
import json
import unittest
from research.hf_h3_two_gpu_probe import main
from research.hf_h3_context_parallel_probe import main as cp_main


class TwoGPUProbeTests(unittest.TestCase):
    def test_actual_class_probe_is_plan_only_without_opt_in(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cp_main(['--output','/not/created/cp']),0)
        plan=json.loads(output.getvalue())
        self.assertTrue(plan['plan_only'])
        self.assertFalse(plan['pretrained_weights'])
        self.assertFalse(plan['identity_verified'])

    def test_default_plan_only(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--output','/not/created/probe']),0)
        plan=json.loads(output.getvalue())
        self.assertTrue(plan['plan_only'])
        self.assertEqual(plan['world_size'],2)
        self.assertFalse(plan['h3_model_loaded'])
        self.assertFalse(plan['whole_model_cp_verified'])
