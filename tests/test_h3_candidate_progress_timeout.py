"""Candidate-only receiver progress allowance; no cloud/device operations."""
import ast
import inspect
import unittest
from research import launch_h3_direct_conversation as launcher


class CandidateProgressTests(unittest.TestCase):
    def test_only_continuous_headless_ordered_candidate_opts_in(self):
        tree = ast.parse(inspect.getsource(launcher.run))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'candidate_progress'
                                  for t in n.targets))
        code = compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                       '<candidate progress>', 'exec')
        for continuous in (False, True):
            for headless in (False, True):
                for ordered in (None, {'selected': True}):
                    with self.subTest(continuous=continuous, headless=headless, ordered=ordered):
                        # stable_selfie is a real run() option; attended_visual=False keeps the
                        # evaluated branch the device-free candidate allowance, not an attended one.
                        scope = dict(continuous_av=continuous, headless_interrupt=headless,
                                     ordered_references=ordered, stable_selfie=False,
                                     attended_visual=False)
                        exec(code, scope)
                        expected = ({'http_timeout': 120, 'stream_progress_timeout': 120}
                                    if continuous and headless and ordered is not None else {})
                        self.assertEqual(scope['candidate_progress'], expected)
        consumer = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'consumer' for t in n.targets)
                        and isinstance(n.value, ast.Call))
        self.assertTrue(any(k.arg is None and isinstance(k.value, ast.Name)
                            and k.value.id == 'candidate_progress' for k in consumer.value.keywords))


if __name__ == '__main__':
    unittest.main()
