"""CPU-only refusal evidence contract; no model/device imports."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from research.h3_personal.fast_continuation_render import rejection_evidence
from research.h3_personal.fast_ref2va import utilization_name
from research.h3_personal.utilization_policy import DENOISE


class RejectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime = SimpleNamespace(coverage=Path(self.tmp.name))
        self.manifest = dict(session_id='fresh-session', index=1)
        self.rows = []
        for rank in (0, 1):
            row = dict(schema='h3-window-utilization-v1', session_id='fresh-session',
                window_index=1, rank=rank, denoise_policy=dict(DENOISE), faulted=True,
                candidate_rejected=True, state='faulted', owner_terminal='faulted', denoise_installed=False,
                parameter_identity_preserved=True, representative_preflight_complete=False,
                backend_compiles=4, regions=50, compile_seconds=1., extra_status_seconds=.1,
                extra_status_collectives=2, representative_metrics={'gate_passed': False})
            self.rows.append(row)
        self.write()

    def write(self):
        for rank, row in enumerate(self.rows):
            (self.runtime.coverage / utilization_name('fresh-session', 1, rank)).write_text(json.dumps(row))

    def test_both_rank_terminal_refusal_never_complete(self):
        result = rejection_evidence(self.runtime, self.manifest, dict(DENOISE))
        self.assertIs(result['owner_complete'], False)
        self.assertEqual(result['owner_terminal'], 'faulted')
        self.assertIsNone(result['trace_policy'])
        self.assertEqual(len(result['ranks']), 2)

    def test_missing_rank_refused(self):
        (self.runtime.coverage / utilization_name('fresh-session', 1, 1)).unlink()
        with self.assertRaises(FileNotFoundError):
            rejection_evidence(self.runtime, self.manifest, dict(DENOISE))

    def test_malformed_stale_or_native_failure_not_rejection(self):
        for key, value in [('candidate_rejected', False), ('faulted', False),
                           ('session_id', 'stale'), ('state', 'ready'),
                           ('backend_compiles', True), ('backend_compiles', 9),
                           ('denoise_policy', None), ('representative_metrics', None)]:
            with self.subTest(key=key, value=value):
                original = self.rows[1][key]
                self.rows[1][key] = value
                self.write()
                with self.assertRaises((ValueError, RuntimeError, AssertionError)):
                    rejection_evidence(self.runtime, self.manifest, dict(DENOISE))
                self.rows[1][key] = original


if __name__ == '__main__':
    unittest.main()
