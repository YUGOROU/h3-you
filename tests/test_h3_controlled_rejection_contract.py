"""CPU-only controlled producer -> window writer -> renderer -> worker contract."""
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_h3_utilization_end_to_end_contract as fixtures
from research.h3_personal.denoise_execution import (
    DenoiseController, RequestAdapter, CandidateRejected)
from research.h3_personal import fast_ref2va as forward
from research.h3_personal import fast_continuation_render as renderer
from research import hf_h3_utilization_comparison as worker


class ControlledRejectionTests(unittest.TestCase):
    session = 'controlled-session'
    setUp = fixtures.EndToEndContractTests.setUp

    def produce(self, mode='overbudget', peer_native=False):
        for rank in (0, 1):
            self.rank = rank
            status = NS(min=lambda ok, label: False,
                        rejection_vote=lambda code, label: max(code, 2 if peer_native else 1))
            c = DenoiseController(NS(), [object() for _ in range(50)], status=status,
                compiler=lambda: None, backend=lambda: None,
                max_compile_count=worker.DENOISE_POLICY['max_backend_compiles'],
                max_compile_seconds=180)
            c.state = 'ready'
            c.identity = ()
            c.receipt['identity_unchanged'] = True
            adapter = RequestAdapter.__new__(RequestAdapter)
            adapter.controller, adapter.session_id = c, self.session
            adapter.window_index, adapter.policy = 0, dict(worker.DENOISE_POLICY)
            if mode == 'overbudget':
                ticks = iter((0., 181.))
                c.budget.clock = lambda: next(ticks)
                compute = lambda *args: c.budget()
            else:
                unsupported = type('Unsupported', (RuntimeError,),
                                   {'__module__': 'torch._dynamo.exc'})
                def compute(*args):
                    raise unsupported('pre-backend rejection')
            with patch.object(adapter, '_probe_compute', side_effect=compute):
                with self.assertRaises(RuntimeError if peer_native else CandidateRejected):
                    adapter._probe(None, None, ())
            self.assertEqual('rejection_kind' in c.receipt, not peer_native)
            adapter._write_receipt()
            forward._write_utilization(NS(), dict(session_id=self.session, index=0),
                None, worker.DENOISE_POLICY, adapter, 0., 1., False)

    def evidence(self):
        return renderer.rejection_evidence(NS(coverage=self.coverage),
            dict(session_id=self.session, index=0), worker.DENOISE_POLICY)

    def test_overbudget_is_finding_not_success(self):
        self.produce()
        evidence = self.evidence()
        rows = worker.rejected_evidence({'utilization': evidence}, 'candidate_warmup')
        self.assertEqual(len(rows), 2)
        self.assertFalse(evidence['owner_complete'])
        self.assertTrue(all(r['compile_seconds'] == 181 for r in evidence['ranks']))

    def test_prebackend_unsupported_without_metrics(self):
        self.produce('unsupported')
        evidence = self.evidence()
        worker.rejected_evidence({'utilization': evidence}, 'candidate_warmup')
        self.assertTrue(all(r['backend_compiles'] == 0 and
                            r['representative_metrics'] is None for r in evidence['ranks']))

    def test_peer_native_failure_is_not_controlled(self):
        self.produce('unsupported', peer_native=True)
        with self.assertRaises((ValueError, RuntimeError)):
            self.evidence()

    def test_invalid_proof_fields_rejected(self):
        for key, value in [('rejection_kind', 'unknown'), ('rejection_kind', None),
                           ('regions', 49), ('backend_compiles', 999),
                           ('representative_metrics', {}), ('session_id', 'stale'),
                           ('parameter_identity_preserved', False)]:
            with self.subTest(key=key, value=value):
                self.session = 'controlled-' + key + str(value)
                self.produce('unsupported')
                path = self.coverage / forward.utilization_name(self.session, 0, 1)
                row = json.loads(path.read_text())
                if value is None:
                    row.pop(key)
                else:
                    row[key] = value
                path.write_text(json.dumps(row))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.evidence()


if __name__ == '__main__':
    unittest.main()
