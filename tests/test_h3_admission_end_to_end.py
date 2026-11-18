"""Real pinned native prepare rejection through both-rank receipt consumers, CPU only."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_h3_local_denoise_compile as native_fixtures
from research.h3_personal.denoise_execution import (
    CandidateUnavailable, DenoiseController, RequestAdapter)
from research.h3_personal import fast_ref2va as forward
from research.h3_personal import fast_continuation_render as renderer
from research import hf_h3_utilization_comparison as worker


class FakeModel:
    def __init__(self):
        self.parameter = NS(dtype='torch.bfloat16', shape=(5376, 5376))

    def named_parameters(self):
        return [('transformer_blocks.0.weight', self.parameter)]

    def state_dict(self, keep_vars=False):
        assert keep_vars
        return dict(self.named_parameters())


class FakeStatus:
    def __init__(self):
        self.labels = []
        self.votes = []

    def min(self, ok, label):
        self.labels.append(label)
        return ok

    def rejection_vote(self, code, label):
        self.votes.append((code, label))
        return code


class AdmissionEndToEndTests(unittest.TestCase):
    session = '../private-admission-session'

    @classmethod
    def setUpClass(cls):
        native_fixtures.Tests.setUpClass()

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.coverage = Path(temp.name)
        self.rank = 0
        # No Tensor, compiler, or CUDA API: only receipt rank lookup is available.
        for context in (
            patch.dict('sys.modules', {'torch': NS(distributed=NS(get_rank=lambda: self.rank))}),
            patch.dict(os.environ, {'H3_REF2VA_COVERAGE_DIR': str(self.coverage)}),
        ):
            context.start()
            self.addCleanup(context.stop)

    def produce(self):
        fixture = native_fixtures.Tests()
        for rank in (0, 1):
            self.rank = rank
            block, namespace, events, _ = fixture.make()
            import copy
            blocks = [block] + [copy.copy(block) for _ in range(49)]
            import types
            for item in blocks:
                item.forward = types.MethodType(type(block).forward, item)
            block.ff.chunk_tokens = 4
            original = [b.forward for b in blocks]
            status = FakeStatus()

            def forbidden(*args, **kwargs):
                self.fail('prepare rejection must not compile or execute tensors')

            owner = DenoiseController(FakeModel(), blocks, status=status,
                compiler=forbidden, backend=forbidden,
                max_compile_count=worker.DENOISE_POLICY['max_backend_compiles'],
                max_compile_seconds=180)
            identity = owner._identity()
            with self.assertRaises(CandidateUnavailable) as caught:
                owner.install(source=fixture.source, namespace=namespace,
                    admission=fixture.config()['admission'],
                    stage_shapes=fixture.config()['stage_shapes'],
                    gates=dict(source_pinned=True, real_bf16_weights=True,
                        static_adapter_receipt_verified=True, main50_verified=True))
            self.assertEqual(caught.exception.__cause__.code, 'ffn-configuration-unsupported')
            self.assertEqual(owner.state, 'faulted')
            self.assertEqual(owner._identity(), identity)
            self.assertEqual([b.forward for b in blocks], original)
            self.assertFalse(any(hasattr(b, '_h3_region_b_dispatch') for b in blocks))
            self.assertEqual(events, [])
            self.assertEqual(status.labels, ['prepare-main50'])
            self.assertEqual(len(status.votes), 1)
            self.assertEqual(status.votes[0][1], 'prepare-main50')
            self.assertGreater(status.votes[0][0], 0)
            self.assertIs(owner.receipt['source_pinned_verified'], True)
            self.assertIs(owner.receipt['main50_verified'], True)
            self.assertIs(owner.receipt['identity_unchanged'], True)
            adapter = RequestAdapter.__new__(RequestAdapter)
            adapter.controller, adapter.session_id = owner, self.session
            adapter.window_index, adapter.policy = 0, dict(worker.DENOISE_POLICY)
            adapter._write_receipt()
            forward._write_utilization(NS(), dict(session_id=self.session, index=0),
                None, worker.DENOISE_POLICY, adapter, 0., 1., False)
        for path in self.coverage.iterdir():
            self.assertEqual(path.parent, self.coverage)
            self.assertNotIn('private-admission-session', path.name)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def evidence(self):
        return renderer.rejection_evidence(NS(coverage=self.coverage),
            dict(session_id=self.session, index=0), worker.DENOISE_POLICY)

    def test_real_typed_prepare_finding_reaches_both_rank_consumers(self):
        self.produce()
        evidence = self.evidence()
        rows = worker.rejected_evidence({'utilization': evidence}, 'candidate_warmup')
        self.assertEqual(len(rows), 2)
        self.assertFalse(evidence['owner_complete'])
        self.assertEqual({r['rank'] for r in evidence['ranks']}, {0, 1})
        for row in evidence['ranks']:
            self.assertEqual(row['prepare_error_code'], 'ffn-configuration-unsupported')
            self.assertEqual(row['backend_compiles'], 0)
            self.assertIsNone(row['representative_metrics'])
            self.assertEqual(row['regions'], 50)
            self.assertIs(row['parameter_identity_preserved'], True)
            self.assertIs(row['candidate_rejected'], True)
            self.assertEqual(row['rejection_kind'], 'controlled-admission-rejection')

    def test_one_bad_rank_proof_denies_finding(self):
        self.produce()
        evidence = self.evidence()
        evidence['ranks'][1]['source_pinned_verified'] = False
        with self.assertRaises((ValueError, RuntimeError)):
            worker.rejected_evidence({'utilization': evidence}, 'candidate_warmup')
        path = self.coverage / forward.utilization_name(self.session, 0, 1)
        row = json.loads(path.read_text())
        row['source_pinned_verified'] = False
        path.write_text(json.dumps(row))
        with self.assertRaises((ValueError, RuntimeError)):
            self.evidence()


if __name__ == '__main__':
    unittest.main()
