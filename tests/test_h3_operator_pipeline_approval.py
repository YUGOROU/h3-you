"""Controller/launcher closure tests, independent of encode/train gate implementation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import hf_identity_pipeline as driver
from research import launch_hf_identity_pipeline as launcher


class OperatorApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.documents = {}
        for key in ('country_evidence', 'license_approval_evidence'):
            p = self.root / (key + '.txt')
            p.write_text(key)
            self.documents[key] = {'path': p.name, 'sha256': driver.digest(p)}
        self.evidence = dict(schema='h3-compute-approval-v1', evidence_mode='operator_hf_acceptance',
                             execution_country=None, provider='huggingface_jobs', risk_acknowledged=True,
                             approved=True, approved_by='operator')
        for key, entry in self.documents.items():
            self.evidence[key] = dict(entry, path=str(self.root / entry['path']),
                                      description='HF hosting; country unknown, risk accepted')
        self.territory = dict(compute_country=None, provider='huggingface_jobs', risk_acknowledged=True,
                              approved=True, approved_by='operator', evidence_kind='operator_hf_acceptance',
                              evidence_file=self.evidence['country_evidence']['path'],
                              evidence_sha256=self.evidence['country_evidence']['sha256'])

    def check_driver(self, evidence, territory):
        # Mock only lower-level gates: these tests target cross-document closure.
        with patch.object(driver, 'pinned', return_value=evidence), \
             patch.object(driver.encoder, 'license_gate'), patch.object(driver.trainer, 'territory_gate'):
            driver.approvals({'approval': {'path': 'unused', 'sha256': 'unused'},
                              'territory_approval': territory})

    def test_unknown_country_acceptance_not_hosting_proof(self):
        self.check_driver(self.evidence, self.territory)

    def test_controller_rejects_mixed_modes_and_claims(self):
        for key, value in [('evidence_kind', 'provider_compute_country_statement'),
                           ('compute_country', 'JP'), ('provider', 'other'),
                           ('risk_acknowledged', False), ('approved_by', '  '),
                           ('evidence_sha256', '0' * 64), ('evidence_file', 'other')]:
            with self.subTest(key=key):
                t = dict(self.territory, **{key: value})
                with self.assertRaises(ValueError):
                    self.check_driver(self.evidence, t)

    def test_strict_hosting_rejection_preserved(self):
        e = dict(self.evidence, evidence_mode='provider_compute_country_statement', execution_country='JP')
        t = dict(self.territory, evidence_kind='provider_compute_country_statement', compute_country='JP')
        with self.assertRaises(ValueError):
            self.check_driver(e, t)

    def launcher_gates(self, territory):
        e = copy.deepcopy(self.evidence)
        for key, entry in self.documents.items():
            e[key]['path'] = '/deployment/' + entry['path']
        p = self.root / 'approval.json'
        p.write_text(json.dumps(e))
        documents = dict(self.documents, approval={'path': p.name, 'sha256': driver.digest(p)})
        t = dict(territory)
        t['evidence_file'] = e['country_evidence']['path']
        with patch.object(driver.encoder, 'license_gate'), patch.object(driver.trainer, 'territory_gate'):
            return launcher.inspect_spec({'documents': documents, 'pipeline': {'territory_approval': t}}, self.root)

    def test_launcher_acceptance_and_mismatch(self):
        gate = 'explicit territory/license evidence or HF operator risk acceptance'
        self.assertNotIn(gate, self.launcher_gates(self.territory))
        for key, value in [('compute_country', 'JP'), ('risk_acknowledged', False),
                           ('evidence_kind', 'provider_compute_country_statement'),
                           ('evidence_sha256', '0' * 64)]:
            with self.subTest(key=key):
                expected = ('explicit existing permitted execution-country/license evidence'
                            if key == 'evidence_kind' else gate)
                self.assertIn(expected, self.launcher_gates(dict(self.territory, **{key: value})))


if __name__ == '__main__':
    unittest.main()
