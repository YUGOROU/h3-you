"""Operator acceptance preserves uncertainty; strict territory proof stays strict."""
import copy
from pathlib import Path
import tempfile
import unittest

from research import hf_identity_encode as encoder
from research import hf_identity_train as trainer


class OperatorApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        country = root / 'country.txt'
        license_file = root / 'acceptance.txt'
        country.write_text('Execution country unknown; official HF deployment observed.')
        license_file.write_text('Operator accepts contract interpretation risk, not rights-holder authorization.')
        common = dict(evidence_mode='operator_hf_acceptance', provider='huggingface_jobs',
                      approved=True, approved_by='operator', risk_acknowledged=True)
        self.encode = dict(common, schema='h3-compute-approval-v1', execution_country=None,
                           country_evidence=dict(path=str(country), sha256=encoder.cache.sha256(country),
                                                 description=country.read_text()),
                           license_approval_evidence=dict(path=str(license_file),
                               sha256=encoder.cache.sha256(license_file), description='Operator risk acceptance'))
        self.train = dict(common, compute_country=None, evidence_kind='operator_hf_acceptance',
                          evidence_file=str(license_file), evidence_sha256=trainer.sha256(license_file))

    def test_explicit_acceptance_without_country_or_blanket_permission(self):
        encoder.license_gate(self.encode)
        trainer.territory_gate(self.train)

    def test_required_operator_fields(self):
        for gate, original, country_key in ((encoder.license_gate, self.encode, 'execution_country'),
                                           (trainer.territory_gate, self.train, 'compute_country')):
            changes = {'provider': ['other', None], 'approved': [False, 1],
                       'approved_by': ['', '  ', True], 'risk_acknowledged': [False, 1],
                       country_key: ['JP', 'US', 'unknown'],
                       'license_allows_execution_country': [True]}
            for key, values in changes.items():
                for value in values:
                    with self.subTest(gate=gate.__name__, key=key, value=value), self.assertRaises(ValueError):
                        gate(dict(original, **{key: value}))
            for key in ('evidence_mode', 'provider', 'approved', 'approved_by',
                        'risk_acknowledged', country_key):
                bad = dict(original)
                del bad[key]
                with self.subTest(missing=key, gate=gate.__name__), self.assertRaises(ValueError):
                    gate(bad)

    def test_encoder_distinct_pins_and_description(self):
        for description in ('provider location', 'unknown HF deployment observed',
                            'official HF deployment observed', 'unknown official deployment observed'):
            bad = copy.deepcopy(self.encode)
            bad['country_evidence']['description'] = description
            with self.assertRaises(ValueError):
                encoder.license_gate(bad)
        bad = copy.deepcopy(self.encode)
        bad['country_evidence'] = bad['license_approval_evidence']
        with self.assertRaises(ValueError):
            encoder.license_gate(bad)
        for key in ('country_evidence', 'license_approval_evidence'):
            bad = copy.deepcopy(self.encode)
            bad[key]['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                encoder.license_gate(bad)

    def test_train_kind_and_pin(self):
        for change in ({'evidence_kind': 'rights_holder_execution_authorization'},
                       {'evidence_sha256': '0' * 64}, {'evidence_file': '/nonexistent/operator-proof'}):
            with self.assertRaises(ValueError):
                trainer.territory_gate(dict(self.train, **change))

    def test_strict_mode_unchanged(self):
        enc = dict(self.encode, execution_country='JP', license_allows_execution_country=True)
        enc.pop('evidence_mode')
        train = dict(self.train, compute_country='JP', evidence_kind='provider_compute_country_statement')
        train.pop('evidence_mode')
        encoder.license_gate(enc)
        trainer.territory_gate(train)
        for country in (None, 'US', 'GB', 'KR', 'FR', 'DE', 'ZZ', ''):
            with self.subTest(country=country):
                with self.assertRaises(ValueError):
                    encoder.license_gate(dict(enc, execution_country=country))
                with self.assertRaises(ValueError):
                    trainer.territory_gate(dict(train, compute_country=country))
        with self.assertRaises(ValueError):
            encoder.license_gate(dict(enc, license_allows_execution_country=False))
        with self.assertRaises(ValueError):
            trainer.territory_gate(dict(train, evidence_kind='operator_hf_acceptance'))


if __name__ == '__main__':
    unittest.main()
