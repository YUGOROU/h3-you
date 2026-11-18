"""CPU-only admission consumer regressions."""
import copy
import unittest
from research import hf_h3_utilization_comparison as worker
import test_h3_utilization_comparison_worker as fixtures


class AdmissionConsumers(unittest.TestCase):
    def profile(self):
        profile = fixtures.UtilizationWorkerTests().rejection_profile()
        for row in profile['utilization']['ranks']:
            row.update(rejection_kind='controlled-admission-rejection',
                prepare_error_code='ffn-configuration-unsupported',
                source_pinned_verified=True, main50_verified=True,
                parameter_dtype_histogram={'torch.bfloat16': 500},
                native_type_hashes=['a' * 64], regions=50,
                state='faulted', owner_terminal='faulted', denoise_installed=False,
                representative_preflight_complete=False, representative_metrics=None,
                backend_compiles=0, extra_status_collectives=2)
        return profile

    def test_both_rank_proof_and_sanitization(self):
        profile = self.profile()
        profile['utilization']['ranks'][0]['secret'] = 'do not copy'
        rows = worker.rejected_evidence(profile, 'candidate_warmup')
        self.assertNotIn('secret', rows[0])
        self.assertEqual(rows[1]['prepare_error_code'], 'ffn-configuration-unsupported')
        self.assertEqual(rows[0]['backend_compiles'], 0)
        self.assertIsNone(rows[0]['representative_metrics'])

    def test_bad_peer_denied(self):
        for key, value in [('prepare_error_code', 'unexpected-error'),
                           ('native_type_hashes', ['not-sha']),
                           ('source_pinned_verified', None),
                           ('main50_verified', 1),
                           ('parameter_dtype_histogram', {'private.name': 1}),
                           ('parameter_identity_preserved', False),
                           ('rejection_kind', 'unknown')]:
            profile = copy.deepcopy(self.profile())
            profile['utilization']['ranks'][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.rejected_evidence(profile, 'candidate_warmup')
