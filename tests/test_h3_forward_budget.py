import unittest
from research.h3_personal.forward_budget import proposed_allocation, validate_allocation, exposure_outlook


class ForwardBudgetTests(unittest.TestCase):
    def outlook(self, **overrides):
        args = dict(provider='modal', hourly_rate='6.774192', seconds_per_update='84.75',
                    setup_seconds=1800, final_seconds=1200, preview_seconds=2700)
        args.update(overrides)
        return exposure_outlook(proposed_allocation(), **args)

    def test_allocation_protects_pipeline_inference_and_contingency(self):
        a = validate_allocation(proposed_allocation())
        self.assertEqual(sum(a['modal'].values()), 5541)
        self.assertEqual(sum(a['huggingface'].values()), 6000)
        self.assertEqual(sum(x['training'] for x in a.values()), 7000)
        self.assertEqual(sum(x['future_inference'] for x in a.values()), 2500)
        self.assertEqual(sum(x['pipeline'] for x in a.values()), 1000)
        self.assertEqual(sum(x['contingency'] for x in a.values()), 1041)

    def test_pools_cannot_be_transferred_or_oversubscribed(self):
        a = proposed_allocation()
        a['modal']['training'] += 1
        a['huggingface']['training'] -= 1
        with self.assertRaises(ValueError):
            validate_allocation(a)

    def test_baseline_insufficient_and_faster_hypothesis_not_quality(self):
        baseline = self.outlook()
        self.assertFalse(baseline['one_coverage_epoch_fits'])
        fast = self.outlook(seconds_per_update='25')
        self.assertTrue(fast['one_coverage_epoch_fits'])
        self.assertFalse(fast['quality_verified'])
        self.assertFalse(fast['throughput_measured'])
        self.assertTrue(fast['advance_approval_required'])
        self.assertEqual(fast['training_cap_usd'], '45')

    def test_overhead_can_exhaust_training_budget(self):
        o = self.outlook(setup_seconds=100000)
        self.assertEqual(o['conditional_updates'], 0)
        self.assertEqual(o['required_seconds_per_update_for_coverage'], 0)

    def test_bad_inputs_rejected(self):
        for args in (dict(hourly_rate='NaN'), dict(seconds_per_update='Infinity'),
                     dict(setup_seconds=True), dict(samples=0), dict(provider='unknown')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.outlook(**args)


if __name__ == '__main__':
    unittest.main()
