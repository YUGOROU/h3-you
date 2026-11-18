import unittest
from research import modal_gpu_plan as plan


class GPUPlanTests(unittest.TestCase):
    def test_official_prices_and_regional_multipliers(self):
        self.assertAlmostEqual(plan.hourly('H200'), 6.774192)
        self.assertAlmostEqual(plan.hourly('B200'), 8.484192)
        self.assertAlmostEqual(plan.hourly('B300'), 9.333792)
        self.assertAlmostEqual(plan.hourly('H200', 'jp'), 11.854836)
        self.assertAlmostEqual(plan.hourly('B200', 'jp'), 14.847336)
        self.assertAlmostEqual(plan.hourly('H200', 'ap'), 7.7903208)
        with self.assertRaises(ValueError): plan.hourly('H200', 'ap-southeast-1')
        with self.assertRaises(ValueError): plan.hourly('B200:2')

    def test_b300_rejected_by_current_cuda_comparison_no_quality_promises(self):
        value = plan.comparison(cuda='12.8', region='jp')
        self.assertFalse(value['B300']['cuda_minimum_met'])
        self.assertTrue(value['B200']['cuda_minimum_met'])
        self.assertFalse(value['B200']['production_guard_supported'])
        self.assertFalse(value['B200']['h3_gpu_verified'])
        self.assertFalse(value['B200']['quality_verified_on_this_gpu'])
        self.assertAlmostEqual(value['B200']['cost_break_even_speedup'], 1.25243, places=4)

    def test_no_fake_base_price_or_single_country_for_broad_pool(self):
        p = {'cloud': 'aws', 'region': 'jp', 'country': 'JP'}
        e = dict(p, provider='modal', approved=True, region_multiplier=1.75)
        self.assertAlmostEqual(plan.validate_placement(p, e), 11.854836)
        with self.assertRaises(ValueError): plan.validate_placement(p, dict(e, region_multiplier=1))
        with self.assertRaises(ValueError):
            plan.validate_placement(dict(p, region='ap'), dict(e, region='ap', region_multiplier=1.15))


if __name__ == '__main__': unittest.main()
