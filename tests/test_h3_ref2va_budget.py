import unittest
from decimal import Decimal
from research.h3_personal.ref2va_budget import budget_bound,HF_CAP,MODAL_SHARED
from research.h3_personal.realtime_first import budget_bound as legacy

class RefBudgetTests(unittest.TestCase):
    def test_new_pool_does_not_double_count_old_remaining(self):
        proof=budget_bound(timeout_seconds=900)
        self.assertEqual(HF_CAP,Decimal('221.672274')+Decimal('9.40'))
        self.assertEqual(proof['job_bound'],Decimal('2.500005'))
        self.assertEqual(proof['remaining'],HF_CAP-Decimal('169.35')-proof['job_bound'])
        self.assertEqual(MODAL_SHARED,Decimal('20.44'))

    def test_hf_cannot_spend_modal_funds_or_reset_origin(self):
        for total in (HF_CAP,Decimal('169.34'),Decimal('NaN')):
            with self.assertRaises(ValueError):budget_bound(ledger_bound=total)

    def test_new_live_pool_replaces_old_remainder(self):
        proof=budget_bound(ledger_bound=Decimal('221.672274'),timeout_seconds=1800)
        self.assertEqual(proof['remaining'],Decimal('9.40')-proof['job_bound'])
        self.assertEqual(proof['job_bound'],Decimal('5.000010'))

    def test_legacy_and_rate_limits_not_expanded(self):
        with self.assertRaises(ValueError):legacy(cap=HF_CAP)
        with self.assertRaises(ValueError):budget_bound(hourly_rate=Decimal('10.01'))
        with self.assertRaises(ValueError):budget_bound(timeout_seconds=3001)
