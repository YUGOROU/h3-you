import copy
from types import SimpleNamespace as NS
import unittest
from research.h3_personal.execution_budget import PAIR,cap
from research.h3_personal.compute_price_approval import RATE
from research.launch_hf_identity_pipeline import budget
from test_h3_compute_price_approval import proof


class ExplicitPairTests(unittest.TestCase):
    def proposal(self):
        p=proof();p['scope']='h3-fp8-streaming-experiment-v1';p['parallel_pair']=copy.deepcopy(PAIR);return p
    def ledger(self):
        return dict(spent_usd=143.39,cumulative_cap_usd=170,approved_parallel_pair=copy.deepcopy(PAIR),reservations={
            'cpu':dict(flavor='cpu-upgrade',upper_bound_usd=.09),
            'ancillary':dict(upper_bound_usd=.91)})
    def test_default_unchanged_and_explicit_pair_cap(self):
        self.assertEqual(cap(None),150);self.assertEqual(cap(proof()),150);self.assertEqual(cap(self.proposal()),170)
        p=self.proposal();p['parallel_pair']['cumulative_cap_usd']=200
        with self.assertRaises(ValueError):cap(p)
    def test_first_allocation_and_second_parallel(self):
        ledger=self.ledger();p=self.proposal()
        total,upper=budget(ledger,[],RATE,4500,price_approval=p)
        self.assertAlmostEqual(total,144.39);self.assertAlmostEqual(upper,12.500025)
        ledger['reservations']['first']=dict(job_id='first',state='submitted_no_retry',pair_id=PAIR['id'],
            scope='h3-corrected-native-continuation-v1',flavor='h200x2',rate_usd_hour=RATE,timeout_seconds=4500,upper_bound_usd=upper)
        active=NS(id='first',status=NS(stage='RUNNING'))
        self.assertAlmostEqual(budget(ledger,[active],RATE,4500,price_approval=p)[0],156.890025)
        ledger['reservations']['first']['scope']=p['scope']
        with self.assertRaises(ValueError):budget(ledger,[active],RATE,4500,price_approval=p)
    def test_no_unrelated_job_or_missing_authority_or_longer_timeout(self):
        ledger=self.ledger();p=self.proposal()
        with self.assertRaises(ValueError):budget(ledger,[],RATE,4501,price_approval=p)
        del ledger['approved_parallel_pair']
        with self.assertRaises(ValueError):budget(ledger,[],RATE,4500,price_approval=p)
        ledger=self.ledger()
        with self.assertRaises(ValueError):budget(ledger,[NS(id='unknown',status=NS(stage='RUNNING'))],RATE,4500,price_approval=p)
    def test_not_general_experiment_or_unbounded_spend_waiver(self):
        p=self.proposal();p['scope']='h3-four-condition-matrix-v1'
        with self.assertRaises(ValueError):cap(p)
        ledger=self.ledger();ledger['spent_usd']=169
        with self.assertRaises(ValueError):budget(ledger,[],RATE,4500,price_approval=self.proposal())
