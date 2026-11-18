import copy
import unittest
from types import SimpleNamespace
from research.h3_personal.compute_price_approval import validate,request_limit,RATE
from research.launch_hf_identity_pipeline import hardware_rate,budget


def proof():
    return dict(schema='h3-hf-price-precision-approval-v1',approved=True,approved_by='test-user',
        approval_text='explicit finite test approval',provider='huggingface_jobs',flavor='h200x2',
        scope='h3-four-condition-matrix-v1',max_rate_usd_hour=RATE,
        max_timeout_seconds=4800,max_total_usd=13.34)


class PriceApprovalTests(unittest.TestCase):
    def test_exact_sku_price_only(self):
        self.assertEqual(validate(proof()),RATE)
        for key,value in [('flavor','b300'),('max_rate_usd_hour',11),('approved',False),('max_total_usd',50)]:
            p=proof();p[key]=value
            with self.assertRaises(ValueError):validate(p)
        api=SimpleNamespace(list_jobs_hardware=lambda:[SimpleNamespace(name='h200x2',
            accelerator='2xH200',unit_label='minute',unit_cost_usd=.166667)])
        self.assertEqual(hardware_rate(api,'h200x2',proof()),RATE)
        with self.assertRaises(ValueError):hardware_rate(api,'h200x2')

    def test_inference_only_finite_provider_bound(self):
        r=dict(hf_price_precision_approval=proof(),rate_usd_hour=RATE,preview_only=True,
               overall_seconds=4800,territory_approval={'provider':'huggingface_jobs'})
        self.assertEqual(request_limit(r),RATE)
        for patch in [dict(preview_only=False),dict(overall_seconds=4801),dict(rate_usd_hour=11),
                      dict(territory_approval={'provider':'modal'})]:
            with self.assertRaises(ValueError):request_limit(dict(r,**patch))

    def test_total_and_concurrency_cap(self):
        ledger=dict(spent_usd=113.49,reservations={'cpu':dict(upper_bound_usd=.09,flavor='cpu-upgrade')})
        total,upper=budget(ledger,[],RATE,4800,price_approval=proof())
        self.assertAlmostEqual(upper,13.33336)
        with self.assertRaises(ValueError):budget(ledger,[],RATE,4801,price_approval=proof())
        with self.assertRaises(ValueError):budget(ledger,[],RATE,4800)
        active=SimpleNamespace(id='x',status=SimpleNamespace(stage='RUNNING'))
        with self.assertRaises(ValueError):budget(ledger,[active],RATE,4800,price_approval=proof())
