import unittest
from research.h3_personal.turbo_adapter import validate_contract
from research.launch_hf_identity_pipeline import hardware_rate
from types import SimpleNamespace


class TurboAdapterTests(unittest.TestCase):
    def fixture(self):
        return dict(metadata={'alpha':'8'},tensor_count=624,other_tensor_names=[],
                    peft_modules=[dict(rank=128) for _ in range(312)])

    def test_fixed_rank_is_not_alpha(self):
        self.assertEqual(len(validate_contract(self.fixture())),312)
        r=self.fixture();r['metadata']['alpha']='128'
        with self.assertRaises(ValueError):validate_contract(r)

    def test_no_partial_or_custom_head(self):
        r=self.fixture();r['peft_modules'].pop()
        with self.assertRaises(ValueError):validate_contract(r)
        r=self.fixture();r['other_tensor_names']=['pdd.head']
        with self.assertRaises(ValueError):validate_contract(r)

    def test_two_gpu_price_is_total_configuration(self):
        api=SimpleNamespace(list_jobs_hardware=lambda:[SimpleNamespace(name='h200x2',
            accelerator='2xH200',unit_label='hour',unit_cost_usd=10)])
        self.assertEqual(hardware_rate(api,'h200x2'),10)
        with self.assertRaises(ValueError):hardware_rate(api,'b300')
        api.list_jobs_hardware=lambda:[SimpleNamespace(name='h200x2',accelerator='2xH200',
            unit_label='minute',unit_cost_usd=.166667)]
        # Do not silently round $10.00002 down through the explicit $10 cap.
        with self.assertRaises(ValueError):hardware_rate(api,'h200x2')
