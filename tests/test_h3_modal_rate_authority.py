import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from research.modal_gpu_plan import authorized_rate_limit, hourly


class RateAuthorityTests(unittest.TestCase):
    def test_only_pinned_modal_japan_h200_50usd_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rate.json'
            proof = dict(schema='h3-modal-rate-approval-v1', approved=True, approved_by='synthetic-user',
                approval_text='synthetic explicit approval', provider='modal', gpu='H200', region='jp',
                rate_usd_hour=hourly('H200', 'jp'), max_total_usd=50)
            path.write_text(json.dumps(proof))
            r = dict(rate_usd_hour=proof['rate_usd_hour'], overall_seconds=12380,
                territory_approval=dict(provider='modal', compute_country='JP'),
                compute_rate_approval=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            self.assertEqual(authorized_rate_limit(r), proof['rate_usd_hour'])
            with self.assertRaises(ValueError): authorized_rate_limit(dict(r, overall_seconds=20000))
            with self.assertRaises(ValueError): authorized_rate_limit(dict(r, rate_usd_hour=14.847336))
            proof['max_total_usd'] = 51
            path.write_text(json.dumps(proof))
            with self.assertRaises(ValueError): authorized_rate_limit(r)
        self.assertEqual(authorized_rate_limit(dict(rate_usd_hour=5)), 10)
        with self.assertRaises((KeyError, ValueError)):
            authorized_rate_limit(dict(rate_usd_hour=11.854836))


if __name__ == '__main__': unittest.main()
