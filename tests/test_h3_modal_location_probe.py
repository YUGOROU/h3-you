import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from research.probe_modal_ref2va_location import classify,build_probe,RATE


class ProbeTests(unittest.TestCase):
    def test_only_documented_country_tokens_permit(self):
        for region,country in [('jp','JP'),('au','AU'),('ca','CA'),('mx','MX')]:
            self.assertEqual(classify(dict(MODAL_REGION=region,MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_GCP')),
                             dict(decision='allowed_candidate',country=country))

    def test_excluded_and_ambiguous_never_permit(self):
        for region in ('us','us-east','eu-west','uk'):
            self.assertEqual(classify(dict(MODAL_REGION=region,MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_AWS'))['decision'],'excluded')
        for region in ('ap','ap-northeast','us-east-1-fake','','jp '):
            self.assertNotEqual(classify(dict(MODAL_REGION=region,MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_AWS'))['decision'],'allowed_candidate')
        self.assertEqual(classify(dict(MODAL_REGION='jp'))['decision'],'unknown')

    def test_unplaced_no_secrets_no_source_and_finite_resources(self):
        captured={}
        class App:
            def __init__(self,name):pass
            def function(self,**kwargs):
                captured.update(kwargs)
                return lambda fn:fn
        fake=NS(App=App,Image=NS(debian_slim=lambda **kwargs:object()))
        with patch.dict(sys.modules,modal=fake):build_probe()
        for key in ('region','routing_region','secrets','volumes'):self.assertNotIn(key,captured)
        self.assertEqual(captured['gpu'],'H200:2')
        self.assertEqual(captured['cpu'],(1,1));self.assertEqual(captured['memory'],(1024,1024))
        self.assertTrue(captured['single_use_containers']);self.assertTrue(captured['block_network'])
        self.assertFalse(captured['include_source']);self.assertEqual(captured['retries'],0)
        self.assertLess(RATE,10)

if __name__=='__main__':unittest.main()
