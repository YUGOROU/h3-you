"""No weights/GPU/network: preview conditioning and same-Job contracts."""
import unittest
from research.hf_identity_preview import case_inputs

class PreviewContracts(unittest.TestCase):
    def test_fixed_short_sample(self):
        sample={'id':'sample1','num_frames':124}
        self.assertIs(case_inputs({'samples':[sample]},'sample1'),sample)

    def test_missing_duplicate_and_long_cases_fail(self):
        for samples in ([],[{'id':'sample1','num_frames':124}]*2,[{'id':'sample1','num_frames':360}]):
            with self.assertRaises(ValueError):
                case_inputs({'samples':samples},'sample1')
