import os
import unittest
from unittest.mock import patch
from research.h3_personal.live_native_backend import LiveNativeBackend
from research.h3_personal.visual_ancestry_policy import ENV,VALUE,MANIFEST_KEY,COMPARATOR,CANDIDATE


class ReachedManifestValidation(Exception): pass


class ModeTests(unittest.TestCase):
    def test_dedicated_comparison_still_accepts_paired_manifest(self):
        owner=object.__new__(LiveNativeBackend);owner._visual_policy=None;owner.visual_intents=False
        for arm in (COMPARATOR,CANDIDATE):
            manifest={MANIFEST_KEY:dict(schema=VALUE,policy=arm)}
            with patch.dict(os.environ,{ENV:VALUE}),patch('research.h3_personal.live_native_backend.validate_manifest',side_effect=ReachedManifestValidation):
                with self.assertRaises(ReachedManifestValidation):owner(manifest)
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaises(ValueError):owner(manifest)

    def test_live_option_cannot_be_overridden(self):
        owner=object.__new__(LiveNativeBackend);owner._visual_policy=CANDIDATE;owner.visual_intents=False
        with patch.dict(os.environ,{ENV:VALUE}):
            with self.assertRaises(ValueError):owner({MANIFEST_KEY:dict(schema=VALUE,policy=COMPARATOR)})


if __name__=='__main__':unittest.main()
