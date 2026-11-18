import dataclasses
import unittest

from research.h3_personal.accepted_profiles import (
    AcceptedQualityProfile, PROCESSOR_SHA256, choose_profile,
)
from research.h3_personal.conditioning_experiment import validate_manifest


class AcceptedProfilesTests(unittest.TestCase):
    def profile(self, name=None, **kwargs):
        return choose_profile(name, ordered_source_sha256=('a' * 64, 'b' * 64),
                              prompt_sha256=kwargs.pop('prompt_sha256', 'c' * 64), **kwargs)

    def test_default_and_explicit_only(self):
        p = self.profile()
        self.assertEqual(p.name, 'reference1024')
        self.assertEqual(p.expected_rt, (1824, 1867))
        q = self.profile('reference512')
        self.assertEqual(q.expected_prepared_geometry, (512, 896))
        self.assertEqual(q.expected_rt, (448, 491))
        self.assertNotEqual(p.cache_identity, q.cache_identity)
        for name in ('native2048', 'reference2048', '', False, 512, 'auto'):
            with self.assertRaises(ValueError):
                self.profile(name)

    def test_frozen_and_wire(self):
        p = self.profile()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.name = 'reference512'
        record = p.wire_record()
        record['ordered_source_sha256'].reverse()
        self.assertEqual(p.ordered_source_sha256[0], 'a' * 64)
        self.assertNotIn('raw_prompt', record)
        self.assertEqual(len(p.digest), 64)
        self.assertEqual(p.digest, self.profile().digest)
        self.assertNotEqual(p.digest, self.profile(prompt_sha256='d' * 64).digest)
        self.assertNotEqual(p.digest, dataclasses.replace(
            p, ordered_source_sha256=p.ordered_source_sha256[::-1]).digest)

    def test_strict_fixed_contract(self):
        p = self.profile()
        for field, value in [('fps', True), ('fps', 24.0), ('window', 125),
                             ('overlap', 21), ('forwards', 3), ('geometry', (256.0, 448)),
                             ('geometry', (252, 448)), ('exported_geometry', (256, 448)),
                             ('training_original_immutable', 1), ('resize_grid', 16),
                             ('precision', 'FP8'), ('processor_sha256', 'd' * 64)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                dataclasses.replace(p, **{field: value})
        for value in ('raw prompt', 'A' * 64, True):
            with self.assertRaises(ValueError):
                self.profile(prompt_sha256=value)
        with self.assertRaises(ValueError):
            AcceptedQualityProfile('reference1024', ['a' * 64], 'c' * 64)

    def test_binding_rejects_drift(self):
        p = self.profile()
        binding = dict(ordered_source_sha256=p.ordered_source_sha256,
                       prompt_sha256=p.prompt_sha256,
                       processor_sha256=PROCESSOR_SHA256, geometry=(256, 448))
        self.assertIs(p.validate_binding(**binding), p)
        for key, value in [('ordered_source_sha256', p.ordered_source_sha256[::-1]),
                           ('ordered_source_sha256', ('e' * 64,)),
                           ('prompt_sha256', 'd' * 64), ('processor_sha256', 'd' * 64),
                           ('geometry', (256.0, 448)), ('geometry', (252, 448))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.validate_binding(**dict(binding, **{key: value}))

    def test_explicit_legacy_adapter(self):
        for name, edge in [('reference1024', 1024), ('reference512', 512)]:
            manifest = self.profile(name).legacy_conditioning_manifest()
            self.assertEqual(validate_manifest(manifest), manifest)
            self.assertEqual(set(manifest), {'schema', 'short_edges', 'active_short_edge'})
            self.assertEqual(manifest['active_short_edge'], edge)


if __name__ == '__main__':
    unittest.main()
