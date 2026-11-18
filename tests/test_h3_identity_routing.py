import unittest

from research.h3_personal.identity_routing import target_video_routing_plan


class RoutingContractTests(unittest.TestCase):
    def plan(self, **overrides):
        args = dict(sequence_length=20, video_indices=[3, 4, *range(10, 20)],
                    video_loss_mask=[False, False, *([True] * 10)],
                    bypass_ratio=0.5, seed=42, training=True, latent_frames=32)
        args.update(overrides)
        return target_video_routing_plan(**args)

    def test_only_target_video_is_bypassed(self):
        p = self.plan()
        self.assertEqual(len(p.bypass_indices), 5)
        self.assertTrue(set(p.bypass_indices) <= set(range(10, 20)))
        self.assertTrue(set(range(10)) <= set(p.keep_indices))
        self.assertEqual(p.keep_indices, tuple(sorted(p.keep_indices)))
        self.assertEqual(set(p.keep_indices) | set(p.bypass_indices), set(range(20)))
        self.assertFalse(set(p.keep_indices) & set(p.bypass_indices))

    def test_seeded_reproducibility_without_global_rng(self):
        import random
        before = random.getstate()
        self.assertEqual(self.plan(), self.plan())
        self.assertEqual(before, random.getstate())

    def test_inference_stills_and_zero_ratio_are_identity(self):
        for overrides in ({"training": False}, {"latent_frames": 1}, {"bypass_ratio": 0}):
            with self.subTest(overrides=overrides):
                p = self.plan(**overrides)
                self.assertEqual(p.keep_indices, tuple(range(20)))
                self.assertEqual(p.bypass_indices, ())

    def test_never_bypasses_all_targets(self):
        p = self.plan(bypass_ratio=0.999)
        self.assertEqual(len(set(p.keep_indices) & set(range(10, 20))), 1)

    def test_single_target_retained(self):
        p = self.plan(video_indices=[3, 10], video_loss_mask=[False, True])
        self.assertEqual(p.bypass_indices, ())

    def test_mask_maps_video_order_not_sequence_tail(self):
        p = self.plan(video_indices=[19, 3, 10, 4], video_loss_mask=[True, False, True, False])
        self.assertEqual(len(p.bypass_indices), 1)
        self.assertTrue(set(p.bypass_indices) <= {10, 19})
        self.assertTrue({3, 4} <= set(p.keep_indices))

    def test_invalid_contracts_fail_closed_even_in_inference(self):
        invalid = [dict(bypass_ratio=x) for x in (-1, 1, float('nan'), float('inf'), True)]
        invalid += [dict(video_indices=[3, 3], video_loss_mask=[False, True]),
                    dict(video_indices=[20], video_loss_mask=[True]),
                    dict(video_loss_mask=[True]), dict(video_loss_mask=[1] * 12),
                    dict(video_loss_mask=[False] * 12), dict(sequence_length=True),
                    dict(start_block=47, end_block=2), dict(end_block=51),
                    dict(training='yes'), dict(seed=True), dict(latent_frames=0)]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.plan(**override)


if __name__ == '__main__':
    unittest.main()
