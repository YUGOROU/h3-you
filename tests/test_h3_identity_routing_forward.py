import unittest
from types import SimpleNamespace as NS
from research.h3_personal.identity_routing_forward import routing_forward, validate_routing_config
try:
    import torch
except ImportError:
    torch = None


def config():
    condition = NS(type='reference', modality='image', probability=1.0)
    return NS(training_strategy=NS(identity_tread_ratio=0.5, identity_tread_start=2,
        identity_tread_end=47, identity_guidance_scale=None, uses_references=True,
        video=NS(is_generated=True, conditions=[condition]), audio=NS(is_generated=False, conditions=[])),
        model=NS(variant='ref2va', training_mode='lora'),
        optimization=NS(batch_size=1, gradient_accumulation_steps=1),
        acceleration=NS(strategy='ddp', quantization='none', mixed_precision_mode='bf16'), lora=NS(dropout=0))


class ConfigTests(unittest.TestCase):
    def test_valid_and_guidance_rejection(self):
        c = config()
        validate_routing_config(c)
        c.training_strategy.identity_guidance_scale = 4
        with self.assertRaises(ValueError):
            validate_routing_config(c)

    def test_unsupported_conditions_audio_and_accumulation(self):
        for change in ('audio', 'reference', 'accumulation'):
            c = config()
            if change == 'audio':
                c.training_strategy.audio.is_generated = True
            elif change == 'reference':
                c.training_strategy.video.conditions[0].probability = 0.5
            else:
                c.optimization.gradient_accumulation_steps = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_routing_config(c)


@unittest.skipIf(torch is None, 'optional torch CPU environment')
class ForwardTests(unittest.TestCase):
    def packed(self):
        clean = torch.randn(1, 1, 2)
        return NS(video=torch.ones(1, 5, 1), audio=clean.clone(), clean_audio=clean,
            audio_loss_mask=torch.tensor([False]), video_loss_mask=torch.tensor([False, True, True, True, True]),
            timestep=torch.tensor([0.4, 1.0]), timestep_indices=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
            sequence_length=8, layout_kwargs=dict(video_indices=torch.tensor([2, 4, 5, 6, 7]),
                audio_indices=torch.tensor([3]), position_ids=torch.arange(24).reshape(8, 3)),
            to_model_kwargs=lambda: {'sentinel': 'unchanged'})

    def call(self, packed, ratio=0.5):
        captured = {}
        routing_forward(lambda **kw: captured.update(kw), packed, ratio=ratio, seed=42, start_block=2, end_block=47)
        return captured

    def test_only_targets_routed_inputs_unchanged(self):
        p = self.packed()
        kw = self.call(p)
        self.assertEqual(kw['sentinel'], 'unchanged')
        plan = kw['identity_routing_plan']
        self.assertTrue(set(plan.bypass_indices) <= {4, 5, 6, 7})
        self.assertTrue({0, 1, 2, 3} <= set(plan.keep_indices))
        self.assertTrue(torch.equal(p.audio, p.clean_audio))

    def test_no_grad_and_default_omit_routing_argument(self):
        self.assertNotIn('identity_routing_plan', self.call(self.packed(), ratio=0))
        with torch.no_grad():
            self.assertNotIn('identity_routing_plan', self.call(self.packed()))

    def test_noisy_audio_or_nonclean_time_rejected(self):
        for change in ('noise', 'time', 'loss'):
            p = self.packed()
            if change == 'noise':
                p.audio += 1
            elif change == 'time':
                p.timestep[1] = 0.99
            else:
                p.audio_loss_mask[0] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.call(p)


if __name__ == '__main__':
    unittest.main()
