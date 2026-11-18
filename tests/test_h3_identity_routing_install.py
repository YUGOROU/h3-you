"""Source overlay and tiny real H3 forward tests; no remote weights/Jobs."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from research.install_identity_routing import DIFFUSERS_SHA256, patch_transformer, stage, inspect_stage
from research.h3_personal.identity_routing import target_video_routing_plan
from research.h3_personal import identity_routing_tensors
try:
    import torch
except ImportError:
    torch = None


class SourceStageTests(unittest.TestCase):
    def test_atomic_stage_receipt_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch('research.install_identity_routing.plan',
                return_value={'trainer/src/test.py': 'x = 1\n', 'diffusers/src/test.py': 'y = 2\n'}):
            destination = stage('source', 'diffusers', Path(tmp) / 'overlay')
            receipt = inspect_stage(destination)
            self.assertFalse(receipt['production_enabled'])
            self.assertEqual(len(receipt['files']), 2)
            with self.assertRaises(FileExistsError):
                stage('source', 'diffusers', destination)
            (destination / 'trainer/src/test.py').write_text('x = 3\n')
            with self.assertRaises(ValueError):
                inspect_stage(destination)

    def test_stage_failure_never_publishes_partial_tree(self):
        with tempfile.TemporaryDirectory() as tmp, patch('research.install_identity_routing.plan',
                return_value={'trainer/src/test.py': 'x = 1\n'}), patch.object(Path, 'write_text',
                side_effect=OSError('synthetic disk failure')):
            destination = Path(tmp) / 'overlay'
            with self.assertRaises(OSError):
                stage('source', 'diffusers', destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_nonpristine_source_rejected_without_dependencies(self):
        with self.assertRaises(ValueError):
            patch_transformer('untrusted source')


@unittest.skipIf(torch is None, 'optional torch CPU environment')
@unittest.skipUnless(importlib.util.find_spec('diffusers'), 'optional pinned Diffusers source')
class ActualForwardTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec('diffusers')
        self.source = (Path(spec.origin).parent / 'models/transformers/transformer_minimax_h3.py').read_text()
        if hashlib.sha256(self.source.encode()).hexdigest() != DIFFUSERS_SHA256:
            self.skipTest('local H3 transformer is not the exact pinned source')
        name = 'diffusers.models.transformers._h3_routing_test'
        module = types.ModuleType(name)
        module.__package__ = 'diffusers.models.transformers'
        module.__file__ = '<synthetic-source-overlay>'
        # Restore only our aliases, not the whole module registry: removing
        # lazily imported torch modules would cause duplicate kernel registration.
        for alias, value in {name: module, 'h3_trainer.identity_routing_tensors': identity_routing_tensors}.items():
            old = sys.modules.get(alias)
            sys.modules[alias] = value
            def restore(alias=alias, old=old):
                if old is None:
                    sys.modules.pop(alias, None)
                else:
                    sys.modules[alias] = old
            self.addCleanup(restore)
        exec(compile(patch_transformer(self.source), module.__file__, 'exec'), module.__dict__)
        self.cls = module.MiniMaxH3Transformer3DModel

    def model(self):
        return self.cls(num_attention_heads=1, attention_head_dim=8, hidden_size=6,
            num_layers=3, num_refiner_layers=1, ffn_dim=8, in_channels=1,
            audio_in_channels=2, patch_size=(1, 1, 1), text_dim=4, freq_dim=4,
            time_embed_hidden_dim=6, time_embed_dim=4, rope_freq_dim=1)

    def kwargs(self):
        return dict(hidden_states=torch.randn(1, 5, 1, requires_grad=True),
            audio_hidden_states=torch.randn(1, 1, 2), encoder_hidden_states=torch.randn(1, 2, 4),
            timestep=torch.tensor([0.4, 1.0]), timestep_indices=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
            token_tags=torch.tensor([1, 1, 0, 2, 0, 0, 0, 0]),
            position_ids=torch.arange(24).reshape(8, 3), video_indices=torch.tensor([2, 4, 5, 6, 7]),
            audio_indices=torch.tensor([3]), text_indices=torch.tensor([0, 1]), return_dict=False)

    def plan(self):
        return target_video_routing_plan(sequence_length=8, video_indices=[2, 4, 5, 6, 7],
            video_loss_mask=[False, True, True, True, True], bypass_ratio=0.5, seed=42,
            start_block=0, end_block=2, training=True, latent_frames=4)

    def test_default_output_matches_pristine_model(self):
        from diffusers import MiniMaxH3Transformer3DModel
        model = self.model()
        original = MiniMaxH3Transformer3DModel.from_config(model.config)
        original.load_state_dict(model.state_dict(), strict=True)
        inputs = self.kwargs()
        with torch.no_grad():
            actual, expected = model(**inputs), original(**inputs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_loop_reduces_and_restores_rows(self):
        model = self.model()
        model.requires_grad_(False)
        model.enable_gradient_checkpointing()
        seen = []
        handles = [b.register_forward_pre_hook(lambda m, args, i=i: seen.append((i, args[0].shape[1])))
                   for i, b in enumerate(model.transformer_blocks)]
        self.addCleanup(lambda: [h.remove() for h in handles])
        inputs = self.kwargs()
        video, audio = model(**inputs, identity_routing_plan=self.plan())
        self.assertEqual(seen[:3], [(0, 6), (1, 6), (2, 8)])
        self.assertEqual(video.shape, (1, 5, 1))
        self.assertEqual(audio.shape, (1, 1, 2))
        video.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs['hidden_states'].grad).all())
        self.assertTrue(all(n == (8 if i == 2 else 6) for i, n in seen))
        with torch.no_grad(), self.assertRaises(ValueError):
            model(**inputs, identity_routing_plan=self.plan())

    def test_modified_source_rejected(self):
        with self.assertRaises(ValueError):
            patch_transformer(self.source + '\n')


if __name__ == '__main__':
    unittest.main()
