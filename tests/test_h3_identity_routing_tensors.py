import importlib.util
import unittest
from research.h3_personal.identity_routing import target_video_routing_plan, RoutingPlan
from research.h3_personal.identity_routing_tensors import route_packed_activations
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch CPU environment')
class RoutingTensorTests(unittest.TestCase):
    def fixture(self, training=True):
        p = target_video_routing_plan(sequence_length=8, video_indices=[1, 4, 5, 6, 7],
            video_loss_mask=[False, True, True, True, True], bypass_ratio=0.5,
            seed=42, training=training, latent_frames=32)
        h = torch.arange(24, dtype=torch.float64).reshape(1, 8, 3).requires_grad_()
        cos, sin = torch.randn(8, 6), torch.randn(8, 6)
        adaln = torch.arange(8)
        mask = torch.eye(8, dtype=torch.bool)
        return h, (cos, sin), adaln, mask, p

    def test_joint_slice_and_identity_bypass_gradient(self):
        h, rope, adaln, mask, p = self.fixture()
        state = route_packed_activations(h, rope, adaln, mask, p)
        idx = list(p.keep_indices)
        self.assertTrue(torch.equal(state.rotary_emb[0], rope[0][idx]))
        self.assertTrue(torch.equal(state.adaln_indices, adaln[idx]))
        self.assertTrue(torch.equal(state.attention_mask, mask[idx][:, idx]))
        out = state.rejoin(state.hidden_states * 2)
        expected = h.detach().clone()
        expected[:, idx] *= 2
        torch.testing.assert_close(out, expected)
        out.sum().backward()
        gradient = torch.ones_like(h)
        gradient[:, idx] = 2
        torch.testing.assert_close(h.grad, gradient)
        self.assertTrue(torch.equal(out.detach()[:, list(p.bypass_indices)], h.detach()[:, list(p.bypass_indices)]))

    def test_inference_identity_and_unmasked_path(self):
        h, rope, adaln, _, p = self.fixture(training=False)
        state = route_packed_activations(h, rope, adaln, None, p)
        self.assertIsNone(state.attention_mask)
        torch.testing.assert_close(state.rejoin(state.hidden_states), h)

    @unittest.skipUnless(importlib.util.find_spec('diffusers'), 'optional Diffusers environment')
    def test_actual_h3_block_checkpoint_slice_and_gradient(self):
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
        from torch.utils.checkpoint import checkpoint
        h, rope, _, mask, p = self.fixture()
        adaln = torch.arange(8) % 6  # two time embeddings x three modalities
        block = MiniMaxH3TransformerBlock(hidden_size=3, num_attention_heads=1,
            attention_head_dim=128, ffn_dim=32, time_embed_dim=16,
            norm_eps=1e-5, qk_norm_eps=1e-5).double()
        block.requires_grad_(False)
        temb = torch.randn(2, 16, dtype=torch.float64)
        reference_h = h.detach().clone().requires_grad_()
        reference = block(reference_h, temb, adaln, rope, mask)
        state = route_packed_activations(h, rope, adaln, mask, p)
        reduced = checkpoint(block, state.hidden_states, temb, state.adaln_indices,
                             state.rotary_emb, state.attention_mask, use_reentrant=False)
        actual = state.rejoin(reduced)
        idx = list(p.keep_indices)
        # Diagonal mask isolates each row: kept values/gradients must match the
        # full block exactly; bypass rows remain identity rather than block output.
        torch.testing.assert_close(actual[:, idx], reference[:, idx])
        actual.sum().backward()
        reference[:, idx].sum().backward()
        expected_grad = reference_h.grad.clone()
        expected_grad[:, list(p.bypass_indices)] = 1
        torch.testing.assert_close(h.grad, expected_grad)

    def test_invalid_partition_mask_and_rejoin_rejected(self):
        h, rope, adaln, mask, p = self.fixture()
        for bad in (RoutingPlan((0, 0), (), 2, 47), RoutingPlan((0,), tuple(range(8)), 2, 47)):
            with self.assertRaises(ValueError):
                route_packed_activations(h, rope, adaln, mask, bad)
        with self.assertRaises(ValueError):
            route_packed_activations(h, rope, adaln, mask.float(), p)
        state = route_packed_activations(h, rope, adaln, mask, p)
        with self.assertRaises(ValueError):
            state.rejoin(state.hidden_states[:, :-1])


if __name__ == '__main__':
    unittest.main()
