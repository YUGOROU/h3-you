"""Synthetic CPU tensors only; no H3/adapter downloads or GPU Jobs."""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from research.h3_personal.identity_frozen_adapter import (
    FrozenAdapterSession, FrozenProjection, configure_subject_scope, verify_subject_scope,
)

try:
    import torch
except ImportError:
    torch = None


class LazyImports(unittest.TestCase):
    def test_no_heavy_import_at_module_import(self):
        subprocess.run([sys.executable, '-S', '-c',
                        'import research.h3_personal.identity_frozen_adapter; '
                        'import sys; assert "torch" not in sys.modules'], check=True)


@unittest.skipIf(torch is None, 'optional torch CPU environment')
class FrozenBranchTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).double()
        model.requires_grad_(False)
        return model

    def projection(self, target='0'):
        return FrozenProjection(target, torch.tensor([[1., 2., 3.]], dtype=torch.float64),
                                torch.tensor([[2.], [-1.]], dtype=torch.float64), 0.5)

    def test_output_input_grad_and_frozen_storage(self):
        model, p = self.model(), self.projection()
        p.a.requires_grad_(True)
        p.b.requires_grad_(True)
        x = torch.tensor([[2., 1., -1.]], dtype=torch.float64, requires_grad=True)
        weights = model[0].weight.detach().clone()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with FrozenAdapterSession(model, [p]):
            actual = model(x)
            effective = weights + p.scale * p.b.detach() @ p.a.detach()
            torch.testing.assert_close(actual, x @ effective.T)
            actual.sum().backward()
            torch.testing.assert_close(x.grad, effective.sum(0).unsqueeze(0))
            self.assertIsNone(p.a.grad)
            self.assertIsNone(p.b.grad)
            self.assertTrue(all(not b.requires_grad for b in model.buffers()))
            self.assertEqual(set(model.state_dict()), set(before))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]))
        torch.testing.assert_close(model(x), x @ weights.T)
        self.assertEqual(len(model[0]._forward_hooks), 0)
        self.assertEqual(list(model.buffers()), [])

    def test_nested_suspend_exception_and_cleanup(self):
        model, p = self.model(), self.projection()
        x = torch.ones(1, 3, dtype=torch.float64)
        baseline = model(x)
        session = FrozenAdapterSession(model, [p])
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with session:
                changed = model(x)
                self.assertFalse(torch.equal(changed, baseline))
                with self.assertRaisesRegex(RuntimeError, 'nested'):
                    with session.suspended():
                        torch.testing.assert_close(model(x), baseline)
                        with session.suspended():
                            torch.testing.assert_close(model(x), baseline)
                        raise RuntimeError('nested')
                torch.testing.assert_close(model(x), changed)
                raise RuntimeError('synthetic')
        session.close()  # idempotent
        torch.testing.assert_close(model(x), baseline)
        with self.assertRaises(RuntimeError):
            session.__enter__()
        with self.assertRaises(RuntimeError):
            with session.suspended():
                pass

    def test_keyword_input_and_buffer_dtype_movement(self):
        model, p = self.model(), self.projection()
        with FrozenAdapterSession(model, [p]):
            model.float()
            x = torch.ones(1, 3)
            expected = x @ (model[0].weight + p.scale * p.b.float() @ p.a.float()).T
            torch.testing.assert_close(model[0](input=x), expected)
            self.assertEqual({b.dtype for b in model.buffers()}, {torch.float32})

    def test_existing_session_cannot_be_overwritten(self):
        model, p = self.model(), self.projection()
        with FrozenAdapterSession(model, [p]):
            with self.assertRaises(ValueError):
                with FrozenAdapterSession(model, [p]):
                    pass
            self.assertEqual(len(model[0]._forward_hooks), 1)
        self.assertEqual(len(model[0]._forward_hooks), 0)

    def test_clone_isolates_caller_storage(self):
        model, p = self.model(), self.projection()
        with FrozenAdapterSession(model, [p]):
            x = torch.ones(1, 3, dtype=torch.float64)
            expected = model(x)
            p.a.zero_()
            p.b.zero_()
            torch.testing.assert_close(model(x), expected)

    def test_invalid_projection_and_duplicate_fail_before_mutation(self):
        model, p = self.model(), self.projection()
        bad = [FrozenProjection('0', p.a, torch.zeros(3, 1), 1),
               FrozenProjection('0', p.a, p.b, float('nan')),
               FrozenProjection('0', p.a * float('inf'), p.b, 1)]
        for projections in ([p, p], *([p, q] for q in bad)):
            with self.subTest(projections=len(projections)), self.assertRaises(ValueError):
                with FrozenAdapterSession(model, projections):
                    pass
            self.assertEqual(len(model[0]._forward_hooks), 0)
            self.assertFalse(hasattr(model[0], '_h3_conditioning_owner'))
        model[0].weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            with FrozenAdapterSession(model, [p]):
                pass

    def test_registration_failure_rolls_back(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)).double()
        model.requires_grad_(False)
        p = self.projection()
        q = FrozenProjection('1', torch.ones(1, 2), torch.ones(2, 1), 1)
        with patch.object(model[1], 'register_forward_hook', side_effect=RuntimeError('registration')):
            with self.assertRaises(RuntimeError):
                with FrozenAdapterSession(model, [p, q]):
                    pass
        for layer in model:
            self.assertEqual(len(layer._forward_hooks), 0)
            self.assertFalse(hasattr(layer, '_h3_conditioning_owner'))
            self.assertEqual(list(layer.buffers()), [])

    def test_nonreentrant_checkpoint_recompute_preserves_gradient(self):
        from torch.utils.checkpoint import checkpoint
        model, p = self.model(), self.projection()
        x = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
        with FrozenAdapterSession(model, [p]):
            out = checkpoint(model, x, use_reentrant=False)
            out.square().sum().backward()
            effective = model[0].weight + p.scale * p.b @ p.a
            expected = 2 * (x.detach() @ effective.T) @ effective
            torch.testing.assert_close(x.grad, expected)

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'optional PEFT environment')
    def test_actual_peft_canonical_export_reload_excludes_frozen_branch(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from research.h3_personal import identity_checkpoint as cp

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(3, 3, bias=False)
                self.to_k = torch.nn.Linear(3, 3, bias=False)
                self.to_v = torch.nn.Linear(3, 3, bias=False)
                self.to_out = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False))

            def forward(self, x):
                return self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))

        base = Tiny()
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0,
                            target_modules=list(cp.TARGETS), bias='none')
        model = get_peft_model(base, config)
        p = FrozenProjection('base_model.model.to_q', torch.ones(1, 3), torch.ones(3, 1), 0.5)
        before = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
        pins = {'sources': cp.SOURCE_PINS, 'models': {
            'transformer_ref': {'repo_id': 'test/synthetic', 'revision': 'a' * 40}}}
        x = torch.ones(2, 3)
        optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad], lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'adapter'
            with FrozenAdapterSession(model, [p]) as session:
                cp._config(model)  # sole default adapter, strict constraints unchanged
                model(x).square().sum().backward()
                optimizer.step()
                state = get_peft_model_state_dict(model)
                self.assertEqual(set(state), set(before))
                self.assertTrue(any(not torch.equal(state[k], before[k]) for k in state))
                self.assertTrue(all('_h3_conditioning' not in k for k in model.state_dict()))
                cp.export_adapter(model, destination, pins=pins)
                with session.suspended():
                    subject_only_output = model(x).detach()
            fresh_base = Tiny()
            fresh_base.load_state_dict(base_state, strict=True)
            fresh = get_peft_model(fresh_base, config)
            cp.strict_reload(fresh, destination, freshly_constructed=True, expected_pins=pins)
            cp.inspect_verified(destination, expected_pins=pins)
            torch.testing.assert_close(fresh(x), subject_only_output)
            self.assertTrue(all('_h3_conditioning' not in k for k in get_peft_model_state_dict(fresh)))


@unittest.skipIf(torch is None, 'optional torch CPU environment')
@unittest.skipUnless(importlib.util.find_spec('peft'), 'optional PEFT environment')
class SubjectScopeRuntimeTests(unittest.TestCase):
    def model(self):
        from peft import LoraConfig, get_peft_model
        from research.h3_personal import identity_checkpoint as cp

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(3, 3, bias=False)
                self.to_k = torch.nn.Linear(3, 3, bias=False)
                self.to_v = torch.nn.Linear(3, 3, bias=False)
                self.to_out = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False))

            def forward(self, x):
                return x + 0.01 * self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def forward(self, x):
                return self.attn(x)

        class TinyPackedStructure(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(50)])
                self.token_refiner = torch.nn.Module()
                self.token_refiner.refiner_blocks = torch.nn.ModuleList([Block() for _ in range(2)])

            def forward(self, x):
                for block in self.token_refiner.refiner_blocks:
                    x = block(x)
                for block in self.transformer_blocks:
                    x = block(x)
                return x

        return get_peft_model(TinyPackedStructure(), LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0, target_modules=list(cp.TARGETS), bias='none'))

    def test_scoped_update_full_export_and_fresh_reload(self):
        from research.h3_personal import identity_checkpoint as cp
        from peft import get_peft_model_state_dict
        model = self.model()
        base = {k: v.clone() for k, v in model.base_model.model.state_dict().items()}
        scope = configure_subject_scope(model)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        verify_subject_scope(model, scope, optimizer=optimizer)
        before = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
        self.assertEqual(len(before), 416)
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        verify_subject_scope(model, scope, optimizer=optimizer)
        after = get_peft_model_state_dict(model)
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))
        self.assertEqual(set(before), set(after))
        pins = {'sources': cp.SOURCE_PINS, 'models': {
            'transformer_ref': {'repo_id': 'test/synthetic-52-blocks', 'revision': 'a' * 40}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter'
            cp.export_adapter(model, path, pins=pins)
            fresh = self.model()
            fresh.base_model.model.load_state_dict(base, strict=True)
            cp.strict_reload(fresh, path, freshly_constructed=True, expected_pins=pins)
            torch.testing.assert_close(fresh(torch.ones(1, 3)), model(torch.ones(1, 3)))

    def test_nonzero_excluded_delta_rejected_before_flag_mutation(self):
        model = self.model()
        flags = {k: p.requires_grad for k, p in model.named_parameters()}
        key = 'base_model.model.transformer_blocks.0.attn.to_q.lora_B.default.weight'
        with torch.no_grad():
            dict(model.named_parameters())[key].fill_(1)
        with self.assertRaises(ValueError):
            configure_subject_scope(model)
        self.assertEqual(flags, {k: p.requires_grad for k, p in model.named_parameters()})

    def test_wrong_optimizer_and_post_update_drift_rejected(self):
        model = self.model()
        scope = configure_subject_scope(model)
        bad_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaises(ValueError):
            verify_subject_scope(model, scope, optimizer=bad_optimizer)
        key = scope.frozen[0]
        parameters = dict(model.named_parameters())
        parameters[key].grad = torch.ones_like(parameters[key])
        with self.assertRaises(ValueError):
            verify_subject_scope(model, scope)
        parameters[key].grad = None
        bkey = next(k for k in scope.frozen if '.lora_B.' in k)
        with torch.no_grad():
            parameters[bkey].fill_(1)
        with self.assertRaises(ValueError):
            verify_subject_scope(model, scope)


if __name__ == '__main__':
    unittest.main()
