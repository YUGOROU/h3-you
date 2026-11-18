"""No weights/network. Stdlib tests always run; tensor tests are optional."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import install_identity_guidance as installer
from h3_personal import identity_guidance as guidance

try:
    import torch
except ImportError:
    torch = None

SNAPSHOT = Path(os.environ.get("H3_AVIAD_AUDIT_SOURCE", "/tmp/h3-identity-audit/aviad"))


class StaticTests(unittest.TestCase):
    def test_gate_and_import_are_stdlib_safe(self):
        self.assertFalse(guidance.C_GATE_ENABLED)
        subprocess.run([sys.executable, "-S", "-c",
                        f"import sys; sys.path.insert(0, {str(ROOT / 'research')!r}); "
                        "import install_identity_guidance; from h3_personal import identity_guidance; "
                        "assert 'torch' not in sys.modules"], check=True)

    def test_settings_and_cache_binding(self):
        for scale, threshold in [(float('nan'), .15), (0, .15), (4, -1), (4, float('inf'))]:
            with self.assertRaises(ValueError):
                guidance.check_settings(scale, threshold)
        with tempfile.TemporaryDirectory() as tmp:
            cond, ref = Path(tmp) / 'cond', Path(tmp) / 'ref'
            cond.write_bytes(b'conditional'); ref.write_bytes(b'reference')
            meta = dict(schema=guidance.CACHE_SCHEMA, aviad_revision=guidance.AVIAD_REVISION,
                        sample_id='a', probe_prompt=' ',
                        conditional_sha256=guidance.file_sha256(cond), reference_sha256=guidance.file_sha256(ref))
            guidance.validate_cache_metadata(meta, 'a', cond, ref)
            with self.assertRaises(ValueError):
                guidance.validate_cache_metadata({}, 'a', cond, ref)
            ref.write_bytes(b'other reference')
            with self.assertRaises(ValueError):
                guidance.validate_cache_metadata(meta, 'a', cond, ref)

    def test_training_contract_rejects_unsupported_paths(self):
        strategy = NS(identity_guidance_scale=4., identity_guidance_sigma_min=.15,
                      video=NS(is_generated=True, conditions=[NS(type='reference', modality='image', probability=1.)]),
                      audio=NS(is_generated=False, conditions=[]))
        cfg = NS(training_strategy=strategy, model=NS(variant='ref2va', training_mode='lora'),
                 optimization=NS(batch_size=1), lora=NS(dropout=0.),
                 acceleration=NS(strategy='ddp', quantization='none', mixed_precision_mode='bf16'))
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}):
            guidance.validate_training_config(cfg)
            for obj, key, value in [(strategy.audio, 'is_generated', True),
                                    (strategy.video.conditions[0], 'probability', .5),
                                    (cfg.model, 'variant', 'fl2va'),
                                    (cfg.optimization, 'batch_size', 2),
                                    (cfg.acceleration, 'strategy', 'deepspeed_zero3'),
                                    (cfg.acceleration, 'quantization', 'nf4-bnb')]:
                old = getattr(obj, key)
                setattr(obj, key, value)
                with self.assertRaises(ValueError):
                    guidance.validate_training_config(cfg)
                setattr(obj, key, old)
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}):
            with self.assertRaises(ValueError):
                guidance.validate_training_config(cfg)

    def test_context_is_exact(self):
        for text in ('missing', 'xx'):
            with self.assertRaises(ValueError):
                installer.substitute(text, 'x', 'y')
        self.assertEqual(installer.substitute('x', 'x', 'y'), 'y')

    def test_hash_and_file_set_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'src/h3_trainer/config.py'
            path.parent.mkdir(parents=True); path.write_text('not pinned')
            with patch.object(installer, 'PINS', {'src/h3_trainer/config.py': installer.PINS['src/h3_trainer/config.py']}):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    installer.plan(root)
                (root / 'src/extra.py').write_text('')
                with self.assertRaisesRegex(ValueError, 'file set'):
                    installer.plan(root)

    def test_install_new_tree_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            with patch.object(installer, 'plan', return_value={'src/h3_trainer/a.py': 'x = 1\n'}):
                installer.install(Path(tmp), out)
                self.assertTrue((out / 'src/h3_trainer/identity_guidance.py').exists())
                self.assertFalse((out / 'weights').exists())
                with self.assertRaises(FileExistsError):
                    installer.install(Path(tmp), out)

    @unittest.skipUnless(SNAPSHOT.exists(), 'optional pinned local source snapshot unavailable')
    def test_real_exact_patch_plan(self):
        planned = installer.plan(SNAPSHOT)
        for path, text in planned.items():
            ast.parse(text)
        trainer = planned['src/h3_trainer/trainer.py']
        self.assertEqual(trainer.count('prediction_video, prediction_audio = training_forward('), 2)
        self.assertIn('validate_training_config(config)', trainer)
        strategy = planned['src/h3_trainer/training_strategies/flexible.py']
        self.assertIn('return attach_probe(packed, batch, sigmas', strategy)
        self.assertIn('guidance_video_loss(prediction_video, packed, self.config)', strategy)
        builder = planned['src/h3_trainer/preprocessing/builder.py']
        self.assertIn('encode_pair(self, record, row)', builder)
        self.assertIn('write_uncond_cache(options.output, sample_id, path, uncond, uncond_tags)', builder)
        self.assertIn('load_uncond_cache(self, record, text_payload)', planned['src/h3_trainer/datasets.py'])
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'trainer'
            installer.install(SNAPSHOT, destination)
            # Reinstallation and source drift are rejected, not silently tolerated.
            with self.assertRaises(ValueError):
                installer.plan(destination)


@dataclass
class PackedStub:
    clean_video: object
    noise_video: object
    video_loss_mask: object
    metadata: dict = field(default_factory=dict)


@unittest.skipIf(torch is None, 'torch absent; use a separate optional ML environment')
class TensorTests(unittest.TestCase):
    def test_math_and_detachment(self):
        clean = torch.tensor([[[3.], [5.]]], requires_grad=True)
        noise = torch.ones_like(clean, requires_grad=True)
        probe = torch.tensor([[[7.], [9.]]], requires_grad=True)
        target = guidance.guided_target(clean, noise, probe, 4)
        torch.testing.assert_close(target, torch.tensor([[[-13.], [-11.]]]))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(guidance.guided_target(clean, noise, probe, 1), clean - noise)
        with self.assertRaises(ValueError):
            guidance.guided_target(clean, noise, probe[:, :1], 4)

    def test_pre_shift_threshold_and_probe_order_lora_active(self):
        for u, expected_calls in [(0.149, 1), (0.15, 2), (0.9, 2)]:
            packed = PackedStub(torch.tensor([[[3.]]]), torch.tensor([[[1.]]]), torch.tensor([False, True]))
            probe = object()
            packed.metadata.update(identity_probe=probe, identity_base_sigma=u)
            calls = []
            # Represents an active trainable LoRA delta; the same forward is used twice.
            delta = torch.nn.Parameter(torch.tensor(2.))
            def forward(value):
                calls.append((value, torch.is_grad_enabled(), float(delta.detach())))
                return delta * torch.ones(1, 2, 1), torch.ones(1, 1, 1)
            guidance.training_forward(forward, packed, NS(identity_guidance_scale=4., identity_guidance_sigma_min=.15))
            self.assertEqual(len(calls), expected_calls)
            self.assertIs(calls[-1][0], packed)
            self.assertTrue(calls[-1][1])
            if expected_calls == 2:
                self.assertIs(calls[0][0], probe)
                self.assertFalse(calls[0][1])
                self.assertEqual(calls[0][2], calls[-1][2])
            self.assertFalse(packed.metadata['identity_video_target'].requires_grad)

    def test_mask_video_only_gradient_and_no_broadcast(self):
        packed = PackedStub(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.tensor([False, True]))
        packed.metadata['identity_video_target'] = torch.tensor([[[2.]]], requires_grad=True)
        pred = torch.tensor([[[999.], [4.]]], requires_grad=True)
        audio_pred = torch.tensor([[[12.]]], requires_grad=True)
        cfg = NS(identity_guidance_scale=4.)
        # Actual strategy separately sets audio_weight=0 and audio_loss=zero.
        loss = guidance.video_loss(pred, packed, cfg)
        self.assertEqual(float(loss.detach()), 4.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[0.], [4.]]]))
        self.assertIsNone(audio_pred.grad)
        self.assertIsNone(packed.metadata['identity_video_target'].grad)
        with self.assertRaises(ValueError):
            guidance.video_loss(pred.expand(2, -1, -1), packed, cfg)

    def test_cache_encoder_preserves_reference_objects(self):
        refs = [object()]
        calls = []
        def encode(prompt, references):
            calls.append((prompt, references))
            return torch.ones(2, 3), torch.ones(2, dtype=torch.int64)
        text_pass = NS(options=NS(references=True, keyframes=()),
                       _rebuild_reference_presentation=lambda record, row: refs,
                       encoders=NS(encode_ref2va_prompt=encode))
        guidance.encode_pair(text_pass, {'caption': 'identity', 'reference_geometry': [[0, 1, 2, 2, 0]]}, {'reference_image': 'synthetic'})
        self.assertEqual([call[0] for call in calls], ['identity', ' '])
        self.assertIs(calls[0][1], calls[1][1])

    @unittest.skipUnless(SNAPSHOT.exists(), 'optional pinned snapshot unavailable')
    def test_installed_dataset_strategy_and_real_packer_cpu(self):
        # Run isolated so imports cannot contaminate other project tests.
        if importlib.util.find_spec('safetensors') is None or importlib.util.find_spec('diffusers') is None:
            self.skipTest('optional safetensors/Diffusers integration dependencies absent')
        with tempfile.TemporaryDirectory() as tmp:
            tree = installer.install(SNAPSHOT, Path(tmp) / 'trainer')
            script = r'''
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from safetensors.torch import save_file
from h3_trainer.constants import Geometry, MINIMAX_H3_TEXT_TAG as T, MINIMAX_H3_VIDEO_TAG as V
from h3_trainer.datasets import PrecomputedDataset, bucket_collate
from h3_trainer.config import FlexibleStrategyConfig
from h3_trainer.flow_matching import SigmaPair
from h3_trainer.training_strategies.flexible import FlexibleStrategy
from h3_trainer.identity_guidance import write_uncond_cache, training_forward
root = Path(__import__('sys').argv[1]); root.mkdir()
g = Geometry.create(32, 32, 22)
for directory in ('latents', 'audio_latents', 'conditions', 'reference_latents'):
    (root/directory).mkdir()
def save(directory, payload):
    save_file(payload, str(root/directory/'a.safetensors'))
save('latents', {'latents': torch.ones(g.video_rows, 4)})
save('audio_latents', {'latents': torch.ones(g.audio_rows, 32)})
save('conditions', {'prompt_embeds': torch.ones(5, 8), 'text_token_tags': torch.tensor([T,V,V,T,T])})
save('reference_latents', {'latents': torch.ones(1, 4), 'geometry': torch.tensor([[0,1,2,2,0]])})
write_uncond_cache(root, 'a', root/'conditions/a.safetensors', torch.zeros(4,8), torch.tensor([T,V,V,T]))
record = dict(id='a', width=32, height=32, num_frames=22, video_rows=g.video_rows,
              audio_rows=g.audio_rows, text_rows=5, has_audio=True)
(root/'index.json').write_text(json.dumps({'samples':[record]}))
cfg = FlexibleStrategyConfig.model_validate(dict(identity_guidance_scale=4.,
    video=dict(is_generated=True, conditions=[dict(type='reference', modality='image', probability=1.)]),
    audio=dict(is_generated=False, latents_dir='audio_latents')))
dataset = PrecomputedDataset(root, cfg.data_sources(), val_split_every=0)
batch = bucket_collate([dataset[0]])
strategy = FlexibleStrategy(cfg)
packed = strategy.prepare(batch, SigmaPair.from_u(.15), 'cpu', torch.float32)
probe = packed.metadata['identity_probe']
assert probe.video is packed.video and probe.audio is packed.audio
assert probe.clean_audio is packed.clean_audio
assert probe.text.shape[1] == 4 and packed.text.shape[1] == 5
assert torch.equal(packed.audio, packed.clean_audio)
# Text length moves absolute indices and the shared rotary clock, not spatial grids.
for key in ('video_indices','audio_indices'):
    i, j = packed.layout_kwargs[key], probe.layout_kwargs[key]
    a, b = packed.layout_kwargs['position_ids'], probe.layout_kwargs['position_ids']
    expected = a.index_select(0,i).clone()
    expected[:,0] += probe.text.shape[1] - packed.text.shape[1]
    torch.testing.assert_close(expected, b.index_select(0,j))
# Per-row modality times also match; clean driving audio has t=1.
assert torch.equal(packed.timestep, probe.timestep)
assert torch.equal(packed.timestep_indices[packed.layout_kwargs['audio_indices']],
                   probe.timestep_indices[probe.layout_kwargs['audio_indices']])
parameter = torch.nn.Parameter(torch.tensor(2.))
calls = []
def forward(p):
    calls.append(torch.is_grad_enabled())
    return parameter * torch.ones_like(p.video), parameter * torch.ones_like(p.audio)
pv, pa = training_forward(forward, packed, cfg)
loss = strategy.compute_loss(pv, pa, packed)
assert calls == [False, True]
assert loss.audio_weight == 0 and float(loss.audio) == 0
loss.total.backward()
assert parameter.grad is not None
# Below threshold must be exactly raw flow with no probe forward.
packed = strategy.prepare(batch, SigmaPair.from_u(.149), 'cpu', torch.float32)
calls.clear(); pv, pa = training_forward(forward, packed, cfg)
assert calls == [True]
raw = packed.clean_video - packed.noise_video
assert torch.equal(packed.metadata['identity_video_target'], raw)
# A reference-free probe with valid file bindings still fails the vision-tag check.
write_uncond_cache(root, 'a', root/'conditions/a.safetensors', torch.zeros(4,8), torch.tensor([T,T,T,T]))
try: dataset[0]
except ValueError: pass
else: raise AssertionError('reference-free cache accepted')
# Stale reference bindings and absent caches fail closed at actual dataset access.
write_uncond_cache(root, 'a', root/'conditions/a.safetensors', torch.zeros(4,8), torch.tensor([T,V,V,T]))
save('reference_latents', {'latents': torch.zeros(1, 4), 'geometry': torch.tensor([[0,1,2,2,0]])})
try: dataset[0]
except ValueError: pass
else: raise AssertionError('stale reference cache accepted')
(root/'identity_uncond/a.safetensors').unlink()
try: dataset[0]
except Exception: pass
else: raise AssertionError('missing cache accepted')
'''
            env = dict(os.environ, PYTHONPATH=str(tree / 'src'))
            result = subprocess.run([sys.executable, '-c', script, str(Path(tmp) / 'cache')],
                                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
