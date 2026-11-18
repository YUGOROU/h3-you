"""Exact-source staging contracts; optional local tiny Aviad/PEFT execution."""
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research import install_identity_checkpoint as installer

SOURCE = Path(os.environ.get('H3_TRAINER_SOURCE', '/tmp/h3-identity-audit/aviad'))
AVAILABLE = (SOURCE / 'src/h3_trainer/checkpointing.py').is_file()


def pipeline():
    # Evaluate the wrapper without importing any ML dependencies.
    fake = NS(SOURCE_PINS={}, export_adapter=lambda *a, **k: None)
    namespace = {}
    with patch.dict(sys.modules, {'h3_trainer.identity_checkpoint': fake}):
        exec(installer.PIPELINE, namespace)
    return namespace


def config():
    return NS(checkpoints=NS(identity_peft_export=True, save_training_state='off'),
              model=NS(variant='ref2va', training_mode='lora', load_checkpoint=None),
              acceleration=NS(strategy='ddp', quantization='none'),
              lora=NS(rank=16, alpha=16, dropout=0,
                      target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))


class Contracts(unittest.TestCase):
    def test_runtime_preflight_no_resume_or_unsupported_backend(self):
        validate = pipeline()['validate_checkpoint_config']
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}):
            validate(config(), NS(world_size=1))
            for section, key, value in (
                ('model', 'load_checkpoint', '/old'), ('model', 'variant', 'fl2va'),
                ('model', 'training_mode', 'heads'), ('lora', 'rank', 8),
                ('lora', 'alpha', 32), ('lora', 'dropout', .1),
                ('lora', 'target_modules', ['to_q']),
                ('acceleration', 'strategy', 'deepspeed_zero3'),
                ('acceleration', 'quantization', 'int8'),
                ('checkpoints', 'save_training_state', 'full')):
                c = config()
                setattr(getattr(c, section), key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate(c, NS(world_size=1))
            with self.assertRaises(ValueError):
                validate(config(), NS(world_size=2))
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}), self.assertRaises(ValueError):
            validate(config(), NS(world_size=1))
        c = config()
        c.checkpoints.identity_peft_export = False
        c.model.load_checkpoint = '/old'
        validate(c, NS(world_size=1))  # legacy mode is unchanged, not strict validation

    def test_lazy_source_installer(self):
        subprocess.run([sys.executable, '-S', '-c',
            'from research import install_identity_checkpoint; import sys; '
            'assert all(n not in sys.modules for n in ("torch", "peft", "safetensors"))'],
            check=True)


@unittest.skipUnless(AVAILABLE, 'local audited Aviad source fixture unavailable')
class SourceTests(unittest.TestCase):
    def test_composition_ast_and_exact_helper(self):
        sources = installer.plan(SOURCE)
        previous = installer.sampling.plan(SOURCE)
        changed = {'src/h3_trainer/config.py', 'src/h3_trainer/trainer.py',
                   'src/h3_trainer/checkpointing.py'}
        for name in previous.keys() - changed:
            self.assertEqual(sources[name], previous[name])
        for name, text in sources.items():
            ast.parse(text, filename=name)
        self.assertEqual(sources['src/h3_trainer/identity_checkpoint.py'],
            (Path(installer.__file__).parent / 'h3_personal/identity_checkpoint.py').read_text())
        checkpoint = sources['src/h3_trainer/checkpointing.py']
        self.assertLess(checkpoint.index('export_checkpoint(model, staging)'),
                        checkpoint.index('if target.exists():'))
        self.assertLess(checkpoint.index('export_checkpoint(model, staging)'),
                        checkpoint.index('os.replace(staging, target)'))
        self.assertIn('identity_peft_export=config.checkpoints.identity_peft_export',
                      sources['src/h3_trainer/trainer.py'])

    def test_install_new_tree_and_reject_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'trainer'
            self.assertEqual(installer.install(SOURCE, target), target)
            self.assertEqual({str(p.relative_to(target)) for p in target.rglob('*.py')},
                             set(installer.plan(SOURCE)))
            with self.assertRaises(FileExistsError):
                installer.install(SOURCE, target)
            self.assertFalse(list(Path(tmp).glob('.h3-checkpoint-*')))

    def test_unknown_sha_set_symlink_context_and_ast_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            # Copy only the audited source subset, never cached weights or media.
            for name in installer.sampling.components.guidance.PINS:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / name, dest)
            path = root / 'src/h3_trainer/checkpointing.py'
            original = path.read_bytes()
            path.write_bytes(original + b'\n# unknown\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                installer.install(root, Path(tmp) / 'output')
            self.assertFalse((Path(tmp) / 'output').exists())
            path.write_bytes(original)
            extra = root / 'src/h3_trainer/unknown.py'
            extra.write_text('')
            with self.assertRaisesRegex(ValueError, 'file set'):
                installer.plan(root)
            extra.unlink()
            path.unlink()
            path.symlink_to(SOURCE / 'src/h3_trainer/checkpointing.py')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                installer.plan(root)
        sources = installer.sampling.plan(SOURCE)
        altered = deepcopy(sources)
        altered['src/h3_trainer/checkpointing.py'] = altered[
            'src/h3_trainer/checkpointing.py'].replace('    is_main_process: bool = True,', '')
        with patch.object(installer.sampling, 'plan', return_value=altered):
            with self.assertRaisesRegex(ValueError, 'context'):
                installer.plan(SOURCE)
        with patch.object(installer, 'PIPELINE', 'not python !!!'):
            with self.assertRaises(SyntaxError):
                installer.plan(SOURCE)


HAS_PEFT = all(importlib.util.find_spec(n) is not None
               for n in ('torch', 'peft', 'safetensors', 'diffusers'))

@unittest.skipUnless(AVAILABLE and HAS_PEFT, 'local Aviad and installed ML environment required')
class ActualAviadTinyPEFT(unittest.TestCase):
    def test_actual_trainer_save_and_canonical_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = installer.install(SOURCE, Path(tmp) / 'trainer')
            # Separate interpreter: execute installed modules without sys.modules leakage.
            subprocess.run([sys.executable, '-c', REAL_TEST, str(staged), tmp], check=True)


REAL_TEST = r'''
import ast, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
import torch
from diffusers.loaders import PeftAdapterMixin
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from h3_trainer import checkpointing as ck
from h3_trainer import identity_checkpoint as cp
from h3_trainer.identity_checkpoint_pipeline import PINS
from h3_trainer.lora import set_trainable

class Tiny(torch.nn.Module, PeftAdapterMixin):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(3, 3)
        self.to_k = torch.nn.Linear(3, 3)
        self.to_v = torch.nn.Linear(3, 3)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(3, 3))
model = Tiny()
set_trainable(model, 'lora', NS(rank=16, alpha=16, dropout=0,
    target_modules=list(cp.TARGETS), init_lora_weights=True))
with torch.no_grad():
    for name, p in model.named_parameters():
        if 'lora_' in name: p.add_(0.123456789)

# Compile ONLY the actual installed Trainer.save method, not a reimplementation.
trainer_ast = ast.parse((Path(sys.argv[1]) / 'src/h3_trainer/trainer.py').read_text())
method = next(node for cls in trainer_ast.body if isinstance(cls, ast.ClassDef)
              for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'save')
ns = dict(torch=torch, json=json, save_checkpoint=ck.save_checkpoint,
          prune_checkpoints=ck.prune_checkpoints)
exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual-trainer-save', 'exec'), ns)
class Config(NS):
    def model_dump_json(self): return '{}'
barriers, comfy = [], []
worker = NS(config=Config(checkpoints=NS(precision='bfloat16', save_training_state='off',
    identity_peft_export=True, keep_last_n=-1)), output_dir=Path(sys.argv[2]) / 'out',
    _core_model=lambda: model, state=ck.TrainingState(step=3), engine=None,
    optimizer=None, scheduler=None, context=NS(is_main=True, barrier=lambda: barriers.append(1)),
    _export_adapter=lambda p: comfy.append(p))
original_replace = ck.os.replace
seen = []
def replace(src, dst):
    src, dst = Path(src), Path(dst)
    assert src.name.endswith('.tmp') and not dst.exists()
    cp.inspect_export(src / 'identity-peft', expected_pins=PINS)
    assert not (src / 'identity-peft' / cp.VERIFIED).exists()
    seen.append(src)
    return original_replace(src, dst)
with patch.object(ck.os, 'replace', side_effect=replace):
    ns['save'](worker, 3)
target = ck.checkpoint_dir(worker.output_dir, 3)
assert len(seen) == len(barriers) == len(comfy) == 1
canonical = target / 'identity-peft'
cp.inspect_export(canonical, expected_pins=PINS)
assert not (canonical / cp.VERIFIED).exists()
assert not (target / 'optimizer.pt').exists()
assert not any((canonical / n).exists() for n in ('optimizer.pt', 'training_state.json'))
assert PINS['models']['transformer_ref']['revision'] == cp.SOURCE_PINS['h3']
expected = get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)
saved = load_file(str(canonical / cp.WEIGHTS))
assert saved.keys() == expected.keys()
for key in saved:
    assert saved[key].dtype == expected[key].dtype == torch.float32
    assert torch.equal(saved[key], expected[key].cpu())
raw = load_file(str(target / 'adapter.safetensors'))
assert raw and all(t.dtype == torch.bfloat16 for t in raw.values())
assert json.loads((canonical / cp.CONFIG).read_text())['r'] == 16

# Disabled legacy saves remain raw-only. Non-main/skipped gathers never export.
with patch.object(ck, 'export_checkpoint', side_effect=AssertionError('must not export')):
    legacy = ck.save_checkpoint(worker.output_dir, 4, model, worker.state)
    assert not (legacy / 'identity-peft').exists()
    assert ck.save_checkpoint(worker.output_dir, 5, model, worker.state,
        save_training_state='off', identity_peft_export=True, is_main_process=False) is None
    with patch.object(ck, 'gather_trainable_state', return_value=None):
        assert ck.save_checkpoint(worker.output_dir, 5, model, worker.state,
            save_training_state='off', identity_peft_export=True) is None

# Failed export cleans staging and preserves the already-published same-step target.
receipt = (canonical / cp.EXPORTED).read_bytes()
with patch.object(ck, 'export_checkpoint', side_effect=ValueError('export failed')):
    try: ns['save'](worker, 3)
    except ValueError: pass
    else: raise AssertionError('failure swallowed')
assert (canonical / cp.EXPORTED).read_bytes() == receipt
assert not target.with_suffix('.tmp').exists()
for kw in ({'engine': object(), 'save_training_state': 'off'}, {'save_training_state': 'full'}):
    try: ck.save_checkpoint(worker.output_dir, 6, model, worker.state, identity_peft_export=True, **kw)
    except ValueError: pass
    else: raise AssertionError('unsupported boundary accepted')
model.peft_config['default'].r = 8
try: ns['save'](worker, 7)
except ValueError: pass
else: raise AssertionError('unsupported actual config accepted')
assert not ck.checkpoint_dir(worker.output_dir, 7).exists()
assert not ck.checkpoint_dir(worker.output_dir, 7).with_suffix('.tmp').exists()
'''

if __name__ == '__main__':
    unittest.main()
