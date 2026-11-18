"""Local-only contracts; optional torch/PEFT tests need no downloaded models."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h3_personal import identity_checkpoint as cp

try:
    import torch
except ImportError:
    torch = None

PINS = {"sources": cp.SOURCE_PINS, "models": {
    "transformer_ref": {"repo_id": "test/local-only", "revision": "a" * 40}}}
CONFIG = dict(peft_type="LORA", r=16, lora_alpha=16, lora_dropout=0,
              target_modules=list(cp.TARGETS), bias="none")


class Contracts(unittest.TestCase):
    def test_lazy_import_and_gate(self):
        subprocess.run([sys.executable, "-S", "-c",
                        f"import sys; sys.path.insert(0, {str(ROOT / 'research')!r}); "
                        "from h3_personal import identity_checkpoint as c; "
                        "assert not c.C_GATE_ENABLED; "
                        "assert all(x not in sys.modules for x in ('torch','peft','safetensors'))"],
                       check=True)

    def test_pins_and_config(self):
        cp.validate_pins(PINS)
        cp.validate_config(CONFIG)
        for key, value in [('r', 8), ('lora_alpha', 32), ('lora_dropout', .1),
                           ('target_modules', ['to_q']), ('rank_pattern', {'to_q': 8})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                cp.validate_config(dict(CONFIG, **{key: value}))
        for pins in ({}, dict(PINS, sources={}), dict(PINS, models={}),
                     dict(PINS, models={'x': {'repo_id': 'x', 'revision': 'main'}})):
            with self.assertRaises(ValueError):
                cp.validate_pins(pins)

    def test_diffusers_active_adapters_method(self):
        model = NS(peft_config={'default': NS(to_dict=lambda: deepcopy(CONFIG))},
                   active_adapters=lambda: ['default'])
        self.assertEqual(cp._config(model), CONFIG)
        model.active_adapters = lambda: ['other']
        with self.assertRaises(ValueError):
            cp._config(model)

    def test_partial_artifact_never_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / cp.WEIGHTS).write_bytes(b'partial')
            with self.assertRaises(FileNotFoundError):
                cp.inspect_export(path, expected_pins=PINS)
            (path / cp.EXPORTED).write_text('{}')
            with self.assertRaises(ValueError):
                cp.inspect_export(path, expected_pins=PINS)


@unittest.skipIf(torch is None, "optional torch environment")
class TensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'checkpoint'
        self.calls = 0
        # JSON fake backend intentionally needs neither PEFT nor safetensors.
        def save(state, path):
            Path(path).write_text(json.dumps({k: {'dtype': str(v.dtype).split('.')[-1],
                'shape': list(v.shape), 'data': v.flatten().tolist()} for k, v in state.items()}))
        def load(path, device='cpu'):
            return {k: torch.tensor(v['data'], dtype=getattr(torch, v['dtype'])).reshape(v['shape'])
                    for k, v in json.loads(Path(path).read_text()).items()}
        def setter(model, state, adapter_name):
            self.calls += 1
            model.state = {k: v.clone() for k, v in state.items()}
            return NS(missing_keys=['base.weight'], unexpected_keys=[])
        self.b = cp.Boundary(lambda m, **kw: m.state, setter, save, load, torch)
        self.model = self.fresh()
        self.model.state['layer.lora_A.weight'].add_(2)
        cp.export_adapter(self.model, self.path, pins=PINS, boundary=self.b)

    def fresh(self):
        config = deepcopy(CONFIG)
        return NS(peft_config={'default': NS(to_dict=lambda: config)},
                  active_adapters=['default'], state={
                      'layer.lora_A.weight': torch.zeros(16, 3),
                      'layer.lora_B.weight': torch.zeros(2, 16)})

    def reload(self, model=None, **kwargs):
        return cp.strict_reload(model or self.fresh(), self.path, freshly_constructed=True,
                                expected_pins=PINS, boundary=kwargs.get('boundary', self.b))

    def reseal(self):
        marker = json.loads((self.path / cp.EXPORTED).read_text())
        marker['digests'] = {n: cp.sha256(self.path / n) for n in marker['digests']}
        (self.path / cp.EXPORTED).write_text(json.dumps(marker))

    def test_round_trip_and_no_resume(self):
        self.assertFalse((self.path / cp.VERIFIED).exists())
        with self.assertRaises(FileNotFoundError):
            cp.inspect_verified(self.path, expected_pins=PINS)
        fresh = self.fresh()
        self.reload(fresh)
        cp.inspect_verified(self.path, expected_pins=PINS)
        for k in fresh.state:
            self.assertTrue(torch.equal(fresh.state[k], self.model.state[k]))
        self.assertEqual(set(p.name for p in self.path.iterdir()),
                         {cp.WEIGHTS, cp.CONFIG, cp.RECEIPT, cp.EXPORTED, cp.VERIFIED})
        self.assertEqual(self.calls, 1)

    def test_reject_tensor_contract_before_set(self):
        original = self.b.load(str(self.path / cp.WEIGHTS))
        for kind in ('missing', 'extra', 'shape', 'dtype', 'nonfinite'):
            with self.subTest(kind=kind):
                state = {k: v.clone() for k, v in original.items()}
                key = next(iter(state))
                if kind == 'missing': del state[key]
                if kind == 'extra': state['extra'] = torch.ones(1)
                if kind == 'shape': state[key] = torch.ones(1)
                if kind == 'dtype': state[key] = state[key].double()
                if kind == 'nonfinite': state[key].flatten()[0] = float('nan')
                self.b.save(state, str(self.path / cp.WEIGHTS))
                self.reseal()  # defeats digest guard to exercise strict semantic checks
                with self.assertRaises(ValueError): self.reload()
                self.assertFalse((self.path / cp.VERIFIED).exists())
        self.assertEqual(self.calls, 0)

    def test_altered_config_and_corrupt_bytes(self):
        for name in (cp.CONFIG, cp.WEIGHTS, cp.RECEIPT):
            original = (self.path / name).read_bytes()
            (self.path / name).write_bytes(original + b'corrupt')
            with self.assertRaises(ValueError): self.reload()
            (self.path / name).write_bytes(original)
        config = json.loads((self.path / cp.CONFIG).read_text())
        config['init_lora_weights'] = False
        (self.path / cp.CONFIG).write_text(json.dumps(config))
        self.reseal()
        with self.assertRaises(ValueError): self.reload()
        self.assertEqual(self.calls, 0)

    def test_setter_omission_unexpected_and_inequality(self):
        for result in (NS(missing_keys=['x.lora_A.default.weight'], unexpected_keys=[]),
                       NS(missing_keys=[], unexpected_keys=['extra']),
                       NS(missing_keys=['base'], unexpected_keys=[])):
            b = cp.Boundary(self.b.get, lambda *a, **kw: result,
                            self.b.save, self.b.load, torch)
            with self.assertRaises(ValueError): self.reload(boundary=b)
            self.assertFalse((self.path / cp.VERIFIED).exists())

    def test_explicit_freshness_and_stale_marker(self):
        self.reload()
        with self.assertRaises(ValueError):
            cp.strict_reload(self.fresh(), self.path, freshly_constructed=False,
                             expected_pins=PINS, boundary=self.b)
        self.assertFalse((self.path / cp.VERIFIED).exists())

    def test_marker_failure_no_success(self):
        self.reload()
        with patch.object(cp.os, 'replace', side_effect=OSError('marker failure')):
            with self.assertRaises(OSError): self.reload()
        self.assertFalse((self.path / cp.VERIFIED).exists())
        self.assertFalse(list(self.path.glob('.*.tmp')))
        self.reload()
        (self.path / cp.VERIFIED).write_text('{}')
        with self.assertRaises(ValueError): cp.inspect_verified(self.path, expected_pins=PINS)

    def test_atomic_export_failure_and_immutable_destination(self):
        destination = self.path.parent / 'failed'
        def partial(state, path):
            Path(path).write_bytes(b'partial')
            raise OSError('disk failure')
        b = cp.Boundary(self.b.get, self.b.set, partial, self.b.load, torch)
        with self.assertRaises(OSError):
            cp.export_adapter(self.model, destination, pins=PINS, boundary=b)
        self.assertFalse(destination.exists())
        self.assertFalse(list(destination.parent.glob('.failed-*')))
        with patch.object(cp.os, 'rename', side_effect=OSError('publish failure')):
            with self.assertRaises(OSError):
                cp.export_adapter(self.model, destination, pins=PINS, boundary=self.b)
        self.assertFalse(destination.exists())
        with self.assertRaises(FileExistsError):
            cp.export_adapter(self.model, self.path, pins=PINS, boundary=self.b)
        cp.inspect_export(self.path, expected_pins=PINS)

    def test_lossless_cpu_contiguous_and_nonfinite_export(self):
        self.model.state = {k: v.t().double() for k, v in self.model.state.items()}
        dest = self.path.parent / 'double'
        cp.export_adapter(self.model, dest, pins=PINS, boundary=self.b)
        for k, v in self.b.load(str(dest / cp.WEIGHTS)).items():
            self.assertEqual(v.dtype, torch.float64)
            self.assertTrue(torch.equal(v, self.model.state[k]))
        self.model.state[next(iter(self.model.state))].flatten()[0] = float('inf')
        # Transposed flatten can copy; fill the actual tensor instead.
        next(iter(self.model.state.values())).fill_(float('inf'))
        with self.assertRaises(ValueError):
            cp.export_adapter(self.model, self.path.parent / 'bad', pins=PINS, boundary=self.b)


@unittest.skipIf(torch is None or importlib.util.find_spec('peft') is None,
                     'optional installed PEFT + torch only')
class RealPEFT(unittest.TestCase):
    def test_small_local_model_actual_boundary(self):
        from peft import LoraConfig, get_peft_model
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(3, 3)
                self.to_k = torch.nn.Linear(3, 3)
                self.to_v = torch.nn.Linear(3, 3)
                self.to_out = torch.nn.Sequential(torch.nn.Linear(3, 3))
        def fresh():
            return get_peft_model(Block(), LoraConfig(r=16, lora_alpha=16,
                                  lora_dropout=0, target_modules=list(cp.TARGETS)))
        trained = fresh()
        with torch.no_grad():
            for name, param in trained.named_parameters():
                if 'lora_' in name: param.add_(.125)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter'
            cp.export_adapter(trained, path, pins=PINS)
            cp.strict_reload(fresh(), path, freshly_constructed=True, expected_pins=PINS)
            cp.inspect_verified(path, expected_pins=PINS)


if __name__ == '__main__':
    unittest.main()
