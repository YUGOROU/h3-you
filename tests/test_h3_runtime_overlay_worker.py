import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
from research import apply_identity_runtime_overlay as apply, install_identity_routing as overlay, hf_identity_train as worker

PIN = Path('/tmp/h3-identity-audit/aviad')
DIFF = Path('/tmp/h3-engine-review/diffusers-transformer.py')


@unittest.skipUnless(PIN.is_dir() and DIFF.is_file(), 'optional pristine source audit fixtures')
class OverlayWorkerTests(unittest.TestCase):
    def test_training_shadow_and_separate_pristine_verification_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = overlay.stage(PIN, DIFF, root / 'stage')
            base = root / 'pristine/diffusers'
            module = base / 'models/transformers/transformer_minimax_h3.py'
            module.parent.mkdir(parents=True)
            module.write_bytes(DIFF.read_bytes())
            (base / '__init__.py').write_text('# synthetic import-only fixture, not an H3 model\n')
            work = root / 'work'
            work.mkdir()
            request = dict(work=str(work), runtime_overlay=dict(stage=str(stage),
                stage_sha256=apply.sha256(stage / 'OVERLAY_MANIFEST.json'), base_diffusers=str(base),
                base_inventory_sha256=apply.inventory_digest(apply.inventory(base))))
            runtime = worker.materialize_runtime(request)
            self.assertTrue((work / 'runtime-overlay-pin.json').is_file())
            for training in (False, True):
                with self.subTest(training=training), patch.dict(sys.modules), patch.object(sys, 'path', list(sys.path)):
                    for name in list(sys.modules):
                        if name == 'diffusers' or name.startswith('diffusers.'):
                            sys.modules.pop(name)
                    sys.path.insert(0, str(base.parent))
                    trainer = ModuleType('h3_trainer')
                    trainer.__file__ = str(runtime / 'src/h3_trainer/__init__.py')
                    config = ModuleType('h3_trainer.config')
                    config.H3TrainerConfig = Mock()
                    sys.modules['h3_trainer'] = trainer
                    sys.modules['h3_trainer.config'] = config
                    with patch.object(worker, 'config_payload', return_value={}):
                        worker.load_runtime(request, training=training)
                        diffusers = importlib.import_module('diffusers')
                        expected = runtime / 'diffusers-shadow/diffusers' if training else base
                        self.assertEqual(Path(diffusers.__file__).parent, expected)
                        if training:
                            with self.assertRaises(ValueError): worker.load_runtime(request, training=False)
            (runtime / 'RUNTIME_OVERLAY.json').write_text('{}')
            with self.assertRaises(ValueError): worker.load_runtime(request)


if __name__ == '__main__': unittest.main()
