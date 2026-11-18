from pathlib import Path
import tempfile
import unittest
from research import apply_identity_runtime_overlay as apply, install_identity_routing as source

PIN = Path('/tmp/h3-identity-audit/aviad')
DIFF = Path('/tmp/h3-engine-review/diffusers-transformer.py')


@unittest.skipUnless(PIN.is_dir() and DIFF.is_file(), 'optional pristine source audit fixtures')
class ApplyTests(unittest.TestCase):
    def test_isolated_source_copy_baseline_untouched_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = source.stage(PIN, DIFF, root / 'stage')
            base = root / 'pristine-diffusers'
            module = base / 'models/transformers/transformer_minimax_h3.py'
            module.parent.mkdir(parents=True)
            module.write_bytes(DIFF.read_bytes())
            (base / '__init__.py').write_text('# synthetic package inventory\n')
            baseline = apply.inventory_digest(apply.inventory(base))
            pin = apply.sha256(stage / 'OVERLAY_MANIFEST.json')
            target = apply.install(stage, stage_sha256=pin, base_diffusers=base,
                                   base_inventory_sha256=baseline, destination=root / 'runtime')
            receipt = apply.inspect(target, stage_sha256=pin)
            self.assertFalse(receipt['site_packages_modified'])
            self.assertFalse(receipt['actual_h3_gpu_verified'])
            self.assertEqual(apply.sha256(module), source.DIFFUSERS_SHA256)
            self.assertNotEqual(apply.sha256(target / 'diffusers-shadow/diffusers/models/transformers/transformer_minimax_h3.py'), source.DIFFUSERS_SHA256)
            self.assertTrue((target / 'src/h3_trainer/identity_routing_forward.py').is_file())
            (target / 'src/h3_trainer/config.py').write_text('pass\n')
            with self.assertRaises(ValueError): apply.inspect(target, stage_sha256=pin)
            with self.assertRaises(ValueError):
                apply.install(stage, stage_sha256=pin, base_diffusers=base,
                              base_inventory_sha256='0'*64, destination=root / 'bad')
            self.assertFalse((root / 'bad').exists())
            self.assertEqual(list(root.glob('.h3-runtime-overlay-*')), [])


if __name__ == '__main__': unittest.main()
