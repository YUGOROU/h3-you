import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from research import modal_identity_worker as worker


class CarrierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / 'research/modal_identity_worker.py'
        source.parent.mkdir()
        source.write_bytes(Path(worker.__file__).read_bytes())
        (self.root / 'entry.py').write_text('raise RuntimeError("must not execute locally")\n')
        self.spec = dict(schema='h3-modal-carrier-v1', files={str(p.relative_to(self.root)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*.py')},
            entrypoint='entry.py', seconds=1200, startup_seconds=600, cap_usd=4, region_multiplier=1)

    def test_bounded_exposure_includes_startup_and_scaledown(self):
        self.assertAlmostEqual(worker.validate(self.spec, self.root), worker.RATE * 1802 / 3600)
        bad = dict(self.spec, cap_usd=1)
        with self.assertRaises(ValueError): worker.validate(bad, self.root)

    def test_source_tamper_and_extra_files_rejected(self):
        (self.root / 'entry.py').write_text('pass\n')
        with self.assertRaises(ValueError): worker.validate(self.spec, self.root)
        (self.root / 'extra.json').write_text('{}')
        with self.assertRaises(ValueError): worker.validate(self.spec, self.root)

    def test_deadlines_premium_and_secret_fields_fail_closed(self):
        for changes in (dict(seconds=True), dict(seconds=0), dict(region_multiplier=1.15), dict(token='not-a-token')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                worker.validate(dict(self.spec, **changes), self.root)

    def test_import_does_not_mutate_bound_closure_with_image_environment(self):
        import json, os, subprocess, sys
        from research import modal_gpu_plan
        plan = self.root / 'research/modal_gpu_plan.py'
        plan.write_bytes(Path(modal_gpu_plan.__file__).read_bytes())
        self.spec['files'][str(plan.relative_to(self.root))] = hashlib.sha256(plan.read_bytes()).hexdigest()
        spec_path = self.root.parent / (self.root.name + '-spec.json')
        spec_path.write_text(json.dumps(self.spec))
        self.addCleanup(spec_path.unlink)
        code = ('import json;from research.modal_identity_worker import validate;'
                'validate(json.load(open(__import__("sys").argv[1])), ".")')
        subprocess.run([sys.executable, '-c', code, str(spec_path)], cwd=self.root,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(self.root)),
                       check=True, capture_output=True)
        self.assertFalse(list(self.root.rglob('*.pyc')))
        worker.validate(self.spec, self.root)

    def test_sdk_definition_hard_limits_and_no_remote_call(self):
        options = []
        class Image:
            @classmethod
            def from_registry(cls, image):
                self.assertEqual(image, worker.IMAGE)
                return cls()
            def add_local_dir(image, *args, **kwargs): return image
            def env(image, value):
                self.assertEqual(value, {'PYTHONPATH': '/deployment', 'PYTHONDONTWRITEBYTECODE': '1'})
                return image
        class App:
            def __init__(app, name): pass
            def function(app, **kwargs):
                options.append(kwargs)
                return lambda fn: fn
        sdk = NS(Image=Image, App=App, Secret=NS(
            from_name=lambda *a, **kw: NS(name=a[0], keys=kw['required_keys']),
            from_dict=lambda values: NS(values=values)))
        app, function = worker.build_app(self.spec, self.root, hf_secret_name='private-hub', modal_module=sdk)
        self.assertTrue(callable(function))
        opts = options[0]
        self.assertEqual(opts['gpu'], 'H200')
        self.assertEqual(opts['cpu'], (4, 4))
        self.assertEqual(opts['memory'], (262144, 262144))
        self.assertEqual(opts['ephemeral_disk'], 524288)
        self.assertEqual(opts['retries'], 0)
        self.assertEqual(opts['max_containers'], 1)
        self.assertEqual(opts['min_containers'], 0)
        self.assertTrue(opts['single_use_containers'])
        self.assertTrue(opts['serialized'])
        self.assertFalse(opts['include_source'])
        self.assertEqual(opts['secrets'][0].keys, ['HF_TOKEN'])
        _, dynamic = worker.build_app(self.spec, self.root, hf_token='synthetic-user-token', modal_module=sdk)
        self.assertEqual(options[1]['secrets'][0].values, {'HF_TOKEN': 'synthetic-user-token'})
        self.assertNotIn('hf_token', dynamic.__code__.co_freevars)
        self.assertNotIn('secret', dynamic.__code__.co_freevars)
        import json
        pricing = self.root / 'pricing.json'
        pricing.write_text(json.dumps(dict(provider='modal', approved=True, cloud='provider_selected',
                                           region='jp', country='JP', region_multiplier=1.75)))
        regional = copy.deepcopy(self.spec)
        regional['files']['pricing.json'] = hashlib.sha256(pricing.read_bytes()).hexdigest()
        regional.update(cap_usd=8, region_multiplier=1.75, placement=dict(cloud='provider_selected',
            region='jp', country='JP', pricing_evidence=dict(path='pricing.json', sha256=regional['files']['pricing.json'])))
        worker.build_app(regional, self.root, hf_token='synthetic-user-token', modal_module=sdk)
        self.assertEqual(options[-1]['region'], 'jp')
        self.assertNotIn('cloud', options[-1])
        with self.assertRaises(ValueError):
            worker.build_app(self.spec, self.root, hf_token='synthetic-user-token',
                             hf_secret_name='private-hub', modal_module=sdk)


if __name__ == '__main__': unittest.main()
