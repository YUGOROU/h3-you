"""CPU-only contracts: fake CUDA/imports, no network, models or Jobs."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from research import hf_restore_identity_runtime as restore
from research import prepare_identity_runtime as prep
from research.install_identity_guidance import PINS


class RuntimeRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        repo = Path(restore.__file__).resolve().parents[1]
        for path in (repo / 'research').rglob('*.py'):
            target = self.root / path.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copytree(prep.CONFIG, self.root / 'config')
        self.expectations = self.root / 'config/expectations.json'
        self.c, _ = prep.expectations(self.expectations, prep.sha256(self.expectations))
        self.source = self.root / 'upstream'
        for name in set(PINS) | {'pyproject.toml', 'LICENSE'}:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('source fixture')
        self.lock = {'schema': 'h3-encode-runtime-v1', 'versions': self.c['versions'],
                     'modules': {k: v['sha256'] for k, v in self.c['modules'].items()},
                     'ffmpeg': {'path': '/usr/bin/ffmpeg', 'sha256': 'a' * 64}}
        self.lock_path = self.root / 'runtime-lock.json'
        self.write(self.lock_path, self.lock)
        self.apt = {'package_stanza': 'Package: ffmpeg\nArchitecture: amd64\nVersion: 7:4.4.2-0ubuntu0.22.04.1\nSHA256: ' + 'b' * 64,
                    'deb_sha256': 'b' * 64, 'keyring_sha256': 'c' * 64,
                    'version_output': 'ffmpeg fixture\n'}
        self.report = {'schema': 'h3-identity-runtime-prep-report-v1', 'complete': True,
                       'cpu_imports_verified': True, 'weights_loaded': False, 'gpu_verified': False,
                       'image': self.c['image'], 'source_revisions': self.c['source_revisions'],
                       'expectations_sha256': prep.sha256(self.expectations),
                       'runtime_lock_sha256': prep.sha256(self.lock_path),
                       'independent_expectations': {k: self.c[k] for k in
                           ('versions', 'modules', 'transformers_source_revision', 'diffusers_inventory_sha256')},
                       'ffmpeg_package_evidence': self.apt}
        self.report_path = self.root / 'runtime-report.json'
        self.write(self.report_path, self.report)
        self.interpreter_path = self.root / 'base-interpreter.json'
        self.interpreter = {'schema': 'h3-runtime-base-interpreter-v1', 'complete': True,
                            'image': self.c['image'], 'cpu_report_sha256': prep.sha256(self.report_path),
                            'python': '/opt/conda/bin/python3.11', 'python_sha256': 'd' * 64}
        self.write(self.interpreter_path, self.interpreter)

    def write(self, path, document):
        path.write_text(json.dumps(document))

    def build(self):
        # Source fixtures aren't upstream bytes; source plan validation is separately tested by CPU prep.
        with patch.object(prep, 'check_trainer', return_value=self.source.resolve()):
            return restore.build_manifest(mount_root=self.root, cpu_report=self.report_path,
                runtime_lock=self.lock_path, expectations=self.expectations,
                interpreter_receipt=self.interpreter_path, source_root=self.source)

    def test_literal_launcher_contract_and_actual_code_pins(self):
        m = self.build()
        self.assertEqual(m['schema'], restore.SCHEMA)
        self.assertEqual(m['native_source_inventory'], 'PINS23+LICENSE')
        self.assertEqual(m['python_sha256'], 'd' * 64)
        self.assertEqual(m['runtime_lock_sha256'], prep.sha256(self.lock_path))
        self.assertEqual(m['source_hashes']['pyproject.toml'], self.c['trainer_pyproject_sha256'])
        self.assertIn('scripts/process_dataset.py', m['source_hashes'])
        self.assertEqual(m['wheels'], self.c['wheels'])
        for name, digest in m['bootstrap_files'].items():
            self.assertEqual(digest, prep.sha256(self.root / name))
        self.assertEqual(m['entrypoint']['sha256'], prep.sha256(restore.__file__))

    def test_no_guessed_interpreter(self):
        self.interpreter.pop('python_sha256')
        self.write(self.interpreter_path, self.interpreter)
        with self.assertRaisesRegex(ValueError, 'interpreter receipt'):
            self.build()

    def test_report_lock_binding_and_false_flags_rejected(self):
        for field, value in [('complete', False), ('cpu_imports_verified', False),
                             ('weights_loaded', True), ('runtime_lock_sha256', '0' * 64)]:
            report = dict(self.report, **{field: value})
            self.write(self.report_path, report)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build()

    def test_strict_independent_lock_no_extra_or_invented_pins(self):
        for key in ('versions', 'modules'):
            lock = copy.deepcopy(self.lock)
            lock[key]['invented'] = '0' * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                restore.strict_lock(self.c, lock)
        lock = dict(self.lock, attested=True)
        with self.assertRaises(ValueError):
            restore.strict_lock(self.c, lock)

    def test_staged_code_or_missing_helper_rejected(self):
        path = self.root / 'research/prepare_identity_runtime.py'
        path.write_text(path.read_text() + '\n# mutation\n')
        with self.assertRaisesRegex(ValueError, 'source differs'):
            self.build()
        path.unlink()
        with self.assertRaises(ValueError):
            self.build()

    def test_artifact_symlink_rejected(self):
        self.interpreter_path.unlink()
        self.interpreter_path.symlink_to(self.report_path)
        with self.assertRaises(ValueError):
            self.build()

    def gpu(self, count=1, name='NVIDIA H200', available=True, cuda='12.8'):
        return types.SimpleNamespace(version=types.SimpleNamespace(cuda=cuda),
            cuda=types.SimpleNamespace(is_available=lambda: available,
                device_count=lambda: count, get_device_name=lambda _: name))

    def test_gpu_inspector_accepts_cuda_without_constructing_models(self):
        with patch.object(restore.importlib, 'import_module', return_value=self.gpu()) as imports:
            restore.inspect_gpu(self.c)
        imports.assert_called_once_with('torch')

    def test_gpu_inspector_rejects_cpu_wrong_build_multi_gpu(self):
        for kwargs in ({'available': False}, {'count': 2}, {'name': 'A100'}, {'cuda': '12.7'}):
            with self.subTest(kwargs=kwargs), patch.object(restore.importlib, 'import_module',
                    return_value=self.gpu(**kwargs)), self.assertRaises(ValueError):
                restore.inspect_gpu(self.c)

    def test_gpu_probe_uses_independent_checks_not_cpu_probe(self):
        from research import hf_identity_encode
        with patch.object(prep, 'check_trainer') as trainer, \
             patch.object(prep, 'probe', side_effect=AssertionError('CPU rejection must stay')), \
             patch.object(hf_identity_encode, 'verify_runtime') as bytechecks, \
             patch.object(restore, 'inspect_gpu') as gpu, \
             patch.object(restore.importlib, 'import_module') as imports, \
             patch.object(restore.importlib.metadata, 'version', side_effect=self.c['versions'].__getitem__):
            restore.probe(self.c, self.source, self.lock)
        trainer.assert_called_once_with(self.source, self.c)
        bytechecks.assert_called_once_with(self.source, self.lock)
        gpu.assert_called_once_with(self.c)
        self.assertEqual([c.args[0] for c in imports.call_args_list],
                         ['torchvision', 'torchaudio', 'peft', 'h3_trainer.trainer'])

    def test_no_cpu_cuda_mask_in_restore(self):
        source = Path(restore.__file__).read_text()
        self.assertNotIn('CUDA_VISIBLE_DEVICES', source)
        self.assertNotIn('from_pretrained(', source)
        self.assertNotIn('run_job(', source)


if __name__ == '__main__':
    unittest.main()
