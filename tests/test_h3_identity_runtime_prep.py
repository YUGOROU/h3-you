"""CPU/stdlib contracts only: no network, package installation, model or Jobs API."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from research import prepare_identity_runtime as prep
from research import hf_prepare_identity_runtime as worker


class RuntimePrepTests(unittest.TestCase):
    def test_reviewed_configuration_and_independent_modules(self):
        p = prep.CONFIG / 'expectations.json'
        with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            c, tree = prep.expectations(p, prep.sha256(p))
        self.assertEqual(c['image'], prep.IMAGE)
        self.assertEqual(c['versions']['transformers'], '5.15.0')
        self.assertEqual(c['versions']['diffusers'], '0.40.0.dev0')
        self.assertEqual(c['versions']['torch'], '2.8.0+cu128')
        self.assertIn('peft', c['versions'])
        self.assertGreater(len(tree['git_blob_sha1']), 900)
        self.assertEqual({k: v['version'] for k, v in c['wheels'].items()},
                         {k: v for k, v in c['versions'].items() if k in c['wheels']})
        for entry in c['modules'].values():
            self.assertIn('/src/', entry['url'])
            self.assertNotIn('/main/', entry['url'])

    def test_document_byte_pin_fails_closed(self):
        with self.assertRaises(ValueError):
            prep.expectations(prep.CONFIG / 'expectations.json', '0' * 64)

    def test_allowlist_rejects_traversal(self):
        for value in ('/tmp/file', '../x', 'src/../x', 'a\\b', 'a//b', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prep.safe_name(value)

    def archive(self, root, entries):
        archive = root / 'source.tgz'
        with tarfile.open(archive, 'w:gz') as t:
            for name, payload, symlink in entries:
                member = tarfile.TarInfo('diffusers-revision/' + name)
                if symlink:
                    member.type = tarfile.SYMTYPE
                    member.linkname = '/etc/passwd'
                    t.addfile(member)
                else:
                    member.size = len(payload)
                    t.addfile(member, io.BytesIO(payload))
        return archive

    def test_extract_only_verified_allowlisted_source_not_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            content = b'x = 1\n'
            a = self.archive(root, [('src/diffusers/a.py', content, False),
                                    ('weights.safetensors', b'forbidden', False)])
            out = root / 'out'
            prep.extract_source(a, out, {'src/diffusers/a.py': prep.git_blob(content)})
            self.assertEqual((out / 'src/diffusers/a.py').read_bytes(), content)
            self.assertFalse((out / 'weights.safetensors').exists())

    def test_extract_rejects_mutation_symlink_duplicate_and_missing(self):
        scenarios = [([('src/a.py', b'wrong', False)], {'src/a.py': prep.git_blob(b'good')}),
                     ([('src/a.py', b'', True)], {'src/a.py': prep.git_blob(b'')}),
                     ([('src/a.py', b'x', False)] * 2, {'src/a.py': prep.git_blob(b'x')}),
                     ([], {'src/a.py': prep.git_blob(b'x')})]
        for entries, pins in scenarios:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                with self.assertRaises(ValueError):
                    prep.extract_source(self.archive(root, entries), root / 'out', pins)

    def test_fetch_rejects_model_and_untrusted_hosts_without_network(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            for url in ('https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/model.safetensors',
                        'http://files.pythonhosted.org/pkg.whl', 'https://evil.example/code'):
                with self.assertRaises(ValueError):
                    prep.fetch(url, '/unused', 100)

    def test_native_digest_requires_independent_package_and_binary_equality(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            deb, extracted, binary = (root / n for n in ('package.deb', 'extracted', 'ffmpeg'))
            deb.write_bytes(b'authenticated package')
            extracted.write_bytes(b'authentic executable')
            binary.write_bytes(extracted.read_bytes())
            expected = prep.sha256(deb)
            self.assertEqual(prep.verify_native_package(deb, expected, extracted, binary),
                             prep.sha256(binary))
            with self.assertRaises(ValueError):
                prep.verify_native_package(deb, '0' * 64, extracted, binary)
            binary.write_bytes(b'whatever happened to be on PATH')
            with self.assertRaises(ValueError):
                prep.verify_native_package(deb, expected, extracted, binary)
            binary.unlink()
            binary.symlink_to(extracted)
            with self.assertRaises(ValueError):
                prep.verify_native_package(deb, expected, extracted, binary)

    def test_watchdog_kills_process_group(self):
        fake = Mock(pid=12345)
        fake.wait.side_effect = [subprocess.TimeoutExpired('child', 1), 0]
        with patch('subprocess.Popen', return_value=fake), patch('os.killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.bounded(['child'], time.monotonic() + 1)
            kill.assert_called_once_with(12345, worker.signal.SIGKILL)
        self.assertEqual(fake.wait.call_count, 2)

    def test_expired_deadline_never_starts_process(self):
        with patch('subprocess.Popen') as popen, self.assertRaises(ValueError):
            worker.bounded(['child'], time.monotonic() - 1)
        popen.assert_not_called()

    def receipts(self, root):
        os.chmod(root, 0o700)
        prep.atomic_json(root / 'runtime-lock.json', {'schema': 'h3-encode-runtime-v1'})
        prep.atomic_json(root / 'runtime-report.json', {'complete': True, 'cpu_imports_verified': True, 'image': prep.IMAGE,
                         'runtime_lock_sha256': prep.sha256(root / 'runtime-lock.json')})
        prep.atomic_json(root / 'base-interpreter-receipt.json', {
            'schema': 'h3-runtime-base-interpreter-v1', 'complete': True, 'image': prep.IMAGE,
            'python': sys.executable, 'python_sha256': prep.sha256(sys.executable),
            'cpu_report_sha256': prep.sha256(root / 'runtime-report.json')})

    def test_upload_is_atomic_three_artifacts_private_checked_twice(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.receipts(root)
            api = Mock()
            api.repo_info.return_value = argparse.Namespace(private=True, sha='a' * 40)
            worker.upload(api, 'owner/runtime', root, 'runtimes/' + 'a' * 64,
                          operation=lambda **kw: kw)
            self.assertEqual(api.repo_info.call_count, 2)
            api.create_commit.assert_called_once()
            kwargs = api.create_commit.call_args.kwargs
            self.assertEqual(len(kwargs['operations']), 3)
            self.assertEqual({x['path_in_repo'].split('/')[-1] for x in kwargs['operations']},
                             {'runtime-lock.json', 'runtime-report.json', 'base-interpreter-receipt.json'})
            self.assertEqual(kwargs['repo_type'], 'dataset')
            self.assertEqual(kwargs['parent_commit'], 'a' * 40)

    def test_upload_rejects_missing_or_unbound_interpreter_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.receipts(root)
            api = Mock()
            receipt_path = root / 'base-interpreter-receipt.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['cpu_report_sha256'] = '0' * 64
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                worker.upload(api, 'owner/runtime', root, 'runtimes/' + 'a' * 64)
            receipt_path.unlink()
            with self.assertRaises(FileNotFoundError):
                worker.upload(api, 'owner/runtime', root, 'runtimes/' + 'a' * 64)
            api.create_commit.assert_not_called()

    def test_upload_rejects_public_and_mismatched_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.receipts(root)
            api = Mock()
            api.repo_info.return_value = argparse.Namespace(private=False)
            with self.assertRaises(ValueError):
                worker.upload(api, 'owner/runtime', root, 'runtimes/' + 'a' * 64)
            api.create_commit.assert_not_called()
            (root / 'runtime-lock.json').write_text('tampered')
            with self.assertRaises(ValueError):
                worker.upload(api, 'owner/runtime', root, 'runtimes/' + 'a' * 64)
            api.create_commit.assert_not_called()

    def test_execute_failure_has_sanitized_report_and_no_lock(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'runtime'
            args = argparse.Namespace(expectations=prep.CONFIG / 'expectations.json',
                expectations_sha256=prep.sha256(prep.CONFIG / 'expectations.json'),
                source_root=Path(d), output=output, deadline_seconds=600, snapshot_repo=None)
            def fail(command, deadline):
                prep.atomic_json(output / 'candidate-lock.json', {'bad': True})
                raise RuntimeError('SECRET_TOKEN_VALUE')
            with patch('sys.platform', 'linux'), patch.object(prep, 'check_trainer'), \
                 patch.dict(os.environ, H3_RUNTIME_IMAGE=prep.IMAGE), self.assertRaises(RuntimeError):
                worker.execute(args, runner=fail)
            report = (output / 'runtime-report.json').read_text()
            self.assertNotIn('SECRET_TOKEN_VALUE', report)
            self.assertFalse(json.loads(report)['complete'])
            self.assertFalse((output / 'candidate-lock.json').exists())
            self.assertFalse((output / 'runtime-lock.json').exists())
            self.assertFalse((output / 'base-interpreter-receipt.json').exists())
            self.assertEqual((output / 'runtime-report.json').stat().st_mode & 0o777, 0o600)

    def test_success_lock_versions_are_independent_not_import_capture(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'runtime'
            p = prep.CONFIG / 'expectations.json'
            args = argparse.Namespace(expectations=p, expectations_sha256=prep.sha256(p),
                source_root=Path(d), output=output, deadline_seconds=600, snapshot_repo=None)
            def ready(command, deadline):
                prep.atomic_json(output / 'runtime-lock.json', {'verified': True})
                prep.atomic_json(output / 'apt-evidence.json', {'authenticated': True})
            with patch('sys.platform', 'linux'), patch.object(prep, 'check_trainer'), \
                 patch.dict(os.environ, H3_RUNTIME_IMAGE=prep.IMAGE):
                self.assertEqual(worker.execute(args, runner=ready), 0)
            report = json.loads((output / 'runtime-report.json').read_text())
            receipt_path = output / 'base-interpreter-receipt.json'
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt['schema'], 'h3-runtime-base-interpreter-v1')
            self.assertEqual(receipt['python'], str(Path(sys.executable).absolute()))
            self.assertEqual(receipt['python_sha256'], prep.sha256(sys.executable))
            self.assertEqual(receipt['cpu_report_sha256'], prep.sha256(output / 'runtime-report.json'))
            self.assertEqual(receipt['image'], prep.IMAGE)
            self.assertTrue(receipt['complete'])
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
            self.assertTrue(report['complete'])
            self.assertFalse(report['weights_loaded'])
            self.assertFalse(report['gpu_verified'])
            self.assertEqual(report['independent_expectations']['versions'], json.loads(p.read_text())['versions'])


if __name__ == '__main__':
    unittest.main()
