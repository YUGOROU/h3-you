import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import prepare_identity_model_lock as lock


class FakeIO:
    def __init__(self):
        self.info = {'id': lock.REPO, 'sha': lock.REVISION, 'siblings': []}
        self.bytes = {}
        self.calls = []
        for group in sorted(lock.GROUPS | {'transformer_ref'}):
            self.git(group + '/config.json', b'{}')
            if group in {'vae', 'audio_vae', 'text_encoder', 'transformer_ref'}:
                self.lfs(group + '/weights.safetensors')
                self.git(group + '/weights.safetensors.index.json',
                         b'{"weight_map":{"x":"weights.safetensors"}}')
        self.git('model_index.json', b'{}')
        self.git('tokenizer/tokenizer.json', b'{"tokenizer":true}')
        self.git('processor/tokenizer.json', b'{"tokenizer":true}')
        self.git('README.md', b'ignored')
        self.git('vae/.cache/download/x.json', b'ignored')
        self.git('transformer/config.json', b'ignored')

    def git(self, name, data):
        self.info['siblings'].append({'rfilename': name, 'size': len(data), 'blobId': lock.git_blob(data)})
        self.bytes[name] = data

    def lfs(self, name):
        self.info['siblings'].append({'rfilename': name, 'size': 1000000000, 'blobId': 'a' * 40,
                                     'lfs': {'size': 1000000000, 'sha256': 'b' * 64}})

    def get(self, url, limit):
        self.calls.append((url, limit))
        if url == lock.API_URL:
            return json.dumps(self.info).encode()
        name = url.split('/resolve/' + lock.REVISION + '/')[1]
        if name not in self.bytes:
            raise AssertionError('weight download attempted')
        return self.bytes[name]


class ModelLockTests(unittest.TestCase):
    def test_exact_worker_contract_and_receipt(self):
        io = FakeIO()
        artifacts, receipt = lock.prepare(io)
        model = json.loads(artifacts['model-lock.json'])
        transformer = json.loads(artifacts['transformer-assets.json'])
        self.assertEqual(model['schema'], 'h3-local-model-lock-v1')
        self.assertEqual(model['revision'], lock.REVISION)
        self.assertEqual(model['files']['vae/weights.safetensors'], 'b' * 64)
        self.assertEqual(model['files']['tokenizer/tokenizer.json'], lock.sha256(b'{"tokenizer":true}'))
        self.assertFalse(any(n.startswith('transformer') for n in model['files']))
        self.assertEqual(transformer['transformer_files'], sorted(transformer['files']))
        self.assertEqual(set(receipt['expected_files']), set(model['files']) | set(transformer['files']))
        self.assertEqual(receipt['artifact_sha256']['model-lock.json'], lock.sha256(artifacts['model-lock.json']))
        self.assertEqual(receipt['totals']['component_lfs_bytes'], 3000000000)
        self.assertEqual(receipt['totals']['transformer_lfs_bytes'], 1000000000)
        self.assertLess(receipt['totals']['small_download_bytes'], receipt['totals']['small_inventory_bytes'])
        self.assertEqual(len(io.calls), 4)  # API + three distinct Git blobs
        self.assertTrue(all(lock.REVISION in u for u, _ in io.calls))
        self.assertFalse(receipt['weights_downloaded'])

    def test_existing_encoder_accepts_exact_inventory(self):
        from research.hf_identity_encode import verify_models
        io = FakeIO()
        weight = b'fake weight bytes, not a downloaded model'
        for item in io.info['siblings']:
            if item.get('lfs'):
                item['size'] = item['lfs']['size'] = len(weight)
                item['lfs']['sha256'] = lock.sha256(weight)
        artifacts, _ = lock.prepare(io)
        model = json.loads(artifacts['model-lock.json'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in model['files']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(weight if name.endswith('.safetensors') else io.bytes[name])
            self.assertEqual(verify_models(root, model), model['files'])
            (root / 'tokenizer' / 'extra.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'exact relevant'):
                verify_models(root, model)

    def test_metadata_failures_before_download(self):
        mutations = [lambda i: i.update(sha='a' * 40), lambda i: i.update(id='other/model'),
                     lambda i: i['siblings'].append(copy.deepcopy(i['siblings'][0])),
                     lambda i: i['siblings'][0].update(rfilename='../config.json'),
                     lambda i: i['siblings'][0].update(blobId=None),
                     lambda i: i['siblings'][0].update(size=-1),
                     lambda i: i['siblings'][1]['lfs'].update(size=1),
                     lambda i: i['siblings'][1]['lfs'].update(sha256='wrong')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                io = FakeIO()
                mutate(io.info)
                with self.assertRaises((ValueError, TypeError)):
                    lock.prepare(io)
                self.assertEqual(len(io.calls), 1)

    def test_no_alternative_hash_or_large_git_download(self):
        io = FakeIO()
        io.info['siblings'][0].update(size=lock.SMALL_CAP + 1, blobId='c' * 40)
        with self.assertRaisesRegex(ValueError, 'budget'):
            lock.prepare(io)
        self.assertEqual(len(io.calls), 1)
        io = FakeIO()
        io.git('vae/model.bin', b'weight')
        with self.assertRaisesRegex(ValueError, 'refuse'):
            lock.prepare(io)
        self.assertEqual(len(io.calls), 1)

    def test_byte_and_size_corruption(self):
        for data in (b'[]', b'longer'):
            io = FakeIO()
            # config blobs are deduplicated to the last sorted representative.
            io.bytes['vae/config.json'] = data
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                lock.prepare(io)

    def test_missing_components_and_shards(self):
        io = FakeIO()
        io.info['siblings'] = [s for s in io.info['siblings'] if not s['rfilename'].startswith('vae/')]
        with self.assertRaisesRegex(ValueError, 'component'):
            lock.prepare(io)
        io = FakeIO()
        io.git('vae/bad.safetensors.index.json', b'{"weight_map":{"x":"missing.safetensors"}}')
        with self.assertRaisesRegex(ValueError, 'missing indexed'):
            lock.prepare(io)

    def test_deterministic_lock_and_private_outputs(self):
        artifacts, _ = lock.prepare(FakeIO())
        io = FakeIO()
        io.info['siblings'].reverse()
        other, _ = lock.prepare(io)
        self.assertEqual(artifacts['model-lock.json'], other['model-lock.json'])
        self.assertEqual(artifacts['transformer-assets.json'], other['transformer-assets.json'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'assets'
            lock.save(root, artifacts)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in root.iterdir()))
            with self.assertRaisesRegex(ValueError, 'replace'):
                lock.save(root, artifacts)
            root.chmod(0o755)
            with self.assertRaisesRegex(ValueError, 'private'):
                lock.save(root, artifacts)


if __name__ == '__main__':
    unittest.main()
