"""Small CPU-only fakes; no original media, model download, or job API."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from research import hf_identity_encode as worker
from research.h3_personal import component_encode as encode


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def plan(self, root):
        root.mkdir()
        for name in ('v.mp4', 'a.wav', 'r.jpg'):
            (root / name).write_bytes(name.encode())
        return encode.make_plan([dict(id='fake', video=root/'v.mp4', audio=root/'a.wav',
            reference_image=root/'r.jpg', caption='synthetic', width=768, height=1344,
            num_frames=124)], {k: 'a'*64 for k in worker.cache.KINDS},
            attested_by='synthetic-test', media_shards=2)

    def test_relocation_and_receipt(self):
        a, b = self.plan(self.root/'a'), self.plan(self.root/'b')
        self.assertEqual(worker.semantic_plan(a), worker.semantic_plan(b))
        self.assertEqual(set(a['units']), set(b['units']))
        engine = worker.RelocatedEngine(self.root/'out', a, None)
        engine._receipt('media-0', {})
        moved = worker.RelocatedEngine(self.root/'out', b, None)
        self.assertEqual(moved._read('media-0'), {})
        b['samples'][0]['caption'] = 'changed'
        with self.assertRaises(ValueError):
            moved._read('media-0')

    def test_merge_requires_every_stage(self):
        plan = self.plan(self.root/'media')
        engine = worker.RelocatedEngine(self.root/'out', plan, None)
        engine._receipt('media-0', {})
        with self.assertRaises((ValueError, FileNotFoundError)):
            engine.finish()
        self.assertFalse((engine.root/'index.json').exists())

    def test_real_constructor_contract_with_fake_cuda(self):
        calls = []
        class FakeEncoders:
            def __init__(self, **kwargs):
                calls.append(kwargs)
        module = types.SimpleNamespace(H3Encoders=FakeEncoders)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {'h3_trainer.preprocessing.encoders': module, 'torch': torch}), \
             patch.object(encode, 'PinnedBackend', side_effect=lambda e, r: e):
            for stage in ('media', 'reference', 'text'):
                worker.construct(stage, '/external/models', '/source')
        self.assertEqual([(c['need_video'], c['need_audio'], c['need_text']) for c in calls],
                         [(True, True, False), (True, False, False), (False, False, True)])
        self.assertTrue(all(c['device']=='cuda' and c['text_device_map'] is None for c in calls))

    def models(self):
        root = self.root/'models'
        root.mkdir()
        (root/'model_index.json').write_text('{}')
        for group in ('vae', 'audio_vae', 'text_encoder', 'tokenizer', 'processor'):
            (root/group).mkdir()
            (root/group/'config.json').write_text('{}')
            if group in ('vae', 'audio_vae', 'text_encoder'):
                (root/group/'model.safetensors').write_bytes(b'fake-weights')
        files = {str(p.relative_to(root)): worker.cache.sha256(p) for p in root.rglob('*') if p.is_file()}
        return root, dict(schema='h3-local-model-lock-v1', revision='b'*40, files=files)

    def test_actual_file_hashes_not_attestor_flags(self):
        root, lock = self.models()
        hashes = worker.verify_models(root, lock)
        runtime = {'ffmpeg': {'path': '/one', 'sha256': 'c'*64}, 'versions': {'fake': '1'}}
        first = worker.identities(hashes, runtime)
        runtime['ffmpeg']['path'] = '/relocated'
        self.assertEqual(first, worker.identities(hashes, runtime))
        (root/'vae/model.safetensors').write_bytes(b'wrong')
        lock['attestor_verified'] = True
        with self.assertRaises(ValueError):
            worker.verify_models(root, lock)

    def test_loader_subprocess_is_bounded_and_script_compiles(self):
        root = self.root/'composed'
        for name in ('__init__.py', 'datasets.py', 'component_cache.py'):
            path = root/'src/h3_trainer'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic source')
        lock = dict(schema='h3-composed-loader-lock-v1', files={
            str(p.relative_to(root)): worker.cache.sha256(p) for p in root.rglob('*.py')})
        class Process:
            def __init__(self, command, **kwargs):
                compile(command[2], '<loader-validation>', 'exec')
                assert kwargs['start_new_session'] is True
            def wait(self, timeout=None):
                assert timeout == 600
                return 0
        with patch.object(worker.subprocess, 'Popen', Process):
            worker.validate_loader(self.root/'out', root, lock)

    def test_missing_and_extra_model_files(self):
        root, lock = self.models()
        (root/'text_encoder/extra.json').write_text('{}')
        with self.assertRaises(ValueError):
            worker.verify_models(root, lock)

    def test_license_distinct_country_evidence(self):
        country, approval = self.root/'country', self.root/'approval'
        country.write_bytes(b'provider execution country')
        approval.write_bytes(b'human license review')
        evidence = dict(schema='h3-compute-approval-v1', execution_country='JP', approved=True,
            approved_by='synthetic', license_allows_execution_country=True,
            country_evidence=dict(path=str(country), sha256=worker.cache.sha256(country), description='provider location'),
            license_approval_evidence=dict(path=str(approval), sha256=worker.cache.sha256(approval), description='license review'))
        worker.license_gate(evidence)
        for excluded in ('US', 'GB', 'KR', 'FR', 'DE'):
            with self.subTest(country=excluded), self.assertRaises(ValueError):
                worker.license_gate(dict(evidence,execution_country=excluded))
        evidence['country_evidence'] = evidence['license_approval_evidence']
        with self.assertRaises(ValueError):
            worker.license_gate(evidence)
        with self.assertRaises(ValueError):
            worker.license_gate({'collection_url': 'https://huggingface.co/collections/example'})

    def test_snapshots_private_bounded_allowlisted(self):
        out = self.root/'out'
        (out/'objects').mkdir(parents=True)
        for name in ('a', 'b'):
            (out/'objects'/name).write_bytes(b'1234')
        (out/'media-0.json').write_text('{}')
        commits = []
        class API:
            private = True
            def repo_info(self, *a, **kw):
                return types.SimpleNamespace(private=self.private)
            def create_commit(self, **kw):
                commits.append(kw['operations'])
        api = API()
        fake = types.SimpleNamespace(CommitOperationAdd=lambda **kw: kw)
        with patch.dict(sys.modules, {'huggingface_hub': fake}):
            snap = worker.PrivateSnapshots(api, 'private/repo', out, 'components/'+'a'*64, max_bytes=5)
            snap.upload(['objects/a', 'objects/b', 'media-0.json'])
            self.assertEqual(len(commits), 3)
            self.assertTrue(commits[-1][0]['path_in_repo'].endswith('media-0.json'))
            api.private = False
            with self.assertRaises(ValueError):
                snap.upload(['media-0.json'])
            with self.assertRaises(ValueError):
                snap.upload(['model_index.json'])


if __name__ == '__main__':
    unittest.main()
