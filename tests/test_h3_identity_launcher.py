"""CPU-only launcher boundaries. No Hub calls, auth changes, uploads or paid jobs."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research import hf_bootstrap_identity_pipeline as boot
from research import launch_hf_identity_pipeline as launch
from research.build_identity_bundle import FILES
from research.install_identity_guidance import PINS


class FakeAPI:
    def __init__(self, stage, unknown=False):
        self.stage, self.calls, self.unknown = stage, [], unknown
        self.jobs = []

    def whoami(self):
        return {'name': 'tester', 'isPro': True}

    def repo_info(self, name, **kw):
        siblings = []
        if kw.get('files_metadata'):
            for path, sha in boot.load(self.stage / 'upload-allowlist.json').items():
                p = self.stage / path
                siblings.append(NS(rfilename=path, size=p.stat().st_size, lfs=NS(sha256=sha)))
        return NS(private=True, sha=kw.get('revision'), siblings=siblings)

    def auth_check(self, *args, **kw):
        pass

    def list_jobs(self, **kw):
        return self.jobs

    def list_jobs_hardware(self):
        return [NS(name='h200', unit_label='hour', unit_cost_usd=5, accelerator='gpu')]

    def run_job(self, **kw):
        self.calls.append(kw)
        ledger = boot.load(self.stage.parent / 'ledger.json')
        assert any(v.get('state') == 'reserved_unknown_no_retry' for v in ledger['reservations'].values())
        assert (self.stage / 'submission_guard.json').exists()
        if self.unknown:
            raise TimeoutError('ambiguous response')
        return NS(id='new-job')


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_runtime_manifest_git_blob_separator(self):
        path = self.root / 'manifest.json'
        boot.dump(path, {'bootstrap_files': {}})
        sha = boot.sha256(path)
        blob = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0' + path.read_bytes()).hexdigest()
        item = NS(rfilename='manifest.json', size=path.stat().st_size, lfs=None, blob_id=blob)
        api = NS(repo_info=lambda *a, **kw: NS(private=True, sha='a'*40, siblings=[item]))
        spec = {'runtime_bootstrap': {'repo': 'tester/runtime', 'revision': 'a'*40,
                                     'manifest': {'path': 'manifest.json', 'sha256': sha}},
                'documents': {'runtime_manifest': {'path': 'manifest.json', 'sha256': sha}}}
        launch.verify_runtime_artifacts(api, spec, self.root)
        item.blob_id = 'b'*40
        with self.assertRaises(ValueError):
            launch.verify_runtime_artifacts(api, spec, self.root)

    def ledger(self):
        return {'spent_usd': 29.83, 'reservations': {'cpu': {
            'upper_bound_usd': .09, 'flavor': 'cpu-upgrade', 'job_id': 'raw'}}}

    def prepared(self):
        stage = self.root / 'stage'
        stage.mkdir()
        spec = {'schema': 'h3-identity-deployment-v1', 'image': 'python@sha256:' + 'a'*64,
                'pipeline': {'output_repo': 'tester/out', 'cache_repo': 'tester/cache', 'overall_seconds': 21600},
                'runtime_bootstrap': {'repo': 'tester/runtime', 'revision': 'c'*40, 'timeout_seconds': 1800}}
        boot.dump(stage / 'deployment.json', spec)
        (stage / 'hf_bootstrap_identity_pipeline.py').write_text('pass\n')
        (stage / 'documents').mkdir()
        for key in launch.DOCUMENT_KEYS | {'image_evidence'}:
            boot.dump(stage / 'documents' / (key + '.json'), {})
        boot.dump(stage / 'upload-allowlist.json', {
            str(p.relative_to(stage)): boot.sha256(p) for p in stage.rglob('*') if p.is_file()})
        boot.dump(self.root / 'ledger.json', self.ledger())
        return stage

    def submission(self, stage, api):
        return launch.submit(stage, self.root / 'ledger.json', 'tester/deploy', 'b'*40,
                             api, lambda **kw: kw, 'test-secret-do-not-persist')

    def reviewed_spec(self):
        from research.hf_identity_encode import REQUIRED_MODULES
        from research.h3_personal.identity_checkpoint import SOURCE_PINS
        docs = self.root / 'reviewed'
        docs.mkdir()
        image = 'pytorch/pytorch@sha256:' + 'a'*64
        pins = {}
        def document(key, value):
            path = docs / (key + '.json')
            boot.dump(path, value)
            pins[key] = {'path': str(path), 'sha256': boot.sha256(path)}
        document('country_evidence', {'synthetic_test_only': 'country statement'})
        document('license_approval_evidence', {'synthetic_test_only': 'distinct license statement'})
        # A synthetic permitted-country approval is ONLY a test fixture, never deployment evidence.
        country = {'path': '/deployment/documents/country_evidence.json',
                   'sha256': pins['country_evidence']['sha256'], 'description': 'synthetic provider statement'}
        license_proof = {'path': '/deployment/documents/license_approval_evidence.json',
                   'sha256': pins['license_approval_evidence']['sha256'], 'description': 'synthetic license statement'}
        document('approval', dict(schema='h3-compute-approval-v1', execution_country='JP',
            country_evidence=country, license_approval_evidence=license_proof,
            approved=True, approved_by='synthetic-test-only', license_allows_execution_country=True))
        document('model_lock', dict(schema='h3-local-model-lock-v1', revision=boot.MODEL_PIN,
                                    files={'model_index.json': 'd'*64}))
        versions = {n: 'synthetic-version' for n in ('torch', 'diffusers', 'transformers', 'safetensors',
                     'Pillow', 'numpy', 'accelerate', 'peft', 'huggingface-hub')}
        document('runtime_lock', dict(schema='h3-encode-runtime-v1', versions=versions,
                                     modules={n: 'd'*64 for n in REQUIRED_MODULES}))
        document('model_receipt', dict(schema='h3-model-asset-source-receipt-v1',
            repo='MiniMaxAI/MiniMax-H3', revision=boot.MODEL_PIN,
            artifact_sha256={'model-lock.json': pins['model_lock']['sha256']},
            expected_files={'model_index.json': {'sha256': 'd'*64, 'method': 'verified-git-blob-bytes'},
                'transformer_ref/model.safetensors': {'sha256': 'c'*64, 'method': 'revision-api-lfs-sha256'}},
            weights_downloaded=False, jobs_submitted=False))
        document('runtime_manifest', dict(schema='h3-identity-gpu-runtime-bootstrap-v1',
            image=image, gpu_bootstrap_supported=True, native_source_inventory='PINS23+LICENSE',
            runtime_lock_sha256=pins['runtime_lock']['sha256'], source_revisions=SOURCE_PINS,
            entrypoint={'path': 'external/bootstrap.py', 'sha256': 'f'*64}, module='external.bootstrap',
            python_sha256='e'*64, bootstrap_files={'external/bootstrap.py': 'f'*64}))
        evidence = self.root / 'image.json'
        boot.dump(evidence, dict(image=image, os='linux', architecture='amd64', digest_verified=True))
        spec = dict(schema='h3-identity-launch-spec-v1', image=image, documents=pins,
            image_evidence={'path': str(evidence), 'sha256': boot.sha256(evidence)},
            input_repo=boot.INPUT_REPO, input_revision=boot.INPUT_REVISION,
            raw_revision=boot.RAW_REVISION, original_revision=boot.ORIGINAL_REVISION,
            input_document_sha256={k: boot.FREEZE_SHA if k == 'selection_path' else 'b'*64 for k in boot.DOCUMENTS},
            runtime_bootstrap=dict(repo='tester/runtime', revision='c'*40, module='external.bootstrap',
                entrypoint={'path': 'external/bootstrap.py', 'sha256': 'f'*64},
                manifest={'path': 'manifest.json', 'sha256': pins['runtime_manifest']['sha256']}, timeout_seconds=1800),
            pipeline=dict(schema='h3-identity-pipeline-v1', flavor='h200', overall_seconds=21600,
                rate_usd_hour=5, spent_and_reserved_usd=29.92, stage_seconds=9000, max_updates=2000,
                source_revision=boot.TRAINER_PIN, runtime_source_revisions=SOURCE_PINS,
                model_repo='MiniMaxAI/MiniMax-H3', transformer_files=['transformer_ref/model.safetensors'],
                runtime_expected_versions=versions, runtime_provenance='synthetic-test-only',
                output_repo='tester/out', cache_repo='tester/cache', territory_approval=dict(
                    compute_country='JP', approved=True, approved_by='synthetic-test-only',
                    evidence_kind='provider_compute_country_statement', evidence_file=country['path'],
                    evidence_sha256=country['sha256'])))
        path = self.root / 'spec.json'
        boot.dump(path, spec)
        return path

    def test_actual_offline_contract_accepts_reviewed_synthetic_pins(self):
        spec = self.reviewed_spec()
        report = launch.stage(spec, self.root / 'stage')
        self.assertEqual(len(report['remaining_gates']), 2)  # upload/live read-only checks remain
        self.assertEqual(launch.inspect_spec(boot.load(self.root / 'stage/deployment.json'), self.root / 'stage'), [])

    def test_native_subset_and_country_policy_are_not_assumed(self):
        spec = self.reviewed_spec()
        launch.stage(spec, self.root / 'stage')
        deployed = boot.load(self.root / 'stage/deployment.json')
        approval_path = self.root / 'stage/documents/approval.json'
        approval = boot.load(approval_path); approval['execution_country'] = 'US'
        boot.dump(approval_path, approval)
        deployed['documents']['approval']['sha256'] = boot.sha256(approval_path)
        self.assertIn('explicit existing permitted execution-country/license evidence',
                      launch.inspect_spec(deployed, self.root / 'stage'))
        runtime_path = self.root / 'stage/documents/runtime_manifest.json'
        runtime = boot.load(runtime_path); runtime['native_source_inventory'] = 'full-checkout-assumed'
        boot.dump(runtime_path, runtime)
        deployed['documents']['runtime_manifest']['sha256'] = boot.sha256(runtime_path)
        deployed['runtime_bootstrap']['manifest']['sha256'] = boot.sha256(runtime_path)
        self.assertIn('independent immutable runtime artifacts/bootstrap interface',
                      launch.inspect_spec(deployed, self.root / 'stage'))

    def test_default_missing_runtime_region_prepares_no_service_calls(self):
        source = self.root / 'spec.json'
        boot.dump(source, {'schema': 'h3-identity-launch-spec-v1'})
        with patch('research.launch_hf_identity_pipeline.submit', side_effect=AssertionError):
            report = launch.stage(source, self.root / 'stage')
        self.assertTrue(report['plan_only'])
        self.assertTrue(report['remaining_gates'])
        self.assertEqual(report['upload_files'], ['deployment.json', 'hf_bootstrap_identity_pipeline.py'])

    def test_secret_fields_rejected_before_stage(self):
        source = self.root / 'spec.json'
        boot.dump(source, {'schema': 'h3-identity-launch-spec-v1', 'HF_TOKEN': 'no'})
        with self.assertRaises(ValueError):
            launch.stage(source, self.root / 'stage')

    def test_budget_retains_cpu_reserve(self):
        spent, upper = launch.budget(self.ledger(), [], 5, 23400)
        self.assertAlmostEqual(spent, 29.92)
        self.assertEqual(upper, 32.5)
        ledger = self.ledger(); ledger['reservations'] = {}
        with self.assertRaises(ValueError):
            launch.budget(ledger, [], 5, 23400)

    def test_all_nonterminal_jobs_accounted_conservatively(self):
        active = NS(id='active', status=NS(stage='RUNNING'))
        ledger = self.ledger()
        with self.assertRaises(ValueError):
            launch.budget(ledger, [active], 5, 23400)
        ledger['reservations']['other'] = dict(job_id='active', upper_bound_usd=2,
            rate_usd_hour=2, timeout_seconds=3600)
        self.assertAlmostEqual(launch.budget(ledger, [active], 5, 23400)[0], 31.92)
        ledger['reservations']['other']['upper_bound_usd'] = .1
        with self.assertRaises(ValueError):
            launch.budget(ledger, [active], 5, 23400)

    def test_unknown_status_is_not_terminal(self):
        with self.assertRaises(ValueError):
            launch.budget(self.ledger(), [NS(id='unknown', status=None)], 5, 23400)

    def test_cumulative_and_concurrent_caps(self):
        ledger = self.ledger(); ledger['spent_usd'] = 120
        with self.assertRaises(ValueError):
            launch.budget(ledger, [], 5, 23400)
        ledger = self.ledger()
        ledger['reservations']['other'] = dict(job_id='active', upper_bound_usd=6,
            rate_usd_hour=6, timeout_seconds=3600)
        with self.assertRaises(ValueError):
            launch.budget(ledger, [NS(id='active', status=NS(stage='SCHEDULING'))], 5, 23400)
        with self.assertRaises(ValueError):
            launch.budget(dict(spent_usd=float('nan'), reservations={}), [], 5, 23400)

    def test_single_submission_mounts_secrets_and_reserve_before_call(self):
        stage = self.prepared(); api = FakeAPI(stage)
        # Other tests exercise actual offline gates; isolate the atomic submission boundary.
        with patch.object(launch, 'inspect_spec', return_value=[]), patch.object(launch, 'verify_runtime_artifacts'):
            report = self.submission(stage, api)
            with self.assertRaises(ValueError):
                self.submission(stage, api)
        self.assertEqual(report['timeout_seconds'], 23400)
        self.assertEqual(len(api.calls), 1)
        call = api.calls[0]
        self.assertEqual(call['flavor'], 'h200')
        self.assertEqual(call['secrets'], {'HF_TOKEN': 'test-secret-do-not-persist'})
        self.assertNotIn('HF_TOKEN', call['env'])
        self.assertEqual([v['mount_path'] for v in call['volumes']], ['/deployment', '/inputs', '/runtime-inputs'])
        self.assertEqual(call['volumes'][1]['revision'], boot.INPUT_REVISION)
        self.assertNotIn('test-secret-do-not-persist', json.dumps(call['command']))
        self.assertNotIn('test-secret-do-not-persist', (self.root / 'ledger.json').read_text())

    def test_unknown_api_outcome_keeps_reservation_and_no_retry(self):
        stage = self.prepared(); api = FakeAPI(stage, unknown=True)
        with patch.object(launch, 'inspect_spec', return_value=[]), patch.object(launch, 'verify_runtime_artifacts'):
            with self.assertRaises(TimeoutError):
                self.submission(stage, api)
            with self.assertRaises(ValueError):
                self.submission(stage, api)
        self.assertEqual(len(api.calls), 1)
        self.assertTrue(any(v.get('state') == 'reserved_unknown_no_retry' for v in
                            boot.load(self.root / 'ledger.json')['reservations'].values()))

    def test_missing_gates_no_gpu(self):
        stage = self.prepared(); api = FakeAPI(stage)
        with self.assertRaises(ValueError):
            self.submission(stage, api)
        self.assertFalse(api.calls)

    def test_runtime_mount_missing_files_blocks_before_reservation(self):
        spec_path = self.reviewed_spec()
        launch.stage(spec_path, self.root / 'stage')
        stage = self.root / 'stage'
        boot.dump(self.root / 'ledger.json', self.ledger())
        api = FakeAPI(stage)
        with self.assertRaises(ValueError):
            self.submission(stage, api)
        self.assertEqual(boot.load(self.root / 'ledger.json'), self.ledger())
        self.assertFalse(api.calls)

    def test_changed_deployment_bytes_no_gpu(self):
        stage = self.prepared(); api = FakeAPI(stage)
        (stage / 'hf_bootstrap_identity_pipeline.py').write_text('evil')
        with self.assertRaises(ValueError):
            self.submission(stage, api)
        self.assertFalse(api.calls)

    def test_uploaded_bytes_require_full_commit_and_exact_git_lfs(self):
        stage = self.prepared(); api = FakeAPI(stage)
        allow = boot.load(stage / 'upload-allowlist.json')
        launch.verify_uploaded(api, 'tester/deploy', 'a'*40, stage, allow)
        with patch.object(api, 'repo_info', return_value=NS(private=True, sha='a'*40,
            siblings=[NS(rfilename='deployment.json', size=1, lfs=NS(sha256='f'*64))])):
            with self.assertRaises(ValueError):
                launch.verify_uploaded(api, 'tester/deploy', 'a'*40, stage, allow)

    def archive(self, extra=None):
        entries = {name: b'# code\n' for name in FILES}
        entries.update({'upstream/' + name: b'# upstream\n' for name in PINS})
        entries['upstream/LICENSE'] = b'MIT\n'
        manifest = dict(schema='h3-identity-source-bundle-v1', trainer_revision=boot.TRAINER_PIN,
            files={k: hashlib.sha256(v).hexdigest() for k, v in entries.items()},
            contains_weights=False, contains_credentials=False)
        entries['CODE_MANIFEST.json'] = json.dumps(manifest).encode()
        path = self.root / 'bundle.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            for name, data in entries.items():
                member = tarfile.TarInfo(name); member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            if extra:
                tar.addfile(extra)
        return path

    def test_archive_full_39_allowlist_and_manifest_root(self):
        archive = self.archive()
        with patch.object(boot, 'BUNDLE_SHA', boot.sha256(archive)), patch.object(boot, 'BUNDLE_BYTES', archive.stat().st_size):
            manifest = boot.extract_bundle(archive, self.root / 'out')
        self.assertEqual(len(manifest['files']), 39)
        self.assertTrue((self.root / 'out/CODE_MANIFEST.json').is_file())

    def test_archive_traversal_link_duplicate_extra_all_rejected(self):
        for name in ('../escape', '/absolute', 'CODE_MANIFEST.json', 'extra.py'):
            with self.subTest(name=name):
                extra = tarfile.TarInfo(name)
                if name == 'extra.py':
                    extra.type = tarfile.SYMTYPE; extra.linkname = '/etc/passwd'
                archive = self.archive(extra)
                with patch.object(boot, 'BUNDLE_SHA', boot.sha256(archive)), patch.object(boot, 'BUNDLE_BYTES', archive.stat().st_size):
                    with self.assertRaises(ValueError):
                        boot.extract_bundle(archive, self.root / 'out')
                self.assertFalse((self.root / 'out').exists())

    def test_wrong_bundle_pin_rejected_before_extraction(self):
        archive = self.archive()
        with self.assertRaises(ValueError):
            boot.extract_bundle(archive, self.root / 'out')

    def test_path_safety(self):
        for name in ('../x', '/x', 'a//x', 'a/./x', 'a\\x'):
            with self.assertRaises(ValueError):
                boot.relative(name)
        (self.root / 'file').write_text('a')
        (self.root / 'link').symlink_to(self.root / 'file')
        with self.assertRaises(ValueError):
            boot.inside(self.root, 'link')


if __name__ == '__main__':
    unittest.main()
