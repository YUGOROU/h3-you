import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from research import modal_bootstrap_identity as boot, modal_identity_worker as carrier
from research import provision_modal_identity_secret as secrets


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        def document(name, value):
            path = self.root / name
            path.write_text(json.dumps(value))
            return {'path': name, 'sha256': boot.wire.sha256(path)}
        runtime = document('runtime.json', {'schema': 'h3-identity-gpu-runtime-bootstrap-v1', 'source_root': 'upstream'})
        closure = document('closure.json', {'schema': 'h3-identity-redesign-closure-v1',
            'deployed': False, 'actual_h3_gpu_verified': False, 'files': {'runtime.json': runtime['sha256']},
            'runtime_manifest_sha256': runtime['sha256']})
        approval = document('approval.json', {'provider': 'modal'})
        self.request = dict(schema='h3-modal-bootstrap-v1', closure=closure, runtime_manifest=runtime,
            input_repo=boot.wire.INPUT_REPO, input_revision=boot.wire.INPUT_REVISION, runtime_seconds=1200,
            input_document_sha256={key: 'a'*64 for key in boot.wire.DOCUMENTS}, pipeline=dict(
                territory_approval={'provider': 'modal'}, model_lock=runtime, runtime_lock=runtime,
                approval=approval, checkpoint_escrow=True, cache_reuse={}, overall_seconds=3600,
                rate_usd_hour=carrier.RATE, inference={'sample_ids': ['synthetic'], 'seconds': 600}))

    def test_provider_binding_before_input_or_model_fetch(self):
        self.request['pipeline']['territory_approval'] = {'provider': 'huggingface_jobs', 'evidence_mode': 'operator_hf_acceptance'}
        with patch.object(boot.pipeline, 'approvals') as approve:
            with self.assertRaisesRegex(ValueError, 'HF acceptance'):
                boot.validate(self.root, self.request)
            approve.assert_not_called()

    def test_local_source_spec_binding_without_network_or_gpu(self):
        with patch.object(boot.pipeline, 'approvals') as approve:
            request, manifest = boot.validate(self.root, self.request)
            approve.assert_called_once()
            self.assertEqual(request['source_root'], str(self.root / 'upstream'))
            self.assertEqual(request['output'], '/work/identity')
            self.assertEqual(manifest['source_root'], 'upstream')
        (self.root / 'runtime.json').write_text('{}')
        with self.assertRaises(ValueError): boot.validate(self.root, self.request)

    def test_safe_diagnostics_never_return_extra_fields(self):
        path = self.root / 'diagnostic.json'
        path.write_text(json.dumps({'schema': 'h3-modal-bootstrap-status-v1', 'stage': 'runtime',
            'failure_type': 'ValueError', 'token': 'synthetic-secret-never-return'}))
        self.assertEqual(carrier.safe_diagnostic(path), {'stage': 'runtime', 'failure_type': 'ValueError'})
        path.write_text('x'*1025)
        self.assertIsNone(carrier.safe_diagnostic(path))
        path.write_text(json.dumps({'schema': 'h3-modal-bootstrap-status-v1', 'stage': 'runtime',
                                   'failure_type': 'secret with spaces'}))
        self.assertIsNone(carrier.safe_diagnostic(path))

    def test_managed_secret_create_once_no_overwrite(self):
        manager = Mock()
        manager.list.return_value = []
        result = secrets.ensure_secret(manager=manager, name='project-secret', token='synthetic-token')
        self.assertTrue(result['created'])
        manager.create.assert_called_once_with('project-secret', {'HF_TOKEN': 'synthetic-token'}, allow_existing=False)
        manager.reset_mock()
        manager.list.return_value = [NS(name='project-secret')]
        result = secrets.ensure_secret(manager=manager, name='project-secret', token='synthetic-token')
        self.assertFalse(result['created'])
        manager.create.assert_not_called()

    def test_cached_token_ignores_harness_environment(self):
        path = self.root / 'cached-token'
        path.write_text('synthetic-user-token\n')
        with patch.dict('os.environ', {'HF_TOKEN': 'synthetic-harness-token', 'HF_HOME': '/not-used'}):
            self.assertEqual(secrets.cached_token(path), 'synthetic-user-token')
        path.write_text('')
        with self.assertRaises(ValueError): secrets.cached_token(path)

    def test_permission_check_does_not_download_payloads(self):
        api = Mock()
        api.repo_info.return_value = NS(private=True)
        result = secrets.check_hub_permissions(api, input_repo='private/input', cache_repo='private/cache', output_repo='private/output')
        self.assertTrue(result['private_repository_permissions_checked'])
        self.assertEqual(api.auth_check.call_count, 3)
        api.snapshot_download.assert_not_called()
        api.repo_info.return_value = NS(private=False)
        with self.assertRaises(ValueError):
            secrets.check_hub_permissions(api, input_repo='private/input', cache_repo='private/cache', output_repo='private/output')


if __name__ == '__main__': unittest.main()
