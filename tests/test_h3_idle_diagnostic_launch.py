"""Cloud-mocked launcher safety tests; never create real resources."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from research import launch_h3_idle_diagnostic as launch


class LaunchTests(unittest.TestCase):
    def test_anchor_is_prepare_only(self):
        for flags in (['--idle-anchor'], ['--idle-anchor', '--submit']):
            with patch.object(launch, 'canonical') as canonical:
                with self.assertRaises(SystemExit):
                    launch.main(['/tmp/offline', *flags])
                canonical.assert_not_called()

    def test_prepare_forwards_anchor(self):
        with patch.object(launch, 'canonical'), patch.object(launch, 'prepare', return_value=Path('/tmp/stage')) as prepare, patch.object(launch, 'inspect', return_value=({'publication_repo': 'repo'}, {})), patch('research.provision_modal_identity_secret.cached_token', return_value='test'), patch('huggingface_hub.set_client_factory'):
            self.assertEqual(launch.main(['/tmp/offline', '--prepare', '--idle-anchor']), 0)
            self.assertTrue(prepare.call_args.kwargs['idle_anchor'])

    def test_single_post_and_callback(self):
        client = Mock()
        client.post.return_value = NS(status_code=201, json=lambda: {'id': 'a'*24})
        callback = Mock()
        payload = dict(flavor='h200x2', timeoutSeconds=1200, attempts=1)
        job = launch.submit_job(client, 'secret', payload, callback)
        self.assertEqual(job.id, 'a'*24)
        callback.assert_called_once_with(job)
        client.post.assert_called_once_with('https://huggingface.co/api/jobs/YUGOROU',
            headers={'Authorization': 'Bearer secret'}, json=payload, follow_redirects=False)

    def test_invalid_id_not_owned_or_retried(self):
        client = Mock()
        client.post.return_value = NS(status_code=201, json=lambda: {'id': '../bad'})
        callback = Mock()
        with self.assertRaises(RuntimeError):
            launch.submit_job(client, 'secret', dict(flavor='h200x2', timeoutSeconds=1200, attempts=1), callback)
        callback.assert_not_called()
        self.assertEqual(client.post.call_count, 1)

    def test_ports_and_nonfinite_rejected(self):
        for extra in ({'expose': {}}, {'timeoutSeconds': 1201}, {'attempts': True}, {'flavor': 'h200'}):
            client = Mock()
            with self.assertRaises(ValueError):
                launch.submit_job(client, 'secret', dict(dict(flavor='h200x2', timeoutSeconds=1200, attempts=1), **extra), Mock())
            client.post.assert_not_called()

    def test_submission_receipt_failure_cancels_owned_job_and_blocks_replay(self):
        with TemporaryDirectory() as tmp:
            stage = Path(tmp)/'deployment'
            stage.mkdir()
            for name in ('spec.json', 'upload-allowlist.json', 'research/bootstrap_h3_fast_ref2va.py'):
                p = stage/name
                p.parent.mkdir(exist_ok=True)
                p.write_text('{}')
            inventory = {'files': [{'path': 'spec.json'}, {'path': 'research/bootstrap_h3_fast_ref2va.py'}]}
            names = [r['path'] for r in inventory['files']] + ['upload-allowlist.json']
            entries = []
            for name in names:
                body = (stage/name).read_bytes()
                entries.append(NS(rfilename=name, size=len(body), lfs=None,
                    blob_id=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()))
            api = Mock()
            api.whoami.return_value = {'name': 'YUGOROU', 'isPro': True}
            api.list_jobs.return_value = []
            api.create_commit.return_value = NS(oid='b'*40)
            api.repo_info.return_value = NS(sha='b'*40, private=True, siblings=entries)
            control = Mock()
            control.post.side_effect = [NS(status_code=201), NS(status_code=201, json=lambda: {'id': 'a'*24})]
            spec = {'publication_repo': 'YUGOROU/h3-fast-ref2va-smd4-gpu-idle-fresh', 'image': 'image'}
            durable = launch.durable_json
            def persist(path, row):
                if path.name == 'submission.json':
                    raise OSError('disk failure')
                durable(path, row)
            with patch.object(launch, 'canonical'), patch.object(launch, 'inspect', return_value=(spec, inventory)), patch('research.launch_h3_fast_ref2va.pre_submit') as gates, patch('httpx.Client') as factory, patch.object(launch, 'durable_json', side_effect=persist):
                factory.return_value.__enter__.return_value = control
                with self.assertRaises(OSError):
                    launch.submit(api, stage, 'secret')
                api.cancel_job.assert_called_once_with(job_id='a'*24)
                self.assertEqual(gates.call_count, 2)
                payload = control.post.call_args.kwargs['json']
                self.assertNotIn('expose', payload)
                self.assertEqual(payload['secrets'], {'HF_TOKEN': 'secret'})
                self.assertEqual(json.loads((stage.parent/'submission_guard.json').read_text())['state'], 'unknown_no_replay')
                with self.assertRaises(ValueError):
                    launch.submit(api, stage, 'secret')
                self.assertEqual(control.post.call_count, 2)

    def test_offline_main_does_not_authenticate(self):
        with patch.object(launch, 'inspect', return_value=({'publication_repo': 'repo'}, {})), patch.object(launch, 'canonical') as canonical:
            self.assertEqual(launch.main(['/tmp/offline']), 0)
            canonical.assert_not_called()


if __name__ == '__main__':
    unittest.main()
