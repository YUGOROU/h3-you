"""Offline native probe and finite owner contracts; no GPU/API/device access."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, MagicMock, patch

from research import hf_h3_interrupt_probe as worker
from research import run_h3_interrupt_probe as runner
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import (
    ResamplingContext, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256)


def spec():
    from research.hf_h3_streaming_trial import SECTION
    from research.launch_h3_conversation_scenario import CONVERSATION
    return dict(interrupt_probe=dict(worker.SECTION), streaming_trial=dict(SECTION),
                conversation_scenario=dict(CONVERSATION),
                locked_speech=dict(samples=577920, sample_rate=48000),
                attention_mode='fa4-packed', layout='portrait',
                runtime_policy='resident-visual-cached', config={'timeout_seconds': 1200})


class ProbeTests(unittest.TestCase):
    def probe(self, root, *, inject=True):
        seen = []
        def backend(manifest):
            c = manifest['streaming_audio']; seen.append(c)
            if c['sequence'] == 1 and inject:
                time.sleep(.04)
            end = c['committed_before'] + (124 if c['sequence'] == 0 else 102)
            emitted = end-(0 if c['conversation_final'] else 5)
            before = 0 if c['sequence'] == 0 else c['committed_before']-5
            return dict(video=b'native-fake-only', count=emitted-before,
                        committed_frame=end, emitted_frame=emitted,
                        conditioning_prefix_sha256=(None if c['audio_offer'] is None else
                                                   c['audio_offer']['prefix_sha256']),
                        receipt_sha256='b'*64)
        result = worker.run_probe(Path(root)/'probe', pcm=b'\x01\x00'*577920,
            binding=LiveAVBinding('probe', 'a'*64, 1, max_segments=8, max_total_frames=838),
            context=ResamplingContext('native-48k32k-prefix128', NATIVE_TORCH,
                                      NATIVE_TORCHAUDIO, REFERENCE_SHA256),
            backend=backend)
        return result, seen

    def test_seven_windows_keep_complete_new_speech_after_old_tail_quarantine(self):
        with tempfile.TemporaryDirectory() as root:
            result, seen = self.probe(root)
        self.assertEqual([row['role'] for row in seen],
                         ['listening', 'speech', 'listening', 'speech', 'speech', 'speech', 'listening'])
        self.assertFalse(seen[1]['source_final'])
        self.assertFalse(seen[1]['audio_offer']['eos'])
        self.assertEqual(seen[2]['previous_speech_interrupted']['committed_frame'], 226)
        self.assertEqual(seen[2]['previous_speech_interrupted']['source_id'], 'probe-speech-1')
        self.assertEqual(seen[2]['logical_index'], 0)
        self.assertTrue(seen[5]['source_final'])
        self.assertTrue(seen[5]['audio_offer']['eos'])
        self.assertEqual(result['admission']['committed_frame'], 736)
        self.assertEqual(result['admission']['emitted_frame'], 736)
        self.assertEqual(result['admission']['interrupted_turns'], 1)
        self.assertEqual(result['admission']['completed_turns'], 1)
        self.assertFalse(result['devices_used'])
        self.assertFalse(result['new_live_input'])
        self.assertFalse(result['playback_verified'])
        self.assertEqual(result['saved_pcm_model'], 'legacy-Irodori-approved-fixture')
        self.assertFalse(result['current_vox_route_verified'])
        self.assertFalse(result['source_media_uploaded'])
        self.assertLess(result['local_invalidation_seconds'], .2)
        self.assertEqual([row['index'] for row in result['windows']], list(range(7)))
        # The mixed five-frame old speech tail is quarantined as an A/V unit.
        self.assertEqual(result['windows'][1]['headless_disposition'], 'discarded-interrupted')
        self.assertEqual(result['windows'][2]['headless_disposition'], 'discarded-interrupted')
        self.assertFalse(any(2 in row['source_epochs'] and row['headless_disposition'] != 'selected'
                             for row in result['windows']))
        self.assertTrue(result['new_source_pcm_exact'])
        self.assertEqual(result['new_source_selected_samples'], 577920)
        self.assertEqual(result['new_source_selected_sha256'],
                         hashlib.sha256(b'\x01\x00'*577920).hexdigest())

    def test_bad_source_and_timeout_rejected_before_backend(self):
        for pcm, timeout in ((b'', 180), (b'\0'*1155840, 0),
                             (b'\0'*1155840, float('nan'))):
            backend = Mock()
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                worker.run_probe('/unused', pcm=pcm, binding=None, context=None,
                                 backend=backend, timeout=timeout)
            backend.assert_not_called()

    def test_exact_spec_and_no_combined_diagnostic(self):
        self.assertEqual(worker.validate_spec(spec()), worker.SECTION)
        for update in ({'interrupt_probe': dict(worker.SECTION, devices=True)},
                       {'interrupt_probe': dict(worker.SECTION, devices=0)},
                       {'config': {'timeout_seconds': 1500}},
                       {'late_listening_diagnostic': {}},
                       {'runtime_policy': 'baseline'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                worker.validate_spec(dict(spec(), **update))

    def test_offline_cli_never_constructs_owner(self):
        with patch.object(runner, 'run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main([]), 0)
        run.assert_not_called()
        summary = json.loads(output.getvalue())
        self.assertEqual(summary['scope'], 'offline-only')
        self.assertEqual(summary['hardware'], 'original h200x2')
        self.assertFalse(summary['devices'])

    def test_no_replay_and_private_path_precede_api(self):
        with tempfile.TemporaryDirectory() as root, patch.object(runner, 'PRIVATE', Path(root)):
            with self.assertRaises(ValueError):
                runner.run(Path(root))
            with self.assertRaises(ValueError):
                runner.run('/tmp/not-authorized-probe')

    def test_raw_submit_reserves_unknown_fence_before_single_post(self):
        with tempfile.TemporaryDirectory() as root:
            private = Path(root); campaign = private/'campaign'; stage = campaign/'deployment'
            (stage/'research').mkdir(parents=True)
            bootstrap = stage/'research/bootstrap_h3_fast_ref2va.py'; bootstrap.write_text('pass\n')
            (stage/'upload-allowlist.json').write_text('{}')
            inventory = dict(files=[dict(path='research/bootstrap_h3_fast_ref2va.py')])
            local_spec = dict(publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-test', image='pinned-image')
            api = Mock(); api.whoami.return_value = dict(name='YUGOROU', isPro=True)
            api.create_commit.return_value = NS(oid='c'*40)
            entries = []
            for name in ('research/bootstrap_h3_fast_ref2va.py', 'upload-allowlist.json'):
                body = (stage/name).read_bytes()
                entries.append(NS(rfilename=name, size=len(body), lfs=None,
                    blob_id=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()))
            api.repo_info.return_value = NS(sha='c'*40, private=True, siblings=entries)
            from research.h3_personal.direct_job_api import submit_job as real_submit
            created = Mock(); job = NS(id='a'*24)
            client = Mock()
            client.post.return_value = NS(status_code=201, json=lambda: {'id': job.id})
            context_client = MagicMock(); context_client.__enter__.return_value = client
            def post(client, token, payload, on_created):
                self.assertTrue((campaign/'submission_guard.json').exists())
                self.assertEqual(json.loads((private/runner.FENCE).read_text())['directory'], str(campaign))
                self.assertEqual(payload['flavor'], 'h200x2')
                self.assertEqual(payload['timeoutSeconds'], 1200)
                self.assertEqual(payload['attempts'], 1)
                self.assertEqual(payload['expose'], {'ports': [7860], 'portsPublic': []})
                self.assertNotIn('network', payload)
                return real_submit(client, token, payload, on_created=on_created)
            modules = {'huggingface_hub': NS(CommitOperationAdd=lambda **kw: NS(**kw)),
                       'httpx': NS(Client=Mock(return_value=context_client))}
            with patch.dict('sys.modules', modules), patch.object(runner, 'PRIVATE', private), \
                 patch.object(runner, 'inspect', return_value=(local_spec, inventory)), \
                 patch('research.launch_h3_fast_ref2va.pre_submit') as pins, \
                 patch('research.h3_personal.job_network.assert_idle') as idle, \
                 patch('research.h3_personal.direct_job_api.submit_job', side_effect=post) as submit:
                self.assertEqual(runner.submit(api, stage, 'fake-token', created).id, job.id)
                self.assertEqual(pins.call_count, 2)
                self.assertEqual(idle.call_count, 2)
                submit.assert_called_once()
                created.assert_called_once()
                self.assertEqual(created.call_args.args[0].id, job.id)
                client.post.assert_called_once()
                with self.assertRaisesRegex(ValueError, 'NEVER replay'):
                    runner.submit(api, stage, 'fake-token', created)
                submit.assert_called_once()

    def test_known_owned_job_cancels_after_failed_cleanup_inspection(self):
        with tempfile.TemporaryDirectory() as root:
            private = Path(root).resolve(); campaign = private/'fresh'
            api = Mock()
            api.inspect_job.side_effect = [OSError('temporary control failure'), NS(status=NS(stage='CANCELED'))]
            modules = {'huggingface_hub': NS(HfApi=lambda **_: api,
                hf_hub_download=Mock(), set_client_factory=Mock()),
                'httpx': NS(Client=MagicMock())}
            def prepare(directory):
                stage = directory/'deployment'; stage.mkdir(parents=True)
                return stage
            def submit(api, stage, token, created):
                (private/runner.FENCE).write_text(json.dumps({'directory': str(campaign)}))
                created(NS(id='exact-owned'))
                raise KeyboardInterrupt()
            with patch.dict('sys.modules', modules), patch.object(runner, 'PRIVATE', private), \
                 patch.object(runner, 'prepare', side_effect=prepare), \
                 patch.object(runner, 'submit', side_effect=submit), \
                 patch('research.provision_modal_identity_secret.cached_token', return_value='fake'), \
                 patch('research.launch_h3_relay_conversation.canonical'):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(campaign)
            api.cancel_job.assert_called_once_with(job_id='exact-owned', namespace='YUGOROU')
            self.assertFalse((private/runner.FENCE).exists())
            result = json.loads((campaign/'result.json').read_text())
            self.assertTrue(result['cleanup_confirmed'])
            self.assertEqual(result['terminal'], 'CANCELED')
            self.assertFalse(result['complete'])

    def test_uncertain_create_keeps_fence_but_explicit_rejection_clears_it(self):
        from research.h3_personal.direct_job_api import JobCreationRejected
        for status, known in ((422, True), (500, False)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                private = Path(root).resolve(); campaign = private/'fresh'
                api = Mock()
                modules = {'huggingface_hub': NS(HfApi=lambda **_: api,
                    hf_hub_download=Mock(), set_client_factory=Mock()),
                    'httpx': NS(Client=MagicMock())}
                def prepare(directory):
                    stage = directory/'deployment'; stage.mkdir(parents=True)
                    return stage
                def submit(api, stage, token, created):
                    (private/runner.FENCE).write_text(json.dumps({'directory': str(campaign)}))
                    (campaign/'submission_guard.json').write_text('{"state":"unknown_no_replay"}')
                    raise JobCreationRejected(status, 'sanitized')
                with patch.dict('sys.modules', modules), patch.object(runner, 'PRIVATE', private), \
                     patch.object(runner, 'prepare', side_effect=prepare), \
                     patch.object(runner, 'submit', side_effect=submit), \
                     patch('research.provision_modal_identity_secret.cached_token', return_value='fake'), \
                     patch('research.launch_h3_relay_conversation.canonical'):
                    with self.assertRaises(JobCreationRejected):
                        runner.run(campaign)
                self.assertEqual((private/runner.FENCE).exists(), not known)
                self.assertTrue((campaign/'submission_guard.json').exists())
                result = json.loads((campaign/'result.json').read_text())
                self.assertEqual(result['cleanup_confirmed'], known)
                api.cancel_job.assert_not_called()  # unknown ID is never guessed
                api.inspect_job.assert_not_called()

    def test_uncertain_publication_adopts_only_exact_empty_private_repo_without_post_retry(self):
        class TransportError(Exception):pass
        repo = 'YUGOROU/fresh-private-probe'
        for identity, private, names, allowed in (
            (repo, True, ['.gitattributes'], True),
            (repo, True, ['other.py'], False),
            (repo, False, ['.gitattributes'], False),
            ('YUGOROU/other', True, ['.gitattributes'], False)):
            with self.subTest(identity=identity, private=private, names=names):
                api = Mock(); api.create_repo.side_effect = TransportError()
                api.repo_info.return_value = NS(id=identity, private=private,
                    siblings=[NS(rfilename=name) for name in names])
                with patch.dict('sys.modules', {'httpx': NS(TransportError=TransportError)}):
                    if allowed:
                        runner.create_publication_once(api, repo)
                    else:
                        with self.assertRaises(RuntimeError):
                            runner.create_publication_once(api, repo)
                api.create_repo.assert_called_once()
                api.repo_info.assert_called_once_with(repo, repo_type='dataset')

    def test_terminal_confirmation_is_exact_owned_job(self):
        api = Mock(); api.inspect_job.return_value = NS(status=NS(stage='CANCELED'))
        self.assertEqual(runner.confirm_terminal(api, NS(id='owned'), seconds=.1), 'CANCELED')
        api.inspect_job.assert_called_once_with(job_id='owned', namespace='YUGOROU')


if __name__ == '__main__':
    unittest.main()
