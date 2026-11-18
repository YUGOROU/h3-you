"""Offline mocked lifecycle; no provider, model, capture or device operation."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from research.h3_personal import voxcpm_https_lease as module
from research.prepare_h3_voxcpm_rtc import deployment as rtc_deployment
from research.run_h3_native_coupled import deployment
from research.h3_personal.voxcpm_endpoint_protocol import PINS


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.private = Path(self.temp.name).resolve()
        self.parent = self.private/'parent'; self.parent.mkdir(mode=0o700)
        self.config = dict(schema='h3-native-voxcpm-rtc-v3',
            url='https://voice.endpoints.huggingface.cloud', session='a'*32, boot='b'*32,
            expires_at=9999, pins=dict(PINS), preroll=dict(sample_rate=48000,
                reserve_samples=48000, max_reserve_samples=96000, max_wait_seconds=2.0))
        self.old = rtc_deployment(module.DEFAULT_REPOSITORY, module.DEFAULT_REVISION,
                                 module.NAME, 'a'*32, 'b'*32, 9999)
        self.write_parent()
        self.current = deepcopy(self.old)
        self.current['status'] = dict(state='paused', url=self.config['url'])
        self.calls = []
        self.guard = self.private/module.GUARD.name
        self.client = Mock()
        self.client.read.side_effect = self.ready

    def write_parent(self):
        for name, data in (('intent.json', self.old), ('native-config.json', self.config),
                           ('terminal.json', dict(name=module.NAME, state='paused',
                                                  independent_confirmation=True))):
            (self.parent/name).write_text(json.dumps(data))

    def ready(self, kind, deadline):
        self.assertEqual(kind, 'tts_ready')
        self.assertLessEqual(deadline, 530)
        env = self.current['model']['env']
        return dict(PINS, session=env['H3_TTS_SESSION'], boot=env['H3_TTS_BOOT'],
                    expires_at=int(env['H3_TTS_EXPIRES_AT']), epoch=0)

    def control(self, token, path, payload=None, method='GET'):
        self.assertEqual(token, 'fake-token')
        self.calls.append((path, method, deepcopy(payload)))
        if method == 'PUT':
            self.current['model'] = deepcopy(payload['model'])
            self.current['model'].pop('secrets')
            self.current['compute'] = deepcopy(payload['compute'])
        elif method == 'POST':
            self.assertTrue(path.endswith('/resume'))
            self.current['status']['state'] = 'running'
        return deepcopy(self.current)

    def pause(self, token, path):
        self.current['status']['state'] = 'paused'
        return 'paused'

    @contextmanager
    def mocks(self):
        with ExitStack() as stack:
            patches = dict(PRIVATE=self.private, GUARD=self.guard, DEFAULT_PARENT=self.parent,
                _credentials=Mock(return_value=(Mock(), 'fake-token')),
                preflight=Mock(return_value=({'approved': True}, {'instanceType': 'nvidia-l4'})),
                upload_verified=Mock(return_value='c'*40),
                copy_runtime=Mock(side_effect=lambda path, spec: path.mkdir(mode=0o700)),
                control=Mock(side_effect=self.control), pause_owned=Mock(side_effect=self.pause),
                RemoteVoxCPM=Mock(return_value=self.client))
            for name, value in patches.items():
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(module.time, 'time', return_value=10000))
            stack.enter_context(patch.object(module.time, 'monotonic', return_value=500))
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            yield

    def test_existing_endpoint_put_native_ready_and_independent_pause(self):
        root = self.private/'new'
        with self.mocks():
            with module.lease(root) as value:
                self.assertEqual(set(value), {'config', 'token', 'key', 'root'})
                self.assertEqual(value['config']['schema'], 'h3-native-voxcpm-backend-v1')
                self.assertEqual(value['config']['expires_at'], 11200)
                self.assertEqual(value['config']['pins'], PINS)
                self.assertTrue(self.guard.exists())
                self.assertEqual(self.current['model']['env']['H3_VOX_MEDIA_TRANSPORT'], 'https-native-v1')
                self.assertEqual(self.current['model']['command'],
                                 ['python', '/repository/entrypoint.py', '/repository'])
                self.assertEqual(self.current['compute']['scaling'], {'minReplica': 1, 'maxReplica': 1})
                self.client.read.assert_called_once()
                self.client.close.assert_called_once()
                contents = '\n'.join(p.read_text() for p in root.glob('*.json'))
                self.assertNotIn('fake-token', contents)
                self.assertNotIn(value['key'], contents)
                self.assertNotIn('H3_NATIVE_TTS_KEY', os.environ)
            module.pause_owned.assert_called_once()
            self.assertEqual(self.calls[-1][1], 'GET')
            self.assertFalse(self.guard.exists())
            terminal = json.loads((root/'terminal.json').read_text())
            self.assertTrue(terminal['independent_confirmation'])
            self.assertTrue(terminal['deployment_applied'])
        puts = [row for row in self.calls if row[1] == 'PUT']
        posts = [row for row in self.calls if row[1] == 'POST']
        self.assertEqual(len(puts), 1); self.assertEqual(len(posts), 1)
        self.assertEqual(puts[0][0], '/v2/endpoint/YUGOROU/'+module.NAME)
        self.assertEqual(posts[0][0], puts[0][0]+'/resume')

    def test_cancel_optin_binds_env_client_config_and_fetches_metadata_before_pause(self):
        from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL
        proof=dict(binding={},evidence=dict(runtime={},completions=[]))
        self.client.evidence.return_value=proof
        with self.mocks():
            with module.lease(self.private/'cancel',native_tts_cancel=True) as value:
                self.assertEqual(value['config']['schema'],'h3-native-voxcpm-cancel-v4')
                self.assertEqual(value['config']['cancel_protocol'],CANCEL_PROTOCOL)
                self.assertEqual(self.current['model']['env']['H3_VOX_CANCEL_PROTOCOL'],CANCEL_PROTOCOL)
                self.assertEqual(module.RemoteVoxCPM.call_args.kwargs['cancel_protocol'],CANCEL_PROTOCOL)
                self.assertFalse(self.client.evidence.called)
            self.assertTrue(self.client.evidence.called)
            self.assertEqual(json.loads((self.private/'cancel/native-completion-evidence.json').read_text()),proof)
            self.assertEqual(self.current['status']['state'],'paused')
            self.assertFalse(self.guard.exists())

    def test_cancel_parent_is_explicit_and_no_silent_downgrade(self):
        from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL
        self.config.update(schema='h3-native-voxcpm-cancel-v4',cancel_protocol=CANCEL_PROTOCOL)
        self.old=deployment('YUGOROU/h3-voxcpm-https-runtime-reviewed','d'*40,
                            module.NAME,'a'*32,'b'*32,9999)
        self.old['model']['env'].update(H3_VOX_MEDIA_TRANSPORT='https-native-v1',
                                        H3_VOX_CANCEL_PROTOCOL=CANCEL_PROTOCOL)
        self.write_parent()
        with self.mocks():
            self.assertEqual(module._parent(self.parent)[1],self.old)
            del self.old['model']['env']['H3_VOX_CANCEL_PROTOCOL'];self.write_parent()
            with self.assertRaises(ValueError):module._parent(self.parent)

    def test_explicit_future_https_parent_is_reviewed_not_default_fallback(self):
        self.config['schema'] = 'h3-native-voxcpm-backend-v1'
        self.old = deployment('YUGOROU/h3-voxcpm-https-runtime-reviewed', 'd'*40,
                              module.NAME, 'a'*32, 'b'*32, 9999)
        self.old['model']['env']['H3_VOX_MEDIA_TRANSPORT'] = 'https-native-v1'
        self.write_parent(); self.current = dict(deepcopy(self.old), status=dict(
            state='paused', url=self.config['url']))
        with self.mocks():
            with self.assertRaisesRegex(ValueError, 'default RTC lineage'):
                with module.lease(self.private/'default'):
                    self.fail('unreviewed default admitted')
            with module.lease(self.private/'explicit', parent_campaign=self.parent):
                pass

    def test_identity_revision_hardware_auth_and_parent_terminal_drift_rejected(self):
        for mutate in (
            lambda current: current['model'].update(revision='e'*40),
            lambda current: current['compute'].update(instanceType='nvidia-t4'),
            lambda current: current.update(type='public'),
            lambda current: current['provider'].update(region='other'),
            lambda current: current['compute']['scaling'].update(maxReplica=3),
            lambda current: current['status'].update(state='running')):
            self.current = dict(deepcopy(self.old), status=dict(state='paused', url=self.config['url']))
            mutate(self.current)
            with self.mocks(), self.assertRaises(ValueError):
                with module.lease(self.private/'rejected'):
                    self.fail('drift admitted')
            self.assertFalse(self.guard.exists())
        (self.parent/'terminal.json').write_text('{"state":"paused"}')
        with self.mocks(), self.assertRaisesRegex(ValueError, 'independently paused'):
            with module.lease(self.private/'rejected'):
                self.fail('bad parent admitted')

    def test_ready_failure_and_body_failure_pause_without_exposing_secret(self):
        for stage in ('ready', 'body'):
            self.current = dict(deepcopy(self.old), status=dict(state='paused', url=self.config['url']))
            with self.mocks():
                if stage == 'ready':
                    self.client.read.side_effect = RuntimeError('native READY rejected')
                with self.assertRaises(RuntimeError):
                    with module.lease(self.private/stage):
                        raise RuntimeError('body failed')
                module.pause_owned.assert_called_once()
                self.assertFalse(self.guard.exists())
            self.client.read.side_effect = self.ready

    def test_uncertain_put_still_pauses_and_confirms_exact_old_parent(self):
        def uncertain(token, path, payload=None, method='GET'):
            if method == 'PUT':
                raise TimeoutError('uncertain PUT')
            return self.control(token, path, payload, method)
        with self.mocks():
            module.control.side_effect = uncertain
            with self.assertRaises(TimeoutError):
                with module.lease(self.private/'uncertain'):
                    self.fail('uncertain PUT admitted')
            module.pause_owned.assert_called_once()
            self.assertFalse(self.guard.exists())
            terminal = json.loads((self.private/'uncertain/terminal.json').read_text())
            self.assertFalse(terminal['deployment_applied'])

    def test_unconfirmed_cleanup_retains_guard_and_locks_are_released(self):
        with self.mocks():
            module.pause_owned.side_effect = RuntimeError('pause unknown')
            with self.assertRaisesRegex(RuntimeError, 'pause unknown'):
                with module.lease(self.private/'unconfirmed'):
                    pass
            self.assertTrue(self.guard.exists())
            # A second owner sees the fence, not a leaked process lock.
            with self.assertRaisesRegex(RuntimeError, 'unresolved prior resource'):
                with module.lease(self.private/'second'):
                    self.fail('fenced resource admitted')

    def test_independent_get_and_unknown_cleanup_revision_keep_guard(self):
        for fail_get in (True, False):
            self.current = dict(deepcopy(self.old), status=dict(state='paused', url=self.config['url']))
            root = self.private/('failed-get' if fail_get else 'unknown-revision')
            with self.mocks():
                def bad_pause(token, path):
                    self.current['status']['state'] = 'paused'
                    if fail_get:
                        module.control.side_effect = RuntimeError('independent GET failed')
                    else:
                        self.current['model']['revision'] = 'f'*40
                    return 'paused'
                module.pause_owned.side_effect = bad_pause
                with self.assertRaises((RuntimeError, ValueError)):
                    with module.lease(root):
                        pass
                self.assertTrue(self.guard.exists())
            self.guard.unlink()

    def test_insufficient_remaining_lease_never_resumes(self):
        with self.mocks():
            module.time.time.side_effect = [10000, 10800]
            with self.assertRaisesRegex(TimeoutError, 'before Endpoint update'):
                with module.lease(self.private/'expired-stage'):
                    self.fail('insufficient staging lease admitted')
            self.assertFalse(self.guard.exists())
            self.assertFalse(any(method != 'GET' for _, method, _ in self.calls))
            module.pause_owned.assert_not_called()
        self.calls.clear()
        self.current = dict(deepcopy(self.old), status=dict(state='paused', url=self.config['url']))
        with self.mocks():
            module.time.time.side_effect = [10000, 10000, 10800]
            with self.assertRaisesRegex(TimeoutError, 'before Endpoint resume'):
                with module.lease(self.private/'expired-put'):
                    self.fail('insufficient resume lease admitted')
            module.pause_owned.assert_called_once()
            self.assertFalse(self.guard.exists())
            self.assertFalse(any(method == 'POST' for _, method, _ in self.calls))

    def test_competing_lock_refuses_before_provider_access(self):
        with self.mocks(), patch.object(module.fcntl, 'flock', side_effect=BlockingIOError):
            with self.assertRaises(BlockingIOError):
                with module.lease(self.private/'competing'):
                    self.fail('overlapping owner admitted')
            module._credentials.assert_not_called()
            module.control.assert_not_called()
            self.assertFalse((self.private/'competing').exists())

    def test_preexisting_fence_credentials_bad_paths_and_bounds_precede_update(self):
        with self.mocks():
            for remaining in (419, 901, True, None):
                with self.assertRaises(ValueError):
                    with module.lease(self.private/'bad', min_remaining=remaining):
                        self.fail('bad bound')
            self.guard.write_text('{}')
            with self.assertRaises(RuntimeError):
                with module.lease(self.private/'fenced'):
                    self.fail('fence ignored')
            self.guard.unlink()
            with patch.dict(os.environ, {'H3_NATIVE_TTS_KEY': 'existing'}), self.assertRaises(ValueError):
                with module.lease(self.private/'creds'):
                    self.fail('credentials reused')
            module.control.assert_not_called()
            (self.private/'existing').mkdir()
            with self.assertRaises(ValueError):
                with module.lease(self.private/'existing'):
                    self.fail('directory replay')
            (self.private/'alias').symlink_to(self.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                with module.lease(self.private/'fresh', parent_campaign=self.private/'alias'):
                    self.fail('parent symlink')


if __name__ == '__main__':
    unittest.main()
