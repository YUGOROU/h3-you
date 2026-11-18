"""CPU-only controller tests: no network, GPU imports, or paid calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research import hf_identity_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def document(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': pipeline.digest(path)}

    def test_budget_and_deadline_caps(self):
        base = dict(schema='h3-identity-pipeline-v1', overall_seconds=23401,
                    rate_usd_hour=10, spent_and_reserved_usd=0)
        with self.assertRaises(ValueError):
            pipeline.validate(base)
        base.update(overall_seconds=21600, rate_usd_hour=10.01)
        with self.assertRaises(ValueError):
            pipeline.validate(base)
        base.update(rate_usd_hour=10, spent_and_reserved_usd=100)
        with self.assertRaises(ValueError):
            pipeline.validate(base)

    def test_byte_pin(self):
        pin = self.document('pin.json', {'a': 1})
        self.assertEqual(pipeline.pinned(pin['path'], pin['sha256']), {'a': 1})
        Path(pin['path']).write_text('{}')
        with self.assertRaises(ValueError):
            pipeline.pinned(pin['path'], pin['sha256'])

    def test_missing_approval_defaults_plan_only(self):
        pin = self.document('request.json', {'schema': 'h3-identity-pipeline-v1'})
        with patch.object(pipeline, 'run') as run:
            self.assertEqual(pipeline.main(['--request', pin['path'], '--request-sha256', pin['sha256']]), 0)
            run.assert_not_called()

    def test_approval_precedes_sdk(self):
        download = unittest.mock.Mock()
        with patch.object(pipeline, 'approvals', side_effect=ValueError('blocked')):
            with self.assertRaises(ValueError):
                pipeline.provision({}, self.root / 'model', NS(), download)
        download.assert_not_called()

    def provision_fixture(self, lfs=False, corrupt=False, symlink=False):
        data = b'asset bytes'
        lock = self.document('model-lock.json', {'revision': 'a' * 40, 'files':
                             {'model_index.json': hashlib.sha256(data).hexdigest()}})
        r = {'model_lock': lock, 'model_repo': 'MiniMaxAI/MiniMax-H3', 'transformer_files': [],
             'cache_repo': 'owner/cache', 'output_repo': 'owner/adapter'}
        blob = hashlib.sha1(b'blob 11\0' + data).hexdigest()
        item = NS(rfilename='model_index.json', size=len(data), blob_id=blob,
                  lfs=NS(sha256=hashlib.sha256(data).hexdigest()) if lfs else None)
        api = NS(model_info=lambda *a, **kw: NS(sha='a' * 40, siblings=[item]),
                 repo_info=lambda *a, **kw: NS(private=True), auth_check=unittest.mock.Mock())
        def download(**kw):
            self.assertEqual(kw['revision'], 'a' * 40)
            self.assertEqual(kw['max_workers'], 1)
            path = Path(kw['local_dir']) / 'model_index.json'
            if symlink:
                outside = self.root / 'outside'
                outside.write_bytes(data)
                path.symlink_to(outside)
            else:
                path.write_bytes(b'bad' if corrupt else data)
        with patch.object(pipeline, 'approvals'), patch.object(pipeline, 'encoder_inventory'):
            return pipeline.provision(r, self.root / 'model', api, download)

    def test_private_repository_access_before_provision(self):
        r = dict(cache_repo='owner/cache', output_repo='owner/adapter')
        for blocked in ('public-cache', 'public-output', 'no-write', 'missing'):
            with self.subTest(blocked=blocked):
                api = unittest.mock.Mock()
                api.repo_info.side_effect = (
                    RuntimeError('missing') if blocked == 'missing' else
                    lambda repo, **kw: NS(private=not (
                        (blocked == 'public-cache' and kw['repo_type'] == 'dataset') or
                        (blocked == 'public-output' and kw['repo_type'] == 'model'))))
                if blocked == 'no-write':
                    api.auth_check.side_effect = PermissionError('denied')
                download = unittest.mock.Mock()
                with patch.object(pipeline, 'approvals'):
                    with self.assertRaises((ValueError, RuntimeError, PermissionError)):
                        pipeline.provision(r, self.root / 'model', api, download)
                api.model_info.assert_not_called()
                download.assert_not_called()
                self.assertFalse((self.root / 'model').exists())
        api = unittest.mock.Mock()
        api.repo_info.return_value = NS(private=True)
        pipeline.private_repositories(r, api)
        self.assertEqual(api.repo_info.call_args_list, [
            unittest.mock.call('owner/cache', repo_type='dataset'),
            unittest.mock.call('owner/adapter', repo_type='model')])
        self.assertEqual(api.auth_check.call_args_list, [
            unittest.mock.call('owner/cache', repo_type='dataset', write=True),
            unittest.mock.call('owner/adapter', repo_type='model', write=True)])
        api.create_repo.assert_not_called()
        api.create_commit.assert_not_called()

    def valid_request(self):
        versions = {key: 'test-version' for key in ('torch', 'diffusers', 'transformers',
                    'safetensors', 'Pillow', 'numpy', 'accelerate', 'peft', 'huggingface-hub')}
        return dict(schema='h3-identity-pipeline-v1', overall_seconds=21600,
                    rate_usd_hour=1, spent_and_reserved_usd=0, flavor='h200',
                    output_repo='owner/adapter', cache_repo='owner/cache', max_updates=10,
                    stage_seconds=9000, source_revision=pipeline.trainer.helper().SOURCE_PINS['trainer'],
                    runtime_source_revisions=dict(pipeline.trainer.helper().SOURCE_PINS),
                    model_repo='MiniMaxAI/MiniMax-H3', transformer_files=['transformer_ref/config.json'],
                    model_lock=self.document('model-lock.json', {
                        'schema': 'h3-local-model-lock-v1', 'revision': pipeline.trainer.MODEL_PIN,
                        'files': {'model_index.json': 'a' * 64}}),
                    runtime_lock=self.document('runtime-lock.json', {
                        'schema': 'h3-encode-runtime-v1', 'versions': versions,
                        'modules': {key: 'b' * 64 for key in pipeline.encoder.REQUIRED_MODULES}}),
                    runtime_expected_versions=versions, runtime_provenance='CPU fixture only',
                    approval=self.document('approval.json', {}),
                    input_documents={key: self.document(key + '.json', {}) for key in (
                        'selection_path', 'manifest_path', 'summary_path', 'source_map_path',
                        'reference_mapping_path', 'caption_mapping_path')})

    def test_request_private_repo_and_actual_pins(self):
        r = self.valid_request()
        self.assertIs(pipeline.validate(r), r)
        for changes in ({'cache_repo': None}, {'cache_repo': r['output_repo']},
                        {'output_repo': '../bad'}, {'cache_repo': 'owner/cache/token'},
                        {'source_revision': '0' * 40}, {'runtime_source_revisions': {}},
                        {'model_repo': 'owner/replacement'},
                        {'model_lock': self.document('wrong-model.json', {
                            'schema': 'h3-local-model-lock-v1', 'revision': '0' * 40})}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pipeline.validate(dict(r, **changes))

    def test_every_encoder_command_persists_to_dataset(self):
        r = self.valid_request()
        r['source_root'] = '/remote/source'
        out = self.root / 'out'
        out.mkdir()
        pipeline.write_json(out / 'inputs.json', {})
        pipeline.write_json(out / 'loader-lock.json', {})
        for stage, rank in (('media', 0), ('media', 1), ('reference', 0), ('text', 0), ('merge', 0)):
            cmd = pipeline.encoder_command(r, out, stage, 701, rank)
            self.assertEqual(cmd[3], stage)
            self.assertEqual(cmd[cmd.index('--rank') + 1], str(rank))
            self.assertEqual(cmd[cmd.index('--snapshot-repo') + 1], r['cache_repo'])
            self.assertNotIn(r['output_repo'], cmd)
            self.assertEqual(cmd[cmd.index('--deadline-seconds') + 1], '701')
            self.assertEqual(cmd[cmd.index('--output') + 1], str(out / 'components'))
            for name in ('model-lock', 'runtime-lock', 'approval'):
                self.assertEqual(cmd[cmd.index('--' + name + '-sha256') + 1],
                                 r[name.replace('-', '_')]['sha256'])

    def test_train_request_unique_safe_run_id_and_unchanged_wall_budget(self):
        r = self.valid_request()
        r.update(source_root='/remote/source', territory_approval={})
        out = self.root / 'out'
        (out / 'components').mkdir(parents=True)
        pipeline.write_json(out / 'components/index.json', {})
        pipeline.write_json(out / 'inventory.json', {})
        with patch.object(pipeline.trainer, 'validate_request') as validate:
            first = pipeline.train_request(r, out, 1700)
            second = pipeline.train_request(r, out, 1700)
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertRegex(first['run_id'], r'^h3-[0-9a-f]{32}$')
        self.assertEqual(second['overall_seconds'], 1700)  # Child owns its 300s save guard.
        self.assertEqual(second['max_updates'], r['max_updates'])
        self.assertEqual(second['pins']['sources'], pipeline.trainer.helper().SOURCE_PINS)
        self.assertEqual(second['pins']['models']['transformer_ref']['revision'], pipeline.trainer.MODEL_PIN)
        self.assertEqual(json.loads((out / 'train-request.json').read_text()), second)
        self.assertEqual(validate.call_count, 2)

    def test_git_inventory(self):
        self.assertEqual(set(self.provision_fixture()), {'model_index.json'})

    def test_lfs_inventory(self):
        self.assertEqual(set(self.provision_fixture(lfs=True)), {'model_index.json'})

    def test_corrupt_inventory(self):
        with self.assertRaises(ValueError):
            self.provision_fixture(corrupt=True)

    def test_symlink_inventory(self):
        with self.assertRaises(ValueError):
            self.provision_fixture(symlink=True)

    def test_cached_training_skips_encoders_and_reserves_same_job_preview(self):
        out = self.root / 'cached'
        r = dict(output=str(out), overall_seconds=21600, stage_seconds=9000,
                 cache_reuse={}, inference={'seconds':2700})
        calls = []
        def runner(cmd, deadline):
            calls.append(cmd)
            if 'verify-upload' in cmd:
                (out / 'training').mkdir()
                pipeline.write_json(out / 'training/worker-status.json', {
                    'uploaded':True, 'status':'final_strict_reload_verified_quality_unverified'})
        with patch.object(pipeline, 'approvals'), patch.object(pipeline, 'encoder_command') as encoder:
            pipeline.run(r, self.root / 'request', 'a'*64, runner=runner)
        encoder.assert_not_called()
        stages = [c[c.index('--internal-stage')+1] for c in calls]
        self.assertEqual(stages, ['hardware','prepare','train','verify-upload','preview'])
        self.assertLessEqual(int(calls[2][-1]), 17700)
        self.assertLessEqual(int(calls[-1][-1]), 2700)

    def test_serial_stages_and_verify_reserve(self):
        out = self.root / 'out'
        r = dict(output=str(out), overall_seconds=21600, stage_seconds=9000)
        commands = []
        def runner(cmd, deadline):
            commands.append((cmd, deadline))
            if 'verify-upload' in cmd:
                (out / 'training').mkdir()
                pipeline.write_json(out / 'training/worker-status.json', {
                    'uploaded': True, 'status': 'final_strict_reload_verified_quality_unverified'})
        def enc(r, out, stage, seconds, rank):
            return ['fake', stage, str(rank), str(seconds)]
        with patch.object(pipeline, 'approvals'), patch.object(pipeline, 'encoder_command', side_effect=enc):
            pipeline.run(r, self.root / 'request', 'a' * 64, runner=runner)
        stages = [c[0][1:3] for c in commands[2:7]]
        self.assertEqual(stages, [['media', '0'], ['media', '1'], ['reference', '0'],
                                  ['text', '0'], ['merge', '0']])
        train = commands[-2][0]
        self.assertLessEqual(int(train[-1]), 20400)
        self.assertLessEqual(int(commands[-1][0][-1]), 1200)
        self.assertLessEqual(commands[-2][1], pipeline.time.monotonic() + 20400)
        self.assertTrue(json.loads((out / 'pipeline-report.json').read_text())['readiness'])
        self.assertEqual(out.stat().st_mode & 0o777, 0o700)

    def test_real_command_builders_with_fake_workers_and_elapsed_budget(self):
        r = self.valid_request()
        out = self.root / 'out'
        r.update(output=str(out), source_root='/remote/source')
        clock = [1000.0]
        calls = []
        def runner(cmd, deadline):
            calls.append((cmd, deadline))
            if '--internal-stage' in cmd:
                stage = cmd[cmd.index('--internal-stage') + 1]
                if stage == 'prepare':
                    pipeline.write_json(out / 'inputs.json', {})
                    pipeline.write_json(out / 'loader-lock.json', {})
                if stage == 'verify-upload':
                    (out / 'training').mkdir()
                    pipeline.write_json(out / 'training/worker-status.json', {
                        'uploaded': True, 'status': 'final_strict_reload_verified_quality_unverified'})
                clock[0] += {'hardware': 10, 'prepare': 100, 'train': 17000,
                             'verify-upload': 100}[stage]
            else:
                self.assertEqual(cmd[cmd.index('--snapshot-repo') + 1], 'owner/cache')
                clock[0] += 100
        with patch.object(pipeline, 'approvals'), patch.object(pipeline.time, 'monotonic',
                                                             side_effect=lambda: clock[0]):
            pipeline.run(r, self.root / 'request', 'a' * 64, runner=runner)
        self.assertEqual(len(calls), 9)  # No watcher, pilot, retry or duplicate encoding.
        train, verify = calls[-2:]
        self.assertEqual(int(train[0][-1]), 19790)
        self.assertEqual(train[1], 21400)  # Overall end22600 minus1200.
        self.assertEqual(int(verify[0][-1]), 1200)
        status = json.loads((out / 'pipeline-report.json').read_text())
        self.assertTrue(status['readiness'])
        self.assertFalse(status['quality_verified'])
        self.assertFalse(status['c_gate_enabled'])

    def test_exhausted_reserve_stops_before_next_worker(self):
        out = self.root / 'out'
        clock = [1000.0]
        calls = []
        def runner(cmd, deadline):
            calls.append(cmd)
            clock[0] += 2
        with patch.object(pipeline, 'approvals'), patch.object(pipeline.time, 'monotonic',
                                                             side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ValueError, 'reserve exhausted'):
                pipeline.run(dict(output=str(out), overall_seconds=1201, stage_seconds=9000),
                             self.root / 'request', 'a' * 64, runner=runner)
        self.assertEqual(len(calls), 1)
        status = json.loads((out / 'pipeline-report.json').read_text())
        self.assertFalse(status['readiness'])
        self.assertEqual(status['stage'], 'prepare')

    def test_valid_offline_plan_never_checks_hub_or_executes(self):
        pin = self.document('request.json', self.valid_request())
        with patch.object(pipeline, 'approvals'), patch.object(pipeline, 'run') as run, \
                patch.object(pipeline, 'private_repositories') as access:
            self.assertEqual(pipeline.main(['--request', pin['path'],
                                            '--request-sha256', pin['sha256']]), 0)
        run.assert_not_called()
        access.assert_not_called()

    def test_failure_stops_no_retry_no_secret(self):
        out = self.root / 'out'
        calls = []
        def runner(command, deadline):
            calls.append(command)
            raise RuntimeError('SECRET_TOKEN must not appear')
        with patch.object(pipeline, 'approvals'):
            with self.assertRaises(RuntimeError):
                pipeline.run(dict(output=str(out), overall_seconds=21600, stage_seconds=9000),
                             self.root / 'req', 'a' * 64, runner=runner)
        self.assertEqual(len(calls), 1)
        report = (out / 'pipeline-report.json').read_text()
        self.assertNotIn('SECRET', report)
        self.assertFalse(json.loads(report)['readiness'])

    def test_fake_command_watchdog(self):
        with self.assertRaises(TimeoutError):
            pipeline.trainer.bounded_process([sys.executable, '-c', 'import time; time.sleep(10)'],
                                             deadline=pipeline.time.monotonic() + .05)


if __name__ == '__main__':
    unittest.main()
