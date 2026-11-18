"""Offline launcher contracts: no Hub calls, GPU, ledger edits or submission."""
import json
from pathlib import Path
import socket
import tempfile
import unittest
from decimal import Decimal
from unittest.mock import patch

from research import launch_h3_continuation_diagnostic as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage = self.root/'deployment'
        self.stage.mkdir()
        self.ledger = self.root/'ledger.json'
        self.ledger.write_text(json.dumps(dict(spent_usd=250, reservations={}, authorized_cap_usd=275)))
        self.network = patch.object(socket, 'create_connection', side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def fixture(self, timeout=1800):
        spec = dict(config=dict(timeout_seconds=timeout),
                    execution=dict(timeout_seconds=timeout, max_jobs=1),
                    continuation_diagnostic=launcher.contract(timeout),
                    locked_speech=launcher.locked_speech(), layout='portrait',
                    runtime_policy='resident-visual-cached', attention_mode='fa4-packed')
        for name in (*launcher.EXTRA, launcher.BOOTSTRAP):
            target = self.stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# continuation_diagnostic hf_h3_continuation_diagnostic\n')
        self.save(spec)
        return spec

    def save(self, spec):
        (self.stage/'spec.json').write_text(json.dumps(spec))
        launcher.write_inventory(self.stage)

    def test_contract_and_fixed_speech(self):
        for seconds in (1, 1501, 1800, 3599, 3600):
            result = launcher.contract(seconds)
            self.assertEqual(result['expected_windows'], 3)
            self.assertEqual(result['suffix_frames'], [119, 102, 107])
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['source']['new_speech'])
        for bad in (0, 3601, True, 1.5):
            with self.assertRaises(ValueError):
                launcher.contract(bad)
        self.assertEqual(set(launcher.locked_speech()),
                         {'repo', 'revision', 'path', 'sha256', 'sample_rate', 'samples'})
        self.assertEqual(launcher.locked_speech()['samples'], 577920)

    def test_comparison_contract_is_explicit_and_rejects_tampering(self):
        from research.hf_h3_continuation_diagnostic import CONTINUATION_COMPARISON
        spec = self.fixture()
        spec['continuation_comparison'] = dict(CONTINUATION_COMPARISON)
        self.save(spec)
        with patch.object(launcher.base, 'inspect'):
            self.assertEqual(launcher.inspect(self.stage)[0], spec)
        spec['continuation_comparison']['same_warm_runtime'] = False
        self.save(spec)
        with patch.object(launcher.base, 'inspect'), self.assertRaisesRegex(ValueError, 'comparison'):
            launcher.inspect(self.stage)

    def test_efficiency_contract_fresh_copy_and_strict_inspection(self):
        helper = launcher.continuation_efficiency
        expected = helper.comparison_contract()
        changed = helper.comparison_contract()
        changed['order'].append('changed')
        self.assertEqual(helper.comparison_contract(), expected)
        spec = self.fixture()
        spec['efficiency_comparison'] = expected
        self.save(spec)
        with patch.object(launcher.base, 'inspect'):
            self.assertEqual(launcher.inspect(self.stage)[0], spec)
        for value in (None, {}, dict(expected, profile_stages=0),
                      dict(expected, atol=0), dict(expected, total_frames=True),
                      dict(expected, extra=True), dict(expected, order=['baseline'])):
            spec['efficiency_comparison'] = value
            self.save(spec)
            with patch.object(launcher.base, 'inspect'), self.assertRaisesRegex(ValueError, 'efficiency comparison'):
                launcher.inspect(self.stage)
        for key in ('continuation_candidate', 'continuation_comparison', 'decode_parity_diagnostic'):
            spec = self.fixture()
            spec.update(efficiency_comparison=expected)
            spec[key] = None
            self.save(spec)
            with patch.object(launcher.base, 'inspect'), self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                launcher.inspect(self.stage)

    def test_efficiency_bool_modes_and_prepare_only(self):
        for value in (None, 0, 1, 'true', [], {}):
            with self.assertRaises(ValueError):
                launcher.prepare(self.root, None, self.ledger, None, efficiency_comparison=value)
        modes = ('comparison', 'decode_parity', 'window_audio_context', 'defer_progress_receipts')
        for mode in modes:
            with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                launcher.prepare(self.root, None, self.ledger, None,
                                 efficiency_comparison=True, **{mode: True})
        args = ['--directory', str(self.root), '--ledger', str(self.ledger)]
        flags = [[], ['--submit']] + [['--prepare', '--'+mode.replace('_', '-')] for mode in modes]
        for other in flags:
            with patch.object(launcher, 'prepare') as prepare, patch.object(launcher, 'launch') as launch:
                with self.assertRaises(SystemExit):
                    launcher.main(args + other + ['--efficiency-comparison'])
                prepare.assert_not_called()
                launch.assert_not_called()

    def test_decode_parity_contract_and_malformed_rejection(self):
        expected = dict(schema='h3-same-latent-decode-parity-v1', serial=True,
                        atol=0.0, retain_latents_in_memory_only=True,
                        total_frames=328, arrivals=[37, 30, 30])
        self.assertEqual(launcher.decode_parity_contract(), expected)
        spec = self.fixture()
        spec['decode_parity_diagnostic'] = expected
        self.save(spec)
        with patch.object(launcher.base, 'inspect'):
            self.assertEqual(launcher.inspect(self.stage)[0], spec)
        malformed = [None, {}, dict(expected, serial=1), dict(expected, atol=False),
                     dict(expected, atol=0), dict(expected, arrivals=[37, 30]),
                     dict(expected, total_frames=329), dict(expected, extra=True)]
        for value in malformed:
            spec['decode_parity_diagnostic'] = value
            self.save(spec)
            with patch.object(launcher.base, 'inspect'), self.assertRaisesRegex(ValueError, 'decode parity'):
                launcher.inspect(self.stage)
        spec['decode_parity_diagnostic'] = expected
        spec['continuation_comparison'] = {}
        self.save(spec)
        with patch.object(launcher.base, 'inspect'), self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            launcher.inspect(self.stage)

    def test_decode_parity_opt_in_validation_and_cli(self):
        for value in (None, 0, 1, 'true', []):
            with self.assertRaises(ValueError):
                launcher.prepare(self.root, None, self.ledger, None, decode_parity=value)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            launcher.prepare(self.root, None, self.ledger, None, comparison=True, decode_parity=True)
        args = ['--directory', str(self.root), '--ledger', str(self.ledger)]
        for flags in (['--decode-parity'], ['--submit', '--decode-parity'],
                      ['--prepare', '--comparison', '--decode-parity']):
            with patch.object(launcher, 'prepare') as prepare, patch.object(launcher, 'launch') as launch:
                with self.assertRaises(SystemExit):
                    launcher.main(args + flags)
                prepare.assert_not_called()
                launch.assert_not_called()

    def test_candidate_policies_and_tampering(self):
        helper = launcher.continuation_candidate
        for audio, progress in ((True, False), (False, True), (True, True)):
            spec = self.fixture()
            options = helper.make_options(window_audio_context=audio,
                                          defer_progress_receipts=progress)
            spec['continuation_candidate'] = options
            self.save(spec)
            with patch.object(launcher.base, 'inspect'):
                self.assertEqual(launcher.inspect(self.stage)[0], spec)
            self.assertEqual(options['audio_context'],
                             'window-context-canonical-v1' if audio else 'full-source-v1')
            self.assertEqual(options['progress_receipts'],
                             'bind-and-commit-v1' if progress else 'per-step-v1')
        malformed = [None, {}, True, dict(options, extra=True),
                     dict(options, schema='unknown'), dict(options, audio_context=True),
                     dict(options, progress_receipts='unknown'),
                     dict(options, audio_context='full-source-v1', progress_receipts='per-step-v1')]
        for value in malformed:
            spec = self.fixture()
            spec['continuation_candidate'] = value
            self.save(spec)
            with patch.object(launcher.base, 'inspect'), self.assertRaises(ValueError):
                launcher.inspect(self.stage)
        for key in ('continuation_comparison', 'decode_parity_diagnostic'):
            spec = self.fixture()
            spec.update(continuation_candidate=options)
            spec[key] = {}
            self.save(spec)
            with patch.object(launcher.base, 'inspect'), self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                launcher.inspect(self.stage)

    def test_candidate_strict_bool_and_combinations(self):
        for key in ('window_audio_context', 'defer_progress_receipts'):
            for value in (None, 0, 1, 'true', [], {}):
                with self.assertRaises(ValueError):
                    launcher.prepare(self.root, None, self.ledger, None, **{key: value})
            for mode in ('comparison', 'decode_parity'):
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    launcher.prepare(self.root, None, self.ledger, None,
                                     **{key: True, mode: True})
        args = ['--directory', str(self.root), '--ledger', str(self.ledger)]
        for flags in (['--window-audio-context'], ['--submit', '--defer-progress-receipts'],
                      ['--prepare', '--comparison', '--window-audio-context'],
                      ['--prepare', '--decode-parity', '--defer-progress-receipts']):
            with patch.object(launcher, 'prepare') as prepare, patch.object(launcher, 'launch') as launch:
                with self.assertRaises(SystemExit):
                    launcher.main(args + flags)
                prepare.assert_not_called()
                launch.assert_not_called()

    def test_timeout_projection_preserves_original(self):
        spec = self.fixture(3600)
        before = (self.stage/'spec.json').read_bytes()
        def check(view):
            projected = json.loads((view/'spec.json').read_text())
            expected = json.loads(json.dumps(spec))
            expected['config']['timeout_seconds'] = 1500
            expected['execution']['timeout_seconds'] = 1500
            self.assertEqual(projected, expected)
        with patch.object(launcher.base, 'inspect', side_effect=check) as base:
            checked, _ = launcher.inspect(self.stage)
        base.assert_called_once()
        self.assertEqual(checked, spec)
        self.assertEqual(before, (self.stage/'spec.json').read_bytes())

    def test_tampering_rejected_before_base(self):
        self.fixture()
        (self.stage/launcher.WORKER).write_text('changed')
        with patch.object(launcher.base, 'inspect') as base:
            with self.assertRaises(ValueError):
                launcher.inspect(self.stage)
            base.assert_not_called()

    def test_wrong_contract_live_source_and_missing_runtime(self):
        mutations = [lambda s: s['continuation_diagnostic'].update(expected_windows=2),
                     lambda s: s['locked_speech'].update(sha256='0'*64),
                     lambda s: s.update(live_voice_session={}),
                     lambda s: s.update(modal_live_approval={}),
                     lambda s: s.update(attention_mode='fa2')]
        for mutate in mutations:
            spec = self.fixture()
            mutate(spec)
            self.save(spec)
            with patch.object(launcher.base, 'inspect'), self.assertRaises(ValueError):
                launcher.inspect(self.stage)
        self.fixture()
        (self.stage/launcher.WORKER).unlink()
        launcher.write_inventory(self.stage)
        with patch.object(launcher.base, 'inspect'), self.assertRaises(ValueError):
            launcher.inspect(self.stage)

    def test_base_pin_failure_not_suppressed(self):
        self.fixture()
        with patch.object(launcher.base, 'inspect', side_effect=ValueError('asset pin differs')):
            with self.assertRaisesRegex(ValueError, 'asset pin'):
                launcher.inspect(self.stage)

    def test_pre_submit_reuses_remote_permission_checks(self):
        spec = self.fixture()
        def checked(view):
            return json.loads((view/'spec.json').read_text()), {}
        api = object()
        with patch.object(launcher.base, 'inspect', side_effect=checked), patch.object(launcher.base, 'pre_submit') as remote:
            launcher.pre_submit(api, spec, self.stage)
        remote.assert_called_once()
        self.assertIs(remote.call_args.args[0], api)
        self.assertEqual(remote.call_args.args[1]['config']['timeout_seconds'], 1500)

    def test_default_offline_and_explicit_submit_delegation(self):
        spec = self.fixture()
        args = ['--directory', str(self.root), '--ledger', str(self.ledger)]
        original = self.ledger.read_bytes()
        with patch.object(launcher, 'inspect', return_value=(spec, {})), patch.object(launcher, 'launch', return_value=0) as launch, patch.object(launcher, 'prepare') as prepare:
            self.assertEqual(launcher.main(args), 0)
            self.assertNotIn('--submit', launch.call_args.args[0])
            prepare.assert_not_called()
            launcher.main(args+['--submit'])
            self.assertIn('--submit', launch.call_args.args[0])
            self.assertEqual(launch.call_args.kwargs['approval_request'], launcher.APPROVAL)
        self.assertEqual(self.ledger.read_bytes(), original)

    def test_default_missing_stage_never_prepares(self):
        with patch.object(launcher, 'prepare') as prepare:
            with self.assertRaises(FileNotFoundError):
                launcher.main(['--directory', str(self.root), '--ledger', str(self.ledger)])
            prepare.assert_not_called()

    def test_prepare_packages_fresh_modules_without_launch(self):
        from types import SimpleNamespace
        source = self.root/'source'
        for name in launcher.EXTRA:
            target = source/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# fresh runtime\n')
        destination = self.root/'new'
        fixture_spec = self.fixture()
        # A plain function exercises the preparer's private globals binding.
        def fake_prepare(output, template, ledger, api, **kwargs):
            output.mkdir(parents=True)
            spec = dict(config=dict(timeout_seconds=kwargs['timeout_seconds']),
                        locked_speech=kwargs['locked_speech'])
            (output/'spec.json').write_text(json.dumps(spec))
            return {'job_bound': '5.00001'}
        with patch.object(launcher, 'ROOT', source), patch.object(launcher.staging, 'prepare', fake_prepare), patch.object(launcher, 'inspect', return_value=(fixture_spec, {})), patch.object(launcher, 'pre_submit') as permissions, patch.object(launcher, 'launch') as launch:
            launcher.prepare(destination, self.root/'template', self.ledger, SimpleNamespace(), comparison=True)
            launcher.prepare(self.root/'parity', self.root/'template', self.ledger,
                             SimpleNamespace(), decode_parity=True)
            launcher.prepare(self.root/'efficiency', self.root/'template', self.ledger,
                             SimpleNamespace(), efficiency_comparison=True)
            efficiency_stage = self.root/'efficiency/deployment'
            efficiency = json.loads((efficiency_stage/'spec.json').read_text())
            self.assertEqual(efficiency['efficiency_comparison'],
                             launcher.continuation_efficiency.comparison_contract())
            for key in ('continuation_candidate', 'continuation_comparison', 'decode_parity_diagnostic'):
                self.assertNotIn(key, efficiency)
            self.assertEqual(efficiency['locked_speech']['samples'], 577920)
            for name in ('research/hf_h3_efficiency_comparison.py',
                         'research/h3_personal/parallel_decode_parity.py'):
                self.assertIn(name, launcher.EXTRA)
                self.assertEqual((efficiency_stage/name).read_bytes(), (source/name).read_bytes())
            self.assertEqual(json.loads((efficiency_stage/'upload-allowlist.json').read_text()),
                             launcher.inventory(efficiency_stage))
            parity = json.loads((self.root/'parity/deployment/spec.json').read_text())
            self.assertEqual(parity['decode_parity_diagnostic'], launcher.decode_parity_contract())
            self.assertNotIn('continuation_comparison', parity)
            permissions.assert_called()
            permissions.reset_mock()
            launcher.prepare(self.root/'baseline', self.root/'template', self.ledger, SimpleNamespace())
            baseline = json.loads((self.root/'baseline/deployment/spec.json').read_text())
            self.assertNotIn('decode_parity_diagnostic', baseline)
            self.assertNotIn('continuation_comparison', baseline)
            self.assertNotIn('continuation_candidate', baseline)
            permissions.assert_called_once()
            for audio, progress in ((True, False), (False, True), (True, True)):
                candidate_dir = self.root/f'candidate-{audio}-{progress}'
                launcher.prepare(candidate_dir, self.root/'template', self.ledger,
                                 SimpleNamespace(), window_audio_context=audio,
                                 defer_progress_receipts=progress)
                candidate_stage = candidate_dir/'deployment'
                candidate_spec = json.loads((candidate_stage/'spec.json').read_text())
                self.assertEqual(candidate_spec['continuation_candidate'],
                                 launcher.continuation_candidate.make_options(
                                     window_audio_context=audio, defer_progress_receipts=progress))
                self.assertNotIn('decode_parity_diagnostic', candidate_spec)
                self.assertNotIn('continuation_comparison', candidate_spec)
                helper = 'research/h3_personal/continuation_candidate.py'
                self.assertIn(helper, launcher.EXTRA)
                self.assertEqual((candidate_stage/helper).read_bytes(), (source/helper).read_bytes())
                self.assertEqual(json.loads((candidate_stage/'upload-allowlist.json').read_text()),
                                 launcher.inventory(candidate_stage))
            launch.assert_not_called()
        stage = destination/'deployment'
        spec = json.loads((stage/'spec.json').read_text())
        self.assertEqual(spec['continuation_diagnostic'], launcher.contract(1800))
        self.assertEqual(spec['locked_speech'], launcher.locked_speech())
        self.assertTrue(spec['continuation_comparison']['same_warm_runtime'])
        for name in launcher.EXTRA:
            self.assertEqual((stage/name).read_bytes(), (source/name).read_bytes())
        self.assertEqual(json.loads((stage/'upload-allowlist.json').read_text()), launcher.inventory(stage))

    def test_fresh_budget_no_legacy_cap(self):
        original = self.ledger.read_bytes()
        bound = launcher.budget_for(self.ledger)
        result = bound(ledger_bound=Decimal(250), timeout_seconds=3600)
        self.assertEqual(result['authorized_cap'], Decimal(275))
        self.assertEqual(result['job_bound'], Decimal('10.00002'))
        with self.assertRaises(ValueError):
            bound(ledger_bound=Decimal(274), timeout_seconds=1800)
        with self.assertRaises(ValueError):
            bound(ledger_bound=Decimal(250), timeout_seconds=1800, hourly_rate=Decimal(11))
        self.assertEqual(self.ledger.read_bytes(), original)
        self.ledger.write_text('{"spent_usd":250,"reservations":{}}')
        with self.assertRaises(KeyError):
            bound(ledger_bound=Decimal(250), timeout_seconds=1800)


if __name__ == '__main__':
    unittest.main()
