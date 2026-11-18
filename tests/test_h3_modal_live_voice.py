"""CPU-only carrier contracts. No Modal SDK, APIs, model imports or GPU builds."""
import contextlib
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from research import modal_h3_live_voice as carrier


class PlanTests(unittest.TestCase):
    def test_offline_plan(self):
        with patch.dict(sys.modules, {'modal': None, 'huggingface_hub': None}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(carrier.main([]), 0)
        p = json.loads(output.getvalue())
        self.assertTrue(p['executable'])
        self.assertEqual((p['region'], p['gpu'], p['max_containers'], p['retries']),
                         ('ap', 'H200:2', 1, 0))
        self.assertEqual(p['cpu'], [4, 4])
        self.assertEqual(p['memory_mib'], [262144, 262144])
        self.assertEqual(Decimal(p['rate_usd_hour']),
                         (Decimal('4.5396') * 2 + Decimal('.141912') * 4
                          + Decimal('.024012') * 256) * Decimal('1.15'))
        self.assertLessEqual(Decimal(p['calculated_bound_usd']), Decimal('6'))
        self.assertGreaterEqual(Decimal(p['remaining_after_usd']), Decimal('10'))
        self.assertFalse(p['detached'])
        self.assertFalse(p['watchers'])

    def test_run_is_explicitly_blocked_without_side_effects(self):
        with patch.object(carrier.Ledger, 'reserve') as reserve:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                carrier.main(['--run'])
            reserve.assert_not_called()

    def test_immutable_pins(self):
        p = carrier.plan('owner/private-live', 'a' * 40, 'b' * 64)
        self.assertEqual(p['revision'], 'a' * 40)
        for args in [('owner/private', 'main', 'b' * 64),
                     ('owner/private', 'a' * 40, None),
                     ('../private', 'a' * 40, 'b' * 64)]:
            with self.assertRaises(ValueError):
                carrier.plan(*args)


class GateTests(unittest.TestCase):
    def test_exact_documented_countries_only(self):
        for cloud in ('CLOUD_PROVIDER_AWS', 'CLOUD_PROVIDER_GCP', 'CLOUD_PROVIDER_OCI'):
            for region in ('jp', 'au', 'ca', 'mx'):
                self.assertEqual(carrier.placement_gate(dict(
                    MODAL_REGION=region, MODAL_CLOUD_PROVIDER=cloud))['decision'],
                    'allowed_candidate')
        for region in ('ap', 'ap-northeast', 'ap-southeast', 'ap-melbourne',
                       'kr', 'JP', 'jp-east', 'us', 'eu-west', '', 'japan'):
            with self.assertRaises(ValueError):
                carrier.placement_gate(dict(MODAL_REGION=region,
                    MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_AWS'))
        for cloud in ('', 'aws', None):
            with self.assertRaises(ValueError):
                carrier.placement_gate(dict(MODAL_REGION='jp', MODAL_CLOUD_PROVIDER=cloud))

    def test_script_checks_inside_runtime(self):
        script = carrier.metadata_script()
        self.assertNotIn('HF_TOKEN', script)
        self.assertNotIn('research', script)
        self.assertNotIn('torch', script)
        for region, code in [('jp', 0), ('ap-northeast', 78), ('', 78)]:
            proc = subprocess.run([sys.executable, '-c', script],
                env={'MODAL_REGION': region, 'MODAL_CLOUD_PROVIDER': 'CLOUD_PROVIDER_AWS'},
                capture_output=True, text=True, timeout=5)
            self.assertEqual(proc.returncode, code)
            self.assertEqual(json.loads(proc.stdout)['metadata']['MODAL_REGION'], region)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'modal-budget.json'
        self.initial = dict(schema='h3-ref2va-modal-budget-v1', cap_usd='20.44',
                            spent_bound_usd='1', reservations={})
        self.write(self.initial)
        self.ledger = carrier.Ledger(self.path)
        self.sandbox = Mock()
        self.sandbox.metadata.return_value = dict(MODAL_REGION='jp',
            MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_AWS')
        self.sandbox.terminate.return_value = True
        self.adapter = Mock()
        self.adapter.create.return_value = self.sandbox

    def write(self, data):
        self.path.write_text(json.dumps(data))

    def read(self):
        return json.loads(self.path.read_text())

    def test_allowed_same_instance_order_and_full_phase_charge(self):
        events = []
        self.sandbox.metadata.side_effect = lambda: (events.append('metadata') or
            dict(MODAL_REGION='au', MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_GCP'))
        def payload(sandbox):
            self.assertIs(sandbox, self.sandbox)
            self.assertTrue(self.read()['reservations'])
            events.append('payload')
            return 'resident-result'
        self.sandbox.terminate.side_effect = lambda: (events.append('terminate') or True)
        self.assertEqual(carrier.exercise_lifecycle(self.adapter, self.ledger, payload),
                         'resident-result')
        self.assertEqual(events, ['metadata', 'payload', 'terminate'])
        self.adapter.create.assert_called_once()
        self.assertEqual(self.read()['spent_bound_usd'], '7')
        self.assertEqual(self.read()['reservations'], {})

    def test_unknown_and_excluded_never_obtain_payload_or_credentials(self):
        for region in ('ap-northeast', 'kr', 'us', ''):
            self.write(self.initial)
            self.sandbox.metadata.return_value['MODAL_REGION'] = region
            payload = Mock(side_effect=AssertionError('no staging/token allowed'))
            with self.assertRaises(ValueError):
                carrier.exercise_lifecycle(self.adapter, self.ledger, payload)
            payload.assert_not_called()
            self.assertEqual(self.read()['spent_bound_usd'], '7')

    def test_ambiguous_creation_retains_lock_and_blocks_retry(self):
        self.adapter.create.side_effect = TimeoutError('unknown allocation')
        with self.assertRaises(TimeoutError):
            carrier.exercise_lifecycle(self.adapter, self.ledger, Mock())
        self.assertTrue(self.read()['reservations'])
        with self.assertRaises(ValueError):
            self.ledger.reserve()
        self.sandbox.terminate.assert_not_called()

    def test_ambiguous_termination_retains_reservation(self):
        self.sandbox.terminate.return_value = False
        carrier.exercise_lifecycle(self.adapter, self.ledger, Mock())
        self.assertTrue(self.read()['reservations'])
        self.assertEqual(self.read()['spent_bound_usd'], '1')

    def test_payload_failure_still_terminates(self):
        with self.assertRaises(RuntimeError):
            carrier.exercise_lifecycle(self.adapter, self.ledger,
                                      Mock(side_effect=RuntimeError('bootstrap failed')))
        self.sandbox.terminate.assert_called_once()
        self.assertEqual(self.read()['spent_bound_usd'], '7')

    def test_insufficient_or_wrong_ledger_never_allocates(self):
        for change in ({'spent_bound_usd': '5'}, {'spent_bound_usd': '0'},
                       {'spent_bound_usd': 'NaN'}, {'cap_usd': '192.35'},
                       {'reservations': {'old': {'upper_bound_usd': '1'}}}):
            self.write(dict(self.initial, **change))
            with self.assertRaises(ValueError):
                carrier.exercise_lifecycle(self.adapter, self.ledger, Mock())
        self.adapter.create.assert_not_called()

    def test_no_implicit_ledger_creation(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.ledger.reserve()
        self.assertFalse(self.path.exists())


class ProductionTests(LifecycleTests):
    def invoke(self, region='jp', names=None, terminal=0):
        from types import SimpleNamespace
        self.sdk = Mock()
        self.sdk.Sandbox.list.return_value = []
        self.sdk.App.return_value.run.return_value.__enter__ = Mock(return_value=Mock())
        self.sdk.App.return_value.run.return_value.__exit__ = Mock(return_value=False)
        sb = self.sdk.Sandbox.create.return_value
        sb.object_id = 'sb-private'
        sb.terminate.return_value = terminal
        meta = Mock()
        meta.stdout.read.return_value = json.dumps({'metadata': dict(
            MODAL_REGION=region, MODAL_CLOUD_PROVIDER='CLOUD_PROVIDER_AWS')})
        meta.wait.return_value = 0
        gpu = Mock()
        gpu.stdout.read.return_value = json.dumps(names if names is not None else ['NVIDIA H200'] * 2)
        gpu.wait.return_value = 0
        child = Mock()
        child.stdout = iter(['hf_FAKEcredential\n'])
        child.stderr = iter([])
        child.wait.return_value = 0
        sb.exec.side_effect = [meta, gpu, child]
        args = SimpleNamespace(stage=Path('unused'), deployment='owner/private',
            revision='a'*40, allowlist_sha256='b'*64, budget=self.path,
            output=Path(self.directory.name)/'output')
        with patch.object(carrier, 'validate_stage'), patch(
                'research.provision_modal_identity_secret.cached_token', return_value='hf_FAKEcredential') as token:
            self.token = token
            try:
                return carrier.run_production(args, self.sdk)
            finally:
                self.token_calls = token.call_count

    def test_production_success(self):
        result = self.invoke()
        self.assertEqual(result['state'], 'terminated')
        self.assertEqual(result['process_exitcode'], 0)
        self.assertEqual(self.token_calls, 1)
        kwargs = self.sdk.Sandbox.create.call_args.kwargs
        self.assertNotIn('secrets', kwargs)
        self.assertEqual(kwargs['image'], self.sdk.Image.from_registry.return_value)
        self.sdk.Image.from_registry.assert_called_once_with(carrier.IMAGE)
        self.sdk.Sandbox.create.return_value.terminate.assert_called_once_with(wait=True)
        self.assertNotIn('hf_FAKE', (Path(self.directory.name)/'output/stdout.log').read_text())
        self.assertLess(Decimal(result['charge_bound_usd']), Decimal('6'))

    def test_production_country_denied_no_secret(self):
        with self.assertRaises(RuntimeError):
            self.invoke(region='ap-northeast')
        self.assertEqual(self.token_calls, 0)
        self.sdk.Secret.from_dict.assert_not_called()
        self.assertEqual(self.sdk.Sandbox.create.return_value.exec.call_count, 1)
        self.assertFalse(self.read()['reservations'])

    def test_production_wrong_gpu_no_secret(self):
        with self.assertRaises(RuntimeError):
            self.invoke(names=['NVIDIA H100'] * 2)
        self.assertEqual(self.token_calls, 0)
        self.sdk.Secret.from_dict.assert_not_called()

    def test_production_unknown_termination_keeps_lock(self):
        self.invoke(terminal=None)
        self.assertTrue(self.read()['reservations'])

    def test_installed_sdk_local_signatures(self):
        import inspect
        try:
            import modal
        except ImportError:
            self.skipTest('run with installed Modal interpreter for SDK check')
        create = inspect.signature(modal.Sandbox.create).parameters
        for name in ('app', 'image', 'env', 'secrets', 'timeout', 'idle_timeout',
                     'gpu', 'region', 'cpu', 'memory', 'tags'):
            self.assertIn(name, create)
        self.assertIn('secrets', inspect.signature(modal.Sandbox.exec).parameters)
        self.assertIn('wait', inspect.signature(modal.Sandbox.terminate).parameters)
        self.assertIn('tags', inspect.signature(modal.Sandbox.list).parameters)

    def test_starter_compiles_and_has_no_token_literal(self):
        script = carrier.bootstrap_script('owner/private', 'a'*40, 'b'*64)
        compile(script, '<starter>', 'exec')
        self.assertIn('upload-allowlist.json', script)
        self.assertIn('/work/h3-modal-carrier-source', script)
        self.assertLess(script.index('hashlib.sha256(body)'), script.index("root.mkdir"))


if __name__ == '__main__':
    unittest.main()
