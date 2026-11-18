"""CPU-only contracts; no SDK network, credentials or allocation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research import modal_h3_live_function as m


class FunctionTests(unittest.TestCase):
    def test_prices(self):
        for region, reserve, remaining in [('ap', '7.48', '11.96'), ('jp', '11.32', '8.12')]:
            p = m.plan(region=region)
            self.assertEqual(p['reservation_usd'], reserve)
            self.assertEqual(p['remaining_after_usd_assumes_old_spent1'], remaining)
            self.assertLess(float(p['calculated_bound_usd']), float(reserve))
        self.assertRaises(ValueError, m.plan, region='us')

    def test_cached_image_bounds(self):
        for region, timeout, bound, reserve in [
            ('ap', 1800, '7.47282112', '7.48'),
            ('jp', 1800, '11.3195104', '11.32'),
            ('jp', 900, '6.3697264', '6.37'),
            ('jp', 720, '5.3797696', '5.38'),
            ('jp', 600, '4.7197984', '4.72'),
        ]:
            p = m.plan(region=region, timeout_seconds=timeout)
            self.assertEqual(m.Decimal(p['calculated_bound_usd']), m.Decimal(bound))
            self.assertEqual(p['reservation_usd'], reserve)
            self.assertEqual(p['runtime_image_id'], 'im-EHjNujv578xET3b6yvFTS2')
            self.assertEqual(p['image'], m.IMAGE)
            self.assertEqual(m.Decimal(p['build_margin_usd']), m.Decimal('.10'))

    def test_ram_rate_and_720_budget_contract(self):
        d = m.Decimal
        base = 2 * d('4.5396') + 4 * d('.04716') + 256 * d('.007992')
        self.assertEqual(base, d('11.313792'))
        self.assertEqual(m.BASE_RATE, base)
        for region in ('ap', 'jp', 'au'):
            rate = base * d('1.15' if region == 'ap' else '1.75')
            for timeout in (600, 720, 900, 1200, 1500, 1800):
                p = m.plan(region=region, timeout_seconds=timeout)
                self.assertEqual(p['memory'], [262144, 262144])
                self.assertEqual(p['cpu'], [4, 4])
                self.assertEqual(p['gpu'], 'H200:2')
                self.assertEqual(p['image'], m.IMAGE)
                self.assertEqual(p['runtime_image_id'], 'im-EHjNujv578xET3b6yvFTS2')
                self.assertEqual(d(p['rate_usd_hour']), rate)
                bound = (timeout + 180 + 60) * rate / 3600 + d('.10')
                self.assertEqual(d(p['calculated_bound_usd']), bound)
                self.assertEqual(d(p['reservation_usd']), bound.quantize(d('.01'), rounding=m.ROUND_CEILING))
        p = m.plan(region='jp', timeout_seconds=720)
        self.assertEqual(d(p['rate_usd_hour']), d('19.799136'))
        self.assertEqual(d(p['calculated_bound_usd']), d('5.3797696'))
        self.assertEqual(p['reservation_usd'], '5.38')
        self.assertEqual(m.plan(region='jp', timeout_seconds=1200)['reservation_usd'], '8.02')

    def test_queue_admission_fixed_deadline(self):
        namespace = {}
        exec(m.worker_source().replace('MIN_USEFUL_SECONDS = 900',
                                     'MIN_USEFUL_SECONDS = 540'), namespace)
        # Created at 1000, root timeout 600, queue allowance 180: deadline 1780.
        # At 170s queue delay the guard admits; after 240s useful time is lost.
        deadline = 1000 + 600 + m.STARTUP
        with patch.dict(os.environ, {'MODAL_REGION': 'jp',
                        'MODAL_CLOUD_PROVIDER': 'CLOUD_PROVIDER_GCP'}, clear=True), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'H200\nH200\n')), \
             patch('os.path.isfile', return_value=False), \
             patch('subprocess.Popen', side_effect=AssertionError('no model')):
            with patch('time.time', return_value=1170):
                with self.assertRaisesRegex(RuntimeError, 'pinned_existing_image_python_unavailable'):
                    namespace['worker']('', deadline)
            os.environ.pop('H3_SESSION_NOT_AFTER', None)
            with patch('time.time', return_value=1241):
                self.assertEqual(namespace['worker']('', deadline)['classification'], 'expired')
                self.assertNotIn('H3_SESSION_NOT_AFTER', os.environ)
            with patch('time.time', return_value=1781), \
                 patch('subprocess.run', side_effect=AssertionError('past waiting TTL')):
                self.assertEqual(namespace['worker']('', deadline)['classification'], 'expired')

    def test_profiles_and_expiration(self):
        p = m.plan(region='au', timeout_seconds=1200)
        self.assertEqual(p['reservation_usd'], '8.02')
        self.assertEqual(p['rate_usd_hour'], str(m.BASE_RATE * m.Decimal('1.75')))
        self.assertEqual(m.plan(region='jp',timeout_seconds=720)['reservation_usd'],'5.38')
        for timeout in (600, 720, 900, 1200, 1500, 1800):
            self.assertEqual(m.plan(timeout_seconds=timeout)['timeout'], timeout)
        for timeout in (True, 899, 1801, 1200.0):
            self.assertRaises(ValueError, m.plan, timeout_seconds=timeout)
        namespace = {}
        exec(m.worker_source(), namespace)
        with patch('time.time', return_value=1000), \
             patch('subprocess.run', side_effect=AssertionError('no gate or models')), \
             patch('subprocess.Popen', side_effect=AssertionError('no child')):
            self.assertEqual(namespace['worker']('', 999)['classification'], 'expired')
        with patch.dict(os.environ, {'MODAL_REGION': 'au', 'MODAL_CLOUD_PROVIDER': 'CLOUD_PROVIDER_GCP'}, clear=True), \
             patch('time.time', return_value=1000), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'H200\nH200\n')) as probe, \
             patch('subprocess.Popen', side_effect=AssertionError('no child')):
            self.assertEqual(namespace['worker']('', 1500)['classification'], 'expired')
            self.assertEqual(probe.call_count, 1)
            self.assertNotIn('H3_SESSION_NOT_AFTER', os.environ)

    def test_gate_before_secret_or_bootstrap(self):
        namespace = {}
        exec(m.worker_source(), namespace)
        for region, cloud, decision in [('ap', 'CLOUD_PROVIDER_AWS', 'unknown'),
                                        ('us', 'CLOUD_PROVIDER_AWS', 'excluded'),
                                        ('jp', 'AWS', 'unknown')]:
            with patch.dict(os.environ, {'MODAL_REGION': region, 'MODAL_CLOUD_PROVIDER': cloud}, clear=True), \
                 patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'NVIDIA H200\nNVIDIA H200\n')), \
                 patch('subprocess.Popen', side_effect=AssertionError('must not start child')):
                row = namespace['worker']("raise AssertionError('no downloads')", m.time.time() + 1800)
                self.assertEqual(row['placement']['classification']['decision'], decision)
                self.assertEqual(row['classification'], 'gate_denied')

    def test_jp_contract_gate(self):
        namespace = {}
        exec(m.worker_source(), namespace)
        cases = [
            ('ap-northeast', 'CLOUD_PROVIDER_AWS', 'jp', True, True),
            ('ap', 'CLOUD_PROVIDER_GCP', 'jp', True, True),
            ('jp', 'CLOUD_PROVIDER_OCI', 'jp', True, True),
            ('ap-northeast', 'CLOUD_PROVIDER_AWS', 'jp', False, False),
            ('ap-northeast', 'CLOUD_PROVIDER_AWS', 'jp', 1, False),
            ('us', 'CLOUD_PROVIDER_AWS', 'jp', True, False),
            ('eu', 'CLOUD_PROVIDER_AWS', 'jp', True, False),
            ('uk', 'CLOUD_PROVIDER_AWS', 'jp', True, False),
            ('kr', 'CLOUD_PROVIDER_AWS', 'jp', True, False),
            ('ap-northeast', 'unknown', 'jp', True, False),
            ('ap-northeast', 'CLOUD_PROVIDER_AWS', 'ap', True, False),
            ('ap-northeast', 'CLOUD_PROVIDER_AWS', 'au', True, False),
        ]
        for actual, cloud, requested, flag, accepted in cases:
            with self.subTest(actual=actual, cloud=cloud, requested=requested, flag=flag), \
                 patch.dict(os.environ, {'MODAL_REGION': actual, 'MODAL_CLOUD_PROVIDER': cloud}, clear=True), \
                 patch('time.time', return_value=1000), \
                 patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'H200\nH200\n')) as probe, \
                 patch('subprocess.Popen', side_effect=AssertionError('no model child')):
                row = namespace['worker']('raise AssertionError()', 1500, requested, flag)
                self.assertEqual(row['classification'], 'expired' if accepted else 'gate_denied')
                self.assertEqual(probe.call_count, 1)
                self.assertEqual(row['placement']['requested_region'], requested)
                if accepted:
                    decision = row['placement']['classification']
                    self.assertEqual(decision['country'], 'JP')
                    self.assertEqual(decision['evidence'], 'operator_accepted_modal_jp_contract')
                    self.assertFalse(decision['runtime_country_verified'])
                self.assertNotIn('H3_SESSION_NOT_AFTER', os.environ)

    def test_600_cent_reservation_with_closed_receipts(self):
        self.assertEqual(m.plan(region='jp', timeout_seconds=600)['calculated_bound_usd'], '4.71979840')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'budget.json'
            path.write_text(json.dumps(dict(schema='h3-ref2va-modal-budget-v1',
                cap_usd='20.44', spent_bound_usd='14.5400124', reservations={},
                closed_receipts=[{'upper_bound_usd': '100'}])))
            ledger = m.Ledger(path, 'jp', 600)
            key = ledger.reserve()
            self.assertEqual(json.loads(path.read_text())['reservations'][key]['upper_bound_usd'], '4.72')
            self.assertEqual(m.CAP - m.Decimal('14.5400124') - ledger.reservation, m.Decimal('1.1799876'))

    def test_start_rejects_contract_region_mismatch_before_allocation(self):
        for cli, spec_region in [('ap', 'jp'), ('jp', 'au')]:
            inspector = lambda *args: ({'config': {'timeout_seconds': 600},
                'modal_live_approval': {'region': spec_region, 'trust_explicit_jp_contract': True}},)
            with patch('pathlib.Path.read_text', side_effect=AssertionError('no credentials')):
                self.assertRaises(ValueError, m.start, '/stage', '/output', '/budget',
                                  'a/b', 'a' * 40, 'b' * 64, cli, inspector)

    def test_gpu_denial_and_immediate_placement(self):
        namespace = {}
        exec(m.worker_source(), namespace)
        with patch.dict(os.environ, {'MODAL_REGION': 'jp', 'MODAL_CLOUD_PROVIDER': 'CLOUD_PROVIDER_GCP'}, clear=True), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'NVIDIA H100\n')), \
             patch('subprocess.Popen', side_effect=AssertionError('no child')):
            row = namespace['worker']('', m.time.time() + 1800)
            self.assertFalse(row['placement']['gpu_valid'])
            self.assertEqual(row['classification'], 'gate_denied')

    def test_serialization_and_options(self):
        class SDK:
            class Image:
                @staticmethod
                def from_id(image):
                    if image != 'im-EHjNujv578xET3b6yvFTS2':
                        raise AssertionError('only the existing pinned runtime is allowed')
                    return image
            class App:
                def __init__(self, name):
                    pass
                def function(self, **options):
                    self.options = options
                    return lambda f: f
        with patch('time.time', return_value=1000):
            app, f = m.build_function(object(), 'a/b', 'a' * 40, 'b' * 64, sdk=SDK)
        self.assertEqual(app.options['cpu'], (4, 4))
        self.assertEqual(app.options['memory'], (262144, 262144))
        for key, value in dict(gpu='H200:2', region='ap', timeout=1800,
                               startup_timeout=180, retries=0, max_containers=1,
                               serialized=True, include_source=False,
                               single_use_containers=True, scaledown_window=2).items():
            self.assertEqual(app.options[key], value)
        self.assertEqual(set(f.__code__.co_freevars), {'source', 'bootstrap', 'not_after_epoch', 'requested_region', 'trust_jp_contract'})
        cells = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
        self.assertEqual(cells['not_after_epoch'], 2980)
        self.assertEqual(cells['requested_region'], 'ap')
        self.assertIs(cells['trust_jp_contract'], False)
        app600, f600 = m.build_function(object(), 'a/b', 'a' * 40, 'b' * 64,
            region='jp', sdk=SDK, timeout_seconds=600, trust_jp_contract=True)
        cells600 = dict(zip(f600.__code__.co_freevars, (c.cell_contents for c in f600.__closure__)))
        self.assertEqual(cells600['requested_region'], 'jp')
        self.assertIs(cells600['trust_jp_contract'], True)
        self.assertIn('MIN_USEFUL_SECONDS = 540', cells600['source'])
        self.assertEqual(app600.options['timeout'], 600)
        self.assertRaises(ValueError, m.build_function, object(), 'a/b', 'a' * 40,
                          'b' * 64, sdk=SDK, trust_jp_contract=True)
        import inspect
        self.assertFalse(inspect.isgeneratorfunction(f))
        self.assertNotIn('is_generator', app.options)
        compile(m.bootstrap_script('a/b', 'a' * 40, 'b' * 64), '<bootstrap>', 'exec')

    def test_ledger_and_unknown_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'budget.json'
            path.write_text(json.dumps(dict(schema='h3-ref2va-modal-budget-v1',
                cap_usd='20.44', spent_bound_usd='1', reservations={})))
            ledger = m.Ledger(path)
            key = ledger.reserve()
            self.assertRaises(ValueError, ledger.reserve)
            self.assertEqual(json.loads(path.read_text())['reservations'][key]['upper_bound_usd'], '7.48')
            charge = ledger.terminated(key, 10)
            self.assertLess(float(charge), 2)
            self.assertFalse(json.loads(path.read_text())['reservations'])

    def test_closed_historical_receipt_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'launch.json'
            state = dict(full_reservation_retained=False, charge_bound_usd='5.3797696',
                         pins=dict(memory=[262144, 262144], rate_usd_hour='19.799136'))
            path.write_text(json.dumps(state))
            original = path.read_bytes()
            with patch.object(m, 'Ledger', side_effect=AssertionError('no repricing')), \
                 patch.object(m, '_save', side_effect=AssertionError('no mutation')), \
                 patch.object(m.subprocess, 'run', side_effect=AssertionError('no API')):
                self.assertEqual(m.status(tmp), state)
                self.assertEqual(m.status(tmp, stopping=True), state)
            self.assertEqual(path.read_bytes(), original)

    def test_status_timeout_retains_reservation(self):
        import types
        from unittest.mock import Mock
        class Timeout(Exception):
            pass
        call = Mock()
        call.get.side_effect = Timeout()
        class RemoteTerminal(Exception):
            pass
        sdk = types.SimpleNamespace(FunctionCall=Mock(),
                                    exception=types.SimpleNamespace(TimeoutError=Timeout,FunctionTimeoutError=RemoteTerminal,ExecutionError=RemoteTerminal,UserCodeException=RemoteTerminal,RemoteError=RemoteTerminal))
        sdk.FunctionCall.from_id.return_value = call
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'launch.json'
            path.write_text(json.dumps(dict(call_id='fc-test', app_id='ap-test',
                full_reservation_retained=True)))
            with patch.dict('sys.modules', modal=sdk), patch.object(m, '_confirm_stop') as stopping:
                result = m.status(tmp)
            self.assertEqual(result['status'], 'RUNNING')
            self.assertTrue(result['full_reservation_retained'])
            call.get.assert_called_once_with(timeout=0)
            stopping.assert_not_called()

    def test_confirm_stop_requires_positive_evidence(self):
        with patch.object(m.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([], 0, ''),
            subprocess.CompletedProcess([], 0, '[]')]):
            self.assertFalse(m._confirm_stop('ap-test'))

    def test_dry_no_cache_or_sdk(self):
        with patch('pathlib.Path.read_text', side_effect=AssertionError('no file reads')):
            self.assertEqual(m.plan()['timeout'], 1800)
        self.assertNotIn('secret-value', m.sanitize({'x': 'secret-value hf_ABC123'}, 'secret-value'))
        self.assertNotIn('hf_ABC123', m.sanitize({'x': 'hf_ABC123'}, ''))


if __name__ == '__main__':
    unittest.main()
