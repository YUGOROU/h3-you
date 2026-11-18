"""H3 backup Job billing split: explicit, offline, no cloud or credentials.

The split moves ONLY Job create/inspect/log/cancel/attest/gateway and the
consumer token to the compute account. Asset staging, payload secrets, Pi and
Endpoints stay on the main account, with no fallback between them.
"""
import inspect
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from research import launch_h3_direct_conversation as L

MAIN = 'hf_MainFixtureSecret'
BACKUP = 'hf_BackupFixtureSecret'


def profile(namespace='2nd2nd', token=BACKUP, endpoint='https://huggingface.co',
            whoami='2nd2nd'):
    api = Mock(endpoint=endpoint, token=token)
    api.whoami.return_value = {'name': whoami}
    return NS(api=api, token=token, namespace=namespace,
              assert_idle=Mock(), metadata=lambda: {'namespace': namespace})


class BackupBindingTests(unittest.TestCase):
    def test_accepts_exact_distinct_canonical_backup(self):
        p = profile()
        self.assertIs(L._backup_compute_binding(p, Mock(), MAIN), p)

    def test_rejects_wrong_namespace(self):
        for namespace in ('YUGOROU', 'other', None, '2nd2nd '):
            with self.subTest(namespace=namespace):
                with self.assertRaises(ValueError):
                    L._backup_compute_binding(profile(namespace=namespace), Mock(), MAIN)

    def test_rejects_same_token_as_main(self):
        with self.assertRaises(ValueError):
            L._backup_compute_binding(profile(token=MAIN), Mock(), MAIN)

    def test_rejects_non_canonical_endpoint(self):
        for endpoint in ('http://huggingface.co', 'https://example.invalid', None):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    L._backup_compute_binding(profile(endpoint=endpoint), Mock(), MAIN)

    def test_rejects_owner_mismatch(self):
        with self.assertRaises(ValueError):
            L._backup_compute_binding(profile(whoami='YUGOROU'), Mock(), MAIN)

    def test_rejects_none_and_malformed(self):
        for bad in (None, NS(namespace='2nd2nd')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    L._backup_compute_binding(bad, Mock(), MAIN)

    def test_rejects_token_that_is_not_the_api_token(self):
        p = profile()
        p.api.token = 'different'
        with self.assertRaises(ValueError):
            L._backup_compute_binding(p, Mock(), MAIN)

    def test_rejects_non_callable_assert_idle(self):
        p = profile()
        p.assert_idle = None
        with self.assertRaises(ValueError):
            L._backup_compute_binding(p, Mock(), MAIN)

    def test_no_secret_in_binding_error(self):
        try:
            L._backup_compute_binding(profile(namespace='YUGOROU'), Mock(), MAIN)
        except ValueError as error:
            self.assertNotIn(BACKUP, str(error))
            self.assertNotIn(MAIN, str(error))


class RunValidationTests(unittest.TestCase):
    """run() rejects unsupported backup scopes before any resource is reserved."""

    def _run(self, **kw):
        base = dict(directory='/unused', turns=['a', 'b'])
        base.update(kw)
        return L.run(**base)

    def test_backup_requires_stable_selfie(self):
        with self.assertRaises(ValueError):
            self._run(h3_backup_jobs=True, stable_selfie=False)

    def test_backup_rejects_non_boolean(self):
        for value in (1, 'yes', None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self._run(h3_backup_jobs=value)

    def test_backup_rejects_attended_visual(self):
        with self.assertRaises(ValueError):
            self._run(h3_backup_jobs=True, stable_selfie=True, attended_visual=True)

    def test_backup_rejects_headless_interrupt(self):
        with self.assertRaises(ValueError):
            self._run(h3_backup_jobs=True, stable_selfie=True, headless_interrupt=True)

    def test_backup_rejects_rtc(self):
        with self.assertRaises(ValueError):
            self._run(h3_backup_jobs=True, stable_selfie=True,
                      media_transport='webrtc-rtp-v1')

    def test_candidate_progress_timeout_includes_device_free_stable(self):
        import ast
        tree = ast.parse(inspect.getsource(L.run))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'candidate_progress' for t in n.targets))
        code = compile(ast.Expression(assignment.value), '<candidate-progress>', 'eval')
        for stable, headless, attended, references, expected in (
                (True, False, False, {}, True), (False, True, False, {}, True),
                (True, False, True, {}, False), (True, False, False, None, False),
                (False, False, False, {}, False)):
            value = eval(code, dict(continuous_av=True, stable_selfie=stable,
                headless_interrupt=headless, attended_visual=attended, ordered_references=references))
            self.assertEqual(value, dict(http_timeout=120, stream_progress_timeout=120) if expected else {})

    def test_supported_backup_route_reaches_reference_validation(self):
        # Exercise the supported path, not only short-circuit rejection paths.
        # Stop at the next local gate: no credentials or provider calls.
        with patch.object(L, 'ordered_selection', side_effect=ValueError('local-reference-sentinel')) as selection:
            with self.assertRaisesRegex(ValueError, 'local-reference-sentinel'):
                self._run(h3_backup_jobs=True, stable_selfie=True,
                          visual_intents=True, conversation_anchor=True,
                          tts_mode='native_voxcpm', ordered_references={})
        selection.assert_called_once()

    def test_profile_without_backup_flag_rejected(self):
        with self.assertRaises(ValueError):
            self._run(_compute_profile=profile())


class SubmitFailureTerminalTests(unittest.TestCase):
    def test_confirmation_retries_unknown_and_nonterminal_without_fallback(self):
        import tempfile
        import json
        from pathlib import Path
        api = Mock()
        api.inspect_job.side_effect = [OSError('temporary'), NS(status=NS(stage='RUNNING')),
                                       NS(status=NS(stage='CANCELED'))]
        with tempfile.TemporaryDirectory() as directory, patch.object(L.time, 'sleep') as sleep:
            L.confirm_submit_failure_terminal(api, NS(id='owned'), directory, job_namespace='2nd2nd')
            row = json.loads((Path(directory)/'job-terminal.json').read_text())
            self.assertTrue(row['independent_confirmation'])
            self.assertEqual(row['job_namespace'], '2nd2nd')
            self.assertEqual(sleep.call_count, 2)
        self.assertEqual(api.inspect_job.call_count, 3)
        for call in api.inspect_job.call_args_list:
            self.assertEqual(call.kwargs, dict(job_id='owned', namespace='2nd2nd'))

    def test_unconfirmed_terminal_is_bounded_and_has_no_success_receipt(self):
        import tempfile
        from pathlib import Path
        api = Mock()
        api.inspect_job.return_value = NS(status=NS(stage='RUNNING'))
        with tempfile.TemporaryDirectory() as directory, patch.object(L.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'unconfirmed'):
                L.confirm_submit_failure_terminal(api, NS(id='owned'), directory, job_namespace='2nd2nd')
            self.assertFalse((Path(directory)/'job-terminal.json').exists())
            self.assertEqual(api.inspect_job.call_count, 45)
            self.assertEqual(sleep.call_count, 44)


class CliSurfaceTests(unittest.TestCase):
    def test_flag_exists_and_is_opt_in(self):
        source = inspect.getsource(L.main)
        self.assertIn('--h3-backup-jobs', source)
        self.assertIn("action='store_true'", source)

    def test_cli_rejects_backup_outside_device_free_stable(self):
        with patch('builtins.print'), self.assertRaises(SystemExit):
            L.main(['--directory', '/unused', '--h3-backup-jobs'])

    def test_default_route_unchanged(self):
        """Without the flag, run() must not resolve any compute profile."""
        source = inspect.getsource(L.run)
        self.assertIn('if h3_backup_jobs:', source)
        # The profile is only loaded behind the explicit flag.
        self.assertIn('load_compute_profile(api, token, backup=True)', source)


class NamespacePropagationTests(unittest.TestCase):
    """Compute-side calls must carry the explicit namespace, never main."""

    def test_await_ready_rejects_unknown_namespace(self):
        with self.assertRaises(ValueError):
            L.await_ready(Mock(), Mock(), 'session', 0, job_namespace='other')

    def test_collector_rejects_unknown_namespace(self):
        with self.assertRaises(ValueError):
            L.collect_native_worker_receipts(Mock(), Mock(), '/unused', 0,
                                             job_namespace='other')

    def test_cleanup_binds_namespace(self):
        source = inspect.getsource(L.run)
        cleanup = source[source.index('def cleanup('):]
        self.assertIn("ns = {'namespace':job_namespace}", cleanup)
        self.assertIn('job_api.inspect_job(job_id=job.id, **ns)', cleanup)
        self.assertIn('job_api.cancel_job(job_id=job.id, **ns)', cleanup)

    def test_backup_always_confirms_terminal(self):
        source = inspect.getsource(L.run)
        self.assertIn('if rtc or headless_interrupt or compute_profile is not None:', source)

    def test_consumer_uses_job_token_not_main(self):
        source = inspect.getsource(L.run)
        self.assertIn('hf_token=job_token', source)

    def test_attest_uses_job_api(self):
        source = inspect.getsource(L.run)
        self.assertIn('attest_job(job_api,', source)

    def test_payload_secret_stays_main(self):
        source = inspect.getsource(L.submit)
        # The asset token is staged into payload secrets, never the compute token.
        self.assertIn("native_job_secrets(spec)", source)


if __name__ == '__main__':
    unittest.main()
