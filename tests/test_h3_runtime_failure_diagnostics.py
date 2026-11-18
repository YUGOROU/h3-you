"""Only controlled runtime check labels may reach public job logs."""
import inspect
import unittest
from research import prepare_identity_runtime as prep


class RuntimeFailureDiagnosticsTests(unittest.TestCase):
    def test_python_steps_are_distinct(self):
        self.assertEqual(prep.command_label(['python', '-m', 'venv', '/private']), 'venv creation')
        self.assertEqual(prep.command_label(['python', '-m', 'pip', 'check']), 'pip dependency check')
        self.assertEqual(prep.command_label(['python', '-m', 'pip', 'install', '--require-hashes']), 'pinned wheel installation')
        self.assertEqual(prep.command_label(['python', '-m', 'pip', 'install', '/private']), 'Diffusers source installation')

    def test_dependency_messages_are_allowlisted(self):
        output = 'foo 1.0 requires bar, which is not installed.\nsecret=https://user:pass@example.com\nfoo 1.0 has requirement bar>=2, but you have bar 1.0.'
        lines = prep.safe_dependency_failures(output)
        self.assertEqual(len(lines), 2)
        self.assertNotIn('secret', str(lines))

    def test_import_failure_reports_only_name_and_location(self):
        try:
            raise ModuleNotFoundError('secret-token-example', name='missing_package')
        except ModuleNotFoundError as exc:
            result = prep.failure_summary(exc)
        self.assertEqual(result['import_name'], 'missing_package')
        self.assertTrue(result['locations'])
        self.assertNotIn('secret-token-example', str(result))

    def test_encoder_static_check_is_visible(self):
        self.assertEqual(prep.failure_summary(ValueError('runtime module mismatch'))['check'], 'runtime module mismatch')

    def test_version_mismatches_include_all_packages(self):
        actual = {'foo': '1.0', 'bar': '2.0'}
        result = prep.version_mismatches({'foo': '2.0', 'bar': '3.0'}, actual.__getitem__)
        self.assertEqual([r['package'] for r in result], ['foo', 'bar'])
        self.assertEqual(result[0]['actual'], '1.0')

    def test_untrusted_metadata_is_hidden(self):
        result = prep.version_mismatches({'foo': '1.0'}, lambda _: 'https://user:secret@example.com')
        self.assertEqual(result[0]['actual'], '<invalid>')

    def test_bootstrap_replaces_venv_seed_metadata(self):
        source = inspect.getsource(prep.bootstrap)
        self.assertIn("'--require-hashes', '--upgrade'", source)
        self.assertNotIn("'--ignore-installed'", source)
        self.assertIn("'--no-index', '--no-deps'", source)

    def test_static_check_is_visible(self):
        with self.assertRaises(prep.RuntimeCheckError) as caught:
            prep.require(False, 'base torch version mismatch')
        self.assertEqual(prep.failure_summary(caught.exception)['check'], 'base torch version mismatch')

    def test_arbitrary_exception_detail_is_hidden(self):
        result = prep.failure_summary(ValueError('secret-token-example'))
        self.assertNotIn('secret-token-example', str(result))
        self.assertNotIn('check', result)
