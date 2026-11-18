"""No-network error boundary: only correlated allowlisted diagnostic data escapes."""
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research.run_h3_relay_conversation import PiDirector, DirectorFailure


class DiagnosticTests(unittest.TestCase):
    def director(self, row):
        director = PiDirector.__new__(PiDirector)
        director.child = SimpleNamespace(stdin=BytesIO())
        director._read = lambda: row
        return director

    def packet(self, code='assistant_error', status=None):
        return dict(type='error', turn_id='fixture-turn', code='director_failed',
                    diagnostic=dict(schema='h3-director-failure-v1', code=code, http_status=status))

    def invoke(self, row):
        with patch('research.run_h3_relay_conversation.uuid.uuid4',
                   return_value=SimpleNamespace(hex='fixture-turn')):
            return self.director(row).turn('PRIVATE_USER_INPUT')

    def test_allowlisted_diagnostic_preserved_without_raw_response(self):
        for code, status in [('provider_http_error', 503), ('tool_error', None),
                             ('missing_proposal', None), ('canceled', None),
                             ('assistant_error', None), ('aborted', None),
                             ('runtime_error', None), ('contract_error', None),
                             ('provider_network_error', None)]:
            with self.subTest(code=code), self.assertRaises(DirectorFailure) as caught:
                self.invoke(self.packet(code, status))
            self.assertEqual(caught.exception.diagnostic, self.packet(code, status)['diagnostic'])
            self.assertNotIn('PRIVATE_USER_INPUT', str(caught.exception))
            self.assertFalse(hasattr(caught.exception, 'response'))

    def test_unknown_extra_uncorrelated_or_invalid_status_fails_closed(self):
        rows = [self.packet('PRIVATE_RESPONSE'), self.packet('provider_http_error'),
                self.packet('provider_network_error', 503),
                self.packet(status=True), self.packet(status=200), self.packet(status=600),
                self.packet(status='PRIVATE_TOKEN'), dict(self.packet(), turn_id='wrong'),
                dict(self.packet(), token='PRIVATE_TOKEN'),
                dict(self.packet(), diagnostic=dict(self.packet()['diagnostic'], raw='PRIVATE_PROMPT')),
                dict(self.packet(), diagnostic=['PRIVATE_RESPONSE']),
                dict(type='error', code='director_failed', turn_id='fixture-turn')]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(RuntimeError) as caught:
                self.invoke(row)
            self.assertNotIsInstance(caught.exception, DirectorFailure)
            self.assertEqual(str(caught.exception), 'director failed; no fallback')
            self.assertNotIn('PRIVATE', str(caught.exception))

    def test_mismatched_canceled_diagnostic_and_nonobject_rejected(self):
        for row in [dict(self.packet('runtime_error'), code='canceled'), ['PRIVATE_RESPONSE'], None]:
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                self.invoke(row)


if __name__ == '__main__':
    unittest.main()
