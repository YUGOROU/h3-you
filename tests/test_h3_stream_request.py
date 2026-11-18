"""CPU-only request contract tests; no launcher, devices or inference."""
import time
import unittest

from research.hf_h3_live_voice import validate_request


class StreamRequestTests(unittest.TestCase):
    def row(self, stream=False):
        row = dict(schema='h3-live-voice-input-v1', session_id='session',
                   sequence=1, origin='discord_received_audio_ondevice_asr',
                   asr_final=True, asr_stopped=True,
                   director_model='Qwen/Qwen3.8-27B:cerebras', reply='Hello',
                   expires_at=time.time() + 30)
        if stream:
            row.update(asr_final=False,
                       asr_endpoint_source='application_stable_partial',
                       asr_revision=1)
        return row

    def test_legacy_default(self):
        row = self.row()
        self.assertIs(validate_request(row, 'session', 1), row)
        self.assertIs(validate_request(row, 'session', 1, 'apple-final'), row)

    def test_opt_in_stream(self):
        row = self.row(True)
        self.assertIs(validate_request(row, 'session', 1,
                                      'application-stream-v1'), row)
        self.assertIs(row['asr_final'], False)

    def test_cross_policy_rejected(self):
        for row, policy in [(self.row(True), 'apple-final'),
                            (self.row(), 'application-stream-v1')]:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_request(row, 'session', 1, policy)
        row = self.row(True)
        row['asr_final'] = True
        for policy in ('apple-final', 'application-stream-v1'):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_request(row, 'session', 1, policy)

    def test_stream_invalid_fields(self):
        cases = [('asr_revision', value) for value in
                 (True, False, None, 0, -1, 1.0, '1')]
        cases += [('asr_stopped', False), ('asr_stopped', 1),
                  ('asr_endpoint_source', 'apple-final'),
                  ('asr_endpoint_source', None), ('asr_final', True),
                  ('asr_final', 0), ('origin', 'other')]
        for key, value in cases:
            row = self.row(True)
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request(row, 'session', 1, 'application-stream-v1')
        for key in ('asr_revision', 'asr_endpoint_source', 'asr_stopped', 'asr_final'):
            row = self.row(True)
            del row[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_request(row, 'session', 1, 'application-stream-v1')

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):
            validate_request(self.row(), 'session', 1, 'unknown')


if __name__ == '__main__':
    unittest.main()
