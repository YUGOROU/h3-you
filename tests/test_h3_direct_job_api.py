"""Offline contracts for the single-attempt direct Jobs REST adapter."""
import unittest
from unittest.mock import Mock

from research.h3_personal.direct_job_api import submit_job


JOB_ID = '0123456789abcdef01234567'


def payload():
    return dict(flavor='h200x2', timeoutSeconds=1200, attempts=1,
                expose={'ports': [7860], 'portsPublic': []})


class DirectJobAPITests(unittest.TestCase):
    def test_success_posts_once_without_redirects(self):
        for status in (200, 201):
            with self.subTest(status=status):
                client = Mock()
                client.post.return_value.status_code = status
                client.post.return_value.json.return_value = {'id': JOB_ID}
                body = payload()
                self.assertEqual(submit_job(client, 'test_token', body).id, JOB_ID)
                client.post.assert_called_once_with(
                    'https://huggingface.co/api/jobs/YUGOROU',
                    headers={'Authorization': 'Bearer test_token'},
                    json=body, follow_redirects=False)

    def test_rejection_private_detail_redacts_secrets_and_not_in_exception(self):
        from research.h3_personal.direct_job_api import JobCreationRejected
        client=Mock();client.post.return_value.status_code=400
        client.post.return_value.text='invalid network; test_token private-key hf_sensitive'
        body=payload();body['secrets']={'KEY':'private-key'}
        with self.assertRaises(JobCreationRejected) as caught:submit_job(client,'test_token',body)
        self.assertEqual(caught.exception.status,400)
        self.assertIn('invalid network',caught.exception.detail)
        for secret in ('test_token','private-key','hf_sensitive'):
            self.assertNotIn(secret,caught.exception.detail)
        self.assertNotIn('invalid network',str(caught.exception))
        client.post.assert_called_once()

    def test_created_callback_receives_returned_job(self):
        client = Mock()
        client.post.return_value.status_code = 201
        client.post.return_value.json.return_value = {'id': JOB_ID}
        callback = Mock()
        job = submit_job(client, 'test_token', payload(), on_created=callback)
        callback.assert_called_once_with(job)

    def test_callback_interruption_propagates_without_retry(self):
        for error in (KeyboardInterrupt(), RuntimeError('receipt failed')):
            with self.subTest(error=type(error).__name__):
                client = Mock()
                client.post.return_value.status_code = 201
                client.post.return_value.json.return_value = {'id': JOB_ID}
                callback = Mock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    submit_job(client, 'test_token', payload(), on_created=callback)
                self.assertIs(caught.exception, error)
                self.assertEqual(callback.call_args.args[0].id, JOB_ID)
                callback.assert_called_once()
                client.post.assert_called_once()

    def test_invalid_payloads_never_post(self):
        cases = [None, [], {}]
        for key, values in {
            'flavor': [None, 'h200'],
            'timeoutSeconds': [None, 1200.0, True, 1199, 1201],
            'attempts': [None, True, 1.0, 0, 2],
            'expose': [None, {}, {'ports': [7860], 'portsPublic': [7860]},
                       {'ports': [80], 'portsPublic': []}],
        }.items():
            for value in values:
                cases.append(dict(payload(), **{key: value}))
        for body in cases:
            with self.subTest(body=body):
                client = Mock()
                with self.assertRaisesRegex(ValueError, 'finite authenticated'):
                    submit_job(client, 'test_token', body)
                client.post.assert_not_called()

    def test_invalid_tokens_never_post(self):
        for token in (None, b'token', '', 'a b', 'a\nb', 'a\tb', '\x00', '\x7f', 'é'):
            with self.subTest(token=repr(token)):
                client = Mock()
                with self.assertRaises(ValueError):
                    submit_job(client, token, payload())
                client.post.assert_not_called()

    def test_http_rejections_do_not_parse_or_retry(self):
        for status in (202, 301, 302, 307, 400, 401, 403, 429, 500, 503):
            with self.subTest(status=status):
                client = Mock()
                client.post.return_value.status_code = status
                with self.assertRaisesRegex(RuntimeError, 'creation rejected'):
                    submit_job(client, 'test_token', payload())
                client.post.assert_called_once()
                client.post.return_value.json.assert_not_called()

    def test_unknown_creation_outcome_never_retries(self):
        for value in (None, [], {}, {'id': None}, {'id': 'a' * 23},
                      {'id': 'A' * 24}, {'id': 'g' * 24}, {'id': JOB_ID + '\n'}):
            with self.subTest(value=value):
                client = Mock()
                client.post.return_value.status_code = 201
                client.post.return_value.json.return_value = value
                callback = Mock()
                with self.assertRaisesRegex(RuntimeError, 'inspect namespace before any resubmission'):
                    submit_job(client, 'test_token', payload(), on_created=callback)
                callback.assert_not_called()
                client.post.assert_called_once()

    def test_transport_and_json_failures_propagate_without_retry(self):
        for phase, error in (('post', TimeoutError('uncertain POST')),
                             ('json', ValueError('invalid JSON'))):
            with self.subTest(phase=phase):
                client = Mock()
                client.post.return_value.status_code = 201
                if phase == 'post':
                    client.post.side_effect = error
                else:
                    client.post.return_value.json.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    submit_job(client, 'test_token', payload())
                self.assertIs(caught.exception, error)
                client.post.assert_called_once()


if __name__ == '__main__':
    unittest.main()
