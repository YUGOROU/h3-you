"""Offline strict direct-Job timeout and launcher forwarding regressions."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from research.h3_personal.direct_job_api import submit_job


class DirectAPIEnduranceTests(unittest.TestCase):
    def payload(self, timeout):
        return dict(flavor='h200x2', timeoutSeconds=timeout, attempts=1,
                    expose={'ports': [7860], 'portsPublic': []})

    def client(self):
        return Mock(post=Mock(return_value=SimpleNamespace(
            status_code=201, json=lambda: {'id': 'a' * 24})))

    def test_strict_modes_single_post_and_callback(self):
        for mode, timeout in ((False, 1200), (True, 300)):
            with self.subTest(mode=mode):
                client, callback = self.client(), Mock()
                payload = self.payload(timeout)
                job = submit_job(client, 'token', payload, on_created=callback,
                                 five_minute_test=mode)
                client.post.assert_called_once_with(
                    'https://huggingface.co/api/jobs/YUGOROU',
                    headers={'Authorization': 'Bearer token'}, json=payload,
                    follow_redirects=False)
                callback.assert_called_once_with(job)
                self.assertEqual(job.id, 'a' * 24)
        submit_job(self.client(), 'token', self.payload(1200))

    def test_invalid_modes_and_timeouts_never_post(self):
        for mode in (None, 0, 1, 'true', [], {}):
            client = self.client()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                submit_job(client, 'token', self.payload(300), five_minute_test=mode)
            client.post.assert_not_called()
        for mode in (False, True):
            expected = 300 if mode else 1200
            for timeout in (1200 if mode else 300, 299, 301, 600, True,
                            float(expected), str(expected), None):
                client = self.client()
                with self.subTest(mode=mode, timeout=timeout), self.assertRaises(ValueError):
                    submit_job(client, 'token', self.payload(timeout), five_minute_test=mode)
                client.post.assert_not_called()

    def test_existing_guards_never_post(self):
        for field, value in (('flavor', 'h200'), ('attempts', 2),
                             ('attempts', True), ('expose', {'ports': [7860]})):
            client = self.client()
            payload = self.payload(300)
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                submit_job(client, 'token', payload, five_minute_test=True)
            client.post.assert_not_called()
        for token in ('', 'bad token', None):
            client = self.client()
            with self.assertRaises(ValueError):
                submit_job(client, token, self.payload(300), five_minute_test=True)
            client.post.assert_not_called()

    def test_launcher_always_uses_default_strict_1200(self):
        # Execute the actual launcher call expression without its cloud setup.
        path = Path(__file__).resolve().parents[1] / 'research/launch_h3_direct_conversation.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'submit')
        call = next(n for n in ast.walk(function) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'submit_job')
        code = compile(ast.Expression(call), str(path), 'eval')
        for flag in (True, False, None, 1, 'true'):
            client, callback = self.client(), Mock()
            payload = self.payload(1200)
            forwarding = Mock(wraps=submit_job)
            eval(code, dict(submit_job=forwarding, control=client, token='token',
                            payload=payload, on_created=callback,
                            spec={'live_direct_session': {'five_minute_test': flag}},
                            # Legacy main-account aliases: no compute profile.
                            job_token='token', job_namespace='YUGOROU', compute_profile=None))
            self.assertEqual(forwarding.call_args.kwargs,
                             dict(on_created=callback))
            client.post.assert_called_once()
            callback.assert_called_once()
        forwarding = Mock()
        eval(code, dict(submit_job=forwarding, control=None, token='token',
                        payload={}, on_created=None, spec={},
                        job_token='token', job_namespace='YUGOROU', compute_profile=None))
        self.assertEqual(forwarding.call_args.kwargs, {'on_created': None})


if __name__ == '__main__':
    unittest.main()
