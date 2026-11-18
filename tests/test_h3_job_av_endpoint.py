"""CPU-only direct Job identity and anonymous gateway attestation tests."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.job_av_endpoint import JobEndpoint, attest_job


JOB_ID = '0123456789abcdef01234567'
IMAGE = 'registry.example/image@sha256:' + 'a' * 64
COMMAND = ['python', '-m', 'worker']
HOST = JOB_ID + '--7860.hf.jobs'
URL = 'https://' + HOST


class JobAVEndpointTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock(endpoint='https://huggingface.co')
        self.job = NS(id=JOB_ID, owner=NS(name='YUGOROU'),
                      status=NS(stage='RUNNING'), docker_image=IMAGE,
                      command=COMMAND.copy(), flavor='h200x2')
        self.api.inspect_job.return_value = self.job
        self.factory_patch = patch(
            'research.h3_personal.job_av_endpoint.http.client.HTTPSConnection')
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.connection = self.factory.return_value
        self.connection.getresponse.return_value.status = 401

    def attest(self, **changes):
        args = dict(job_id=JOB_ID, port=7860, expected_image=IMAGE,
                    expected_command=COMMAND.copy())
        args.update(changes)
        return attest_job(self.api, **args)

    def test_denied_anonymous_access_attests_exact_host(self):
        for status in (401, 403, 404):
            with self.subTest(status=status):
                self.factory.reset_mock()
                self.api.inspect_job.reset_mock()
                self.connection.getresponse.return_value.status = status
                endpoint = self.attest(timeout=3)
                self.api.inspect_job.assert_called_once_with(job_id=JOB_ID, namespace='YUGOROU')
                self.factory.assert_called_once_with(HOST, timeout=3)
                self.connection.request.assert_called_once_with(
                    'GET', '/health', headers={'Accept-Encoding': 'identity', 'Cache-Control': 'no-store'})
                self.connection.close.assert_called_once_with()
                self.assertEqual(endpoint.job_id, JOB_ID)
                self.assertTrue(endpoint.permits(URL))
                for other in (URL + '/', URL + '/health', URL + '.evil', 'http://' + HOST):
                    self.assertFalse(endpoint.permits(other))
                with self.assertRaises(FrozenInstanceError):
                    endpoint.base_url = 'https://evil.example'

    def test_default_timeout_and_unattested_endpoint(self):
        self.attest()
        self.factory.assert_called_once_with(HOST, timeout=10)
        self.assertFalse(JobEndpoint(URL, JOB_ID).permits(URL))
        self.assertFalse(JobEndpoint(URL, JOB_ID, object()).permits(URL))

    def test_invalid_bindings_fail_before_inspection_or_network(self):
        cases = []
        for key, values in {
            'job_id': [None, 123, 'A' * 24, 'a' * 23, JOB_ID + '\n', JOB_ID + '.evil'],
            'port': [True, 7860.0, '7860', 80],
            'expected_image': [None, 'image:latest', 'image@sha256:' + 'A' * 64],
            'expected_command': [None, [], ('python',), 'python'],
        }.items():
            cases.extend({key: value} for value in values)
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, 'canonical immutable'):
                    self.attest(**changes)
                self.api.inspect_job.assert_not_called()
                self.factory.assert_not_called()

    def test_trailing_newline_image_reaches_runtime_check(self):
        # Characterize current regex '$' behavior: it accepts a final newline.
        # Exact runtime comparison still rejects a different inspected image.
        with self.assertRaisesRegex(ValueError, 'identity/runtime differs'):
            self.attest(expected_image=IMAGE + '\n')
        self.api.inspect_job.assert_called_once()
        self.factory.assert_not_called()

    def test_noncanonical_api_rejected_before_inspection(self):
        for endpoint in (None, 'http://huggingface.co', 'https://huggingface.co/',
                         'https://huggingface.co.evil'):
            with self.subTest(endpoint=endpoint):
                self.api.endpoint = endpoint
                with self.assertRaises(ValueError):
                    self.attest()
                self.api.inspect_job.assert_not_called()
                self.factory.assert_not_called()

    def test_runtime_mismatches_never_probe_gateway(self):
        for field, value in (
            ('id', 'f' * 24), ('owner', NS(name='other')),
            ('status', NS(stage='PENDING')), ('status', NS(stage='COMPLETED')),
            ('docker_image', IMAGE.replace('a' * 64, 'b' * 64)),
            ('command', list(reversed(COMMAND))), ('flavor', 'h200'),
        ):
            with self.subTest(field=field, value=value):
                original = getattr(self.job, field)
                setattr(self.job, field, value)
                try:
                    with self.assertRaisesRegex(ValueError, 'identity/runtime differs'):
                        self.attest()
                    self.factory.assert_not_called()
                finally:
                    setattr(self.job, field, original)

    def test_gateway_success_redirects_and_errors_fail_closed(self):
        for status in (200, 204, 301, 302, 307, 400, 429, 500, 503):
            with self.subTest(status=status):
                self.factory.reset_mock()
                self.connection.getresponse.return_value.status = status
                with self.assertRaisesRegex(ValueError, 'access not denied'):
                    self.attest()
                self.connection.request.assert_called_once()
                self.connection.close.assert_called_once_with()

    def test_probe_errors_close_connection_and_do_not_retry(self):
        for phase in ('request', 'getresponse'):
            with self.subTest(phase=phase):
                self.factory.reset_mock()
                error = TimeoutError('unknown gateway outcome')
                getattr(self.connection, phase).side_effect = error
                try:
                    with self.assertRaises(TimeoutError) as caught:
                        self.attest()
                    self.assertIs(caught.exception, error)
                    self.factory.assert_called_once()
                    self.connection.request.assert_called_once()
                    self.connection.close.assert_called_once_with()
                finally:
                    getattr(self.connection, phase).side_effect = None

    def test_inspection_failure_never_probes_gateway(self):
        self.api.inspect_job.side_effect = TimeoutError('unknown inspection')
        with self.assertRaises(TimeoutError):
            self.attest()
        self.api.inspect_job.assert_called_once()
        self.factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
