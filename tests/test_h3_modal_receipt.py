import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from research.modal_identity_receipt import publish


class ReceiptTests(unittest.TestCase):
    def test_private_receipt_has_no_raw_paths_or_quality_country_claims(self):
        api = Mock()
        api.repo_info.return_value = NS(private=True)
        api.create_commit.return_value = NS(oid='a'*40)
        operations = []
        def operation(**kwargs):
            operations.append(kwargs)
            return kwargs
        result = publish(api, repo='test/private', request_sha256='b'*64,
            report={'stage': 'train', 'readiness': False, 'failure_type': 'ValueError',
                    'raw_path': '/private/personal.wav', 'token': 'synthetic-not-returned'},
            placement={'cloud': 'aws', 'region': 'synthetic', 'country': 'CA'}, operation=operation)
        value = json.loads(operations[0]['path_or_fileobj'])
        self.assertFalse(value['quality_verified'])
        self.assertFalse(value['observed_country_verified'])
        self.assertNotIn('raw_path', value['pipeline'])
        self.assertNotIn('token', value['pipeline'])
        self.assertEqual(result, 'a'*40)
        api.create_commit.assert_called_once()

    def test_public_repo_fails_without_publication(self):
        api = Mock()
        api.repo_info.return_value = NS(private=False)
        with self.assertRaises(ValueError):
            publish(api, repo='test/public', request_sha256='b'*64, report={}, placement={}, operation=Mock())
        api.create_commit.assert_not_called()


if __name__ == '__main__': unittest.main()
