"""CPU-only backend admission tests: no provider, network, or local model."""
import copy
import os
import time
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.native_tts_backend import (
    config_preroll, prepare_direct_tts, validate_native_config,
)
from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
from research.h3_personal.voxcpm_endpoint_protocol import PINS


def config():
    return dict(schema='h3-native-voxcpm-backend-v1',
                url='https://test.us-east-1.aws.endpoints.huggingface.cloud',
                session='a' * 32, boot='b' * 32, expires_at=time.time() + 600,
                pins=dict(PINS), preroll=dict(sample_rate=48000,
                reserve_samples=48000, max_reserve_samples=96000, max_wait_seconds=2.0))


def spec(value=None, mode='native_voxcpm'):
    row = {'tts_mode': mode}
    if value is not None:
        row['native_tts'] = value
    return {'live_direct_session': row}


class NativeBackendTests(unittest.TestCase):
    def test_detached_pure_validation_and_policy(self):
        value = config()
        value['expires_at'] = 1  # staging may precede lease activation
        original = copy.deepcopy(value)
        result = validate_native_config(value)
        result['pins']['sample_rate'] = 1
        result['preroll']['reserve_samples'] = 1
        self.assertEqual(value, original)
        self.assertEqual(config_preroll(value).reserve_samples, 48000)
        value['preroll'].update(reserve_samples=24000, max_reserve_samples=48000,
                                max_wait_seconds=1)
        self.assertEqual(config_preroll(value).reserve_samples, 24000)

    def test_strict_fields_and_types(self):
        mutations = [
            lambda c: c.update(secret='forbidden'),
            lambda c: c.pop('boot'),
            lambda c: c.update(session='A' * 32),
            lambda c: c.update(boot='a'),
            lambda c: c.update(expires_at=True),
            lambda c: c.update(expires_at=float('nan')),
            lambda c: c.update(expires_at=float('inf')),
            lambda c: c.update(expires_at=0),
            lambda c: c['pins'].update(sample_rate=48000.0),
            lambda c: c['pins'].update(watermark_applied=0),
            lambda c: c['pins'].update(extra=1),
            lambda c: c['preroll'].update(reserve_samples=True),
            lambda c: c['preroll'].update(max_reserve_samples=96001),
            lambda c: c['preroll'].update(max_wait_seconds=True),
            lambda c: c['preroll'].update(max_wait_seconds=6),
            lambda c: c['preroll'].update(extra=1),
        ]
        for mutate in mutations:
            value = config()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_native_config(value)

    def test_endpoint_only_urls(self):
        for url in ('http://x.endpoints.huggingface.cloud',
                    'https://huggingface.co/jobs/x', 'https://example.com',
                    'https://x.endpoints.huggingface.cloud.evil.com',
                    'https://x.endpoints.huggingface.cloud/',
                    'https://x.endpoints.huggingface.cloud:443',
                    'https://user@x.endpoints.huggingface.cloud',
                    'https://x.endpoints.huggingface.cloud?q=1',
                    'https://x.endpoints.huggingface.cloud#x'):
            value = config()
            value['url'] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_native_config(value)

    def test_native_success_no_io_no_local_no_persistence(self):
        value = spec(config())
        original = copy.deepcopy(value)
        local = Mock(side_effect=AssertionError('local forbidden'))
        with patch.dict(os.environ, {'H3_NATIVE_TTS_HF_TOKEN': 'private-token',
                                    'H3_NATIVE_TTS_KEY': 'private-key'}, clear=True), \
                patch('http.client.HTTPSConnection', side_effect=AssertionError('network forbidden')):
            future = prepare_direct_tts(None, value, local)
            self.assertTrue(future.done())
            client = future.result()
            self.assertIsInstance(client, RemoteVoxCPM)
            self.assertFalse(client._ready)
            self.assertNotIn('H3_NATIVE_TTS_HF_TOKEN', os.environ)
            self.assertNotIn('H3_NATIVE_TTS_KEY', os.environ)
            client.close()
        local.assert_not_called()
        self.assertEqual(value, original)
        self.assertNotIn('private-token', repr(value))
        self.assertNotIn('private-key', repr(value))

    def test_native_failures_never_fallback(self):
        expired = config()
        expired['expires_at'] = 1
        for value, env in ((spec(), {}), (spec(config()), {}),
                           (spec(config()), {'H3_NATIVE_TTS_HF_TOKEN': 'private-token'}),
                           (spec(expired), {'H3_NATIVE_TTS_HF_TOKEN': 'private-token',
                                            'H3_NATIVE_TTS_KEY': 'private-key'})):
            local = Mock(side_effect=AssertionError('local forbidden'))
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(Exception) as error:
                    prepare_direct_tts(None, value, local)
                self.assertNotIn('private-token', str(error.exception))
                self.assertNotIn('H3_NATIVE_TTS_HF_TOKEN', os.environ)
                self.assertNotIn('H3_NATIVE_TTS_KEY', os.environ)
            local.assert_not_called()
        local = Mock()
        with patch.dict(os.environ, {'H3_NATIVE_TTS_HF_TOKEN': 't', 'H3_NATIVE_TTS_KEY': 'k'}), \
                patch('research.h3_personal.native_tts_backend.RemoteVoxCPM',
                      side_effect=RuntimeError('client failed')):
            with self.assertRaisesRegex(RuntimeError, 'client failed'):
                prepare_direct_tts(None, spec(config()), local)
        local.assert_not_called()

    def test_digest_binds_policy_and_secrets_are_child_only(self):
        from research.h3_personal.native_tts_backend import config_sha256,native_job_secrets
        value=config();before=copy.deepcopy(value)
        digest=config_sha256(value)
        self.assertEqual(digest,config_sha256(dict(reversed(list(value.items())))))
        value['preroll']['reserve_samples']=24000
        self.assertNotEqual(digest,config_sha256(value))
        with patch.dict(os.environ,{'H3_NATIVE_TTS_HF_TOKEN':'token','H3_NATIVE_TTS_KEY':'key'},clear=True):
            fields=native_job_secrets(spec(before))
            self.assertEqual(fields,dict(os.environ))
            self.assertNotIn('token',repr(before))
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):native_job_secrets(spec(before))
        local=Mock(return_value=object())
        self.assertIs(prepare_direct_tts(None,{'live_direct_session':{}},local),local.return_value)
        local.assert_called_once_with()

    def test_legacy_and_mutual_exclusion(self):
        for mode in ('complete', 'sentence_incremental'):
            local = Mock(return_value=object())
            self.assertIs(prepare_direct_tts(None, spec(mode=mode), local), local.return_value)
            local.assert_called_once_with()
        for value in (spec(config(), 'complete'), spec(mode='unknown'),
                      spec(mode=True), {}, spec(mode=None),
                      dict(spec(config()), native_tts=config())):
            local = Mock()
            with self.assertRaises(ValueError):
                prepare_direct_tts(None, value, local)
            local.assert_not_called()


if __name__ == '__main__':
    unittest.main()
