import hashlib
import time
import unittest
from unittest.mock import patch

from research.run_h3_https_source_probe import drain_two, main


class SourceProbeTests(unittest.TestCase):
    def source(self):
        return dict(config=dict(url='https://fake.endpoints.huggingface.cloud',
                    session='session', boot='boot', expires_at=int(time.time())+600),
                    token='private-token', key='private-key')

    def test_full_old_drain_and_new_request_before_client_close(self):
        calls = []; completions = []
        class Client:
            def __init__(self, *args, **kwargs):pass
            def read(self, *args):calls.append('ready')
            def stream(self, text, epoch, deadline):
                calls.append(('start', epoch))
                data = b'\x00\x00\x00\x3f'*320
                yield data
                completions.append(dict(epoch=epoch, samples=320, chunks=1,
                    native_float_sha256=hashlib.sha256(data).hexdigest(),
                    completion=dict(reason='natural_stop', natural_stop=True, stream_exhausted=True)))
                calls.append(('END', epoch))
            def _request(self, *args):return dict(evidence=dict(completions=completions))
            def close(self):calls.append('close')
        result = drain_two(self.source(), client_factory=Client)
        self.assertEqual(calls, ['ready', ('start', 1), ('END', 1),
                                 ('start', 2), ('END', 2), 'close'])
        self.assertTrue(result['exact_native_pcm_verified'])
        self.assertTrue(result['serial_native_service_reused'])
        self.assertFalse(result['media_saved'])
        self.assertFalse(result['fresh_conversation'])
        self.assertNotIn('private', repr(result))

    def test_inspection_never_opens_lease(self):
        with patch('research.run_h3_https_source_probe.lease') as lease, patch('builtins.print'):
            self.assertEqual(main([]), 0)
        lease.assert_not_called()


if __name__ == '__main__':
    unittest.main()
