"""Real local socketpair + cooperative input, no WAN/model/device access."""
import io
import socket
import threading
import time
import unittest

from research.h3_personal.endpoint_asr import transcribe


class CleanupTests(unittest.TestCase):
    def run_adapter(self, pcm, **kwargs):
        client, server = socket.socketpair()
        source = client.makefile('rb')
        sent = bytearray()
        def drain():
            try:
                while True:
                    data = server.recv(4096)
                    if not data:
                        return
                    sent.extend(data)
            finally:
                server.close()
        thread = threading.Thread(target=drain)
        thread.start()
        try:
            result = transcribe('https://test.endpoints.huggingface.cloud',
                                token='fake', owner_key='fake', pcm=pcm,
                                completion_policy='input-finalized', timeout=.15,
                                _transport=lambda *a: (client, source), **kwargs)
        finally:
            client.close()
            thread.join(1)
        self.assertFalse(thread.is_alive())
        return result, bytes(sent)

    def test_no_input_closes_without_commit_or_invented_final(self):
        for _ in range(10):
            result, data = self.run_adapter(io.BytesIO(b''))
            self.assertEqual(result['errors'], ['no_input_audio'])
            self.assertFalse(result['input_committed'])
            self.assertFalse(result['final'])
            self.assertEqual(result['audio_sent_seconds'], 0)
            self.assertTrue(result['cleanup_complete'])
            # Decode the one masked client message; no empty commit follows.
            self.assertEqual(data[0], 129)
            length = data[1] & 127
            offset = 2
            if length == 126:
                length = int.from_bytes(data[2:4], 'big'); offset = 4
            self.assertEqual(len(data), offset+4+length)

    def test_input_is_stopped_before_joining_blocked_sender(self):
        entered, stopped, exited = (threading.Event() for _ in range(3))
        class Input:
            def read(self, _):
                entered.set()
                stopped.wait(2)
                exited.set()
                return b''
        def stop_input():
            self.assertTrue(entered.is_set())
            stopped.set()
        start = time.monotonic()
        result, _ = self.run_adapter(Input(), on_input_stop=stop_input)
        self.assertTrue(exited.is_set())
        self.assertTrue(result['cleanup_complete'])
        self.assertEqual(result['errors'], ['deadline'])
        self.assertLess(time.monotonic()-start, 1.)

    def test_failed_input_stop_never_claims_clean_cleanup(self):
        def fail():
            raise RuntimeError('private capture detail')
        result, _ = self.run_adapter(io.BytesIO(b''), on_input_stop=fail)
        self.assertIn('input_stop_error', result['errors'])
        self.assertFalse(result['cleanup_complete'])
        self.assertNotIn('private', repr(result))


if __name__ == '__main__':
    unittest.main()
