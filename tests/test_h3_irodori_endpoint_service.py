import base64
import hashlib
import http.client
import json
from pathlib import Path
import threading
import time
import unittest
import wave

from research.h3_personal.irodori_endpoint_service import create_server, SOURCE, MODEL, CODEC


class FakeTTS:
    closed = False
    fail = False
    block = None

    def close(self):
        self.closed = True

    def synthesize(self, text, turn_id, directory, deadline):
        self.directory, self.deadline = directory, deadline
        assert not directory.exists()
        directory.mkdir()
        if self.block:
            self.block.wait(2)
        if self.fail:
            raise RuntimeError('PRIVATE token /voice/path ' + text)
        path = directory / 'audio.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\x00\x00' * 240)
        self.data = path.read_bytes()
        return {'wav': path, 'receipt': dict(path=str(path), sha256=hashlib.sha256(self.data).hexdigest(), watermark_applied=True, source_revision=SOURCE, model_revision=MODEL, codec_revision=CODEC)}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tts = FakeTTS()
        self.expiry = time.time() + 90
        self.server = create_server(self.tts, {'ready': True, 'device': 'fake'}, key='secret', session='session', expires_at=self.expiry, host='127.0.0.1', port=0)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.server.gate.close()

    def call(self, path='/synthesize', payload=None, key='secret', method='POST', headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        h = {'X-H3-TTS-Key': key}
        h.update(headers or {})
        conn.request(method, path, json.dumps(payload) if payload is not None else None, h)
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    def request(self, epoch=1, **changes):
        return dict(dict(session='session', epoch=epoch, text='hello'), **changes)

    def test_valid_cleanup_replay_limit(self):
        status, result = self.call(payload=self.request())
        self.assertEqual(status, 200)
        self.assertEqual(base64.b64decode(result['wav_base64']), self.tts.data)
        self.assertNotIn('path', result['receipt'])
        self.assertFalse(self.tts.directory.exists())
        self.assertLessEqual(self.tts.deadline, time.monotonic() + 120)
        self.assertGreater(self.tts.deadline, time.monotonic())
        self.assertEqual(self.call(payload=self.request())[0], 409)
        self.assertEqual(self.call(payload=self.request(2))[0], 200)
        self.assertEqual(self.call(payload=self.request(3))[0], 429)

    def test_auth_health_bounds(self):
        self.assertEqual(self.call('/health', key='', method='GET'), (200, {'ready': True}))
        self.assertEqual(self.call('/ready', key='', method='GET')[0], 401)
        self.assertEqual(self.call('/ready', method='GET')[1]['device'], 'fake')
        self.assertEqual(self.call(payload=self.request(), key='bad')[0], 401)
        for changes in ({'session': 'wrong'}, {'epoch': True}, {'epoch': 0}, {'text': ' '}, {'text': 'x' * 161}):
            self.assertEqual(self.call(payload=self.request(**changes))[0], 400)
        self.assertEqual(self.call(payload=self.request(), headers={'Transfer-Encoding': 'chunked'})[0], 400)

    def test_error_quarantine_privacy(self):
        self.tts.fail = True
        status, body = self.call(payload=self.request())
        self.assertEqual((status, body), (503, {'error': 'unavailable'}))
        self.assertTrue(self.tts.closed)
        self.assertFalse(self.tts.directory.exists())
        self.assertEqual(self.call('/health', method='GET')[0], 503)

    def test_one_owner_ready_during_busy(self):
        self.tts.block = threading.Event()
        result = []
        worker = threading.Thread(target=lambda: result.append(self.call(payload=self.request())))
        worker.start()
        limit = time.monotonic() + 2
        while not hasattr(self.tts, 'directory') and time.monotonic() < limit:
            time.sleep(.005)
        self.assertEqual(self.call(payload=self.request(2))[0], 429)
        self.assertEqual(self.call('/ready', method='GET')[0], 200)
        self.assertEqual(self.call('/health', method='GET')[0], 200)
        self.tts.block.set()
        worker.join()
        self.assertEqual(result[0][0], 200)


if __name__ == '__main__':
    unittest.main()
