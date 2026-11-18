import http.client
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research.h3_personal.browser_av import BrowserAVServer, BrowserAVSink, hls_arguments
from research.h3_personal.obs_av_sink import output_arguments


class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.output = {'default_is_builtin_speaker': True, 'stable': True}
        original = BrowserAVServer._read
        self.reader = patch.object(BrowserAVServer, '_read', staticmethod(
            lambda directory, name: b'<script>void 0;</script>' if name == 'h3_browser_client.html'
            else original(directory, name)))
        self.reader.start()

    def tearDown(self):
        self.reader.stop()
        self.temp.cleanup()

    def request(self, server, path, body=None, origin=True):
        conn = http.client.HTTPConnection(server.authority, timeout=2)
        headers = {'Origin': server.origin if origin else 'http://evil.invalid'}
        conn.request('GET' if body is None else 'POST', path,
                     body=None if body is None else json.dumps(body), headers=headers)
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    def test_lifecycle_security_and_privacy(self):
        with BrowserAVServer(self.root, lambda: self.output, 3) as s:
            code, headers, _ = self.request(s, '/')
            self.assertEqual(code, 200)
            self.assertIn('sha256-', headers['Content-Security-Policy'])
            self.assertIn('microphone=()', headers['Permissions-Policy'])
            for path in ('/media/', '/foo', '/index.m3u8?x=secret', '/../secret'):
                self.assertEqual(self.request(s, path)[0], 404)
            self.assertEqual(self.request(s, '/api/event', {'event': 'client_ready'}, False)[0], 403)
            self.assertEqual(self.request(s, '/api/event', {'event': 'client_ready'})[0], 200)
            self.assertTrue(s.client_ready.is_set())
            self.assertFalse(s.media_root.exists())
            s.validate_output()
            for event in ({'event': 'oops'}, {'event': 'playing', 'data': 'secret'},
                          {'event': 'playing', 'current_time': float('nan')}):
                self.assertEqual(self.request(s, '/api/event', event)[0], 400)
            s.media_root.mkdir()
            (s.media_root / 'init.mp4').symlink_to('/etc/passwd')
            self.assertEqual(self.request(s, '/init.mp4')[0], 404)
            self.output['stable'] = False
            with self.assertRaises(RuntimeError):
                s.validate_output()
        self.assertFalse(s._thread.is_alive())
        evidence = (self.root / 'browser-av-evidence.json').read_text()
        self.assertNotIn('secret', evidence)
        self.assertNotIn('http:', evidence)
        self.assertEqual((self.root / 'browser-av-evidence.json').stat().st_mode & 0o777, 0o600)

    def test_failure_before_ready_is_recorded_without_granting_readiness(self):
        with BrowserAVServer(self.root, lambda: self.output, 2) as server:
            for kind in ('playing', 'ended'):
                self.assertEqual(self.request(server,'/api/event',{'event':kind})[0],400)
            self.assertEqual(self.request(server,'/api/event',{'event':'failed',
                'elapsed_seconds':.1,'media_seconds':0})[0],200)
            self.assertEqual(server.stream_state,'failed')
            self.assertFalse(server.client_ready.is_set())
            self.assertFalse(server.playback_done.is_set())
        evidence=json.loads((self.root/'browser-av-evidence.json').read_text())
        self.assertEqual(evidence['phases'],{'failed':1})

    def test_eos_and_bound(self):
        with BrowserAVServer(self.root, lambda: self.output, 2) as s:
            s._event({'event': 'client_ready'})
            s.media_root.mkdir()
            (s.media_root / 'index.m3u8').write_text('#EXTM3U\n#EXT-X-ENDLIST\n')
            s.finish('eos')
            s._event({'event': 'ended'})
            self.assertTrue(s.playback_done.is_set())
            self.assertEqual(s.stream_state, 'ended')
            for _ in range(253):
                s._event({'event': 'waiting'})
            with self.assertRaises(ValueError):
                s._event({'event': 'waiting'})
        with self.assertRaises(ValueError):
            BrowserAVServer(self.root, lambda: {}, 241)

    def test_startup_clock_and_schema(self):
        with patch('research.h3_personal.browser_av.time.monotonic', return_value=100):
            with BrowserAVServer(self.root, lambda: self.output, startup_seconds=1200) as s:
                self.assertEqual(s.deadline, 1540)
                self.assertEqual(s.hard_deadline, 1540)
                s._event({'event': 'client_ready'})
                with self.assertRaises(ValueError):
                    s._event({'event': 'client_ready'})
                with patch('research.h3_personal.browser_av.time.monotonic', return_value=900):
                    s.finish('streaming')
                    self.assertEqual(s.deadline, 1140)
                s.finish('streaming')
                self.assertEqual(s.deadline, 1140)
        for value in (True, -1, 1201, float('nan'), '1200'):
            with self.assertRaises(ValueError):
                BrowserAVServer(self.root, lambda: self.output, startup_seconds=value)

    def test_ready_false_and_early_end(self):
        with BrowserAVServer(self.root, lambda: self.output) as s:
            self.output['default_is_builtin_speaker'] = False
            with self.assertRaises(ValueError):
                s._event({'event': 'client_ready'})
            self.assertFalse(s.client_ready.is_set())
            self.output['default_is_builtin_speaker'] = True
            s._event({'event': 'client_ready'})
            s.finish('streaming')
            s._event({'event': 'ended'})
            self.assertFalse(s.playback_done.is_set())
            self.assertEqual(s.stream_state, 'failed')

    def test_finite_server(self):
        with BrowserAVServer(self.root, lambda: {}, .1) as s:
            s._thread.join(1)
            self.assertFalse(s._thread.is_alive())
            self.assertEqual(s.stream_state, 'failed')

    def test_nested_receiver_root_matches_real_pump(self):
        from research.h3_personal.paired_av import LiveAVBinding
        from research.h3_personal.paired_av_pump import ReceiverPump
        root=self.root/'conversation'/'receiver'
        root.mkdir(parents=True,mode=0o700)
        binding=LiveAVBinding('browser-test','a'*64,1,max_segments=8)
        with BrowserAVServer(self.root, lambda: self.output, 2) as server:
            server._event({'event':'client_ready'})
            sink=BrowserAVSink(root,binding,server=server,playback_approved=True)
            try:
                pump=ReceiverPump(object(),binding,root,sink,deadline=time.monotonic()+1)
                self.assertEqual(pump.root,sink.root)
                self.assertEqual(sink.hls_root,server.media_root)
            finally:
                sink.close()
            with self.assertRaises(ValueError):
                BrowserAVSink(self.root.parent/'outside',binding,server=server,playback_approved=True)

    def test_codec_and_pre_factory_gate(self):
        args = hls_arguments(self.root)
        shared = output_arguments(48124)
        self.assertEqual(args[:shared.index('-mpegts_flags')], shared[:shared.index('-mpegts_flags')])
        self.assertIn('fmp4', args)
        self.assertIn('init.mp4', args)
        self.assertNotIn('udp:', ' '.join(args))
        with BrowserAVServer(self.root, lambda: self.output, 2) as s:
            with patch('research.h3_personal.browser_av.HeadlessAVSink.__init__') as factory:
                with self.assertRaises(RuntimeError):
                    BrowserAVSink(self.root, None, server=s, playback_approved=True)
                factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
