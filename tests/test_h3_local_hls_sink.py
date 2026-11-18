"""CPU-only: no player launch, devices, GUI, GPU or external networking."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.local_hls_sink import (
    LocalHLSSink, _server, _validated_binary, hls_arguments)
from research.h3_personal.obs_av_sink import output_arguments


class LocalHLSTests(unittest.TestCase):
    def test_production_codecs_and_hls(self):
        args = hls_arguments('/private/hls')
        obs = output_arguments(48124)
        self.assertEqual(args[:obs.index('-mpegts_flags')], obs[:obs.index('-mpegts_flags')])
        self.assertEqual(args[args.index('-g') + 1], '24')
        self.assertEqual(args[-2:], ['hls', '/private/hls/index.m3u8'])
        self.assertNotIn('udp://', ' '.join(args))
        self.assertIn('/private/hls/seg-%03d.ts', args)

    def test_telemetry_protocol_sanitizes_and_rejects_unknown_enum(self):
        for name in ('playing', 'unknown-secret-url'):
            sink = object.__new__(LocalHLSSink)
            sink._owned_closed = False
            sink._ready = threading.Event()
            sink._player_done = threading.Event()
            sink._cancel = threading.Event()
            sink._failure = None
            sink._finishing = True
            sink._deadline = time.monotonic() + 2
            sink.player_events = []
            readfd, writefd = os.pipe()
            events = [dict(type='local_player_ready', output_uid='BuiltInSpeakerDevice'),
                      dict(type='local_player_telemetry', event=name,
                           elapsed_seconds=1, position_seconds=.5, url='SECRET'),
                      dict(type='local_player_done')]
            os.write(writefd, ''.join(json.dumps(e) + '\n' for e in events).encode())
            os.close(writefd)
            with os.fdopen(readfd, 'rb') as stream:
                sink._player = Mock(stdout=stream)
                sink._player.poll.return_value = 0
                sink._player.wait.return_value = 0
                sink._server = Mock()
                sink._watch_player()
            if name == 'playing':
                self.assertIsNone(sink._failure)
                self.assertEqual(sink.player_events[1]['event'], 'playing')
                self.assertNotIn('SECRET', json.dumps(sink.player_events))
            else:
                self.assertIsNotNone(sink._failure)
                self.assertTrue(sink._cancel.is_set())

    def test_evidence_private_exclusive_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            sink = object.__new__(LocalHLSSink)
            sink.root = Path(d)
            sink._natural_done = True
            sink._evidence_written = False
            sink.player_events = [dict(type='local_player_telemetry', event='playing',
                                       elapsed_seconds=1, position_seconds=.2)]
            sink._save_evidence()
            path = sink.root / 'local-player-evidence.json'
            saved = path.read_bytes()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(saved)['first_playback']['event'], 'playing')
            self.assertTrue(json.loads(saved)['natural_done'])
            sink._evidence_written = False
            sink._natural_done = False
            sink._save_evidence()
            self.assertEqual(path.read_bytes(), saved)

    def test_server_concurrency_rejects_without_thread(self):
        with tempfile.TemporaryDirectory() as d:
            server = _server(d, 32)
            request = Mock()
            try:
                for _ in range(4):
                    self.assertTrue(server._slots.acquire(False))
                with patch.object(server, 'shutdown_request') as shutdown:
                    server.process_request(request, ('127.0.0.1', 1))
                    self.assertIn(b'503', request.sendall.call_args.args[0])
                    shutdown.assert_called_once_with(request)
            finally:
                server.server_close()

    def test_maximum_matches_worker(self):
        with self.assertRaises(ValueError):
            LocalHLSSink('.', None, player_binary='missing', player_sha256='',
                         speaker_uid='BuiltInSpeakerDevice', max_seconds=241,
                         playback_approved=True)

    def test_shared_sink_call_marker(self):
        sink = object.__new__(LocalHLSSink)
        sink._check_player = Mock()
        marker = object()
        with patch('research.h3_personal.local_hls_sink.HeadlessAVSink.accept',
                   return_value=marker) as accept:
            self.assertIs(sink.accept('segment', 'lease'), marker)
            accept.assert_called_once_with('segment', 'lease')
        self.assertEqual(sink._check_player.call_count, 2)

    def test_approval_and_speaker(self):
        with self.assertRaises(PermissionError):
            LocalHLSSink('.', None, player_binary='missing', player_sha256='',
                         speaker_uid='BuiltInSpeakerDevice', max_seconds=30)
        with self.assertRaises(ValueError):
            LocalHLSSink('.', None, player_binary='missing', player_sha256='',
                         speaker_uid='other', max_seconds=30, playback_approved=True)

    def test_binary_hash_permissions_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'player'
            p.write_bytes(b'synthetic executable')
            p.chmod(0o700)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(_validated_binary(p, digest), str(p))
            with self.assertRaises(ValueError):
                _validated_binary(p, '0' * 64)
            p.chmod(0o755)
            with self.assertRaises(ValueError):
                _validated_binary(p, digest)
            link = p.with_name('link')
            link.symlink_to(p)
            with self.assertRaises(ValueError):
                _validated_binary(link, digest)

    def test_loopback_allowlist_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'index.m3u8').write_bytes(b'#EXTM3U\n')
            (root / 'seg-000.ts').write_bytes(b'tiny')
            (root / 'seg-001.ts').symlink_to(root / 'index.m3u8')
            (root / 'seg-002.ts').write_bytes(b'x' * 100)
            server = _server(root, 32)
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={'poll_interval': .01})
            thread.start()
            try:
                def request(path, method='GET'):
                    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                    conn.request(method, path)
                    response = conn.getresponse()
                    result = response.status, response.read()
                    conn.close()
                    return result
                self.assertEqual(request('/index.m3u8'), (200, b'#EXTM3U\n'))
                self.assertEqual(request('/seg-000.ts', 'HEAD'), (200, b''))
                for path in ('/', '/../index.m3u8', '/%2e%2e/index.m3u8',
                             '/index.m3u8?token=no', 'http://example.com/index.m3u8',
                             '/seg-001.ts', '/missing', '/seg-002.ts'):
                    self.assertNotEqual(request(path)[0], 200)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_constructor_error_cleanup_and_forced_pacing(self):
        captured = {}
        def parent(sink, *args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('synthetic constructor failure')
        with patch('research.h3_personal.local_hls_sink._validated_binary', return_value='/fake'), \
             patch('research.h3_personal.local_hls_sink.HeadlessAVSink.__init__', parent):
            with self.assertRaises(RuntimeError):
                LocalHLSSink('.', None, player_binary='/fake', player_sha256='x',
                             speaker_uid='BuiltInSpeakerDevice', max_seconds=30,
                             playback_approved=True)
        self.assertTrue(captured['paced'])
        self.assertEqual(captured['max_frames'], 124)

    def test_finish_requires_endlist_and_natural_done(self):
        with tempfile.TemporaryDirectory() as d:
            sink = object.__new__(LocalHLSSink)
            sink.playlist = Path(d) / 'index.m3u8'
            sink.playlist.write_bytes(b'#EXTM3U\n')
            sink._deadline = time.monotonic() + 1
            sink._monitor = Mock()
            sink._monitor.is_alive.return_value = False
            sink._failure = None
            sink._player_done = threading.Event()
            sink._player = Mock(returncode=0)
            with self.assertRaises(RuntimeError):
                sink._validate_output(time.monotonic() + 1)
            sink.playlist.write_bytes(b'#EXTM3U\n#EXT-X-ENDLIST\n')
            with self.assertRaises(RuntimeError):
                sink._validate_output(time.monotonic() + 1)
            sink._player_done.set()
            self.assertEqual(sink._validate_output(time.monotonic() + 1), sink.playlist)

    def test_failed_player_rejects_accept_and_cleans(self):
        sink = object.__new__(LocalHLSSink)
        sink._failure = 'failed'
        sink.close = Mock()
        with self.assertRaises(RuntimeError):
            sink.accept('not-read', None)
        sink.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
