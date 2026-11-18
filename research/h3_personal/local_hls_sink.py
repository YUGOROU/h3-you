"""Explicitly approved, finite Mac-local H264/AAC HLS playback owner.

No routing changes. Ready means worker/config ready, NOT playlist/playback ready.
"""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import threading
import time

from .headless_av_sink import HeadlessAVSink
from .obs_av_sink import output_arguments


# Shared stdout protocol: arbitrary worker strings are never retained.
PLAYER_EVENTS = frozenset({'playing', 'waiting', 'paused', 'stalled',
                           'awaiting_playlist', 'ready', 'first_video', 'buffering',
                           'ended', 'failed', 'timeout'})


def hls_arguments(directory):
    args = output_arguments(48124)
    return args[:args.index('-mpegts_flags')] + [
        '-hls_time', '1', '-hls_list_size', '0',
        '-hls_segment_filename', str(Path(directory) / 'seg-%03d.ts'),
        '-hls_playlist_type', 'event', '-f', 'hls',
        str(Path(directory) / 'index.m3u8')]


def _validated_binary(path, digest):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('nonsymlink player required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or not info.st_mode & 0o100
                or info.st_size > 64 * 1024 * 1024):
            raise ValueError('private owned executable required')
        with os.fdopen(os.dup(fd), 'rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or actual != digest:
            raise ValueError('player SHA256 mismatch')
    finally:
        os.close(fd)
    return str(path)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, *args):
        self._slots = threading.BoundedSemaphore(4)
        super().__init__(*args)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(.05)
                request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _server(root, max_file_bytes):
    root = Path(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(1)
            self._request_deadline = time.monotonic() + 1

        def do_HEAD(self):
            self._serve(False)

        def do_GET(self):
            self._serve(True)

        def _serve(self, body):
            # No decoding, query strings, alternate hosts/absolute URLs or paths.
            if not re.fullmatch(r'/(index\.m3u8|seg-[0-9]{3,6}\.ts)', self.path):
                self.send_error(404)
                return
            fd = None
            started = False
            try:
                fd = os.open(root / self.path[1:], os.O_RDONLY | os.O_NOFOLLOW)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= max_file_bytes:
                    raise ValueError('file bound')
                started = True
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.apple.mpegurl' if self.path.endswith('.m3u8') else 'video/mp2t')
                self.send_header('Content-Length', str(info.st_size))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                if body:
                    remaining = info.st_size
                    while remaining:
                        left = self._request_deadline - time.monotonic()
                        if left <= 0:
                            raise TimeoutError()
                        self.connection.settimeout(left)
                        block = os.read(fd, min(65536, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (OSError, ValueError):
                if not started:
                    self.send_error(404)
            finally:
                if fd is not None:
                    os.close(fd)

    return _Server(('127.0.0.1', 0), Handler)


class LocalHLSSink(HeadlessAVSink):
    def __init__(self, root, binding, *, player_binary, player_sha256,
                 speaker_uid, max_seconds, playback_approved=False,
                 ready_timeout=10, **kwargs):
        if playback_approved is not True:
            raise PermissionError('explicit local speaker playback approval required')
        if speaker_uid != 'BuiltInSpeakerDevice':
            raise ValueError('built-in speaker UID required')
        if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 10 < max_seconds <= 240):
            raise ValueError('finite playback window >10 and <=240 seconds required')
        if not 0 < ready_timeout <= 10 or kwargs.get('paced', True) is not True:
            raise ValueError('finite readiness and paced input required')
        self._binary = _validated_binary(player_binary, player_sha256)
        self._duration = max_seconds
        self._ready_timeout = ready_timeout
        self._owned_lock = threading.RLock()
        self._server = self._server_thread = self._player = self._monitor = None
        self._owned_closed = False
        self._ready = threading.Event()
        self._player_done = threading.Event()
        self._failure = None
        self.player_events = []
        self._natural_done = False
        self._evidence_written = False
        kwargs['paced'] = True
        kwargs.setdefault('max_frames', 124)
        try:
            super().__init__(root, binding, **kwargs)
            if not self._ready.wait(ready_timeout) or self._failure:
                raise RuntimeError('local player readiness failed')
        except BaseException:
            self.close()
            raise

    def _encoder_tail(self):
        # Called by parent before encoder creation/any frame submission. Worker
        # must acknowledge config without waiting for the as-yet absent playlist.
        self.hls_root = self.directory / 'hls'
        self.hls_root.mkdir(mode=0o700)
        self.playlist = self.hls_root / 'index.m3u8'
        self._server = _server(self.hls_root, 32 * 1024 * 1024)
        self._server_thread = threading.Thread(target=self._server.serve_forever,
                                                kwargs={'poll_interval': .05}, daemon=True)
        self._server_thread.start()
        self._deadline = time.monotonic() + self._duration
        self._player = subprocess.Popen([self._binary], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        config = dict(url=f'http://127.0.0.1:{self._server.server_port}/index.m3u8',
                      output_uid='BuiltInSpeakerDevice', max_seconds=self._duration)
        self._player.stdin.write((json.dumps(config) + '\n').encode())
        self._player.stdin.close()
        self._monitor = threading.Thread(target=self._watch_player, daemon=True)
        self._monitor.start()
        return hls_arguments(self.hls_root)

    def _watch_player(self):
        buffer = bytearray()
        try:
            fd = self._player.stdout.fileno()
            os.set_blocking(fd, False)
            while not self._owned_closed:
                if time.monotonic() >= self._deadline:
                    raise TimeoutError()
                if select.select([fd], [], [], .05)[0]:
                    block = os.read(fd, 4096)
                    if not block:
                        break
                    buffer.extend(block)
                    if len(buffer) > 16384:
                        raise ValueError()
                    while b'\n' in buffer:
                        line, _, rest = buffer.partition(b'\n')
                        buffer = bytearray(rest)
                        event = json.loads(line)
                        if not isinstance(event, dict) or len(self.player_events) >= 256:
                            raise ValueError()
                        kind = event.get('type')
                        if not self._ready.is_set():
                            if kind != 'local_player_ready' or event.get('output_uid') != 'BuiltInSpeakerDevice':
                                raise ValueError()
                            self._ready.set()
                        elif kind == 'local_player_done':
                            self._player_done.set()
                        elif kind != 'local_player_telemetry':
                            raise ValueError()
                        if kind == 'local_player_telemetry':
                            if event.get('event') not in PLAYER_EVENTS:
                                raise ValueError()
                        # Never retain arbitrary worker strings (URL/error/text).
                        safe = {'type': kind}
                        if kind == 'local_player_telemetry':
                            safe['event'] = event['event']
                        if kind == 'local_player_ready':
                            safe['output_uid'] = 'BuiltInSpeakerDevice'
                        for key in ('elapsed_seconds', 'position_seconds'):
                            value = event.get(key)
                            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1000000:
                                safe[key] = value
                        self.player_events.append(safe)
                if self._player.poll() is not None and not select.select([fd], [], [], 0)[0]:
                    break
            if not self._owned_closed:
                if buffer or self._player.wait(timeout=1) != 0 or not self._player_done.is_set() or not getattr(self, '_finishing', False):
                    raise RuntimeError()
        except BaseException:
            if not self._owned_closed:
                self._failure = 'local player failed or exceeded finite window'
                self._cancel.set()
                # Enforce the finite lease even if the caller becomes idle.
                import signal
                if self._player.poll() is None:
                    try:
                        os.killpg(self._player.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self._player.wait()
                self._server.shutdown()
        finally:
            self._ready.set()

    def accept(self, segment_path, lease):
        try:
            self._check_player()
            result = super().accept(segment_path, lease)
            self._check_player()
            return result
        except BaseException:
            self.close()
            raise

    def _check_player(self):
        if self._failure or self._player.poll() is not None or self._player_done.is_set():
            raise RuntimeError('local player unavailable')

    def finish(self):
        self._finishing = True
        return super().finish()

    def _validate_output(self, deadline):
        # Parent has closed both input pipes and successfully waited encoder EOS.
        if not self.playlist.is_file() or b'#EXT-X-ENDLIST' not in self.playlist.read_bytes():
            raise RuntimeError('HLS encoder EOS missing')
        self._finishing = True
        deadline = min(deadline, self._deadline)
        while self._monitor.is_alive():
            if self._failure or time.monotonic() >= deadline:
                raise RuntimeError('local playback did not finish within deadline')
            self._monitor.join(.02)
        if self._failure or not self._player_done.is_set() or self._player.returncode != 0:
            raise RuntimeError('local playback incomplete')
        self._natural_done = True
        return self.playlist

    def _save_evidence(self):
        if self._evidence_written or not hasattr(self, 'root'):
            return
        self._evidence_written = True
        first = next((dict(e) for e in self.player_events
                      if e.get('event') == 'playing'), None)
        evidence = dict(natural_done=self._natural_done,
                        output_uid='BuiltInSpeakerDevice',
                        first_playback=first, events=list(self.player_events))
        try:
            fd = os.open(self.root / 'local-player-evidence.json',
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as output:
                json.dump(evidence, output, allow_nan=False)
                output.write('\n')
        except (OSError, ValueError):
            # Evidence must neither overwrite a prior run nor mask its error.
            pass

    def close(self):
        with self._owned_lock:
            if not self._owned_closed:
                self._owned_closed = True
                if self._player is not None:
                    if self._player.poll() is None:
                        import signal
                        try:
                            os.killpg(self._player.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    self._player.wait()
                if self._monitor is not None and self._monitor is not threading.current_thread():
                    self._monitor.join(2)
                if self._player is not None:
                    self._player.stdout.close()
                    if self._player.stdin is not None:
                        self._player.stdin.close()
                if self._server is not None:
                    self._server.shutdown()
                    self._server.server_close()
                if self._server_thread is not None:
                    self._server_thread.join(2)
        try:
            if hasattr(self, '_close_lock'):
                super().close()
        finally:
            self._save_evidence()
