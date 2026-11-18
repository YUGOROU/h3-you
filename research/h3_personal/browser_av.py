"""Finite loopback browser transport; no browser, device or GPU ownership."""
from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path
import re
import stat
import socket
import threading
import time

from .headless_av_sink import HeadlessAVSink
from .browser_presentation import BrowserPresentationEvidence
from .local_hls_sink import _Server
from .obs_av_sink import output_arguments

EVENTS = frozenset(('client_ready', 'playing', 'waiting', 'paused', 'stalled',
                    'first_video', 'first_append', 'buffering', 'ended', 'failed', 'timeout',
                    'cancelled', 'output_drift'))
SCALARS = frozenset(('elapsed_seconds', 'position_seconds', 'buffered_seconds',
                     'current_time', 'media_seconds', 'ready_state', 'network_state'))
LIMIT = 32 * 1024 * 1024


class _JoinedServer(_Server):
    daemon_threads = False
    block_on_close = True


def hls_arguments(directory):
    args = output_arguments(48124)
    return args[:args.index('-mpegts_flags')] + [
        '-f', 'hls', '-hls_playlist_type', 'event', '-hls_segment_type', 'fmp4',
        '-hls_fmp4_init_filename', 'init.mp4', '-hls_segment_filename',
        str(Path(directory) / 'seg-%03d.m4s'), '-hls_time', '1',
        '-hls_list_size', '0', str(Path(directory) / 'index.m3u8')]


class BrowserAVServer:
    def __init__(self, root, output_status, max_seconds=240, startup_seconds=0,
                 presentation=None):
        if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 0 < max_seconds <= 240 or not callable(output_status)):
            raise ValueError('finite <=240s server and status callable required')
        if (type(startup_seconds) not in (int, float)
                or not math.isfinite(startup_seconds) or not 0 <= startup_seconds <= 1200):
            raise ValueError('finite startup 0..1200s required')
        if presentation is not None and type(presentation) is not BrowserPresentationEvidence:
            raise ValueError('explicit per-turn presentation evidence required')
        self._presentation = presentation
        self.root = Path(root).absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError('nonsymlink root required')
        if not self.root.is_dir():
            raise ValueError('existing root required')
        os.chmod(self.root, 0o700)
        self.startup_seconds = startup_seconds
        self._active_started = False
        self.media_root = self.root / 'media'
        self.output_status = output_status
        self.max_seconds = max_seconds
        self.client_ready = threading.Event()
        self.playback_done = threading.Event()
        self._lock = threading.RLock()
        self.stream_state = 'waiting'
        self._counts = {}
        self._http_counts = {}
        self._events = 0
        self._connections = {}
        self._server = self._thread = None
        self._closed = False
        self._baseline = None

    @property
    def presentation(self):
        """Per-turn presentation evidence, or None when this route omits it."""
        return self._presentation

    def note_turn_submitted(self, epoch, media_seconds):
        """Owner declaration: every AV submission for one turn is in the stream.

        This only records the submitted media end. Presentation completion still
        requires client-reported playback progress reaching that end.
        """
        if self._presentation is None:
            raise RuntimeError('presentation evidence not enabled')
        with self._lock:
            if self._closed or self.stream_state in ('failed', 'ended'):
                raise RuntimeError('server not accepting turn submissions')
            try:
                self._presentation.note_submitted(epoch, media_seconds)
            except ValueError:
                self.finish('failed')
                raise

    def _status(self):
        # Injected parent must supply a JSON-safe, readonly output contract.
        value = self.output_status()
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        if len(encoded.encode()) > 16384 or not isinstance(value, dict):
            raise ValueError('bounded output status required')
        return value, encoded

    def validate_output(self):
        if not self.client_ready.is_set() or self.stream_state in ('failed', 'ended'):
            raise RuntimeError('browser not ready')
        if self._closed or self._server is None or time.monotonic() >= self.deadline:
            raise RuntimeError('server unavailable')
        try:
            value, encoded = self._status()
            if value.get('default_is_builtin_speaker') is not True or encoded != self._baseline:
                raise RuntimeError('output status drift')
        except Exception:
            self.finish('failed')
            raise

    def finish(self, state='eos'):
        if state not in ('streaming', 'eos', 'failed'):
            raise ValueError('invalid stream state')
        with self._lock:
            if self.stream_state not in ('failed', 'ended'):
                if state == 'streaming' and not self._active_started:
                    self._active_started = True
                    self.deadline = min(self.hard_deadline, time.monotonic() + self.max_seconds)
                self.stream_state = state

    def _event(self, event):
        if (not isinstance(event, dict) or set(event) - (SCALARS | {'event'})
                or event.get('event') not in EVENTS):
            raise ValueError('invalid event')
        for key, value in event.items():
            if key != 'event' and (type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 <= value <= 1000000):
                raise ValueError('invalid scalar')
        with self._lock:
            kind = event['event']
            terminal = kind in ('ended', 'failed', 'timeout', 'cancelled', 'output_drift')
            if self._events >= (256 if terminal else 255):
                raise ValueError('event limit')
            if kind == 'client_ready':
                if self.client_ready.is_set() or self._closed or self.stream_state != 'waiting':
                    raise ValueError('ready replay or unavailable')
                value, baseline = self._status()
                if value.get('default_is_builtin_speaker') is not True:
                    raise ValueError('builtin output required')
                self._baseline = baseline
            elif not self.client_ready.is_set() and kind not in ('failed', 'timeout', 'cancelled', 'output_drift'):
                raise ValueError('ready required')
            self._events += 1
            kind = event['event']
            self._counts[kind] = self._counts.get(kind, 0) + 1
            if kind == 'client_ready':
                self.client_ready.set()
            elif kind in ('failed', 'timeout', 'cancelled', 'output_drift'):
                self.stream_state = 'failed'
            elif kind == 'ended':
                try:
                    actual_end = b'#EXT-X-ENDLIST' in self._read(self.media_root, 'index.m3u8').splitlines()
                except (OSError, ValueError):
                    actual_end = False
                if self.stream_state not in ('streaming', 'eos') or not actual_end:
                    self.stream_state = 'failed'
                else:
                    self.stream_state = 'ended'
                    self.playback_done.set()
            if self._presentation is not None:
                if kind in ('failed', 'timeout', 'cancelled', 'output_drift'):
                    self._presentation.stop(kind)
                elif kind == 'ended' and self.stream_state == 'ended':
                    self._presentation.stop('ended')
                elif kind != 'client_ready' and 'media_seconds' in event:
                    try:
                        self._presentation.note_playhead(event['media_seconds'])
                    except ValueError:
                        self.stream_state = 'failed'
                        self._presentation.stop('playhead_invalid')
                        raise

    def __enter__(self):
        if self._server is not None or self._closed:
            raise RuntimeError('single-use server')
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(1)
                self.until = time.monotonic() + 1
                with owner._lock:
                    owner._connections[self.connection] = self.until

            def finish(self):
                try:
                    super().finish()
                finally:
                    with owner._lock:
                        owner._connections.pop(self.connection, None)

            def send_response(self, code, message=None):
                with owner._lock:
                    owner._http_counts[str(code)] = owner._http_counts.get(str(code), 0) + 1
                super().send_response(code, message)

            def end_headers(self):
                self.send_header('Content-Security-Policy', owner.csp)
                self.send_header('Permissions-Policy', 'camera=(), microphone=(), display-capture=()')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('Cache-Control', 'no-store')
                super().end_headers()

            def reply(self, code, body=b'', mime='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if self.command != 'HEAD':
                    for offset in range(0, len(body), 65536):
                        left = min(self.until, owner.deadline) - time.monotonic()
                        if left <= 0:
                            raise TimeoutError()
                        self.connection.settimeout(left)
                        self.wfile.write(body[offset:offset+65536])

            def valid_host(self):
                return self.headers.get_all('Host') == [owner.authority]

            def do_POST(self):
                try:
                    if time.monotonic() >= owner.deadline or owner._closed:
                        return self.reply(410)
                    if (not self.valid_host() or self.headers.get_all('Origin') != [owner.origin]
                            or self.path != '/api/event'):
                        return self.reply(403)
                    length = self.headers.get('Content-Length', '')
                    if (self.headers.get('Transfer-Encoding') or not length.isdecimal()
                            or not 0 < int(length) <= 4096):
                        return self.reply(400)
                    owner._event(json.loads(self.rfile.read(int(length))))
                    self.reply(200, b'{}')
                except (ValueError, TypeError, OSError):
                    self.reply(400)

            def do_HEAD(self):
                self.do_GET()

            def do_GET(self):
                try:
                    if time.monotonic() >= owner.deadline or owner._closed:
                        return self.reply(410)
                    if not self.valid_host():
                        return self.reply(403)
                    if self.path == '/api/status':
                        value, _ = owner._status()
                        return self.reply(200, json.dumps(dict(output=value,
                            startup_seconds=owner.startup_seconds, max_seconds=owner.max_seconds,
                            stream_state=('streaming' if owner.stream_state == 'eos'
                                          else owner.stream_state)), allow_nan=False).encode())
                    if self.path == '/':
                        return self.reply(200, owner.html, 'text/html; charset=utf-8')
                    if not re.fullmatch(r'/(index\.m3u8|init\.mp4|seg-[0-9]{3,6}\.m4s)', self.path):
                        return self.reply(404)
                    body = owner._read(owner.media_root, self.path[1:])
                    mime = 'application/vnd.apple.mpegurl' if self.path.endswith('.m3u8') else 'video/mp4'
                    self.reply(200, body, mime)
                except (OSError, ValueError):
                    self.reply(404)

        self.html = self._read(Path(__file__).parent.parent, 'h3_browser_client.html')
        scripts = re.findall(rb'<script\b[^>]*>(.*?)</script\s*>', self.html, re.S | re.I)
        hashes = ' '.join("'sha256-" + base64.b64encode(hashlib.sha256(s).digest()).decode() + "'" for s in scripts)
        self.csp = ("default-src 'none'; script-src 'self' " + hashes +
                    "; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; "
                    "connect-src 'self'; img-src 'self'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.started = time.monotonic()
        self.hard_deadline = self.started + self.startup_seconds + self.max_seconds
        self.deadline = self.hard_deadline
        if self._presentation is not None and self._presentation.hard_deadline > self.hard_deadline:
            raise ValueError('presentation evidence deadline exceeds finite server session')
        self._server = _JoinedServer(('127.0.0.1', 0), Handler)
        self.authority = f'127.0.0.1:{self._server.server_port}'
        self.origin = f'http://{self.authority}'
        self.url = self.origin + '/'

        def run():
            self._server.timeout = .05
            while not self._closed and time.monotonic() < self.deadline:
                self._server.handle_request()
                with self._lock:
                    expired = [sock for sock, end in self._connections.items()
                               if time.monotonic() >= end]
                for sock in expired:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            with self._lock:
                for sock in self._connections:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            if not self._closed:
                self.finish('failed')
            self._server.server_close()

        self._thread = threading.Thread(target=run, name='h3-browser-server')
        self._thread.start()
        return self

    @staticmethod
    def _read(directory, name):
        # Directory FD prevents a swapped media symlink from escaping the root.
        dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
                    raise ValueError('bounded regular file required')
                return os.read(fd, info.st_size)
            finally:
                os.close(fd)
        finally:
            os.close(dfd)

    def close(self):
        self.finish('failed')
        self._closed = True
        if self._thread is not None:
            self._thread.join(2)
        # _Server workers have one-second socket deadlines. Join them explicitly.
        evidence = dict(stream_state=self.stream_state, phases=dict(self._counts),
                        http_statuses=dict(self._http_counts), event_count=self._events)
        if self._presentation is not None:
            evidence['presentation'] = self._presentation.snapshot()
        try:
            fd = os.open(self.root / 'browser-av-evidence.json',
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as out:
                json.dump(evidence, out, allow_nan=False)
        except FileExistsError:
            pass

    def __exit__(self, *exc):
        self.close()


class BrowserAVSink(HeadlessAVSink):
    def __init__(self, root, binding, *, server, playback_approved=False, **kwargs):
        if playback_approved is not True:
            raise PermissionError('explicit playback approval required')
        if kwargs.get('paced', True) is not True:
            raise ValueError('paced input required')
        if not isinstance(server, BrowserAVServer) or not Path(root).resolve().is_relative_to(server.root):
            raise ValueError('sink root must be within borrowed server private root')
        if server.presentation is not None and kwargs.get('discontinuity_origin') not in (None, (0, 0)):
            # A nonzero origin would shift the client playhead against frame counts.
            raise ValueError('per-turn presentation evidence requires a zero span origin')
        server.validate_output()  # Must precede parent construction / encoder factory.
        self.server = server
        self.hls_root = server.media_root
        self.hls_root.mkdir(mode=0o700)
        self.playlist = self.hls_root / 'index.m3u8'
        self.media_seconds_end = None
        kwargs['paced'] = True
        kwargs.setdefault('max_frames', 124)
        super().__init__(root, binding, **kwargs)
        server.finish('streaming')

    def _encoder_tail(self):
        self.server.validate_output()
        return hls_arguments(self.hls_root)

    def accept(self, *args):
        try:
            self.server.validate_output()
            submission = super().accept(*args)
        except Exception:
            self.close()
            raise
        # Video timeline end of everything submitted so far; declared per turn by
        # the owner through mark_turn_submitted(). Not a presentation claim.
        self.media_seconds_end = submission.frame_stop / 24
        return submission

    def mark_turn_submitted(self, epoch):
        """Declare that this turn's last AV submission has been accepted."""
        if self.server.presentation is None:
            raise RuntimeError('presentation evidence not enabled')
        if self.media_seconds_end is None:
            raise RuntimeError('no submitted media to declare')
        self.server.note_turn_submitted(epoch, self.media_seconds_end)

    def _validate_output(self, deadline):
        if b'#EXT-X-ENDLIST' not in self.server._read(self.hls_root, 'index.m3u8'):
            raise RuntimeError('real encoder EOS required')
        self.server.finish('eos')
        deadline = min(deadline, self.server.deadline)
        while not self.server.playback_done.wait(.02):
            if self.server.stream_state == 'failed' or time.monotonic() >= deadline:
                raise RuntimeError('browser natural end deadline')
        return self.playlist

    def close(self):
        if not getattr(self, 'finished', False):
            self.server.finish('failed')
        if hasattr(self, '_close_lock'):
            super().close()
        # Server and media are owned by the parent through final browser ended.
