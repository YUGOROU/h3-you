"""Same-container, loopback-only adapter to the unchanged native Vox Gate.

Closing this adapter is local cancellation, not provider suspension. Call ready(0)
explicitly before serving; native admission and PCM limits remain authoritative.
"""
import http.client
import json
import math
import socket
import threading
import time

from .voxcpm_endpoint_protocol import CONTENT_TYPE, PINS, _exact_metadata, loads


class PrivateGate:
    def __init__(self, *, key, session, boot, expires_at, max_requests=2, port=7861):
        now = time.time()
        if (any(type(v) is not str or not v or not v.isascii()
                or any(ord(c) < 32 or ord(c) == 127 for c in v)
                for v in (key, session, boot))
                or type(expires_at) not in (int, float)
                or not math.isfinite(expires_at) or not now < expires_at <= now + 1200
                or type(max_requests) is not int or not 1 <= max_requests <= 16
                or type(port) is not int or not 1 <= port <= 65535):
            raise ValueError('invalid private gate configuration')
        self.key, self.session, self.boot = key, session, boot
        self.expires_at, self.max_requests, self.port = expires_at, max_requests, port
        self.candidate = self
        self.closed = False
        self._lock = threading.Lock()
        self._active = set()
        self._sockets = set()
        self._ready = False

    def healthy(self):
        return not self.closed and time.time() < self.expires_at

    def close(self):
        with self._lock:
            self.closed = True
            active = tuple(self._active)
            sockets = tuple(self._sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for connection in active:
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()

    def _remaining(self, deadline):
        remaining = min(deadline - time.monotonic(), self.expires_at - time.time())
        if not self.healthy() or remaining <= 0:
            raise ValueError('private gate unavailable')
        return remaining

    def _exchange(self, method, path, body, content_type, limit, write=None):
        deadline = time.monotonic() + min(120, self.expires_at - time.time())
        connection = http.client.HTTPConnection('127.0.0.1', self.port,
                                               timeout=self._remaining(deadline))
        sock = None
        try:
            with self._lock:
                if not self.healthy():
                    raise ValueError()
                self._active.add(connection)
            # http.client never consults proxy environment or follows redirects.
            connection.connect()
            with self._lock:
                if not self.healthy():
                    raise ValueError()
            connection.sock.settimeout(self._remaining(deadline))
            headers = {'X-H3-TTS-Key': self.key, 'Connection': 'close'}
            if body is not None:
                headers['Content-Type'] = 'application/json'
            connection.request(method, path, body=body, headers=headers)
            connection.sock.settimeout(self._remaining(deadline))
            response = connection.getresponse()
            if (response.status != 200
                    or response.headers.get_all('Content-Type') != [content_type]
                    or response.headers.get_all('Content-Encoding')
                    or response.headers.get_all('Transfer-Encoding')
                    or write is not None and response.headers.get_all('Content-Length')):
                raise ValueError()
            # Keep the socket reference: HTTPConnection detaches it for close responses.
            sock = response.fp.raw._sock
            with self._lock:
                self._sockets.add(sock)
                if not self.healthy():
                    response.close()
                    raise ValueError()
            total = 0
            parts = []
            try:
                while True:
                    if response.length == 0:
                        break  # HTTP Content-Length completion is framed EOF.
                    sock.settimeout(self._remaining(deadline))
                    data = response.read1(min(65536, limit + 1 - total))
                    self._remaining(deadline)
                    if not data:
                        if response.length not in (None, 0):
                            raise ValueError()
                        break
                    total += len(data)
                    if total > limit:
                        raise ValueError()
                    if write is None:
                        parts.append(data)
                    else:
                        write(data)
                return b''.join(parts)
            finally:
                response.close()
        except Exception:
            self.close()
            raise ValueError('private gate operation failed') from None
        finally:
            connection.close()
            with self._lock:
                self._active.discard(connection)
                self._sockets.discard(sock)

    def _binding(self):
        return dict(PINS, session=self.session, boot=self.boot,
                    expires_at=self.expires_at, epoch=0)

    def ready(self, epoch):
        try:
            if type(epoch) is not int or epoch != 0:
                raise ValueError()
            value = loads(self._exchange('GET', '/ready', None, 'application/json', 1024*1024))
            if not _exact_metadata(value, self._binding()):
                raise ValueError()
            self._ready = True
            return value
        except Exception:
            self.close()
            raise ValueError('private gate readiness failed') from None

    def stream(self, request, write):
        try:
            if (not self._ready or type(request) is not dict
                    or set(request) != {'session', 'boot', 'epoch', 'text'}
                    or request['session'] != self.session or request['boot'] != self.boot
                    or type(request['epoch']) is not int or request['epoch'] <= 0
                    or type(request['text']) is not str or not request['text'].strip()
                    or len(request['text']) > 160):
                raise ValueError()
            body = json.dumps(request, allow_nan=False, separators=(',', ':')).encode()
            self._exchange('POST', '/stream', body, CONTENT_TYPE, 8*1024*1024, write)
        except Exception:
            self.close()
            raise ValueError('private gate stream failed') from None

    def export_evidence(self):
        try:
            if not self._ready:
                raise ValueError()
            value = loads(self._exchange('GET', '/evidence', None, 'application/json', 1024*1024))
            if (type(value) is not dict or set(value) != {'binding', 'evidence'}
                    or not _exact_metadata(value['binding'], self._binding())):
                raise ValueError()
            return value['evidence']
        except Exception:
            self.close()
            raise ValueError('private gate evidence failed') from None
