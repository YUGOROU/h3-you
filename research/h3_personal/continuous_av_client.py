"""Experimental one-shot continuous AV download; ACK/input remain base HTTP lanes.

Selecting ContinuousConsumerAdapter is the explicit client opt-in. No reconnect,
replay, deployment or fallback. The server must separately enable continuous_av.
"""
import http.client
import math
import socket
import threading
import time

from .live_relay_client import SocketAdapter, RelayError, _DeadlineSocket
from .local_av_diagnostics import failure_evidence


class ContinuousConsumerAdapter(SocketAdapter):
    def __init__(self, *args, stream_progress_timeout=10.0, **kwargs):
        if (type(stream_progress_timeout) not in (int, float)
                or not math.isfinite(stream_progress_timeout)
                or not 0 < stream_progress_timeout <= 120):
            raise ValueError('finite continuous progress timeout within0..120 required')
        if kwargs.get('role') != 'consumer':
            raise ValueError('continuous carrier requires consumer role')
        super().__init__(*args, **kwargs)
        self._stream_progress_timeout = min(float(stream_progress_timeout), self._http_timeout, self._timeout)
        self._stream_conn = None
        self._stream_socket = None
        self._stream_response = None
        self._stream_attempted = False
        self._stream_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._progress_end = None
        self._stream_bytes = 0

    def _stream_remaining(self):
        if self._stream_stop.is_set():
            raise RelayError('continuous stream closed')
        left = min(self._remaining(), self._progress_end-time.monotonic())
        if left <= 0:
            raise TimeoutError('continuous stream progress timeout')
        return left

    def _start_stream(self):
        if self._stream_attempted:
            raise RelayError('continuous stream cannot replay')
        self._stream_attempted = True
        self._progress_end = time.monotonic() + self._stream_progress_timeout
        cls = http.client.HTTPSConnection if self._tls else http.client.HTTPConnection
        conn = cls(self._host, self._port, timeout=self._stream_remaining())
        self._stream_conn = conn
        conn.request('GET', '/v1/av-stream', headers={
            'Authorization': 'Bearer ' + self._token, 'X-H3-Session': self._session,
            'X-H3-Role': self._role, 'X-H3-Key': self._key,
            'Content-Length': '0', 'Accept-Encoding': 'identity', 'Cache-Control': 'no-store'})
        self._stream_socket = conn.sock
        conn.sock = _DeadlineSocket(conn.sock, self._stream_remaining)
        response = conn.getresponse()
        self._stream_response = response
        for key in ('X-H3-Boot', 'X-H3-Session', 'Transfer-Encoding', 'Cache-Control'):
            if len(response.headers.get_all(key, [])) != 1:
                raise RelayError('invalid continuous headers')
        if (response.status != 200 or response.getheader('X-H3-Boot') != self._boot
                or response.getheader('X-H3-Session') != self._session
                or response.getheader('Transfer-Encoding').lower() != 'chunked'
                or not response.chunked
                or response.getheader('Content-Length') is not None
                or response.getheader('Content-Encoding') is not None
                or 'no-store' not in [x.strip().lower() for x in
                    response.getheader('Cache-Control', '').split(',')]):
            raise RelayError('invalid continuous response')
        # Headers are not AV progress. The initial absolute deadline remains.

    def recv(self, n):
        if type(n) is not int or n < 0:
            raise ValueError('invalid receive size')
        self._check()
        if not n:
            return b''
        with self._stream_lock:
            phase = 'response-headers' if self._stream_response is None else 'response-body'
            started = time.monotonic()
            try:
                if self._stream_response is None:
                    self._start_stream()
                phase = 'response-body'
                self._stream_remaining()
                data = self._stream_response.read1(min(n, 65536))
                if not data:
                    raise RelayError('continuous stream ended; no replay')
                self._stream_bytes += len(data)
                self._progress_end = time.monotonic() + self._stream_progress_timeout
                return data
            except Exception as exc:
                self.last_failure = failure_evidence(dict(phase=phase, lane='read',
                    method='GET', error_type=type(exc).__name__, framing='chunked',
                    http_status=None if self._stream_response is None else self._stream_response.status,
                    stream_bytes=self._stream_bytes, elapsed_seconds=time.monotonic()-started,
                    request_timeout_seconds=self._stream_progress_timeout))
                self._fail()
                self._interrupt_stream()
                raise RelayError('continuous AV failed; no retry') from None

    def _interrupt_stream(self):
        self._stream_stop.set()
        sock = self._stream_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # Do not close HTTPResponse while another thread is blocked in its reader.

    def close(self):
        self._interrupt_stream()
        with self._stream_lock:
            if self._stream_response is not None:
                self._stream_response.close()
            if self._stream_conn is not None:
                self._stream_conn.close()
        super().close()

    def abort(self):
        self._interrupt_stream()
        super().abort()
