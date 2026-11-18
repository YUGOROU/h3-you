"""Explicitly claimed, RAM-only HTTP byte transport for paired_av_pump.

API: SocketAdapter(base_url, session=..., not_after=..., role=...,
role_key=..., hf_token=..., deadline=None, http_timeout=2.0,
allow_test_loopback=False); open_claim() (alias open), send, recv,
settimeout, shutdown, close, abort, send_input, read_input.

send acknowledges local bounded enqueue, NOT relay delivery or peer ACK.
close/shutdown drain the owned writer; only explicit abort cancels the relay.
No reconnect/replay, proxy environment, redirects, devices or deployment hooks.
One caller per read lane; input helpers are synchronous, not socket streams.
"""
import collections
import http.client
import io
import json
import math
import re
import threading
import time
from urllib.parse import urlsplit

MAX_BODY = 524288
MAX_QUEUE = 1048576


class RelayError(OSError):
    """Terminal relay failure (messages never contain credentials)."""


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('invalid ' + name)
    return float(value)


class _DeadlineReader(io.RawIOBase):
    def __init__(self,sock,raw,remaining):
        self._sock=sock;self._raw=raw;self._remaining=remaining
    def readable(self):return True
    def readinto(self,buffer):
        self._sock.settimeout(self._remaining())
        return self._raw.readinto(buffer)
    def close(self):
        try:self._raw.close()
        finally:super().close()


class _DeadlineSocket:
    def __init__(self,sock,remaining):self._sock=sock;self.remaining=remaining
    def __getattr__(self,name):return getattr(self._sock,name)
    def makefile(self,mode):
        if mode!='rb':raise RelayError('unsupported HTTP stream mode')
        raw=self._sock.makefile('rb',buffering=0)
        return io.BufferedReader(_DeadlineReader(self._sock,raw,lambda:self.remaining()))


class SocketAdapter:
    def __init__(self, base_url, *, session, not_after, role, role_key,
                 hf_token, deadline=None, http_timeout=10.0,
                 allow_test_loopback=False, endpoint_permit=None):
        if type(base_url) is not str or type(allow_test_loopback) is not bool:
            raise ValueError('invalid relay URL')
        try:
            u = urlsplit(base_url)
            port = u.port
            loopback = (allow_test_loopback and u.scheme == 'http' and
                        u.hostname == '127.0.0.1' and port is not None)
            public = (u.scheme == 'https' and u.hostname is not None and
                      (re.fullmatch(r'[a-z0-9-]+\.hf\.space', u.hostname) or
                       re.fullmatch(r'[a-f0-9]{24}--7860\.hf\.jobs',u.hostname)) and
                      port in (None, 443))
            if (not (loopback or public) or u.username is not None or
                    u.password is not None or u.path or u.query or u.fragment or
                    '?' in base_url or '#' in base_url or '\\' in base_url):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('invalid relay URL') from None
        if type(session) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session):
            raise ValueError('invalid session')
        if type(role) is not str or role not in ('producer', 'consumer'):
            raise ValueError('invalid role')
        if type(role_key) is not str or not re.fullmatch(r'[0-9a-fA-F]{64}', role_key):
            raise ValueError('invalid role key')
        if (type(hf_token) is not str or not hf_token or
                any(ord(c) < 33 or ord(c) > 126 for c in hf_token)):
            raise ValueError('invalid token')
        self._not_after = _number(not_after, 'not_after')
        self._http_timeout = _number(http_timeout, 'HTTP timeout')
        self._wall = time.time()
        self._mono = time.monotonic()
        end = self._mono + self._not_after - self._wall
        self._deadline = min(end, _number(deadline, 'deadline')) if deadline is not None else end
        if self._deadline <= self._mono:
            raise ValueError('expired relay')
        self._host, self._port, self._tls = u.hostname, port, not loopback
        self._attested=loopback
        self._base_url=base_url;self._permit=endpoint_permit
        self._session, self._role = session, role
        self._key, self._token = role_key, hf_token
        self._timeout = self._deadline-self._mono
        self._cv = threading.Condition()
        self._queue = collections.deque()
        self._queued = 0
        self._cache = bytearray()
        self._connections = {}
        self._locks = {name: threading.Lock() for name in ('claim', 'write', 'read', 'input', 'abort')}
        self._boot = None
        self._attempted = self._opened = self._closing = self._closed = False
        self._error = None
        self._writer = None
        self.last_failure=None

    def __repr__(self):
        return '<SocketAdapter credentials=REDACTED>'

    def _remaining(self):
        now = time.monotonic()
        wall = time.time()
        # Reject clock discontinuity rather than extending a lease in either direction.
        if abs((wall - self._wall) - (now - self._mono)) > 2.0:
            raise RelayError('relay clock drift')
        left = min(self._deadline - now, self._not_after - wall)
        if left <= 0:
            raise TimeoutError('relay deadline expired')
        return left

    def _check(self):
        if self._error is not None:
            raise RelayError('relay transport failed') from None
        if not self._opened or self._closed:
            raise RelayError('relay not open')
        return self._remaining()

    def _fail(self):
        with self._cv:
            self._error = True
            self._cv.notify_all()

    def _request(self, lane, method, path, body=b'', *, operation_end=None):
        if len(body) > MAX_BODY:
            raise ValueError('relay body too large')
        if not self._attested:raise RelayError('unattested endpoint')
        with self._locks[lane]:
            phase='request';status=None
            started=time.monotonic();timeout=None;headers_elapsed=None
            last_progress=None;content_length=None;bytes_read=0;framing='unknown'
            reject_stage=None;reject_flags=None
            try:
                timeout = min(self._http_timeout, self._timeout, self._remaining())
                if operation_end is not None:
                    timeout = min(timeout, operation_end - time.monotonic())
                if timeout <= 0:
                    raise TimeoutError('relay operation expired')
                conn = self._connections.get(lane)
                if conn is None:
                    cls = http.client.HTTPSConnection if self._tls else http.client.HTTPConnection
                    conn = cls(self._host, self._port, timeout=timeout)
                    self._connections[lane] = conn
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                headers = {'X-H3-Session': self._session, 'X-H3-Role': self._role,
                           'X-H3-Key': self._key, 'Authorization': 'Bearer ' + self._token,
                           'Cache-Control': 'no-store', 'Accept-Encoding': 'identity',
                           'Content-Type': 'application/octet-stream'}
                request_end=time.monotonic()+timeout
                def remaining():
                    left=min(self._remaining(),request_end-time.monotonic())
                    if left<=0:raise TimeoutError('relay HTTP absolute timeout')
                    return left
                conn.request(method, path, body=body if method == 'POST' else None, headers=headers)
                if not isinstance(conn.sock,_DeadlineSocket):conn.sock=_DeadlineSocket(conn.sock,remaining)
                else:conn.sock.remaining=remaining
                phase='response-headers'
                response = conn.getresponse();status=response.status
                headers_elapsed=time.monotonic()-started
                # The relay explains every rejection with these two headers. Keep
                # them so a 400 is attributable to a specific server check instead
                # of a bare status; failure_evidence allowlists them by exact value.
                reject_stage=response.getheader('X-H3-Reject-Stage')
                reject_flags=response.getheader('X-H3-Reject-Flags')
                phase='boot-identity'
                boot = response.getheader('X-H3-Boot', '')
                if not re.fullmatch(r'[0-9a-fA-F]{32}', boot):
                    raise RelayError('missing relay boot identity')
                if self._boot is not None and boot != self._boot:
                    raise RelayError('relay restarted')
                phase='cache-policy'
                if 'no-store' not in [x.strip().lower() for x in response.getheader('Cache-Control', '').split(',')]:
                    raise RelayError('relay cache policy missing')
                phase='session-binding'
                session = response.getheader('X-H3-Session')
                if session is not None and session != self._session:
                    raise RelayError('relay session mismatch')
                phase='content-encoding'
                if response.getheader('Content-Encoding') is not None:
                    raise RelayError('encoded relay response')
                phase='content-length'
                length = response.getheader('Content-Length')
                if length is not None and (not re.fullmatch(r'[0-9]+', length) or int(length) > MAX_BODY):
                    raise RelayError('invalid relay body length')
                content_length=int(length) if length is not None else None
                framing='chunked' if response.chunked else ('content-length' if length is not None else 'eof')
                # Bound cumulative response duration as well as individual socket operations.
                data = bytearray();phase='response-body'
                while True:
                    # HTTPResponse has already consumed an exact framed body. Do not
                    # start another read (or deadline check) solely to discover EOF.
                    if response.isclosed():
                        break
                    left = min(self._remaining(), request_end - time.monotonic())
                    if left <= 0:
                        raise TimeoutError('relay HTTP timeout')
                    if conn.sock is not None:
                        conn.sock.settimeout(left)
                    chunk = response.read1(min(65536, MAX_BODY + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    bytes_read=len(data);last_progress=time.monotonic()
                    if len(data) > MAX_BODY:
                        raise RelayError('relay response too large')
                if length is not None and len(data) != int(length):
                    raise RelayError('truncated relay response')
                self._remaining()
                if self._boot is None:
                    self._boot = boot
                return response.status, bytes(data)
            except Exception as exc:
                from .local_av_diagnostics import failure_evidence
                now=time.monotonic()
                self.last_failure=failure_evidence(dict(
                    phase=phase,http_status=status,error_type=type(exc).__name__,
                    lane=lane,method=method,content_length=content_length,
                    bytes_read=bytes_read,elapsed_seconds=now-started,
                    request_timeout_seconds=timeout,headers_elapsed_seconds=headers_elapsed,
                    last_progress_age_seconds=None if last_progress is None else now-last_progress,
                    framing=framing,reject_stage=reject_stage,reject_flags=reject_flags))
                self._fail()
                raise RelayError('relay HTTP operation failed; no retry; phase='+phase+
                                 '; status='+str(status)+'; error='+type(exc).__name__) from None

    def open_claim(self):
        """Claim exactly once. A failed/duplicate claim cannot be retried."""
        with self._cv:
            if self._attempted or self._closed:
                raise RelayError('claim already attempted or closed')
            self._attempted = True
        if self._tls:
            from .live_relay_endpoint import PrivateEndpoint
            from .job_av_endpoint import JobEndpoint
            if type(self._permit) not in (PrivateEndpoint,JobEndpoint) or not self._permit.permits(self._base_url):
                self._fail()
                raise RelayError('private pinned HF endpoint attestation required')
            self._attested=True
        status, raw = self._request('claim', 'POST', '/v1/claim')
        try:
            obj = json.loads(raw)
            if (status != 200 or type(obj) is not dict or
                    obj.get('schema') != 'h3-live-relay-v1' or
                    obj.get('session') != self._session or
                    obj.get('boot_id') != self._boot or
                    type(obj.get('not_after')) not in (int, float) or
                    obj['not_after'] != self._not_after):
                raise ValueError
        except (ValueError, TypeError):
            self.last_failure=dict(phase='claim-schema',http_status=status,error_type='InvalidClaim')
            self._fail()
            raise RelayError('invalid relay claim') from None
        with self._cv:
            self._opened = True
            self._writer = threading.Thread(target=self._write_loop, name='h3-relay-writer')
            self._writer.start()
        return self

    open = open_claim

    def settimeout(self, value):
        value = _number(value, 'timeout')
        with self._cv:
            self._timeout = min(value, self._remaining())

    def send(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('bytes required')
        with self._cv:
            self._check()
            if self._closing:
                raise RelayError('relay closing')
            end = time.monotonic() + min(self._timeout, self._remaining())
            while self._queued >= MAX_QUEUE:
                left = min(end - time.monotonic(), self._check())
                if left <= 0:
                    raise TimeoutError('relay enqueue timeout')
                self._cv.wait(left)
                if self._closing:
                    raise RelayError('relay closing')
            n = min(len(data), MAX_QUEUE - self._queued)
            if n:
                self._queue.append(bytes(data[:n]))
                self._queued += n
                self._cv.notify_all()
            return n

    def _write_loop(self):
        path = '/v1/av' if self._role == 'producer' else '/v1/ack'
        try:
            while True:
                with self._cv:
                    while not self._queue and not self._closing:
                        self._check()
                        self._cv.wait(min(.1, self._remaining()))
                    self._check()
                    if not self._queue:
                        return
                    end = time.monotonic() + .005
                    while self._queued < MAX_BODY and not self._closing:
                        left = min(end - time.monotonic(), self._remaining())
                        if left <= 0:
                            break
                        self._cv.wait(left)
                    parts = []
                    size = 0
                    while self._queue and size < MAX_BODY:
                        part = self._queue.popleft()
                        take = min(len(part), MAX_BODY - size)
                        parts.append(part[:take])
                        if take < len(part):
                            self._queue.appendleft(part[take:])
                        size += take
                    # In-flight bytes remain charged until HTTP acceptance.
                status, _ = self._request('write', 'POST', path, b''.join(parts))
                if status != 202:
                    raise RelayError('relay write rejected')
                with self._cv:
                    self._queued -= size
                    self._cv.notify_all()
        except Exception:
            self._fail()

    def _read(self, lane, path):
        end = time.monotonic() + min(self._timeout, self._check())
        while True:
            self._check()
            if time.monotonic() >= end:
                raise TimeoutError('relay read timeout')
            status, data = self._request(lane, 'GET', path, operation_end=end)
            self._check()
            if status == 200 and data:
                return data
            if status != 204 or data:
                self._fail()
                raise RelayError('invalid relay read')
            time.sleep(min(.01, max(0, end - time.monotonic()), self._remaining()))

    def recv(self, n):
        if type(n) is not int or n < 0:
            raise ValueError('invalid receive size')
        self._check()
        if n == 0:
            return b''
        if not self._cache:
            self._cache.extend(self._read('read', '/v1/ack' if self._role == 'producer' else '/v1/av'))
        data = bytes(self._cache[:n])
        del self._cache[:n]
        return data

    def send_input(self, data):
        self._check()
        if self._role != 'consumer':
            raise RelayError('only consumer writes input')
        if not isinstance(data, bytes) or len(data) > MAX_BODY:
            raise ValueError('bounded bytes required')
        status, _ = self._request('input', 'POST', '/v1/input', data)
        if status != 202:
            self._fail()
            raise RelayError('relay input rejected')
        return len(data)

    def read_input(self):
        self._check()
        if self._role != 'producer':
            raise RelayError('only producer reads input')
        return self._read('input', '/v1/input')

    def abort(self):
        """Explicit failure/cancel only; never invoked by normal close."""
        self._fail()  # Do not drain queued data after an explicit cancellation.
        try:
            if self._attempted and self._attested and not self._closed:
                self._request('abort', 'POST', '/v1/abort')
        except OSError:
            pass
        finally:
            self.close()

    def shutdown(self, how=2):
        self.close()

    def close(self):
        with self._cv:
            if self._closed:
                return
            self._closing = True
            self._cv.notify_all()
        writer = self._writer
        if writer is not None and writer is not threading.current_thread():
            writer.join()  # Every request and queue drain is bounded by the fixed lease.
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        # Reader operations have finite HTTP timeouts; close each only when not in use.
        for lane, lock in self._locks.items():
            with lock:
                conn = self._connections.get(lane)
                if conn is not None:
                    conn.close()
        with self._cv:
            self._queue.clear();self._cache.clear();self._queued=0
        self._connections.clear()
        self._key = self._token = ''
