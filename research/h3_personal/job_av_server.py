"""In-Job RAM transport; expose the owned HTTP port through an authenticated gateway.

No gateway provisioning, credentials, external connection, or model startup here.
"""
import math
import json
import threading
import time

from research.live_relay_server import build_server
from .live_relay import MAX_CHUNK, RelayError


class LocalProducer:
    """Socket-shaped producer sharing the HTTP consumer's finite Relay queues."""

    def __init__(self, relay):
        self.relay = relay
        self._timeout = max(0, relay._deadline - time.monotonic())
        self._cache = bytearray()
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = False
        relay.claim('producer')  # Exactly one owner, never reclaim/replay.

    def settimeout(self, value):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive timeout required')
        self._timeout = value

    def _check(self):
        self.relay._check()  # Caller holds the Relay condition.
        if self._closed:
            raise RelayError(410)

    def _end(self):
        return min(time.monotonic() + self._timeout, self.relay._deadline)

    def send(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('bytes required')
        with self._write_lock:
            end = self._end()
            try:
                with self.relay._cv:
                    self._check()
                    size = min(len(data), MAX_CHUNK, self.relay.config.av_capacity)
                    if not size:
                        return 0
                    while True:
                        self._check()
                        if time.monotonic() >= end:
                            raise TimeoutError('local send deadline')
                        try:
                            ticket = self.relay.reserve('av', size)
                            break
                        except RelayError as exc:
                            if exc.status != 429:
                                raise
                            self.relay._cv.wait(min(.01, end - time.monotonic()))
                    self.relay.commit(ticket, bytes(data[:size]))
                    return size
            except BaseException:
                # A caller cannot safely infer whether a partial frame was delivered.
                self.abort()
                raise

    def _read(self, channel):
        end = self._end()
        with self.relay._cv:
            while True:
                self._check()
                if time.monotonic() >= end:
                    raise TimeoutError('local read deadline')
                if self.relay._queues[channel]:
                    result = self.relay.take(channel)
                    self.relay._cv.notify_all()
                    return result
                # Empty polling is starvation, never socket EOF/source EOS.
                self.relay._cv.wait(min(.05, end - time.monotonic()))

    def recv(self, n):
        if type(n) is not int or n < 0:
            raise ValueError('invalid receive size')
        with self._read_lock:
            with self.relay._cv:
                self._check()
            if n == 0:
                return b''
            if not self._cache:
                self._cache.extend(self._read('ack'))
            result = bytes(self._cache[:n])
            del self._cache[:n]
            return result

    def read_input(self):
        return self._read('input')

    def abort(self):
        with self.relay._cv:
            self._closed = True
            self._cache.clear()
            self.relay.abort()

    def shutdown(self, how=2):
        self.close()

    def close(self):
        self.abort()


class JobAVServer:
    """Explicit port, bounded lease, one owned HTTP serving thread.

    Local transport needs no HTTP credentials/token. Consumer authentication and
    strict HTTP framing remain exactly those of live_relay_server.
    """

    def __init__(self, config, *, port, keepalive=False, continuous_av=False, rtc_gateway=None):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('explicit port required')
        if not 0 < config.not_after - time.time() <= (10800 if config.long_session else 1800):
            raise ValueError('finite unexpired lease required')
        if type(keepalive) is not bool:
            raise ValueError('explicit keepalive bool required')
        if type(continuous_av) is not bool:
            raise ValueError('explicit continuous_av bool required')
        self.server = build_server(('0.0.0.0', port), config,
                                   **({'keepalive': True} if keepalive else {}),
                                   **({'continuous_av': True} if continuous_av else {}),
                                   **({'rtc_gateway': rtc_gateway} if rtc_gateway is not None else {}))
        try:
            self.producer = LocalProducer(self.server.relay)
        except BaseException:
            self.server.server_close()
            raise
        self._thread = None
        self._closed = False

    def __enter__(self):
        if self._thread is not None or self._closed:
            raise RuntimeError('server cannot restart')
        self._thread = threading.Thread(
            target=self.server.serve_forever, kwargs={'poll_interval': .05},
            name='h3-job-av-http')
        self._thread.start()
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.producer.abort()
        if self._thread is not None:
            self.server.shutdown()
        self.server.server_close()
        try:
            print(json.dumps(dict(type='h3_http_server_timing', **self.server.timing()),
                             allow_nan=False), flush=True)
        except Exception:
            pass  # Diagnostics must not mask primary failure/cleanup.
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError('HTTP server thread failed to stop')

    def __exit__(self, *exc):
        self.close()
