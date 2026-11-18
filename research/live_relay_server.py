"""Explicit-run HTTP carrier; production MUST use a private TLS HF gateway.

No TLS, deployment, persistence, device access, or automatic retries are provided.
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from research.h3_personal.live_relay import Relay, RelayConfig, RelayError, MAX_CHUNK


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, config, *, keepalive=False, continuous_av=False, rtc_gateway=None):
        self.rtc_gateway = rtc_gateway
        if type(continuous_av) is not bool:
            raise ValueError('explicit continuous_av bool required')
        self.continuous_av = continuous_av
        self._av_stream_lock = threading.Lock()
        if type(keepalive) is not bool:
            raise ValueError('explicit keepalive bool required')
        self.keepalive = keepalive
        self._timing_lock = threading.Lock()
        self._timing = dict(connections=0,responses=0,body_bytes=0,
                            write_seconds=0.0,max_write_seconds=0.0,write_failures=0)
        self.relay = Relay(config)
        self._slots = threading.BoundedSemaphore(8)
        super().__init__(address, _Handler)

    def timing(self):
        with self._timing_lock:
            return dict(self._timing)

    def process_request(self, request, client_address):
        with self._timing_lock:self._timing['connections']+=1
        if not self._slots.acquire(False):
            try:
                request.settimeout(0.1)
                request.sendall(('HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n'
                                 'Cache-Control: no-store\r\nX-H3-Boot: ' + self.relay.boot_id +
                                 '\r\nConnection: close\r\n\r\n').encode())
            except OSError:
                pass
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

    def service_actions(self):
        try:self.relay.status()
        except RelayError:pass  # Expire RAM even when both clients disappear.

    def handle_error(self, request, client_address):
        pass  # Never log exceptions, request bodies, credentials, or raw input.

    def server_close(self):
        self.relay.abort()
        super().server_close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        self.request.settimeout(self.server.relay.config.body_timeout)
        super().setup()

    def log_message(self, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        self._reply(code)

    def _reply(self, code, body=b'', content_type='application/octet-stream',reject_stage=None):
        self._response_count=getattr(self,'_response_count',0)+1
        reuse=(self.server.keepalive and getattr(self,'_authenticated_request',False)
               and code in (200,202,204) and self.path!='/v1/abort'
               and self.request_version=='HTTP/1.1' and self._response_count<32
               and self.headers.get('Connection','').lower()!='close')
        self.close_connection = not reuse
        self.send_response_only(code)
        self.send_header('X-H3-Boot', self.server.relay.boot_id)
        self.send_header('Cache-Control', 'no-store')
        if reject_stage is not None:
            self.send_header('X-H3-Reject-Stage',reject_stage)
            self.send_header('X-H3-Reject-Flags',getattr(self,'_reject_flags','0'))
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Type', content_type)
        self.send_header('Connection', 'keep-alive' if reuse else 'close')
        started=time.monotonic()
        try:
            self.end_headers()
            if body:self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            with self.server._timing_lock:self.server._timing['write_failures']+=1
            raise
        else:
            elapsed=time.monotonic()-started
            with self.server._timing_lock:
                stats=self.server._timing
                stats['responses']+=1;stats['body_bytes']+=len(body)
                stats['write_seconds']+=elapsed
                stats['max_write_seconds']=max(stats['max_write_seconds'],elapsed)
        if reuse:
            # A POST may have shortened its socket deadline; reset for the next
            # request, but never renew the finite session lease.
            remaining=min(self.server.relay.config.body_timeout,
                          self.server.relay._deadline-time.monotonic())
            if remaining<=0:self.close_connection=True
            else:self.connection.settimeout(remaining)

    def _av_stream(self):
        relay = self.server.relay
        # A drained stream must never be reopened, even after clean disconnect.
        if not self.server._av_stream_lock.acquire(False):
            raise RelayError(409)
        self.close_connection = True
        self.send_response_only(200)
        self.send_header('X-H3-Boot', relay.boot_id)
        self.send_header('X-H3-Session', relay.config.session)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Connection', 'close')
        drained = False
        try:
            self.end_headers()
            self.wfile.flush()
            with self.server._timing_lock:self.server._timing['responses']+=1
            while True:
                try:
                    body = relay.take('av')
                except RelayError:
                    break
                if not body:
                    # No heartbeat bytes: keep waiting inside the original lease.
                    if relay.config.wait_timeout == 0:
                        time.sleep(min(.01, max(0, relay._deadline-time.monotonic())))
                    continue
                drained = True
                self.connection.settimeout(max(.001, min(relay.config.body_timeout,
                    relay._deadline-time.monotonic(), relay.config.not_after-time.time())))
                started=time.monotonic()
                self.wfile.write(('%x\r\n' % len(body)).encode() + body + b'\r\n')
                self.wfile.flush()
                elapsed=time.monotonic()-started
                with self.server._timing_lock:
                    stats=self.server._timing
                    stats['body_bytes']+=len(body);stats['write_seconds']+=elapsed
                    stats['max_write_seconds']=max(stats['max_write_seconds'],elapsed)
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
        except OSError:
            with self.server._timing_lock:self.server._timing['write_failures']+=1
            if drained:
                relay.abort()
            # Headers already sent: never append a second HTTP response.

    def handle_expect_100(self):
        # Do not invite an unauthenticated body or support two-phase framing.
        self._reply(417)
        return False

    def _dispatch(self):
        if self.server.rtc_gateway is not None:
            if self.path.startswith('/rtc/'):
                return self.server.rtc_gateway.handle(self)
            if self.path.split('?', 1)[0] in ('/v1/av', '/v1/av-stream', '/v1/ack'):
                return self._reply(404, b'{}', 'application/json')
        relay = self.server.relay
        self._authenticated_request=False
        ticket = None
        drained = False
        stage='path'
        try:
            if '?' in self.path or '#' in self.path or not self.path.startswith('/'):
                raise RelayError(400)
            stage='headers-duplicate'
            critical=('Content-Length','Content-Type','Authorization','X-H3-Session','X-H3-Role','X-H3-Key')
            self._reject_flags=str(sum(1<<i for i,k in enumerate(critical) if len(self.headers.get_all(k,[]))>1))
            if any(len(self.headers.get_all(k,[]))>1 for k in critical):
                raise RelayError(400)
            stage='headers-framing'
            framing=('Transfer-Encoding','Expect','Trailer','Upgrade')
            self._reject_flags=str(sum(1<<i for i,k in enumerate(framing) if k in self.headers))
            if any(k in self.headers for k in framing):
                raise RelayError(400)
            stage='length'
            length_text = self.headers.get('Content-Length', '0')
            if not length_text.isascii() or not length_text.isdecimal():
                raise RelayError(400)
            length = int(length_text)
            if self.path in ('/','/health') and self.command == 'GET':
                if length:
                    raise RelayError(400)
                self._reply(200, b'{"status":"ok"}', 'application/json')
                return
            if not self.path.startswith('/v1/'):
                raise RelayError(404)
            stage='route-auth'
            role = relay.authenticate(self.headers.get('X-H3-Session'),
                                      self.headers.get('X-H3-Role'), self.headers.get('X-H3-Key'))
            self._authenticated_request=True
            route = self.path[4:]
            if route in ('claim', 'abort', 'status'):
                if self.command != ('GET' if route == 'status' else 'POST'):
                    raise RelayError(405)
                if length:
                    raise RelayError(400)
                if route == 'claim':
                    relay.claim(role)
                    self._reply(200,json.dumps(relay.status()).encode(),'application/json')
                    return
                elif route == 'abort':
                    relay.abort()
                else:
                    self._reply(200, json.dumps(relay.status()).encode(), 'application/json')
                    return
                self._reply(202)
                return
            if route == 'av-stream':
                if not self.server.continuous_av:
                    raise RelayError(404)
                if self.command != 'GET':
                    raise RelayError(405)
                if role != 'consumer':
                    raise RelayError(403)
                relay.claimed(role)
                if length:
                    raise RelayError(400)
                self._av_stream()
                return
            if route not in ('av', 'ack', 'input'):
                raise RelayError(404)
            if self.command not in ('GET', 'POST'):
                raise RelayError(405)
            writer = 'producer' if route == 'av' else 'consumer'
            if role != (writer if self.command == 'POST' else ('consumer' if writer == 'producer' else 'producer')):
                raise RelayError(403)
            relay.claimed(role)
            if self.command == 'GET':
                if length:
                    raise RelayError(400)
                body = relay.take(route)
                drained = bool(body)
                self._reply(200 if body else 204, body)
                return
            if self.headers.get('Content-Type') != 'application/octet-stream':
                raise RelayError(415)
            if 'Content-Length' not in self.headers:
                raise RelayError(411)
            if not 0 < length <= MAX_CHUNK:
                raise RelayError(413)
            ticket = relay.reserve(route, length)
            # Absolute read deadline as well as socket inactivity timeout: slow
            # trickles cannot renew a request indefinitely. read1 avoids read(n)
            # internally looping with a fresh socket timeout per fragment.
            import time
            end = time.monotonic() + relay.config.body_timeout
            body = bytearray()
            while len(body) < length:
                remaining = min(end-time.monotonic(),relay._deadline-time.monotonic(),
                                relay.config.not_after-time.time())
                if remaining <= 0:
                    raise TimeoutError()
                self.connection.settimeout(remaining)
                chunk = self.rfile.read1(length - len(body))
                if not chunk:
                    raise EOFError()
                body.extend(chunk)
            relay.commit(ticket, body)
            ticket = None
            self._reply(202)
        except RelayError as exc:
            self._reply(exc.status,reject_stage=stage)
        except (OSError, EOFError):
            if ticket is not None or drained:
                relay.abort()  # Uncertain partial transfer is terminal, never replay.
            try:
                self._reply(410 if ticket is not None else 400)
            except OSError:
                pass

    do_GET = _dispatch
    do_POST = _dispatch
    do_PUT = _dispatch
    do_DELETE = _dispatch
    do_PATCH = _dispatch
    do_HEAD = _dispatch
    do_OPTIONS = _dispatch


def build_server(address, config, *, keepalive=False, continuous_av=False, rtc_gateway=None):
    """Unstarted finite carrier; optional authenticated 32-response keepalive."""
    return _Server(address, config, keepalive=keepalive, continuous_av=continuous_av,
                   **({'rtc_gateway': rtc_gateway} if rtc_gateway is not None else {}))


def main():
    if os.environ.get('H3_RELAY_RUN') != '1':
        raise SystemExit('Explicit H3_RELAY_RUN=1 required; private TLS gateway required.')
    with build_server(('0.0.0.0', 7860), RelayConfig.from_env()) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
