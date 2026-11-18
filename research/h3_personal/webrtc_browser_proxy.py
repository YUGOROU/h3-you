"""Private finite loopback signaling; no media proxy or browser/device operations.

API: with BrowserProxy(binding, hf_token=..., app_key=...,
owner_attested=True, turn_config=None) as proxy: use proxy.url privately.
Binding requires job_id/session/boot/expires_at/ice_profile. Owner MUST attest
ownership before construction. receipts is a bounded, sanitized in-memory snapshot.
No logging. HTTP tests are not actual browser/playout acceptance.
"""
from collections import deque
from dataclasses import asdict
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time
from types import MappingProxyType

from . import webrtc_connect_client
from .webrtc_ice import ice_urls, HF_CLOUDFLARE_TURN_TLS
from .webrtc_turn import validate_turn_config
from .paired_av import LiveAVBinding


class BrowserProxy:
    def __init__(self, binding, *, hf_token, app_key, owner_attested=False, turn_config=None, media_binding=None, signaling_path='/offer'):
        b = dict(binding)
        if (owner_attested is not True or set(b) != {'job_id', 'session', 'boot', 'expires_at', 'ice_profile'}
                or not re.fullmatch('[0-9a-f]{24}', b['job_id'])
                or any(not isinstance(b[k], str) or not 1 <= len(b[k]) <= 128 for k in ('session', 'boot'))
                or type(b['expires_at']) not in (int, float) or not math.isfinite(b['expires_at'])
                or not 0 < b['expires_at'] - time.time() <= 1800
                or not hf_token or not app_key):
            raise ValueError('finite owner-attested binding required')
        if signaling_path not in ('/offer','/rtc/offer'):raise ValueError('fixed signaling route required')
        self._signaling_path=signaling_path
        urls = ice_urls(b['ice_profile'])
        if b['ice_profile'] == HF_CLOUDFLARE_TURN_TLS:
            servers = [validate_turn_config(turn_config)]
        else:
            if turn_config is not None:
                raise ValueError('unexpected TURN')
            servers = [{'urls': u} for u in urls]
        self.binding = MappingProxyType(b)
        if media_binding is not None:
            if type(media_binding) is not LiveAVBinding or media_binding.session != b['session']:
                raise ValueError('exact source media binding required')
            media_binding.__post_init__()
        self._config = dict(binding=b, iceServers=servers,
                            media_binding=None if media_binding is None else asdict(media_binding))
        self._token, self._key = hf_token, app_key
        self._cap = secrets.token_urlsafe(32)
        self._events = deque(maxlen=64)
        self._lock = threading.Lock()
        self._offer_used = False
        self._close_lock = threading.Lock()
        self._closed = False
        self._slots = threading.BoundedSemaphore(4)
        self._deadline = time.monotonic() + b['expires_at'] - time.time()
        owner = self

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False
            def process_request(self, sock, address):
                if not owner._slots.acquire(False):
                    sock.close()
                    return
                sock.settimeout(3)
                super().process_request(sock, address)
            def process_request_thread(self, sock, address):
                try:
                    super().process_request_thread(sock, address)
                finally:
                    owner._slots.release()
            def handle_error(self, *args):
                pass

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def reply(self, code, value, html=False):
                data = value if html else json.dumps(value, allow_nan=False).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; media-src blob:; frame-ancestors 'none'")
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True
            def handle_route(self):
                try:
                    if self.headers.get('Host') != owner._host or time.monotonic() >= owner._deadline:
                        return self.reply(403, {})
                    if self.command == 'GET' and self.path == '/':
                        return self.reply(200, owner._html, True)
                    if not hmac.compare_digest(self.headers.get('X-H3-Capability', ''), owner._cap):
                        return self.reply(403, {})
                    if self.command == 'GET' and self.path == '/config':
                        return self.reply(200, owner._config)
                    if (self.command != 'POST' or self.path not in ('/offer', '/event')
                            or self.headers.get('Origin') != owner.origin
                            or any(k in self.headers for k in ('Transfer-Encoding','Expect','Upgrade'))
                            or len(self.headers.get_all('Content-Length',[])) != 1
                            or self.headers.get_all('Content-Type',[]) != ['application/json']):
                        return self.reply(403, {})
                    size = int(self.headers.get('Content-Length', '-1'))
                    if not 0 < size <= (70000 if self.path == '/offer' else 256):
                        return self.reply(413, {})
                    # Absolute read deadline, not a resettable per-byte inactivity timer.
                    end = min(owner._deadline, time.monotonic() + 3)
                    body = bytearray()
                    while len(body) < size:
                        self.connection.settimeout(max(.001, end - time.monotonic()))
                        chunk = self.rfile.read1(min(8192, size-len(body)))
                        if not chunk or time.monotonic() >= end:
                            raise ValueError()
                        body.extend(chunk)
                    value = json.loads(body)
                    if self.path == '/event':
                        if set(value) != {'event'} or value['event'] not in ('opened', 'connected', 'frame-presented', 'unmuted', 'error', 'stop'):
                            raise ValueError()
                        with owner._lock:
                            owner._events.append({'event': value['event']})
                        return self.reply(200, {})
                    if (set(value) != {'session', 'boot', 'description'}
                            or any(value[k] != owner.binding[k] for k in ('session', 'boot'))
                            or set(value['description']) != {'type', 'sdp'}
                            or value['description']['type'] != 'offer'
                            or not isinstance(value['description']['sdp'], str)
                            or not 0 < len(value['description']['sdp'].encode()) <= 65536):
                        raise ValueError()
                    with owner._lock:
                        if owner._offer_used:
                            return self.reply(409, {})
                        owner._offer_used = True
                    answer = webrtc_connect_client.request(owner.binding['job_id']+'--7860.hf.jobs',
                        owner._signaling_path, owner._token, owner._key, min(owner._deadline, time.monotonic()+10), value,
                        session_id=owner.binding['session'])
                    if any(answer.get(k) != owner.binding[k] for k in ('session', 'boot')):
                        raise ValueError()
                    return self.reply(200, {k: answer[k] for k in ('session', 'boot', 'description')})
                except Exception:
                    self.reply(400, {})
            do_GET = handle_route
            do_POST = handle_route
        self._html = Path(__file__).parents[1].joinpath('h3_rtc_client.html').read_bytes()
        self._server = Server(('127.0.0.1', 0), Handler)
        self._host = '127.0.0.1:'+str(self._server.server_port)
        self.origin = 'http://'+self._host
        self.url = self.origin+'/#'+self._cap
        self._thread = None

    @property
    def receipts(self):
        with self._lock:
            return tuple(dict(e) for e in self._events)

    def __enter__(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._timer = threading.Timer(max(0, self._deadline-time.monotonic()), self.close)
        self._timer.daemon = True
        self._timer.start()
        return self

    def close(self):
        with self._close_lock:
            if self._closed:return
            self._closed=True
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(2)
            self._thread = None
        self._server.server_close()
        if hasattr(self, '_timer'):
            self._timer.cancel()
        self._token = self._key = self._cap = ''
        self._config = {}
        self.url = ''

    def __exit__(self, *args):
        self.close()
