"""Bounded HTTP-only signaling. Caller creates/owns Session on the supplied loop.

Use SignalServer(...).start() or a synchronous context manager; neither blocks
that loop waiting for HTTP. No TLS: deploy only on an explicitly trusted network
or behind an authenticated TLS transport. No media or credential-body API.
"""
import asyncio
import hmac
import json
import math
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from .webrtc_live_peer import description

LIMIT = 128 * 1024


def _identity(value):
    return (type(value) is str and bool(value) and value.isascii()
            and all(33 <= ord(c) <= 126 for c in value))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


class _HTTP(ThreadingMixIn, HTTPServer):
    daemon_threads = False
    block_on_close = True
    request_queue_size = 4

    def process_request(self, request, address):
        if not self.slots.acquire(False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        pass

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(5)
        return sock, address


class SignalServer:
    """One immutable session binding and one offer claim, even on failure.

    expires_at is an absolute Unix timestamp, bounded by the Session monotonic
    deadline and max_lifetime (600s default, explicit 1200s for later H3).
    Closing stops HTTP only; the caller still owns Session/tracks/loop lifetime.
    """
    def __init__(self, *, session, loop, session_id, key, expires_at,
                 host="127.0.0.1", port=0, max_lifetime=600):
        now = time.time()
        if (not _identity(session_id) or not _identity(key) or len(key) < 32
                or max_lifetime not in (600, 1200)
                or not isinstance(expires_at, (int, float))
                or not math.isfinite(expires_at)
                or not 0 < expires_at - now <= max_lifetime
                or expires_at - now > session.deadline - time.monotonic()
                or not loop.is_running()):
            raise ValueError("invalid signaling configuration")
        self.session = session
        self.loop = loop
        self.session_id = session_id
        self.boot = secrets.token_hex(16)
        self.expires_at = expires_at
        self._deadline = time.monotonic() + expires_at - now
        self._key = key
        self._lock = threading.Lock()
        self._claimed = False
        self._closed = False
        self._future = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
                if len(body) > LIMIT:
                    status, body = 502, b'{"error":"response too large"}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self):
                self.dispatch()

            def do_POST(self):
                self.dispatch()

            def dispatch(self):
                self.close_connection = True
                lengths = self.headers.get_all("Content-Length", [])
                if (any(name in self.headers for name in
                        ("Transfer-Encoding", "Expect", "Upgrade"))
                        or len(lengths) > 1
                        or (lengths and not re.fullmatch(r"[0-9]{1,6}", lengths[0]))):
                    return self.reply(400, {"error": "framing"})
                size = int(lengths[0]) if lengths else 0
                if size > LIMIT:
                    return self.reply(413, {"error": "body too large"})
                if self.command == "GET" and size:
                    return self.reply(400, {"error": "unexpected body"})
                if self.path == "/health" and self.command == "GET":
                    return self.reply(200, {"ready": owner.ready})
                keys = self.headers.get_all("X-H3-RTC-Key", [])
                if (len(keys) != 1 or not _identity(keys[0])
                        or not hmac.compare_digest(keys[0], owner._key)):
                    return self.reply(401, {"error": "unauthorized"})
                if not owner.ready:
                    return self.reply(410, {"error": "expired"})
                if self.path == "/status" and self.command == "GET":
                    return self.reply(200, {
                        "schema": "h3-rtc-signal-v1", "session": owner.session_id,
                        "boot": owner.boot, "expires_at": owner.expires_at,
                        "peerconnectionstate": owner.session.pc.connectionState,
                        "ice_profile": owner.session.ice_profile})
                if self.path != "/offer" or self.command != "POST":
                    return self.reply(404, {"error": "not found"})
                if len(lengths) != 1 or self.headers.get_all("Content-Type", []) != ["application/json"]:
                    return self.reply(400, {"error": "JSON body required"})
                try:
                    end = min(owner._deadline, time.monotonic() + 5)
                    raw = bytearray()
                    while len(raw) < size:
                        left = end - time.monotonic()
                        if left <= 0:
                            raise TimeoutError()
                        self.connection.settimeout(left)
                        part = self.rfile.read1(size - len(raw))
                        if not part:
                            raise ValueError()
                        raw.extend(part)
                    value = json.loads(raw, object_pairs_hook=_object)
                    if (type(value) is not dict
                            or set(value) != {"session", "boot", "description"}
                            or not _identity(value["session"])
                            or type(value["boot"]) is not str
                            or not re.fullmatch(r"[0-9a-f]{32}", value["boot"])):
                        raise ValueError()
                    description(value["description"], "offer")
                except Exception:
                    return self.reply(400, {"error": "invalid offer"})
                if value["session"] != owner.session_id or value["boot"] != owner.boot:
                    return self.reply(409, {"error": "binding mismatch"})
                with owner._lock:
                    if owner._claimed:
                        return self.reply(409, {"error": "already claimed"})
                    if not owner.ready:
                        return self.reply(410, {"error": "expired"})
                    owner._claimed = True
                    future = asyncio.run_coroutine_threadsafe(
                        owner.session.accept_offer(value["description"]), owner.loop)
                    owner._future = future
                try:
                    answer = future.result(timeout=max(0, min(30, owner._deadline - time.monotonic())))
                    description(answer, "answer")
                    self.reply(200, {"session": owner.session_id, "boot": owner.boot,
                                     "description": answer})
                except Exception:
                    future.cancel()
                    self.reply(502, {"error": "negotiation failed"})

            def handle_expect_100(self):
                self.reply(400, {"error": "framing"})
                return False

        self._http = _HTTP((host, port), Handler)
        self._http.slots = threading.BoundedSemaphore(4)
        self._thread = None
        self._timer = None

    @property
    def address(self):
        return self._http.server_address

    @property
    def ready(self):
        return not self._closed and time.monotonic() < self._deadline

    def start(self):
        if self._closed or self._thread is not None:
            raise RuntimeError("server already started or closed")
        self._thread = threading.Thread(target=self._http.serve_forever,
                                        kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        self._timer = threading.Timer(max(0, self._deadline - time.monotonic()), self.close)
        self._timer.daemon = True
        self._timer.start()
        return self

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._future is not None:
                self._future.cancel()
        if self._timer is not None:
            self._timer.cancel()
        if self._thread is not None:
            self._http.shutdown()
            self._thread.join(6)
        self._http.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
