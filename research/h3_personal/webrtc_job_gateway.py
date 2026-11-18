"""Bounded, RAM-only signaling proxy to one explicit loopback RTC sidecar.

No media, redirects, environment proxies, cookies, or HF credentials cross this seam.
"""
import hmac
import http.client
import json
import math
import time


MAX_BODY = 128 * 1024


class RTCGateway:
    """Forward /rtc/{status,offer} to 127.0.0.1:port/{status,offer}.

    Both requests require X-H3-RTC-Key and X-H3-Session. The sidecar receives
    only those application bindings and explicit JSON framing. expires_at is
    an absolute Unix timestamp; each operation has an additional ten-second cap.
    """

    def __init__(self, port, key, session, expires_at, *, operation_timeout=10):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('explicit loopback port required')
        for value in (key, session):
            if not isinstance(value, str) or not value or not value.isascii() or any(
                    ord(c) < 33 or ord(c) > 126 for c in value):
                raise ValueError('ASCII application binding required')
        if type(expires_at) not in (int, float) or not math.isfinite(expires_at) or expires_at <= time.time():
            raise ValueError('unexpired finite lease required')
        if type(operation_timeout) not in (int,float) or not math.isfinite(operation_timeout) or not 0<operation_timeout<=35:
            raise ValueError('finite signaling proxy timeout required')
        self.operation_timeout=operation_timeout
        self.port, self.key, self.session, self.expires_at = port, key, session, expires_at
        self._deadline = time.monotonic() + expires_at - time.time()

    def _remaining(self, deadline):
        remaining = min(deadline - time.monotonic(), self.expires_at - time.time())
        if remaining <= 0:
            raise TimeoutError()
        return remaining

    def handle(self, handler):
        """Reply using the owning HTTP handler; never log upstream data."""
        code, body = 400, b'{}'
        conn = None
        try:
            if time.monotonic() >= self._deadline or time.time() >= self.expires_at:
                code = 410
                return
            deadline = min(self._deadline, time.monotonic() + self.operation_timeout)
            if handler.path not in ('/rtc/status', '/rtc/offer'):
                code = 404
                return
            if handler.command != ('GET' if handler.path == '/rtc/status' else 'POST'):
                code = 405
                return
            headers = handler.headers
            # Reject ambiguity in bindings/framing. Proxy tracing/forwarding
            # headers are discarded, never interpreted or forwarded upstream.
            checked=('Host','Content-Length','Content-Type','X-H3-RTC-Key',
                     'X-H3-Session','Authorization','Connection','Content-Encoding')
            if any(len(headers.get_all(name, [])) > 1 for name in checked):
                return
            if any(name in headers for name in ('Transfer-Encoding', 'Expect', 'Upgrade', 'Trailer', 'Origin')):
                return
            for name, expected in (('X-H3-RTC-Key', self.key), ('X-H3-Session', self.session)):
                actual = headers.get(name, '')
                if not actual.isascii() or not hmac.compare_digest(actual, expected):
                    code = 403
                    return
            text = headers.get('Content-Length', '')
            if not text or not text.isascii() or not text.isdecimal() or len(text) > 6:
                return
            length = int(text)
            if length > MAX_BODY:
                code = 413
                return
            if handler.command == 'GET' and length:
                return
            if handler.command == 'POST' and (not length or headers.get('Content-Type') != 'application/json'):
                return
            payload = bytearray()
            while len(payload) < length:
                handler.connection.settimeout(self._remaining(deadline))
                chunk = handler.rfile.read1(length - len(payload))
                if not chunk:
                    return
                payload.extend(chunk)
            if payload:
                json.loads(payload)
            conn = http.client.HTTPConnection('127.0.0.1', self.port,
                                               timeout=self._remaining(deadline))
            conn.request(handler.command, handler.path[4:], body=bytes(payload), headers={
                'X-H3-RTC-Key': self.key, 'X-H3-Session': self.session,
                'Content-Type': 'application/json', 'Content-Length': str(length),
                'Connection': 'close'})
            conn.sock.settimeout(self._remaining(deadline))
            response = conn.getresponse()
            rh = response.headers
            size = rh.get('Content-Length', '')
            if (any(len(rh.get_all(n)) != 1 for n in rh.keys()) or
                    any(n in rh for n in ('Transfer-Encoding', 'Upgrade', 'Trailer')) or
                    not size.isascii() or not size.isdecimal() or len(size) > 6 or
                    int(size) > MAX_BODY or rh.get_content_type() != 'application/json' or
                    not 200 <= response.status <= 599 or 300 <= response.status < 400):
                code = 502
                return
            result = bytearray()
            while len(result) < int(size):
                if conn.sock is not None:
                    conn.sock.settimeout(self._remaining(deadline))
                chunk = response.read1(int(size) - len(result))
                if not chunk:
                    code = 502
                    return
                result.extend(chunk)
            self._remaining(deadline)
            json.loads(result)
            code, body = response.status, bytes(result)
        except (ValueError, UnicodeError):
            code = 400 if conn is None else 502
        except (OSError, http.client.HTTPException):
            code = 504 if time.monotonic() >= locals().get('deadline', self._deadline) else 502
        finally:
            if conn is not None:
                conn.close()
            handler._reply(code, body, 'application/json')
