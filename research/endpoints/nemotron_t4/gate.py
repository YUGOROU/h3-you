"""Small owner-key gate; native WS frames relayed, never parsed/logged/stored."""
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import select
import socket
import threading
import time

STREAM_SLOT = threading.BoundedSemaphore(1)


class Gate(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        if time.time() >= int(os.environ['H3_ASR_EXPIRES_AT']):
            self.send_error(503)
            return
        if self.path == '/ready':
            connection = http.client.HTTPConnection('127.0.0.1', 7861, timeout=5)
            try:
                connection.request('GET', '/ready')
                response = connection.getresponse()
                body = response.read(16384)
                self.send_response(response.status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self.send_error(503)
            finally:
                connection.close()
            return
        key = os.environ.get('H3_ASR_KEY', '')
        if (self.path != '/v1/audio/transcriptions/realtime' or len(key) < 32 or
                not hmac.compare_digest(self.headers.get('X-H3-ASR-Key', ''), key) or
                self.headers.get('Upgrade', '').lower() != 'websocket'):
            self.send_error(403)
            return
        if not STREAM_SLOT.acquire(blocking=False):
            self.send_error(429)
            return
        try:
            self.relay()
        finally:
            STREAM_SLOT.release()

    def relay(self):
        with socket.create_connection(('127.0.0.1', 7861), timeout=10) as upstream:
            # No HF token or application-key forwarded into native request logs.
            headers = ''.join(f'{k}: {v}\r\n' for k, v in self.headers.items()
                              if k.lower() not in ('authorization', 'x-h3-asr-key', 'host'))
            upstream.sendall(('GET '+self.path+' HTTP/1.1\r\nHost: 127.0.0.1:7861\r\n'+headers+'\r\n').encode())
            self.close_connection = True
            end = min(time.time()+180, int(os.environ['H3_ASR_EXPIRES_AT']))
            total = 0
            while time.time() < end:
                readable, _, _ = select.select([upstream, self.connection], [], [], 1)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    total += len(data)
                    if total > 4*1024*1024:
                        return
                    (self.connection if source is upstream else upstream).sendall(data)


def run():
    ThreadingHTTPServer(('0.0.0.0', 7860), Gate).serve_forever()
