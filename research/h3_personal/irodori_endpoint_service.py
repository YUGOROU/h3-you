"""Finite, app-authenticated native TTS HTTP boundary (stdlib only)."""
import base64
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import socket
import threading
import time
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, HTTPServer
import wave

from .director_irodori import SOURCE, MODEL, CODEC

MAX_WAV = 6 * 1024 * 1024
METRICS = frozenset(('elapsed_seconds', 'synthesis_seconds', 'queue_seconds', 'name',
                     'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes',
                     'cuda_allocated_bytes', 'cuda_reserved_bytes'))


class Gate:
    def __init__(self, tts, ready, *, key, session, expires_at, max_requests=2):
        if not key or not session or not math.isfinite(expires_at) or not 1 <= max_requests <= 2:
            raise ValueError('invalid service configuration')
        self.tts, self.ready = tts, dict(ready)
        self.key, self.session, self.expires_at = key, session, expires_at
        self.max_requests = max_requests
        self.lock = threading.Lock()
        self.slot = threading.Lock()
        self.epoch = self.count = 0
        self.quarantined = self.closed = False

    def healthy(self):
        return not self.quarantined and not self.closed and time.time() < self.expires_at and self.ready.get('ready') is True

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        try:
            self.tts.close()
        except Exception:
            pass

    def quarantine(self):
        self.quarantined = True
        self.close()

    def generate(self, request):
        if not isinstance(request, dict) or set(request) != {'session', 'epoch', 'text'}:
            return 400, {'error': 'invalid request'}
        epoch, text = request['epoch'], request['text']
        if request['session'] != self.session or type(epoch) is not int or epoch < 1 or not isinstance(text, str) or not 1 <= len(text) <= 160 or not text.strip():
            return 400, {'error': 'invalid request'}
        if not self.slot.acquire(False):
            return 429, {'error': 'busy'}
        try:
            with self.lock:
                if not self.healthy() or time.time() >= self.expires_at - 5:
                    return 503, {'error': 'unavailable'}
                if epoch <= self.epoch:
                    return 409, {'error': 'replay'}
                if self.count >= self.max_requests:
                    return 429, {'error': 'limit'}
                self.epoch = epoch
                self.count += 1
            deadline = time.monotonic() + min(120, self.expires_at - time.time() - 5)
            try:
                with TemporaryDirectory(prefix='h3-tts-') as root:
                    directory = Path(root) / 'turn'
                    result = self.tts.synthesize(text, epoch, directory, deadline)
                    path = Path(result['wav']).resolve()
                    if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size > MAX_WAV:
                        raise ValueError()
                    data = path.read_bytes()
                    receipt = dict(result['receipt'])
                    if receipt.get('sha256') != hashlib.sha256(data).hexdigest() or receipt.get('watermark_applied') is not True or (receipt.get('source_revision'), receipt.get('model_revision'), receipt.get('codec_revision')) != (SOURCE, MODEL, CODEC):
                        raise ValueError()
                    with wave.open(io.BytesIO(data), 'rb') as wav:
                        frames, rate, channels = wav.getnframes(), wav.getframerate(), wav.getnchannels()
                        if wav.getcomptype() != 'NONE' or wav.getsampwidth() != 2 or channels not in (1, 2) or not 0 < rate <= 48000 or not 0 < frames / rate <= 30 or len(wav.readframes(frames)) != frames * channels * 2:
                            raise ValueError()
                    receipt.pop('path', None)
                    metrics = {k: v for k, v in result.get('metrics', {}).items() if k in METRICS and ((type(v) in (int, float) and math.isfinite(v)) or (k == 'name' and isinstance(v, str) and v in ('irodori', 'native')))}
                    response = dict(schema='h3-irodori-endpoint-response-v1', session=self.session, epoch=epoch, wav_base64=base64.b64encode(data).decode('ascii'), receipt=receipt, metrics=metrics)
                    json.dumps(response, allow_nan=False)
                    if time.monotonic() >= deadline or not self.healthy():
                        raise ValueError()
                    return 200, response
            except Exception:
                self.quarantine()
                return 503, {'error': 'unavailable'}
        finally:
            self.slot.release()


class _Server(HTTPServer):
    def __init__(self, address, gate):
        self.gate = gate
        self.handlers = threading.BoundedSemaphore(4)
        super().__init__(address, _Handler)

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(5)
        return sock, address

    def process_request(self, request, address):
        if not self.handlers.acquire(False):
            self.shutdown_request(request)
            return
        def run():
            try:
                self.finish_request(request, address)
            except Exception:
                pass
            finally:
                self.shutdown_request(request)
                self.handlers.release()
        threading.Thread(target=run, daemon=True).start()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.connection.settimeout(min(30, max(0.001, self.server.gate.expires_at-time.time())))
        self.wfile.write(data)

    def authorized(self):
        values = self.headers.get_all('X-H3-TTS-Key', [])
        return len(values) == 1 and hmac.compare_digest(values[0].encode(), self.server.gate.key.encode())

    def do_GET(self):
        gate = self.server.gate
        if self.path == '/health':
            return self.reply(200 if gate.healthy() else 503, {'ready': gate.healthy()})
        if not self.authorized():
            return self.reply(401, {'error': 'unauthorized'})
        if self.path != '/ready':
            return self.reply(404, {'error': 'not found'})
        if not gate.healthy():
            return self.reply(503, {'error': 'unavailable'})
        self.reply(200, dict(gate.ready, schema='h3-irodori-endpoint-ready-v1', session=gate.session))

    def do_POST(self):
        if not self.authorized():
            return self.reply(401, {'error': 'unauthorized'})
        if self.path != '/synthesize':
            return self.reply(404, {'error': 'not found'})
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get_all('Transfer-Encoding') or self.headers.get_all('Upgrade') or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() or not 1 <= int(lengths[0]) <= 8192:
            return self.reply(400, {'error': 'invalid request'})
        try:
            size = int(lengths[0])
            # One absolute body deadline, not a renewable per-byte timeout.
            end = time.monotonic() + 5
            body = bytearray()
            while len(body) < size:
                self.connection.settimeout(max(0.001, end - time.monotonic()))
                chunk = self.rfile.read1(size - len(body))
                if not chunk or time.monotonic() >= end:
                    raise ValueError()
                body.extend(chunk)
            def pairs(items):
                result = {}
                for k, v in items:
                    if k in result:
                        raise ValueError()
                    result[k] = v
                return result
            request = json.loads(body, object_pairs_hook=pairs)
        except Exception:
            return self.reply(400, {'error': 'invalid request'})
        status, value = self.server.gate.generate(request)
        try:
            self.reply(status, value)
        except Exception:
            if status == 200:
                self.server.gate.quarantine()
            raise


def create_server(tts, ready, *, key, session, expires_at, host='0.0.0.0', port=7860, max_requests=2):
    return _Server((host, port), Gate(tts, ready, key=key, session=session, expires_at=expires_at, max_requests=max_requests))


def serve(tts, ready, *, key, session, expires_at, host='0.0.0.0', port=7860, max_requests=2):
    server = create_server(tts, ready, key=key, session=session, expires_at=expires_at, host=host, port=port, max_requests=max_requests)
    timer = threading.Timer(max(0, expires_at - time.time()), lambda: (server.gate.close(), server.shutdown()))
    timer.daemon = True
    timer.start()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        timer.cancel()
        server.gate.close()
        server.server_close()
