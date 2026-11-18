"""Finite, serial native Irodori HTTP adapter; close never pauses the provider.

Ready contract: schema=h3-irodori-endpoint-ready-v1, session, device_name,
device_count, model_precision, codec_precision, compile_model, watermark_ready,
source_revision, model_revision, codec_revision, reference_sha256. Extra memory
metrics are allowed. Response receipt uses IrodoriResident's native field names.
Injected transport receives JSON bytes (or None), never a URL.
"""
import base64
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import urlsplit
import wave

from .director_tts_job import SOURCE, MODEL, CODEC, REFERENCE_SHA
from .live_relay_client import _DeadlineSocket

MAX_RESPONSE = 8 * 1024 * 1024
MAX_WAV = 6 * 1024 * 1024


class IrodoriEndpointError(RuntimeError):
    """Sanitized terminal adapter failure."""


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pins(value):
    return (value.get('source_revision'), value.get('model_revision'),
            value.get('codec_revision')) == (SOURCE, MODEL, CODEC)


class RemoteIrodoriTTS:
    def __init__(self, base_url, *, token, key, session, expires_at, transport=None):
        try:
            url = urlsplit(base_url)
            if (url.scheme != 'https' or url.path or url.query or url.fragment
                    or url.username is not None or url.password is not None
                    or url.port is not None or not re.fullmatch(
                        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+endpoints\.huggingface\.cloud', url.hostname or '')
                    or any(not isinstance(v, str) or not v or '\r' in v or '\n' in v
                           for v in (token, key, session))
                    or not _finite(expires_at) or expires_at <= time.time()):
                raise ValueError()
        except Exception:
            raise IrodoriEndpointError('invalid endpoint configuration') from None
        self._host = url.hostname
        self._headers = {'Authorization': 'Bearer ' + token, 'X-H3-TTS-Key': key,
                         'Content-Type': 'application/json'}
        self.session = session
        self.expires_at = expires_at
        self._transport = transport or self._http
        self._closed = False
        self._ready = False
        self._epoch = 0
        self._lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._active_socket = None

    def close(self):
        # Independent of the serial operation lock: cancellation must unblock I/O.
        with self._socket_lock:
            self._closed = True
            sock = self._active_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                sock.close()

    def _remaining(self, deadline):
        remaining = min(deadline - time.monotonic(), self.expires_at - time.time())
        if self._closed or remaining <= 0:
            raise IrodoriEndpointError('endpoint admission closed')
        return remaining

    def _http(self, method, path, payload, headers, deadline):
        conn = http.client.HTTPSConnection(self._host, timeout=self._remaining(deadline))
        try:
            conn.connect()
            sock = conn.sock
            with self._socket_lock:
                self._remaining(deadline)
                self._active_socket = sock
            conn.sock = _DeadlineSocket(sock, lambda: self._remaining(deadline))
            conn.sock.settimeout(self._remaining(deadline))
            conn.request(method, path, body=payload, headers=headers)
            conn.sock.settimeout(self._remaining(deadline))
            response = conn.getresponse()
            lengths = response.headers.get_all('Content-Length', [])
            if len(lengths) > 1 or (lengths and not re.fullmatch(r'[0-9]+', lengths[0])):
                raise IrodoriEndpointError('invalid response length')
            length = int(lengths[0]) if lengths else None
            if length is not None and length > MAX_RESPONSE:
                raise IrodoriEndpointError('response bound exceeded')
            encodings = response.headers.get_all('Content-Encoding', [])
            transfers = response.headers.get_all('Transfer-Encoding', [])
            if (any(v.strip().lower() != 'identity' for v in encodings)
                    or (transfers and (length is not None or transfers != ['chunked']))):
                raise IrodoriEndpointError('unsupported response encoding')
            chunks = []; size = 0
            while (length is None or size < length) and not response.isclosed():
                self._remaining(deadline)
                limit = min(65536, MAX_RESPONSE + 1 - size)
                if length is not None:
                    limit = min(limit, length - size)
                chunk = response.read1(limit)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_RESPONSE:
                    raise IrodoriEndpointError('response bound exceeded')
                chunks.append(chunk)
            if length is not None and size != length:
                raise IrodoriEndpointError('truncated response')
            return response.status, b''.join(chunks)
        finally:
            with self._socket_lock:
                self._active_socket = None
            try:
                conn.close()
            except Exception:
                pass

    def _request(self, method, path, payload, deadline):
        if not _finite(deadline):
            raise ValueError()
        deadline = min(deadline, time.monotonic() + 120,
                       time.monotonic() + self.expires_at - time.time())
        self._remaining(deadline)
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        status, body = self._transport(method, path, data, dict(self._headers), deadline)
        self._remaining(deadline)
        if type(status) is not int or status != 200 or type(body) is not bytes or len(body) > MAX_RESPONSE:
            raise ValueError()
        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result:
                    raise ValueError()
                result[k] = v
            return result
        result = json.loads(body, object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(result, dict):
            raise ValueError()
        return result

    def _run(self, callback):
        if not self._lock.acquire(blocking=False):
            self.close()
            raise IrodoriEndpointError('endpoint admission closed')
        try:
            if self._closed:
                raise ValueError()
            return callback()
        except Exception:
            self.close()
            raise IrodoriEndpointError('endpoint operation failed; admission closed') from None
        finally:
            self._lock.release()

    def read(self, kind, deadline):
        def operation():
            if kind != 'tts_ready' or self._ready:
                raise ValueError()
            r = self._request('GET', '/ready', None, deadline)
            if (r.get('schema') != 'h3-irodori-endpoint-ready-v1'
                    or r.get('session') != self.session or r.get('device_name') != 'NVIDIA L4'
                    or type(r.get('device_count')) is not int or r['device_count'] != 1
                    or r.get('model_precision') != 'fp32' or r.get('codec_precision') != 'fp32'
                    or r.get('compile_model') is not False or r.get('watermark_ready') is not True
                    or not _pins(r) or r.get('reference_sha256') != REFERENCE_SHA):
                raise ValueError()
            self._ready = True
            return r
        return self._run(operation)

    def synthesize(self, text, turn_id, directory, deadline):
        def operation():
            if not self._ready or not isinstance(text, str) or not text.strip() or len(text) > 160:
                raise ValueError()
            self._epoch += 1
            r = self._request('POST', '/synthesize',
                              dict(session=self.session, epoch=self._epoch, text=text), deadline)
            if (r.get('schema') != 'h3-irodori-endpoint-response-v1'
                    or r.get('session') != self.session or type(r.get('epoch')) is not int
                    or r['epoch'] != self._epoch):
                raise ValueError()
            encoded = r['wav_base64']
            if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_WAV + 2) // 3):
                raise ValueError()
            audio = base64.b64decode(encoded, validate=True)
            if len(audio) > MAX_WAV:
                raise ValueError()
            with wave.open(io.BytesIO(audio), 'rb') as wav:
                samples, rate, channels = wav.getnframes(), wav.getframerate(), wav.getnchannels()
                if (wav.getcomptype() != 'NONE' or wav.getsampwidth() != 2
                        or channels not in (1, 2) or not 0 < rate <= 48000
                        or not 0 < samples / rate <= 30
                        or len(wav.readframes(samples)) != samples * channels * 2):
                    raise ValueError()
            receipt = r['receipt']
            if (not isinstance(receipt, dict) or not _pins(receipt)
                    or receipt.get('watermark_applied') is not True
                    or receipt.get('text_sha256') != hashlib.sha256(text.encode()).hexdigest()
                    or receipt.get('sha256') != hashlib.sha256(audio).hexdigest()
                    or any(type(receipt.get(k)) is not int or receipt[k] != v
                           for k, v in [('samples', samples), ('sample_rate', rate), ('channels', channels)])):
                raise ValueError()
            self._remaining(deadline)
            metrics = {k: v for k, v in r.get('metrics', {}).items()
                       if k in ('synthesis_seconds', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes',
                                'cuda_allocated_bytes', 'cuda_reserved_bytes') and _finite(v) and v >= 0}
            root = Path(directory)
            root.mkdir(mode=0o700, parents=True, exist_ok=False)
            path = root / 'reply.wav'
            owned = False
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                owned = True
                with os.fdopen(fd, 'wb') as output:
                    output.write(audio)
                self._remaining(deadline)
            except Exception:
                if owned:
                    try:
                        path.unlink()
                    except Exception:
                        pass
                raise
            return {'wav': str(path), 'receipt': dict(receipt, path=str(path)),
                    'turn_id': turn_id, 'metrics': metrics}
        return self._run(operation)
