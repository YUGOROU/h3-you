"""Bounded localhost-only JSON WebSocket transport for read-only OBS sessions.

No OBS discovery/launch/configuration, credential files, reconnect or background
watcher. Reuses existing RFC6455 frame helpers; a per-read deadline prevents a
slow peer from extending the finite session with partial frames.
"""
import base64
import hashlib
import math
import os
import socket
import time

from research.probe_h3_asr_cpu_space import frame, recv_frame


class LocalOBSWebSocket:
    def __init__(self, *, port=4455, timeout=10.0, clock=time.monotonic):
        if (type(port) is not int or not 1024 <= port <= 65535 or
                type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30):
            raise ValueError('bounded localhost OBS connection required')
        self._clock = clock
        self._deadline = clock()+timeout
        self._read_deadline = self._deadline
        self._socket = None
        self._closed = False
        try:
            self._socket = socket.create_connection(('127.0.0.1', port), timeout=min(5, self._remaining()))
            key = base64.b64encode(os.urandom(16)).decode('ascii')
            self._send_bytes((f'GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                              'Upgrade: websocket\r\nConnection: Upgrade\r\n'
                              'Sec-WebSocket-Version: 13\r\n'
                              'Sec-WebSocket-Protocol: obswebsocket.json\r\n'
                              f'Sec-WebSocket-Key: {key}\r\n\r\n').encode('ascii'))
            header = bytearray()
            while not header.endswith(b'\r\n\r\n'):
                if len(header) >= 16384:
                    raise ValueError('invalid OBS upgrade')
                value = self.read(1)
                if not value:
                    raise EOFError('OBS connection closed')
                header.extend(value)
            lines = bytes(header).split(b'\r\n')
            status = lines[0].split()
            headers = {}
            for line in lines[1:-2]:
                if b':' not in line:
                    raise ValueError('invalid OBS upgrade')
                k, v = line.split(b':', 1)
                k = k.lower()
                if k in headers:
                    raise ValueError('duplicate OBS upgrade header')
                headers[k] = v.strip()
            expected = base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode('ascii')).digest())
            if (len(status) < 2 or status[0] != b'HTTP/1.1' or status[1] != b'101' or
                    headers.get(b'sec-websocket-accept') != expected or
                    headers.get(b'upgrade', b'').lower() != b'websocket' or
                    b'upgrade' not in [x.strip() for x in headers.get(b'connection', b'').lower().split(b',')] or
                    b'sec-websocket-extensions' in headers or
                    headers.get(b'sec-websocket-protocol', b'obswebsocket.json') != b'obswebsocket.json'):
                raise ValueError('invalid OBS upgrade')
        except BaseException:
            self.close()
            raise

    def _remaining(self):
        remaining = min(self._deadline, self._read_deadline)-self._clock()
        if self._closed or remaining <= 0:
            raise TimeoutError('OBS transport deadline')
        return remaining

    def read(self, count):
        # recv_frame/read_exact invoke this for every partial socket read.
        self._socket.settimeout(self._remaining())
        value = self._socket.recv(count)
        self._remaining()
        return value

    def _send_bytes(self, payload):
        self._socket.settimeout(self._remaining())
        self._socket.sendall(payload)

    def send(self, text):
        if not isinstance(text, str):
            raise ValueError('JSON text required')
        self._send_bytes(frame(text.encode('utf-8'), opcode=1))

    def recv(self, *, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('finite receive timeout required')
        self._read_deadline = min(self._deadline, self._clock()+timeout)
        chunks = bytearray()
        started = False
        try:
            for _ in range(128):
                final, opcode, payload = recv_frame(self)
                if opcode in (8, 9, 10):
                    if not final or len(payload) > 125:
                        raise ValueError('invalid OBS control frame')
                    if opcode == 8:
                        raise EOFError('OBS connection closed')
                    if opcode == 9:
                        self._send_bytes(frame(payload, opcode=10))
                    continue
                if opcode == 1 and not started:
                    started = True
                elif opcode != 0 or not started:
                    raise ValueError('unexpected OBS frame')
                chunks.extend(payload)
                if len(chunks) > 65536:
                    raise ValueError('bounded OBS message required')
                if final:
                    return chunks.decode('utf-8')
            raise ValueError('OBS frame limit')
        finally:
            self._read_deadline = self._deadline

    def close(self):
        if not self._closed:
            self._closed = True
            if self._socket is not None:
                self._socket.close()
