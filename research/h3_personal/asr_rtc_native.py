"""Exact RTC IPC↔unchanged private native ASR event API; no WAN WSS path."""
import base64
import hashlib
import os
import socket
import threading
import time

PATH='/v1/audio/transcriptions/realtime'


def relay(sock,deadline,*,port=7861):
    if not isinstance(sock,socket.socket) or sock.family!=socket.AF_UNIX or sock.type!=socket.SOCK_STREAM:
        raise ValueError('owned ASR IPC stream required')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('private native port')
    native=source=None;threads=[];errors=[];lock=threading.Lock();total=0
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('native ASR relay expired')
        return value
    def abort():
        for connection in (sock,native):
            if connection is not None:
                try:connection.shutdown(socket.SHUT_RDWR)
                except OSError:pass
    try:
        native=socket.create_connection(('127.0.0.1',port),timeout=min(5,remaining()))
        native.settimeout(remaining());sock.settimeout(remaining())
        key=base64.b64encode(os.urandom(16)).decode()
        native.sendall((f'GET {PATH} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                       f'Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {key}\r\n\r\n').encode())
        source=native.makefile('rb');status=source.readline(4096).split();headers={}
        for _ in range(32):
            line=source.readline(8192)
            if line==b'\r\n':break
            if not line.endswith(b'\r\n') or b':' not in line:raise ValueError('native handshake')
            name,value=line.split(b':',1);name=name.lower()
            if name in headers:raise ValueError('duplicate native handshake')
            headers[name]=value.strip()
        else:raise ValueError('native handshake header bound')
        expected=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        if (len(status)<2 or status[1]!=b'101' or headers.get(b'sec-websocket-accept')!=expected
                or headers.get(b'upgrade',b'').lower()!=b'websocket'
                or b'upgrade' not in headers.get(b'connection',b'').lower().split(b', ')):
            raise ValueError('native ASR upgrade rejected')
        def copy(read,target):
            nonlocal total
            try:
                while True:
                    remaining();body=read(12288)
                    if not body:
                        try:target.shutdown(socket.SHUT_WR)
                        except OSError:pass
                        return
                    with lock:
                        total+=len(body)
                        if total>4*1024*1024:raise ValueError('native event wire byte bound')
                    target.settimeout(remaining());target.sendall(body)
            except BaseException as error:errors.append(error);abort()
        # read1 respects any bytes buffered while reading the native handshake.
        for read,target in ((sock.recv,native),(source.read1,sock)):
            thread=threading.Thread(target=copy,args=(read,target),daemon=True);threads.append(thread);thread.start()
        for thread in threads:thread.join(max(0,deadline-time.monotonic()))
        if errors:raise RuntimeError('native ASR relay failed') from None
        if any(thread.is_alive() for thread in threads):raise TimeoutError('native ASR relay threads pending')
        return dict(bytes=total,native_eos_verified=False)
    finally:
        abort()
        for thread in threads:thread.join(1)
        if source is not None:source.close()
        if native is not None:native.close()
        sock.close()
