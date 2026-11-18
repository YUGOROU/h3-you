"""Native Gate→owned RTC IPC source. No HTTP media handler or model changes."""
import socket
import time

def stream_native(gate,request,sock,deadline):
    if not isinstance(sock,socket.socket) or sock.family!=socket.AF_UNIX or sock.type!=socket.SOCK_STREAM:
        raise ValueError('owned native RTC socketpair required')
    try:
        def remaining():
            value=min(deadline-time.monotonic(),gate.expires_at-time.time())
            if value<=0:raise TimeoutError('native RTC source expired')
            sock.settimeout(value)
        remaining()
        # TTS is response-only; text arrived via authenticated control. Never
        # interpret received media bytes as a second prompt/control channel.
        if sock.recv(1)!=b'':raise ValueError('unexpected TTS upload')
        def write(body):remaining();sock.sendall(body)
        gate.stream(request,write)
        sock.shutdown(socket.SHUT_WR)
    finally:sock.close()
