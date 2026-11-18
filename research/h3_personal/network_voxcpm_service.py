"""Finite TLS native stream service; bounded handlers, no HTTP audio routes."""
import hmac
import socket
import socketserver
import ssl
import threading
import time
from .network_voxcpm import receive_json,send_json
from .voxcpm_endpoint_service import Gate

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=False
    daemon_threads=True
    block_on_close=False

    def __init__(self,address,candidate,*,certificate,key_file,**gate_config):
        self.gate=Gate(candidate,**gate_config)
        self.context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.minimum_version=ssl.TLSVersion.TLSv1_3
        self.context.load_cert_chain(certificate,key_file)
        self._slots=threading.BoundedSemaphore(4);self._active=set();self._mutex=threading.Lock();self._closing=False
        super().__init__(address,Handler)

    def process_request(self,request,address):
        if not self._slots.acquire(False):request.close();return
        with self._mutex:
            if self._closing:
                self._slots.release();request.close();return
            self._active.add(request)
        try:super().process_request(request,address)
        except BaseException:
            with self._mutex:self._active.discard(request)
            self._slots.release();request.close();raise

    def process_request_thread(self,request,address):
        try:super().process_request_thread(request,address)
        finally:
            with self._mutex:self._active.discard(request)
            self._slots.release()

    def handle_error(self,*args):pass  # Never log text, keys, addresses or media.

    def server_close(self):
        self.gate.close()
        with self._mutex:self._closing=True;active=tuple(self._active)
        for sock in active:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()
        super().server_close()

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server=self.server;gate=server.gate;sock=None
        try:
            end=min(time.monotonic()+5,time.monotonic()+gate.expires_at-time.time())
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError()
            self.request.settimeout(remaining)
            sock=server.context.wrap_socket(self.request,server_side=True,do_handshake_on_connect=False)
            with server._mutex:
                if server._closing:raise RuntimeError('closing')
                server._active.discard(self.request);server._active.add(sock)
            sock.do_handshake();request=receive_json(sock,end)
            if (type(request) is not dict or set(request)!={'schema','kind','key','session','boot','epoch','text'}
                    or request['schema']!='h3-network-voxcpm-rpc-v1'
                    or type(request['key']) is not str or not request['key'].isascii()
                    or not hmac.compare_digest(request['key'],gate.key)
                    or request['session']!=gate.session or request['boot']!=gate.boot
                    or type(request['epoch']) is not int or not gate.healthy()):
                raise ValueError('invalid owner request')
            if request['kind']=='ready' and request['epoch']==0 and request['text'] is None:
                send_json(sock,gate.ready(0),end);return
            if request['kind']=='evidence' and request['epoch']==0 and request['text'] is None:
                export=getattr(gate.candidate,'export_evidence',None)
                if not callable(export):raise ValueError('evidence unavailable')
                send_json(sock,dict(binding=gate.ready(0),evidence=export()),end);return
            if request['kind']!='stream':raise ValueError('unsupported control')
            payload={k:request[k] for k in ('session','boot','epoch','text')}
            deadline=min(time.monotonic()+120,time.monotonic()+gate.expires_at-time.time())
            def write(body):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('native source expired')
                sock.settimeout(remaining);sock.sendall(body)
            gate.stream(payload,write)
            # A native END must be followed by authenticated TLS EOF, not an
            # accepted truncation of an otherwise unfinished connection.
            sock.settimeout(max(.001,min(1.0,deadline-time.monotonic())))
            try:
                plain=sock.unwrap();plain.close()
            except (OSError,ssl.SSLError):pass
        except Exception:pass  # Missing native END/EOF fails the original decoder.
        finally:
            if sock is not None:
                with server._mutex:server._active.discard(sock)
                sock.close()
