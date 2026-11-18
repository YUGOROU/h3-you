"""ASR Endpoint control-only HTTP gateway. Public audio/Upgrade routes absent."""
import asyncio
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time
from .webrtc_job_gateway import RTCGateway
from .voxcpm_endpoint_protocol import loads

class Server(ThreadingHTTPServer):
    daemon_threads=True;block_on_close=False
    def __init__(self,address,manager,loop):
        self.manager=manager;self.loop=loop;self.gateway=None;self.slots=threading.BoundedSemaphore(4)
        super().__init__(address,Handler)
    def process_request(self,sock,address):
        if not self.slots.acquire(False):sock.close();return
        sock.settimeout(5)
        try:super().process_request(sock,address)
        except BaseException:self.slots.release();sock.close();raise
    def process_request_thread(self,sock,address):
        try:super().process_request_thread(sock,address)
        finally:self.slots.release()
    def handle_error(self,*args):pass

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,code,value):
        return self._reply(code,json.dumps(value,allow_nan=False).encode(),'application/json')
    def _reply(self,code,body,content_type):
        self.send_response(code);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('Connection','close');self.end_headers();self.wfile.write(body);self.close_connection=True
    def do_GET(self):self.route()
    def do_POST(self):self.route()
    def route(self):
        future=None
        try:
            if self.server.manager._closed or self.server.manager._failed or time.time()>=self.server.manager.expires_at:
                return self.reply(503,{'error':'RTC lease unavailable'})
            if self.path in ('/rtc/status','/rtc/offer'):
                if self.server.gateway is None:return self.reply(409,{'error':'no active turn'})
                return self.server.gateway.handle(self)
            if any(k in self.headers for k in ('Upgrade','Transfer-Encoding','Expect','Cookie','Origin')):
                return self.reply(400,{'error':'control only'})
            if self.command=='GET' and self.path=='/ready':
                future=asyncio.run_coroutine_threadsafe(self.server.manager.native_ready(),self.server.loop)
                row=future.result(timeout=5)
                if type(row) is not dict or row.get('device')!='cuda':raise ValueError('native not ready')
                return self.reply(200,dict(row,media_transport='webrtc-exact-v1'))
            if self.command!='POST' or self.path!='/rtc/start':return self.reply(404,{'error':'control only'})
            keys=self.headers.get_all('X-H3-RTC-Key',[])
            if len(keys)!=1 or not keys[0].isascii() or not hmac.compare_digest(keys[0],self.server.manager.key):
                return self.reply(401,{'error':'unauthorized'})
            sizes=self.headers.get_all('Content-Length',[])
            if (len(sizes)!=1 or not sizes[0].isascii() or not sizes[0].isdigit() or not 0<int(sizes[0])<=8192
                    or self.headers.get_all('Content-Type',[])!=['application/json']):return self.reply(400,{'error':'bounded JSON required'})
            deadline=time.monotonic()+5;body=bytearray()
            while len(body)<int(sizes[0]):
                self.connection.settimeout(max(.001,deadline-time.monotonic()))
                chunk=self.rfile.read1(int(sizes[0])-len(body))
                if not chunk or time.monotonic()>=deadline:raise ValueError('control body deadline')
                body.extend(chunk)
            row=loads(body)
            if self.headers.get_all('X-H3-Session',[])!=[row.get('session')]:return self.reply(403,{'error':'session mismatch'})
            future=asyncio.run_coroutine_threadsafe(self.server.manager.start(row),self.server.loop)
            result=future.result(timeout=10)
            self.server.gateway=RTCGateway(result['signal_port'],self.server.manager.key,row['session'],result['expires_at'])
            return self.reply(200,{k:v for k,v in result.items() if k!='signal_port'})
        except Exception:
            if future is not None:future.cancel()
            return self.reply(503,{'error':'RTC control unavailable; no fallback'})
