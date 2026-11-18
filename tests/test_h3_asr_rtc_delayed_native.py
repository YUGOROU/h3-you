"""Delayed native termination regression through real transcribe + serve.

Synthetic final followed by delayed WS CLOSE bytes or TCP half-close.
Both cases must preserve native final and verified carrier completion.

TURN config/peer ICE is patched to HOST_ONLY; no Cloudflare/HF/GPU/devices.
"""
import importlib.util
import asyncio
import base64
import hashlib
import http.client
import json
import os
import socket
import socketserver
import struct
import threading
import time
import unittest
from unittest.mock import patch
from research.h3_personal.asr_rtc_service import ASRRTCManager
from research.h3_personal.asr_rtc_gateway import Server
from research.h3_personal.asr_rtc_client import serve
from research.h3_personal.webrtc_live_peer import Session as RealSession
from research.h3_personal.webrtc_exact_channel import Binding
from research.h3_personal.webrtc_upstream_adapter import RTCStream,asr_transport
from research.h3_personal.endpoint_asr import transcribe,FINAL
from research.h3_personal import paired_av_wire as wire

TURN=dict(urls='turns:turn.cloudflare.com:443?transport=tcp',username='test',credential='test')

HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class DelayedNative(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_close_bytes(self):
        await self.run_delayed_native(send_close=True)

    async def test_delayed_halfclose(self):
        await self.run_delayed_native(send_close=False)

    async def run_delayed_native(self, *, send_close):
        seen=bytearray()
        class Native(socketserver.StreamRequestHandler):
            def handle(inner):
                headers={};inner.rfile.readline()
                while True:
                    line=inner.rfile.readline()
                    if line==b'\r\n':break
                    k,v=line.split(b':',1);headers[k.lower()]=v.strip()
                self.assertNotIn(b'authorization',headers);self.assertNotIn(b'x-h3-asr-key',headers)
                key=headers[b'sec-websocket-key'].decode()
                accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
                inner.wfile.write(b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: '+accept+b'\r\n\r\n');inner.wfile.flush()
                def exact(n):
                    result=bytearray()
                    while len(result)<n:
                        body=inner.rfile.read(n-len(result))
                        if not body:raise EOFError()
                        result.extend(body)
                    return bytes(result)
                while True:
                    a,b=exact(2);n=b&127
                    if n==126:n=struct.unpack('!H',exact(2))[0]
                    elif n==127:n=struct.unpack('!Q',exact(8))[0]
                    self.assertTrue(b&128);mask=exact(4);body=exact(n)
                    body=bytes(v^mask[i%4] for i,v in enumerate(body))
                    if a&15==2:seen.extend(body)
                    elif json.loads(body).get('type')=='input_audio_buffer.commit':
                        result=json.dumps(dict(type=FINAL,transcript='native synthetic final',audio_processed=len(seen)/32000)).encode()
                        header=bytes([129])+(bytes([len(result)]) if len(result)<126 else bytes([126])+struct.pack('!H',len(result)))
                        inner.wfile.write(header+result);inner.wfile.flush()
                        # Separate native final from transport termination: the
                        # transcriber closes its local socket on final coverage.
                        time.sleep(.3)
                        if send_close:
                            inner.wfile.write(b'\x88\x02\x03\xe8');inner.wfile.flush()
                        inner.connection.shutdown(socket.SHUT_WR)
                        while inner.rfile.read(1024):pass
                        return
        native=socketserver.ThreadingTCPServer(('127.0.0.1',0),Native);native.daemon_threads=True
        nt=threading.Thread(target=native.serve_forever,kwargs={'poll_interval':.01});nt.start()
        async def ready():return {'device':'cuda'}  # Fake proof, no model/hardware.
        manager=ASRRTCManager(key='k'*40,expires_at=time.time()+60,native_ready=ready,native_port=native.server_address[1])
        server=Server(('127.0.0.1',0),manager,asyncio.get_running_loop())
        st=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});st.start()
        control,parent_control=socket.socketpair();media,parent_media=socket.socketpair()
        def local_request(endpoint,path,token,key,session,deadline,value):
            c=http.client.HTTPConnection('127.0.0.1',server.server_address[1],timeout=5)
            try:
                body=json.dumps(value)
                c.request('POST',path,body,headers={'Content-Type':'application/json','X-H3-RTC-Key':key,'X-H3-Session':session})
                response=c.getresponse();row=json.loads(response.read())
                if response.status!=200:raise RuntimeError('local control failed')
                return row
            finally:c.close()
        def host_peer(**kwargs):return RealSession(deadline=kwargs['deadline'],control=kwargs.get('control',False))
        deadline=time.monotonic()+12
        task=reading=None
        try:
            with patch('research.h3_personal.asr_rtc_service.Session',side_effect=host_peer),\
                 patch('research.h3_personal.webrtc_live_peer.Session',side_effect=host_peer),\
                 patch('research.h3_personal.asr_rtc_client.request',side_effect=local_request),\
                 patch('research.h3_personal.webrtc_turn.fetch_turn_credentials',return_value=TURN),\
                 patch.dict(os.environ,{'H3_ASR_RTC_TOKEN':'token','H3_ASR_RTC_KEY':'k'*40}):
                task=asyncio.create_task(serve(dict(endpoint='https://test.endpoints.huggingface.cloud',session='session',epoch=1,
                    expires_at=manager.expires_at,deadline=deadline),control,media))
                reading=asyncio.create_task(asyncio.to_thread(wire._json,parent_control,deadline))
                completed,_=await asyncio.wait((reading,task),return_when=asyncio.FIRST_COMPLETED)
                if task in completed:task.result()
                row=await reading
                self.assertEqual(row['schema'],'h3-asr-rtc-ipc-ready-v1')
                factory=asr_transport(RTCStream(parent_media,Binding(**row['binding'])))
                pcm=bytes([1,0])*320
                with patch('socket.create_connection',wraps=socket.create_connection):
                    result=await asyncio.to_thread(transcribe,'https://test.endpoints.huggingface.cloud',token='token',owner_key='k'*40,
                        pcm=pcm,timeout=6,max_audio_seconds=.02,completion_policy='input-finalized',_transport=factory)
                try:
                    await task
                except Exception as error:
                    failure=(type(error).__name__,str(error))
                    print(f"native send_close={send_close}: {failure!r}",flush=True)
                else:
                    failure=None
                    receipt=await asyncio.to_thread(wire._json,parent_control,deadline)
            self.assertEqual(seen,pcm);self.assertEqual(result['text'],'native synthetic final')
            self.assertTrue(result['final']);self.assertTrue(result['input_committed']);self.assertEqual(result['errors'],[])
            self.assertIsNone(failure)
            self.assertTrue(receipt['transport_complete']);self.assertFalse(receipt['native_eos_verified'])
            # The public legacy WS path is unavailable, irrespective of key.
            def legacy():
                c=http.client.HTTPConnection('127.0.0.1',server.server_address[1],timeout=2)
                c.request('GET','/v1/audio/transcriptions/realtime',headers={'Upgrade':'websocket'})
                r=c.getresponse();code=r.status;r.read();c.close();return code
            self.assertIn(await asyncio.to_thread(legacy),(400,404))
        finally:
            if task is not None:task.cancel();await asyncio.gather(task,return_exceptions=True)
            await manager.close()
            await asyncio.to_thread(server.shutdown);server.server_close();st.join(2)
            native.shutdown();native.server_close();nt.join(2)
            for sock in (control,parent_control,media,parent_media):
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                sock.close()
            if reading is not None:
                reading.cancel();await asyncio.gather(reading,return_exceptions=True)

if __name__=='__main__':unittest.main()
