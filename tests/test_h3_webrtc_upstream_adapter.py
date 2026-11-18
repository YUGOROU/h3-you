"""Actual RTC + native protocol client/Gate; synthetic samples, no model/device."""
import importlib.util
import asyncio
import socket
import time
import unittest
from unittest.mock import patch
from research.h3_personal.webrtc_exact_channel import Binding,ExactChannel
from research.h3_personal.webrtc_exact_ipc import bridge
from research.h3_personal.webrtc_upstream_adapter import RTCStream,RemoteVoxCPMRTC,asr_transport
from research.h3_personal.webrtc_voxcpm_source import stream_native
from research.h3_personal.voxcpm_endpoint_service import Gate
from research.h3_personal.webrtc_live_peer import Session

HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class UpstreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_gate_to_real_rtc_to_original_decoder(self):
        deadline=time.monotonic()+8;expires=time.time()+60
        left=Session(deadline=deadline,control=True);right=Session(deadline=deadline)
        self.addAsyncCleanup(left.close);self.addAsyncCleanup(right.close)
        await left.set_answer(await right.accept_offer(await left.create_offer()))
        await asyncio.gather(left.wait_control(),right.wait_control())
        binding=Binding('session','boot',1,'voxcpm-native-wire-v1',100000)
        a=ExactChannel(left,binding);b=ExactChannel(right,binding)
        await asyncio.gather(a.start(),b.start())
        consumer,ipc1=socket.socketpair();model,ipc2=socket.socketpair()
        self.addCleanup(consumer.close);self.addCleanup(model.close)
        class Candidate:
            sample_rate=48000
            def stream(self,text):yield [.125,-.25,0.5]*1000
        gate=Gate(Candidate(),key='key',session='session',boot='boot',expires_at=expires)
        request=dict(session='session',boot='boot',epoch=1,text='fresh typed request')
        client=RemoteVoxCPMRTC('https://test.endpoints.huggingface.cloud',boot='boot',token='token',key='key',
            session='session',expires_at=expires,open_stream=lambda *args:RTCStream(consumer,binding))
        client._ready=True
        self.addCleanup(client.close)
        # No HTTPS media connection is allowed even when native decoding fails.
        with patch('http.client.HTTPSConnection',side_effect=AssertionError('HTTP audio forbidden')):
            result=await asyncio.wait_for(asyncio.gather(bridge(a,ipc1),bridge(b,ipc2),
                asyncio.to_thread(stream_native,gate,request,model,deadline),
                asyncio.to_thread(lambda:list(client.stream('fresh typed request',1,deadline)))),6)
        chunks=result[3]
        self.assertTrue(chunks)
        self.assertEqual(sum(len(c) for c in chunks),3000*4)
        self.assertEqual(gate.count,1)

    async def test_asr_owned_transport_never_connects_wss_or_replays(self):
        a,b=socket.socketpair();self.addCleanup(a.close);self.addCleanup(b.close)
        factory=asr_transport(RTCStream(a,Binding('s','b',1,'asr-native-event-wire-v1',10000)))
        with patch('socket.create_connection',side_effect=AssertionError('WAN WSS forbidden')):
            connection,source=factory('unused','token','key',1)
            self.addCleanup(source.close)
            connection.sendall(b'native application frame')
            self.assertEqual(b.recv(100),b'native application frame')
            with self.assertRaises(RuntimeError):factory('unused','token','key',1)

    async def test_actual_asr_pcm_and_native_coverage_over_rtc(self):
        import json,struct
        from research.h3_personal.endpoint_asr import transcribe,FINAL
        deadline=time.monotonic()+8
        left=Session(deadline=deadline,control=True);right=Session(deadline=deadline)
        self.addAsyncCleanup(left.close);self.addAsyncCleanup(right.close)
        await left.set_answer(await right.accept_offer(await left.create_offer()))
        await asyncio.gather(left.wait_control(),right.wait_control())
        binding=Binding('asr','boot',1,'asr-native-event-wire-v1',100000)
        a=ExactChannel(left,binding);b=ExactChannel(right,binding)
        await asyncio.gather(a.start(),b.start())
        client,ipc1=socket.socketpair();native,ipc2=socket.socketpair()
        self.addCleanup(client.close);self.addCleanup(native.close)
        pcm=bytes([1,0])*320;received=bytearray()
        def server():
            native.settimeout(5)
            with native.makefile('rb') as source:
                def exact(n):
                    out=bytearray()
                    while len(out)<n:
                        block=source.read(n-len(out))
                        if not block:raise EOFError()
                        out.extend(block)
                    return bytes(out)
                while True:
                    first,second=exact(2);n=second&127
                    if n==126:n=struct.unpack('!H',exact(2))[0]
                    elif n==127:n=struct.unpack('!Q',exact(8))[0]
                    self.assertTrue(second&128);mask=exact(4);raw=exact(n)
                    body=bytes(v^mask[i%4] for i,v in enumerate(raw))
                    if first&15==2:received.extend(body)
                    elif json.loads(body).get('type')=='input_audio_buffer.commit':
                        value=json.dumps(dict(type=FINAL,transcript='synthetic native result',audio_processed=len(received)/32000)).encode()
                        header=bytes([129])+(bytes([len(value)]) if len(value)<126 else bytes([126])+struct.pack('!H',len(value)))
                        native.sendall(header+value);native.shutdown(socket.SHUT_WR)
                        while source.read(1024):pass
                        return
        factory=asr_transport(RTCStream(client,binding))
        with patch('research.h3_personal.endpoint_asr._connect',side_effect=AssertionError('WAN WSS forbidden')):
            result=await asyncio.wait_for(asyncio.gather(bridge(a,ipc1),bridge(b,ipc2),asyncio.to_thread(server),
                asyncio.to_thread(transcribe,'https://test.endpoints.huggingface.cloud',token='token',owner_key='key',
                    pcm=pcm,timeout=6,max_audio_seconds=.02,completion_policy='input-finalized',_transport=factory)),7)
        self.assertEqual(received,pcm)
        self.assertEqual(result[3]['text'],'synthetic native result')
        self.assertTrue(result[3]['input_committed']);self.assertTrue(result[3]['final'])
        self.assertEqual(result[3]['errors'],[])

    async def test_wrong_purpose_or_tcp_refused(self):
        a,b=socket.socketpair();self.addCleanup(a.close);self.addCleanup(b.close)
        with self.assertRaises(ValueError):asr_transport(RTCStream(a,Binding('s','b',1,'voxcpm-native-wire-v1',100)))
        tcp=socket.socket();self.addCleanup(tcp.close)
        with self.assertRaises(ValueError):RTCStream(tcp,Binding('s','b',1,'asr-native-event-wire-v1',100))

if __name__=='__main__':unittest.main()
