"""Actual local reliable RTC byte transport; no devices/WAN/models."""
import importlib.util
import asyncio
import hashlib
import socket
import time
import unittest
from research.h3_personal.webrtc_live_peer import Session
from research.h3_personal.webrtc_exact_channel import Binding,ExactChannel,MAX_CHUNK
from research.h3_personal.webrtc_exact_ipc import bridge

HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class ExactTests(unittest.IsolatedAsyncioTestCase):
    async def peers(self):
        deadline=time.monotonic()+8
        left=Session(deadline=deadline,control=True);right=Session(deadline=deadline)
        self.addAsyncCleanup(left.close);self.addAsyncCleanup(right.close)
        await left.set_answer(await right.accept_offer(await left.create_offer()))
        await asyncio.gather(left.wait_control(),right.wait_control())
        return left,right

    async def test_verified_done_latches_before_peer_disconnect(self):
        class Peer:
            calls=0
            async def receive(self):
                self.calls+=1
                if self.calls==1:return '{"kind":"done"}'
                raise EOFError('normal peer teardown')
            async def close(self):pass
        peer=Peer();channel=ExactChannel(peer,Binding('s','b',1,'asr-native-event-wire-v1',1000))
        channel._rx_end=True;channel._tx_end=True
        await channel._pump()
        self.assertTrue(channel._peer_done.is_set())
        self.assertFalse(channel._failed.is_set())
        self.assertEqual(peer.calls,1)

    async def test_terminal_send_disconnect_requires_verified_peer_done(self):
        class Peer:
            async def send(self,value):raise ConnectionResetError('after remote consumes done')
        for completed in (True,False):
            channel=ExactChannel(Peer(),Binding('s','b',1,'asr-native-event-wire-v1',1000))
            channel._tx_finished=True;channel._rx_end=True
            if completed:
                channel._peer_done.set();await channel.barrier()
            else:
                with self.assertRaises(ConnectionResetError):await channel.barrier()

    async def test_actual_duplex_exact_fragmentation_and_eof(self):
        left,right=await self.peers();b=Binding('session','boot',1,'voxcpm-native-wire-v1',100000)
        a=ExactChannel(left,b);z=ExactChannel(right,b)
        self.addAsyncCleanup(a.close);self.addAsyncCleanup(z.close)
        await asyncio.gather(a.start(),z.start())
        body1=bytes(range(256))*231;body2=b'ASR-event'*1700;one=bytearray();two=bytearray()
        async def tx(c,body):await c.send(body);return await c.finish()
        async def rx(c,dest):
            async def sink(body):dest.extend(body)
            while await c.receive_into(sink):pass
        result=await asyncio.wait_for(asyncio.gather(tx(a,body1),tx(z,body2),rx(a,two),rx(z,one)),5)
        await asyncio.gather(a.barrier(),z.barrier())
        self.assertEqual(one,body1);self.assertEqual(two,body2)
        self.assertEqual(result[0]['sha256'],hashlib.sha256(body1).hexdigest())
        self.assertFalse(result[0]['native_eos_verified'])
        self.assertEqual(a._in.maxsize,1)
        with self.assertRaises(ValueError):await a.send(b'replay')

    async def test_sink_acceptance_not_queueing_releases_credit(self):
        left,right=await self.peers();b=Binding('s','b',1,'asr-native-event-wire-v1',100000)
        a=ExactChannel(left,b);z=ExactChannel(right,b)
        self.addAsyncCleanup(a.close);self.addAsyncCleanup(z.close)
        await asyncio.gather(a.start(),z.start())
        sending=asyncio.create_task(a.send(b'x'*MAX_CHUNK));self.addAsyncCleanup(self.cancel,sending)
        await asyncio.sleep(.03);self.assertFalse(sending.done())
        release=asyncio.Event()
        async def sink(body):await release.wait()
        receiving=asyncio.create_task(z.receive_into(sink));self.addAsyncCleanup(self.cancel,receiving)
        await asyncio.sleep(.03);self.assertFalse(sending.done())
        release.set();await asyncio.gather(sending,receiving)

    async def cancel(self,task):
        task.cancel();await asyncio.gather(task,return_exceptions=True)

    async def test_boot_mismatch_refused(self):
        left,right=await self.peers()
        a=ExactChannel(left,Binding('s','bootA',1,'voxcpm-native-wire-v1',10000))
        z=ExactChannel(right,Binding('s','bootB',1,'voxcpm-native-wire-v1',10000))
        results=await asyncio.gather(a.start(),z.start(),return_exceptions=True)
        self.assertTrue(all(isinstance(r,Exception) for r in results))

    async def test_bad_eof_hash_and_unconsumed_overflow_are_terminal(self):
        import json
        from research.h3_personal.webrtc_exact_channel import HEADER
        for mode in ('hash','overflow'):
            with self.subTest(mode=mode):
                left,right=await self.peers();b=Binding('s','b',1,'asr-native-event-wire-v1',1000)
                a=ExactChannel(left,b);z=ExactChannel(right,b)
                self.addAsyncCleanup(a.close);self.addAsyncCleanup(z.close)
                await asyncio.gather(a.start(),z.start())
                if mode=='hash':await left.send(json.dumps(dict(kind='end',seq=0,bytes=0,sha256='0'*64)))
                else:
                    await left.send(HEADER.pack(b'H3B1',0,1)+b'x')
                    await left.send(HEADER.pack(b'H3B1',0,1)+b'x')
                await asyncio.wait_for(z._failed.wait(),1)
                self.assertLessEqual(z._in.qsize(),1)

    async def test_boolean_sequence_ack_is_not_integer_ack(self):
        import json
        left,right=await self.peers();b=Binding('s','b',1,'asr-native-event-wire-v1',1000)
        a=ExactChannel(left,b);z=ExactChannel(right,b)
        self.addAsyncCleanup(a.close);self.addAsyncCleanup(z.close)
        await asyncio.gather(a.start(),z.start())
        sending=asyncio.create_task(a.send(b'xyz'))
        await asyncio.sleep(.02)
        await right.send(json.dumps(dict(kind='ack',seq=False,bytes=3,end=False)))
        with self.assertRaises(RuntimeError):await sending
        with self.assertRaises(ValueError):await a.send(b'replay')

    async def test_actual_ipc_rtc_half_close_both_directions(self):
        left,right=await self.peers();b=Binding('s','b',2,'asr-native-event-wire-v1',100000)
        a=ExactChannel(left,b);z=ExactChannel(right,b)
        await asyncio.gather(a.start(),z.start())
        app1,ipc1=socket.socketpair();app2,ipc2=socket.socketpair()
        self.addCleanup(app1.close);self.addCleanup(app2.close)
        app1.setblocking(False);app2.setblocking(False);loop=asyncio.get_running_loop()
        async def app(sock,body):
            await loop.sock_sendall(sock,body);sock.shutdown(socket.SHUT_WR)
            out=bytearray()
            while True:
                data=await loop.sock_recv(sock,8192)
                if not data:return bytes(out)
                out.extend(data)
        tasks=[bridge(a,ipc1),bridge(z,ipc2),app(app1,b'PCM'*16000),app(app2,b'native final event')]
        result=await asyncio.wait_for(asyncio.gather(*tasks),6)
        self.assertEqual(result[2],b'native final event');self.assertEqual(result[3],b'PCM'*16000)

if __name__=='__main__':unittest.main()
