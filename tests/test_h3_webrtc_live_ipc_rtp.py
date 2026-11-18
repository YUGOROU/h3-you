"""Real growing IPC→aiortc RTP→decoded sink, no GPU/browser/device/WAN."""
import importlib.util
import asyncio
import io
import json
import socket
import threading
import time
from types import SimpleNamespace
import unittest

from research.h3_personal.paired_av import LiveAVBinding,AudioRegion,encode
from research.h3_personal.webrtc_live_peer import Session
from research.h3_personal.webrtc_rtp_bridge import RTPBridge
from research.h3_personal.webrtc_producer import RTCProducerPump


HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class LiveIPCRTP(unittest.IsolatedAsyncioTestCase):
    def packet(self,binding,index,final):
        import av
        out=io.BytesIO()
        with av.open(out,'w',format='mp4') as container:
            stream=container.add_stream('mpeg4',rate=24);stream.width=252;stream.height=448;stream.pix_fmt='yuv420p'
            for i in range(24):
                frame=av.VideoFrame(252,448,'yuv420p')
                for n,plane in enumerate(frame.planes):plane.update(bytes([32+index*40 if n==0 else 128])*plane.buffer_size)
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        video=out.getvalue();pcm=bytes(96000)
        e=encode(binding,index,index*24,(index+1)*24,video,pcm,
                 (AudioRegion('silence',index*48000,(index+1)*48000),),final)
        return SimpleNamespace(envelope=e,video=video,pcm=pcm)

    async def test_growing_real_media_then_explicit_unverified_close(self):
        deadline=time.monotonic()+12
        binding=LiveAVBinding('live-ipc-rtc','a'*64,1)
        bridge=RTPBridge(binding,deadline);gate=asyncio.Event();start=[None]
        tracks=bridge.source.make_tracks(gate,lambda:start[0])
        left,right=socket.socketpair();first_received=threading.Event();second_generated=[]
        first=self.packet(binding,0,False)
        def packets():
            yield first
            if not first_received.wait(3):raise TimeoutError('first RTP before second source')
            second_generated.append(True)
            yield self.packet(binding,1,True)
        pump=RTCProducerPump(right,binding,deadline=deadline,overlap_send=True)
        coverage={'video':0,'audio':0};pending=[];consumers=[];end=asyncio.Event()
        async with Session(deadline=deadline,tracks=tracks) as sender,Session(deadline=deadline,receive_audio=True,receive_video=True,control=True) as receiver:
            offer=await receiver.create_offer();await receiver.set_answer(await sender.accept_offer(offer))
            await asyncio.gather(sender.wait_control(),receiver.wait_control())
            async def consume(track):
                while True:
                    frame=await track.recv()
                    if track.kind=='video':
                        coverage['video']=max(coverage['video'],round(float(frame.pts*frame.time_base)*24)+1)
                        first_received.set()
                    else:coverage['audio']=max(coverage['audio'],round(float(frame.pts*frame.time_base)*48000)+frame.samples)
            for _ in range(2):consumers.append(asyncio.create_task(consume(await receiver.receive_track())))
            async def sink_events():
                while True:
                    msg=json.loads(await receiver.receive())
                    if msg['kind']=='window':pending.append(msg)
                    elif msg['kind']=='source_eos':end.set();return
            async def ack():
                sent=0
                while sent<1:
                    if pending and coverage['video']>=pending[0]['frame_stop'] and coverage['audio']>=pending[0]['sample_stop']:
                        w=pending.pop(0)
                        await receiver.send(json.dumps(bridge._message('sink-accepted',index=w['index'],
                            video_frame_stop=min(coverage['video'],bridge.source.frame_stop),
                            audio_sample_stop=min(coverage['audio'],bridge.source.sample_stop))))
                        sent+=1
                    await asyncio.sleep(.005)
                await end.wait();await bridge.source.wait_for_consumed();await asyncio.sleep(.4)
                self.assertEqual(len(pending),1)
                w=pending[0]
                # aiortc finite jitter/encoder drain is not manufactured into
                # exact completion. Genuine original silence is the final window.
                vf=min(coverage['video'],bridge.source.frame_stop);af=min(coverage['audio'],bridge.source.sample_stop)
                kind='sink-accepted' if vf==w['frame_stop'] and af==w['sample_stop'] else 'end-partial'
                await receiver.send(json.dumps(bridge._message(kind,index=w['index'],video_frame_stop=vf,audio_sample_stop=af)))
            start[0]=time.monotonic();gate.set()
            tasks=[asyncio.create_task(bridge.run(sender,left)),asyncio.create_task(asyncio.to_thread(pump.run,packets())),
                   asyncio.create_task(sink_events()),asyncio.create_task(ack())]
            try:
                values=await asyncio.wait_for(asyncio.gather(*tasks),8)
                self.assertTrue(first_received.is_set());self.assertEqual(second_generated,[True])
                self.assertTrue(values[1]['source_complete'])
                self.assertFalse(values[1]['remote_original_pcm_verified'])
                self.assertLessEqual(pump.highwater,2)
                self.assertGreaterEqual(coverage['video'],24);self.assertGreaterEqual(coverage['audio'],48000)
            finally:
                for task in tasks+consumers:task.cancel()
                await asyncio.gather(*tasks,*consumers,return_exceptions=True)
                left.close();right.close()


if __name__=='__main__':unittest.main()
