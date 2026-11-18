import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest

from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av import LiveAVBinding,AudioRegion,encode
from research.h3_personal.paired_av_pump import ProducerPump,ReceiverPump
from research.h3_personal.live_native_backend import LiveNativeBackend
from test_h3_live_native_backend import FakeRuntime,driver,two_turn_packets,export_stub


class DummySink:
    def __init__(self,root,binding,*,fail=False,release=None,entered=None):
        self.root=Path(root).resolve();self.binding=binding;self.fail=fail
        self.release=release;self.entered=entered;self.cancelled=False;self.count=0
    def accept(self,path,lease):
        if self.entered is not None and self.count==0:
            self.entered.set()
            if not self.release.wait(3):raise TimeoutError('test gate')
        if self.fail:raise OSError('synthetic sink failure after verification')
        self.count+=1
    def finish(self):return None
    def cancel(self):self.cancelled=True


def thread_run(fn,results,errors):
    def run():
        try:results.append(fn())
        except BaseException as exc:errors.append(exc)
    t=threading.Thread(target=run);t.start();return t


class PumpTests(unittest.TestCase):
    def test_two_slots_overlap_but_no_third_native_pull_before_sink_ack(self):
        with tempfile.TemporaryDirectory() as root:
            b=LiveAVBinding('pump','a'*64,1,max_segments=8)
            a,c=socket.socketpair();release=threading.Event();entered=threading.Event();second=threading.Event();third=threading.Event()
            sink=DummySink(root,b,release=release,entered=entered)
            producer=ProducerPump(a,b,deadline=time.monotonic()+10)
            receiver=ReceiverPump(c,b,root,sink,deadline=time.monotonic()+10)
            def packets():
                for i in range(3):
                    if i==1:second.set()
                    if i==2:third.set()
                    pcm=bytes(4000);video=b'v'
                    e=encode(b,i,i,i+1,video,pcm,(AudioRegion('silence',i*2000,(i+1)*2000),),i==2)
                    yield Packet(e,video,pcm)
                raise AssertionError('must not pull beyond final pair')
            pr,rr,errors=[],[],[]
            rt=thread_run(receiver.run,rr,errors);pt=thread_run(lambda:producer.run(packets()),pr,errors)
            try:
                self.assertTrue(entered.wait(2));self.assertTrue(second.wait(2));self.assertFalse(third.is_set())
            finally:
                release.set();pt.join(10);rt.join(10)
                if pt.is_alive() or rt.is_alive():a.close();c.close()
            self.assertFalse(pt.is_alive());self.assertFalse(rt.is_alive());self.assertFalse(errors,errors)
            self.assertEqual(pr[0]['queue_highwater'],2);self.assertEqual(pr[0]['acked'],3)
            self.assertTrue(rr[0]['complete']);self.assertFalse(producer._thread.is_alive())
            timing=producer.timing()
            self.assertEqual(timing['clock'],'local_monotonic')
            self.assertEqual(timing['send_semantics'],'local_enqueue_not_peer_delivery')
            self.assertEqual(len(timing['windows']),3)
            for i,row in enumerate(timing['windows']):
                self.assertEqual(row['index'],i)
                self.assertEqual(row['video_bytes'],1)
                self.assertEqual(row['pcm_bytes'],4000)
                ordered=[row[k] for k in ('credit_wait_started','pull_started',
                                         'packet_ready','send_started','send_enqueued')]
                self.assertEqual(ordered,sorted(ordered))
                self.assertGreaterEqual(row['ack_sink_accepted'],row['ack_verified'])
            timing['windows'][0]['pcm_bytes']=0
            self.assertEqual(producer.timing()['windows'][0]['pcm_bytes'],4000)

    def test_sink_failure_stops_replay_and_preserves_warm_model_for_new_owner(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);runtime=FakeRuntime(root);d=driver(root)
            backend=LiveNativeBackend(runtime,root/'native',exporter=export_stub)
            client=root/'client';client.mkdir(mode=0o700)
            sink=DummySink(client,d.binding,fail=True)
            a,c=socket.socketpair();p=ProducerPump(a,d.binding,deadline=time.monotonic()+10)
            receiver=ReceiverPump(c,d.binding,client,sink,deadline=time.monotonic()+10)
            errors=[];rt=thread_run(receiver.run,[],errors)
            def packets():
                try:yield from two_turn_packets(d,backend,[])
                finally:d.close()
            with self.assertRaises((OSError,EOFError,RuntimeError)):
                p.run(packets())
            rt.join(10);self.assertFalse(rt.is_alive());self.assertTrue(errors)
            self.assertTrue(sink.cancelled);self.assertIsNotNone(p.sender.snapshot()['terminal'])
            self.assertFalse(p._thread.is_alive())
            backend.abandon(reason='local-sink-failure')
            self.assertEqual(runtime.controls[-1]['action'],'abandon');self.assertEqual(runtime.start_calls,0)
            with driver(root,'fresh-after-sink-failure') as new:
                again=LiveNativeBackend(runtime,root/'fresh-native',exporter=export_stub)
                new.request_stop();new.step(again);again.retire()
            self.assertEqual(runtime.start_calls,0)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_two_turn_native_protocol_to_wire_to_headless_exact_audio(self):
        from research.h3_personal.headless_av_sink import HeadlessAVSink
        from research.run_h3_realtime_first import encode_rgb_segment
        from research.h3_personal.conversation_scenario import segment_timing
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);runtime=FakeRuntime(root);d=driver(root)
            def exporter(result,path,start,stop,index):
                raw=bytes([index*20,40,100])*(252*448*(stop-start))
                encode_rgb_segment(raw,path,segment_timing(index,start,stop),width=252,height=448)
                path.chmod(0o400);return path.read_bytes()
            backend=LiveNativeBackend(runtime,root/'native',exporter=exporter)
            client=root/'client';client.mkdir(mode=0o700)
            a,c=socket.socketpair();deadline=time.monotonic()+60
            p=ProducerPump(a,d.binding,deadline=deadline);packets=[];original=[]
            with HeadlessAVSink(client,d.binding,max_frames=124,timeout=30) as sink:
                receiver=ReceiverPump(c,d.binding,client,sink,deadline=deadline)
                results=[];errors=[];rt=thread_run(receiver.run,results,errors)
                def generate():
                    for packet in two_turn_packets(d,backend,original):
                        packets.append(packet);yield packet
                try:sent=p.run(generate())
                finally:
                    rt.join(65)
                    if rt.is_alive():c.close();sink.cancel()
                self.assertFalse(rt.is_alive());self.assertFalse(errors,errors)
                self.assertEqual(sent['acked'],6);self.assertLessEqual(sent['queue_highwater'],2)
                output=results[0]['output'];self.assertTrue(results[0]['complete'])
                audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:a','-f','s16le','pipe:1'],timeout=30)
                self.assertEqual(audio,b''.join(x.pcm for x in packets))
                for epoch,pcm in enumerate(original,1):
                    recovered=b''.join(x.pcm[(r.sample_start-x.envelope.sample_start)*2:(r.sample_stop-x.envelope.sample_start)*2]
                        for x in packets for r in x.envelope.regions if r.source_epoch==epoch)
                    self.assertEqual(recovered,pcm)
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_streams','-show_frames','-of','json',str(output)],timeout=30))
                self.assertEqual(len(probe['frames']),634)
                self.assertTrue(all(Fraction(f['pts'])*Fraction(probe['streams'][0]['time_base'])==Fraction(i,24)
                                    for i,f in enumerate(probe['frames'])))
                self.assertEqual(sink.timing['ack_count'],6)
                backend.retire();d.close()
                self.assertEqual(runtime.start_calls,0)
