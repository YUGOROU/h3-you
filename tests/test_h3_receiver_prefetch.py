"""Receive-only overlap using real duplex sockets; no devices or providers."""
import hashlib
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from research.h3_personal import paired_av_wire as wire
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av_pump import ProducerPump, ReceiverPump
from test_h3_paired_av_pump import DummySink, thread_run


def packets(b, count=3):
    for i in range(count):
        pcm=bytes([i+1])*4000; video=b'v'
        e=encode(b,i,i,i+1,video,pcm,(AudioRegion('speech',i*2000,(i+1)*2000,'source',1,i*2000,hashlib.sha256(pcm).hexdigest()),),i==count-1)
        yield Packet(e,video,pcm)


class Sink(DummySink):
    def __init__(self,*args,finish_fail=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.seen=[];self.finish_entered=threading.Event();self.finish_release=threading.Event()
        self.finish_fail=finish_fail
    def accept(self,path,lease):
        super().accept(path,lease)
        self.seen.append((lease.segment.index,lease.audio_pcm))
    def finish(self):
        self.finish_entered.set()
        if not self.finish_release.wait(3):raise TimeoutError('finish gate')
        if self.finish_fail:raise OSError('finish failure')
        return 'finished'


class PrefetchTests(unittest.TestCase):
    def test_overlap_fifo_and_final_finish(self):
        self.exercise(False)
    def test_finish_failure_no_final_ack(self):
        self.exercise(True)
    def exercise(self,fail):
        with tempfile.TemporaryDirectory() as root:
            b=LiveAVBinding('prefetch','a'*64,1,max_segments=8)
            a,c=socket.socketpair();deadline=time.monotonic()+8
            entered=threading.Event();release=threading.Event()
            sink=Sink(root,b,entered=entered,release=release,finish_fail=fail)
            p=ProducerPump(a,b,deadline=deadline);r=ReceiverPump(c,b,root,sink,deadline=deadline,prefetch=True)
            data=list(packets(b));errors=[];rr=[];pr=[]
            rt=thread_run(r.run,rr,errors);pt=thread_run(lambda:p.run(iter(data)),pr,errors)
            try:
                self.assertTrue(entered.wait(2))
                # Wait on the state condition, not polling the transport/generator.
                with r._cv:
                    self.assertTrue(r._cv.wait_for(lambda:r.receiver.snapshot()['verified']==2,2))
                    state=r.receiver.snapshot()
                    self.assertEqual(state['pending']+state['leased'],2)
                    self.assertEqual(state['sink_acked'],0)
                    self.assertLessEqual(len(r._receipts),1)
                with p._cv:self.assertEqual(p.sender.snapshot()['pending'],2)
                release.set();self.assertTrue(sink.finish_entered.wait(2),repr(errors))
                with p._cv:self.assertLess(p.sender.snapshot()['acked'],3)
                with r._cv:self.assertEqual(r.receiver.snapshot()['sink_acked'],2)
            finally:
                release.set();sink.finish_release.set();pt.join(5);rt.join(5)
                if rt.is_alive() or pt.is_alive():r.cancel()
            self.assertFalse(rt.is_alive());self.assertFalse(pt.is_alive());self.assertFalse(r._thread.is_alive())
            self.assertEqual(sink.seen,[(i,x.pcm) for i,x in enumerate(data)])
            if fail:
                self.assertTrue(errors);self.assertEqual(r.receiver.snapshot()['inflight_bytes'],0)
                self.assertFalse(rr)
            else:
                self.assertFalse(errors,errors);self.assertTrue(rr[0]['complete']);self.assertEqual(pr[0]['acked'],3)
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_corrupt_partial_deadline_and_cancel(self):
        for mode in ('corrupt','partial','deadline','cancel','sink'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                b=LiveAVBinding('failure','b'*64,1)
                a,c=socket.socketpair();deadline=time.monotonic()+(.15 if mode=='deadline' else 3)
                sink=DummySink(root,b,fail=mode=='sink')
                r=ReceiverPump(c,b,root,sink,deadline=deadline,prefetch=True)
                errors=[];rt=thread_run(r.run,[],errors)
                try:
                    packet=next(packets(b,1))
                    if mode=='corrupt':
                        wire._write(a,wire._pack(dict(schema='h3-paired-av-wire-v1',envelope=wire.envelope_record(packet.envelope))),deadline)
                        wire._write(a,b'x'+packet.pcm,deadline)
                    elif mode=='partial':
                        a.sendall(wire.MAGIC[:2]);a.shutdown(socket.SHUT_WR)
                    elif mode=='cancel':r.cancel()
                    elif mode=='sink':wire.send_pair(a,packet.envelope,packet.video,packet.pcm,deadline=deadline)
                    rt.join(4)
                finally:
                    a.close()
                    if rt.is_alive():r.cancel();rt.join(3)
                self.assertFalse(rt.is_alive());self.assertFalse(r._thread.is_alive(),repr(errors));self.assertTrue(errors)
                state=r.receiver.snapshot()
                self.assertEqual(state['inflight_bytes'],0);self.assertEqual(state['pending']+state['leased'],0)
                self.assertTrue(sink.cancelled);self.assertEqual(list(Path(root).iterdir()),[])


if __name__=='__main__':unittest.main()
