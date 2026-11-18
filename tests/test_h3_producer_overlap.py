"""Offline bounded generation/enqueue overlap; no provider or playback."""
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal import paired_av_wire as wire
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal.paired_av_pump import ProducerPump, ReceiverPump
import test_h3_paired_av_pump as fixtures


class OverlapTests(unittest.TestCase):
    def test_blocked_send_allows_second_pull_but_never_third_before_sink_ack(self):
        with tempfile.TemporaryDirectory() as root:
            binding=LiveAVBinding('overlap','a'*64,1,max_segments=8)
            a,b=socket.socketpair()
            deadline=time.monotonic()+10
            producer=ProducerPump(a,binding,deadline=deadline,overlap_send=True)
            sink_entered=threading.Event();sink_release=threading.Event()
            sink=fixtures.DummySink(root,binding,entered=sink_entered,release=sink_release)
            receiver=ReceiverPump(b,binding,root,sink,deadline=deadline,prefetch=True)
            send_entered=threading.Event();send_release=threading.Event()
            second=threading.Event();third=threading.Event()
            owner=[];pull_threads=[]
            original=wire.send_pair
            def blocked(sock,envelope,*args,**kwargs):
                if envelope.index==0:
                    send_entered.set()
                    if not send_release.wait(3):raise TimeoutError('test send gate')
                return original(sock,envelope,*args,**kwargs)
            def packets():
                owner.append(threading.get_ident())
                for i in range(3):
                    pull_threads.append(threading.get_ident())
                    if i==1:second.set()
                    if i==2:third.set()
                    video=b'v';pcm=bytes(4000)
                    envelope=encode(binding,i,i,i+1,video,pcm,
                        (AudioRegion('silence',i*2000,(i+1)*2000),),i==2)
                    yield Packet(envelope,video,pcm)
            results=[];received=[];errors=[]
            with patch.object(wire,'send_pair',side_effect=blocked):
                rt=fixtures.thread_run(receiver.run,received,errors)
                pt=fixtures.thread_run(lambda:producer.run(packets()),results,errors)
                try:
                    self.assertTrue(send_entered.wait(2))
                    self.assertTrue(second.wait(2),'second generation blocked behind send')
                    self.assertFalse(third.is_set())
                    send_release.set()
                    self.assertTrue(sink_entered.wait(2))
                    # A verified ACK must not free a slot; sink is still blocked.
                    limit=time.monotonic()+2
                    while time.monotonic()<limit:
                        if any('ack_verified' in r for r in producer.timing()['windows']):break
                        time.sleep(.005)
                    self.assertTrue(any('ack_verified' in r for r in producer.timing()['windows']))
                    self.assertFalse(third.is_set())
                    self.assertTrue(pt.is_alive(),'returned before final sink ACK')
                finally:
                    send_release.set();sink_release.set()
                    pt.join(10);rt.join(10)
            self.assertFalse(pt.is_alive());self.assertFalse(rt.is_alive())
            self.assertFalse(errors,errors)
            self.assertEqual(results[0]['acked'],3)
            self.assertEqual(results[0]['queue_highwater'],2)
            self.assertTrue(received[0]['complete'])
            self.assertEqual(set(pull_threads),set(owner))
            rows=producer.timing()['windows']
            self.assertLess(rows[1]['packet_ready'],rows[0]['send_enqueued'])
            self.assertFalse(producer._send_thread.is_alive())
            self.assertFalse(producer._thread.is_alive())
            self.assertFalse(producer._send_queue)

    def _overlap_factory(self,*args,**kwargs):
        return ProducerPump(*args,**kwargs,overlap_send=True)

    def test_real_ffmpeg_exact_pcm_and_pts_existing_contract_with_overlap(self):
        case=fixtures.PumpTests('test_two_turn_native_protocol_to_wire_to_headless_exact_audio')
        with patch.object(fixtures,'ProducerPump',side_effect=self._overlap_factory):
            case.test_two_turn_native_protocol_to_wire_to_headless_exact_audio()

    def test_sink_failure_existing_contract_with_overlap(self):
        case=fixtures.PumpTests('test_sink_failure_stops_replay_and_preserves_warm_model_for_new_owner')
        with patch.object(fixtures,'ProducerPump',side_effect=self._overlap_factory):
            case.test_sink_failure_stops_replay_and_preserves_warm_model_for_new_owner()

    def test_send_failure_wakes_credit_wait_and_reaps_both_threads(self):
        with tempfile.TemporaryDirectory() as root:
            binding=LiveAVBinding('sendfail','a'*64,1,max_segments=8)
            a,b=socket.socketpair();deadline=time.monotonic()+5
            producer=ProducerPump(a,binding,deadline=deadline,overlap_send=True)
            closed=threading.Event();pulled=[]
            def packets():
                try:
                    for i in range(3):
                        pulled.append(i);video=b'v';pcm=bytes(4000)
                        yield Packet(encode(binding,i,i,i+1,video,pcm,
                            (AudioRegion('silence',i*2000,(i+1)*2000),),i==2),video,pcm)
                finally:closed.set()
            try:
                with patch.object(wire,'send_pair',side_effect=OSError('synthetic send failure')):
                    with self.assertRaises(RuntimeError):producer.run(packets())
                self.assertTrue(closed.is_set())
                self.assertLessEqual(len(pulled),2)
                self.assertFalse(producer._send_thread.is_alive())
                self.assertFalse(producer._thread.is_alive())
            finally:b.close()

    def test_generator_failure_reaps_idle_sender_and_ack_reader(self):
        binding=LiveAVBinding('generatorfail','a'*64,1)
        a,b=socket.socketpair()
        producer=ProducerPump(a,binding,deadline=time.monotonic()+5,overlap_send=True)
        closed=threading.Event()
        def packets():
            try:
                raise ValueError('synthetic generation failure')
                yield
            finally:closed.set()
        try:
            with self.assertRaisesRegex(ValueError,'synthetic generation failure'):
                producer.run(packets())
            self.assertTrue(closed.is_set())
            self.assertFalse(producer._send_thread.is_alive())
            self.assertFalse(producer._thread.is_alive())
        finally:b.close()

    def test_flag_is_explicit_bool(self):
        binding=LiveAVBinding('flag','a'*64,1)
        for value in (1,None,'yes'):
            with self.assertRaises(ValueError):
                ProducerPump(None,binding,deadline=time.monotonic()+1,overlap_send=value)


if __name__=='__main__':unittest.main()
