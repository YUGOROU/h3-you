import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest

from research.h3_personal.paired_av import LiveAVBinding,AudioRegion,Receiver,encode,ProtocolError
from research.h3_personal import paired_av_wire as w


class WireTests(unittest.TestCase):
    def pair(self,index=0,start=0,frames=1,final=False,video=b'video'):
        b=LiveAVBinding('session','a'*64,1)
        pcm=b'\x01\x00'*(frames*2000)
        region=AudioRegion('speech',start*2000,(start+frames)*2000,'source',1,start*2000,hashlib.sha256(pcm).hexdigest())
        return encode(b,index,start,start+frames,video,pcm,(region,),final),video,pcm

    def test_streaming_receive_and_two_distinct_acks(self):
        pair=self.pair(final=True);r=Receiver(pair[0].binding,deadline=time.monotonic()+10)
        a,b=socket.socketpair();self.addCleanup(a.close);self.addCleanup(b.close)
        deadline=time.monotonic()+5
        w.send_pair(a,*pair,deadline=deadline)
        receipt=w.receive_pair(b,r,deadline=deadline)
        w.send_ack(b,receipt.envelope,stage='verified',deadline=deadline)
        w.receive_ack(a,pair[0],stage='verified',deadline=deadline)
        self.assertEqual(r.snapshot()['sink_acked'],0)
        lease=r.lease();self.assertEqual(lease.audio_pcm,pair[2]);r.ack(lease)
        w.send_ack(b,receipt.envelope,stage='sink-accepted',deadline=deadline)
        w.receive_ack(a,pair[0],stage='sink-accepted',deadline=deadline)
        self.assertTrue(r.snapshot()['complete'])

    def test_wrong_ack_stage_never_promotes_receipt(self):
        pair=self.pair();a,b=socket.socketpair();self.addCleanup(a.close);self.addCleanup(b.close)
        w.send_ack(b,pair[0],stage='verified',deadline=time.monotonic()+2)
        with self.assertRaises(ProtocolError):w.receive_ack(a,pair[0],stage='sink-accepted',deadline=time.monotonic()+2)

    def test_strict_headers_before_bodies(self):
        e=self.pair()[0];value=dict(schema='h3-paired-av-wire-v1',envelope=w.envelope_record(e))
        self.assertEqual(w.parse_envelope(json.loads(json.dumps(value))),e)
        for mutate in ('unknown','extra_binding','many_regions'):
            v=json.loads(json.dumps(value))
            if mutate=='unknown':v['envelope']['extra']=1
            if mutate=='extra_binding':v['envelope']['binding']['secret']=1
            if mutate=='many_regions':v['envelope']['regions']*=9
            with self.assertRaises(ProtocolError):w.parse_envelope(v)
        for body in (w.MAGIC+struct.pack('!I',w.MAX_HEADER+1),
                     w.MAGIC+struct.pack('!I',13)+b'{"a":1,"a":2}'):
            a,b=socket.socketpair()
            try:
                a.sendall(body);a.shutdown(socket.SHUT_WR)
                r=Receiver(e.binding,deadline=time.monotonic()+2)
                with self.assertRaises((ProtocolError,EOFError)):w.receive_pair(b,r,deadline=time.monotonic()+1)
                self.assertIsNotNone(r.snapshot()['terminal'])
            finally:a.close();b.close()

    def test_stale_header_rejected_before_reading_unprovided_body(self):
        e=self.pair()[0]
        wrong=dataclasses.replace(e,binding=dataclasses.replace(e.binding,epoch=2))
        a,b=socket.socketpair()
        try:
            a.sendall(w._pack(dict(schema='h3-paired-av-wire-v1',envelope=w.envelope_record(wrong))))
            r=Receiver(e.binding,deadline=time.monotonic()+2)
            with self.assertRaises(ProtocolError):w.receive_pair(b,r,deadline=time.monotonic()+1)
            self.assertIsNotNone(r.snapshot()['terminal'])
        finally:a.close();b.close()

    def test_truncated_pair_is_terminal_no_replay(self):
        e=self.pair()[0];a,b=socket.socketpair()
        try:
            a.sendall(w._pack(dict(schema='h3-paired-av-wire-v1',envelope=w.envelope_record(e))));a.shutdown(socket.SHUT_WR)
            r=Receiver(e.binding,deadline=time.monotonic()+2)
            with self.assertRaises(EOFError):w.receive_pair(b,r,deadline=time.monotonic()+1)
            self.assertIsNotNone(r.snapshot()['terminal'])
            with self.assertRaises(ProtocolError):r.offer(*self.pair())
        finally:a.close();b.close()

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_real_headless_paired_sink_continuous_pts_original_pcm(self):
        from research.h3_personal.headless_av_sink import HeadlessAVSink
        from research.run_h3_realtime_first import encode_rgb_segment
        from research.h3_personal.conversation_scenario import segment_timing
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);binding=LiveAVBinding('session','a'*64,1)
            r=Receiver(binding,deadline=time.monotonic()+30)
            expected=[];cursor=0
            with HeadlessAVSink(d,binding,width=252,height=448,max_frames=124,timeout=20) as sink:
                for index,n in enumerate((5,7)):
                    path=d/f'seg{index}.mp4'
                    encode_rgb_segment(bytes([index*80+25,50,100])*(252*448*n),path,
                        segment_timing(index,cursor,cursor+n),width=252,height=448)
                    pair=self.pair(index,cursor,n,index==1,path.read_bytes())
                    r.offer(*pair);lease=r.lease();sink.accept(path.name,lease);r.ack(lease)
                    expected.append(pair[2]);cursor+=n
                output=sink.finish()
                audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:a','-f','s16le','pipe:1'],timeout=20)
                self.assertEqual(audio,b''.join(expected));self.assertTrue(r.snapshot()['complete'])
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_frames','-show_streams','-of','json',str(output)],timeout=20))
                self.assertEqual(len(info['frames']),12)
                self.assertTrue(all(Fraction(f['pts'])*Fraction(info['streams'][0]['time_base'])==Fraction(i,24) for i,f in enumerate(info['frames'])))
                self.assertEqual(sink.timing['ack_count'],2)


class OBSSinkTests(unittest.TestCase):
    def test_approval_gate_and_fixed_local_destination_no_launch(self):
        from research.h3_personal.obs_av_sink import OBSAVSink,output_arguments
        for p in (True,0,1023,65536,'1234','udp://elsewhere'):
            with self.assertRaises(ValueError):output_arguments(p)
        args=output_arguments(19390)
        self.assertEqual(args[-1],'udp://127.0.0.1:19390?pkt_size=1316')
        self.assertIn('zerolatency',args);self.assertIn('aac',args)
        with self.assertRaises(PermissionError):OBSAVSink('not-accessed',None,port=19390)
        with self.assertRaises(ValueError):OBSAVSink('not-accessed',None,port=19390,playback_approved=True,paced=False)
