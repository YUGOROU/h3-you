"""Real CPU FFmpeg/wire/epoch review integration; not GPU or audible acceptance."""
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest

from research.h3_personal.epoch_av_sink import EpochAVSink
from research.h3_personal.headless_av_sink import ReviewArchive
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av import LiveAVBinding,AudioRegion,PresentationCut,encode
from research.h3_personal.paired_av_pump import ProducerPump,ReceiverPump


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'existing FFmpeg required')
class ReviewEpochIntegration(unittest.TestCase):
    def test_real_cut_wire_partial_credit_archive_and_complete_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();deadline=time.monotonic()+30
            video=root/'synthetic.mp4'
            subprocess.run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i',
                'testsrc2=size=16x16:rate=24','-frames:v','24','-c:v','libx264',
                '-pix_fmt','yuv420p',str(video)],check=True,capture_output=True,timeout=10)
            body=video.read_bytes();binding=LiveAVBinding('fixture','a'*64,1,max_segments=4,max_total_frames=96)
            old=struct.pack('<h',2222)*48000
            tail=struct.pack('<h',2222)*10000
            quiet=bytes(38000*2)
            new=b''.join(struct.pack('<h',1234+(i%100)) for i in range(48000))
            def speech(start,pcm,name,epoch,offset):
                return AudioRegion('speech',start,start+len(pcm)//2,name,epoch,offset,hashlib.sha256(pcm).hexdigest())
            e0=encode(binding,0,0,24,body,old,(speech(0,old,'old',1,0),),False)
            e1=encode(binding,1,24,48,body,tail+quiet,
                (speech(48000,tail,'old',1,48000),AudioRegion('silence',58000,96000)),False,
                presentation_epoch=1)
            e1=replace(e1,presentation_cut=PresentationCut(29,0,1))
            e2=encode(binding,2,48,72,body,new,(speech(96000,new,'new',2,0),),True,presentation_epoch=1)
            packets=[Packet(e0,body,old),Packet(e1,body,tail+quiet),Packet(e2,body,new)]
            archive=ReviewArchive(root/'review',deadline)
            class Sink(EpochAVSink):
                def accept(self,path,lease):
                    result=super().accept(path,lease)
                    if lease.segment.index==0:self.invalidate(1,interrupted_source_epoch=1)
                    return result
            a,b=socket.socketpair();producer=ProducerPump(a,binding,deadline=deadline,overlap_send=True)
            errors=[]
            def produce():
                try:producer.run(iter(packets))
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=produce);thread.start()
            with Sink(root,binding,width=16,height=16,max_frames=124,max_total_frames=96,
                      paired_cuts=True,review_archive=archive,observe_output=True) as sink:
                outcome=ReceiverPump(b,binding,root,sink,deadline=deadline,prefetch=True).run()
                observed=sink.timing
            thread.join(5)
            self.assertFalse(thread.is_alive());self.assertFalse(errors,errors)
            self.assertTrue(outcome['transport_complete']);self.assertFalse(outcome['presentation_complete'])
            self.assertEqual(outcome['partial_acked'],1)
            self.assertEqual(outcome['sink_acked'],2)
            self.assertEqual(producer.sender.snapshot()['partial_acked'],1)
            self.assertEqual(observed['partially_presented_interrupted'],1)
            self.assertEqual(observed['decoded_source_output']['2']['samples'],48000)
            self.assertEqual(observed['decoded_source_output']['2']['sha256'],hashlib.sha256(new).hexdigest())
            saved=archive.snapshot()
            spans=[e for e in saved['entries'] if e['kind']=='encoder-span']
            references=[e for e in saved['entries'] if e['kind']=='full-native-cut-reference']
            self.assertEqual(len(spans),2);self.assertEqual(len(references),1)
            self.assertEqual(spans[-1]['metadata']['origin_frame'],29)
            self.assertEqual(references[0]['metadata']['omitted_frame_range'],[24,29])
            for entry in saved['entries']:
                for item in entry['files']:self.assertTrue((root/'review'/item['name']).exists())
            from research.launch_h3_direct_conversation import create_review_previews
            previews=create_review_previews(root/'review',max_seconds=30)
            self.assertEqual(len(previews['files']),3)
            self.assertTrue(all('file' in x for x in previews['files']),previews)
            self.assertTrue((root/'review/review-previews.json').exists())


if __name__=='__main__':unittest.main()
