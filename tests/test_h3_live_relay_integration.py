"""Synthetic, local HTTP carrier: no HF/deployment/device access."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research.live_relay_server import build_server
from research.h3_personal.live_relay import RelayConfig
from research.h3_personal.live_relay_client import SocketAdapter,RelayError
from research.h3_personal.live_relay_endpoint import attest,PrivateEndpoint
from research.h3_personal.paired_av import LiveAVBinding,AudioRegion,encode
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av_pump import ProducerPump,ReceiverPump


class RelayIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.expiry=time.time()+60
        self.config=RelayConfig('session',self.expiry,'a'*64,'b'*64)
        self.server=build_server(('127.0.0.1',0),self.config)
        self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.clients=[]
    def tearDown(self):
        for c in self.clients:c.close()
        self.server.shutdown();self.server.server_close();self.thread.join()
    def client(self,role):
        c=SocketAdapter(f'http://127.0.0.1:{self.server.server_port}',session='session',not_after=self.expiry,
            role=role,role_key=('a' if role=='producer' else 'b')*64,hf_token='fake-hf-token',allow_test_loopback=True)
        self.clients.append(c);return c.open_claim()
    def test_actual_server_client_empty_wait_does_not_become_eos(self):
        p=self.client('producer');c=self.client('consumer');errors=[]
        def later():
            try:time.sleep(2.2);p.send(b'after-long-poll')
            except Exception as exc:errors.append(exc)
        t=threading.Thread(target=later);t.start()
        c.settimeout(8)
        self.assertEqual(c.recv(100),b'after-long-poll');t.join();self.assertFalse(errors)
        c.send_input(b'new-synthetic-request');self.assertEqual(p.read_input(),b'new-synthetic-request')
        c.abort()
        self.assertTrue(self.server.relay._terminal)
        self.assertTrue(all(not q for q in self.server.relay._queues.values()))
    def test_production_needs_attested_private_endpoint_before_claim(self):
        c=SocketAdapter('https://some-relay.hf.space',session='session',not_after=self.expiry,
                        role='producer',role_key='a'*64,hf_token='fake-hf-token')
        self.clients.append(c)
        with patch('http.client.HTTPSConnection') as connection:
            with self.assertRaises(RelayError):c.open_claim()
            c.abort();connection.assert_not_called()
    def test_private_endpoint_metadata_and_anonymous_denial(self):
        info=NS(id='YUGOROU/h3-live-relay-test1234',private=True,sha='a'*40,host='owner-relay.hf.space')
        api=NS(endpoint='https://huggingface.co',space_info=lambda *a,**kw:info)
        with patch('research.h3_personal.live_relay_endpoint.http.client.HTTPSConnection') as http:
            http.return_value.getresponse.return_value.status=404
            permit=attest(api,space_id=info.id,revision=info.sha)
            self.assertTrue(permit.permits('https://owner-relay.hf.space'))
            headers=http.return_value.request.call_args.kwargs['headers'];self.assertNotIn('Authorization',headers)
            self.assertFalse(PrivateEndpoint(permit.base_url,info.id,info.sha).permits(permit.base_url))
            http.return_value.getresponse.return_value.status=200
            with self.assertRaises(ValueError):attest(api,space_id=info.id,revision=info.sha)
            info.private=False
            with self.assertRaises(ValueError):attest(api,space_id=info.id,revision=info.sha)
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_http_relay_to_pumps_to_headless_exact_pcm_pts(self):
        from research.h3_personal.headless_av_sink import HeadlessAVSink
        from research.run_h3_realtime_first import encode_rgb_segment
        from research.h3_personal.conversation_scenario import segment_timing
        from fractions import Fraction
        p=self.client('producer');c=self.client('consumer');deadline=time.monotonic()+30
        b=LiveAVBinding('session','c'*64,1,max_segments=8);packets=[];start=0
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);client=root/'client';client.mkdir(mode=0o700)
            for index,n in enumerate((5,7)):
                path=root/f'generated-{index}.mp4'
                encode_rgb_segment(bytes([40+index*30,20,90])*(252*448*n),path,
                    segment_timing(index,start,start+n),width=252,height=448)
                video=path.read_bytes();pcm=(index+1).to_bytes(2,'little')*(n*2000)
                region=AudioRegion('speech',start*2000,(start+n)*2000,'synthetic',1,start*2000,hashlib.sha256(pcm).hexdigest())
                envelope=encode(b,index,start,start+n,video,pcm,(region,),index==1)
                packets.append(Packet(envelope,video,pcm));start+=n
            with HeadlessAVSink(client,b,max_frames=124,timeout=20) as sink:
                receiver=ReceiverPump(c,b,client,sink,deadline=deadline);results=[];errors=[]
                def consume():
                    try:results.append(receiver.run())
                    except BaseException as exc:errors.append(exc)
                t=threading.Thread(target=consume);t.start()
                producer=ProducerPump(p,b,deadline=deadline)
                try:report=producer.run(iter(packets))
                finally:t.join(35)
                self.assertFalse(t.is_alive());self.assertFalse(errors,errors)
                self.assertEqual(report['acked'],2);self.assertLessEqual(report['queue_highwater'],2)
                self.assertTrue(results[0]['complete']);self.assertFalse(producer._thread.is_alive())
                audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(results[0]['output']),'-map','0:a','-f','s16le','pipe:1'],timeout=20)
                self.assertEqual(audio,b''.join(p.pcm for p in packets))
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_frames','-show_streams','-of','json',str(results[0]['output'])],timeout=20))
                self.assertEqual(len(probe['frames']),12)
                self.assertTrue(all(Fraction(f['pts'])*Fraction(probe['streams'][0]['time_base'])==Fraction(i,24) for i,f in enumerate(probe['frames'])))
