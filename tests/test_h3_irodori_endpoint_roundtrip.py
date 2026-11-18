import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import wave

from research.h3_personal.irodori_endpoint_service import Gate
from research.h3_personal.irodori_endpoint_client import RemoteIrodoriTTS
from research.h3_personal.director_tts_job import SOURCE,MODEL,CODEC,REFERENCE_SHA


class EndpointRoundTripTests(unittest.TestCase):
    def test_native_receipt_adapter_roundtrip_no_audio_transform(self):
        out=io.BytesIO()
        with wave.open(out,'wb') as w:
            w.setparams((1,2,48000,0,'NONE','not compressed'));w.writeframes(b'\x01\x00'*4800)
        body=out.getvalue();paths=[]

        class TTS:
            def synthesize(self,text,turn_id,directory,deadline):
                self.assert_deadline=deadline
                directory.mkdir();p=directory/'reply.wav';p.write_bytes(body);paths.append(p)
                return dict(wav=str(p),receipt=dict(path=str(p),sha256=hashlib.sha256(body).hexdigest(),
                    samples=4800,sample_rate=48000,channels=1,watermark_applied=True,
                    source_revision=SOURCE,model_revision=MODEL,codec_revision=CODEC,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest()),
                    metrics=dict(synthesis_seconds=.01,cuda_peak_allocated_bytes=100))
            def close(self):pass

        tts=TTS()
        ready=dict(ready=True,device_name='NVIDIA L4',device_count=1,model_precision='fp32',
                   codec_precision='fp32',compile_model=False,watermark_ready=True,
                   source_revision=SOURCE,model_revision=MODEL,codec_revision=CODEC,reference_sha256=REFERENCE_SHA)
        gate=Gate(tts,ready,key='k'*48,session='test',expires_at=time.time()+300)
        def transport(method,path,payload,headers,deadline):
            self.assertIn('Authorization',headers)
            if path=='/ready':
                return 200,json.dumps(dict(ready,schema='h3-irodori-endpoint-ready-v1',session='test')).encode()
            code,value=gate.generate(json.loads(payload))
            return code,json.dumps(value).encode()
        client=RemoteIrodoriTTS('https://fake.us-east-1.aws.endpoints.huggingface.cloud',
            token='TOKEN',key='k'*48,session='test',expires_at=time.time()+300,transport=transport)
        client.read('tts_ready',time.monotonic()+5)
        with tempfile.TemporaryDirectory() as root:
            for i in range(2):
                r=client.synthesize('新しい合成テストです。',str(i),Path(root)/str(i),time.monotonic()+5)
                self.assertEqual(Path(r['wav']).read_bytes(),body)
                self.assertEqual(r['metrics']['cuda_peak_allocated_bytes'],100)
                self.assertFalse(paths[-1].exists())
        self.assertLessEqual(tts.assert_deadline,time.monotonic()+120)
        client.close();gate.close()
