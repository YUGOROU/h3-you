"""Text-turn contracts with fake LLM/TTS/H3, no network/models/GPU."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import wave
from research.h3_personal.conversation_producer import ConversationProducer,ConversationInterrupted,SpokenLLM,LocalZeroShotTTS
from research.h3_personal.generated_transfer import H3_REVISION
from research.h3_personal.component_cache import sha256


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.current=True;self.tts_calls=0;self.h3_calls=0
        self.text='Hello!';self.seconds=.2;self.wrong=False
        self.llm=N(reply=lambda text,history:self.text)
        self.tts=N(synthesize=self.synthesize)
        self.h3=N(generate=self.generate,max_audio_seconds=124/24)
        self.p=ConversationProducer(self.llm,self.tts,self.h3,'a'*64)

    def synthesize(self,text,output):
        self.tts_calls+=1
        with wave.open(str(output),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
            f.writeframes(b'\0\0'*int(self.seconds*16000))

    def generate(self,audio,output,directory,is_current):
        self.h3_calls+=1;output.write_bytes(b'generated-video')
        return {'h3_revision':H3_REVISION,'adapter_sha256':'b'*64 if self.wrong else 'a'*64,
                'driving_audio_sha256':sha256(audio)}

    def run_turn(self,name='turn'):
        return self.p.produce('Hi',self.root/name,is_current=lambda:self.current,
                              validator=lambda root,clip:Fraction(1,4))

    def test_complete_bundle_and_history_commit(self):
        path=self.run_turn();manifest=json.loads(path.read_text())
        self.assertEqual(manifest['audio_source'],'voxcpm2_zero_shot_generated')
        self.assertEqual(set(p.name for p in path.parent.iterdir()),{'manifest.json','clips.json','video.mp4','tts.wav'})
        self.assertEqual(manifest['assets']['original_tts']['sha256'],sha256(path.parent/'tts.wav'))
        self.assertEqual(len(self.p.history),2);self.assertEqual((self.tts_calls,self.h3_calls),(1,1))
        with self.assertRaises(ValueError):self.run_turn()

    def test_cancellation_after_tts_no_video_no_history(self):
        original=self.synthesize
        def cancel(text,path):original(text,path);self.current=False
        self.tts.synthesize=cancel
        with self.assertRaises(ConversationInterrupted):self.run_turn()
        self.assertEqual(self.h3_calls,0);self.assertFalse(self.p.history);self.assertEqual(list(self.root.iterdir()),[])

    def test_oversized_speech_not_truncated(self):
        self.seconds=6
        with self.assertRaises(ValueError):self.run_turn()
        self.assertEqual(self.h3_calls,0);self.assertFalse(self.p.history)

    def test_wrong_adapter_or_invalid_av_no_publication(self):
        self.wrong=True
        with self.assertRaises(ValueError):self.run_turn()
        self.assertFalse(self.p.history);self.assertEqual(list(self.root.iterdir()),[])
        self.wrong=False
        def invalid(*a):raise ValueError('invalid AV')
        with self.assertRaises(ValueError):self.p.produce('Hi',self.root/'turn',validator=invalid)
        self.assertFalse(self.p.history)

    def test_reentrant_and_incomplete_llm_refused(self):
        self.p._lock.acquire()
        with self.assertRaises(RuntimeError):self.run_turn()
        self.p._lock.release();self.text='x'*241
        with self.assertRaises(ValueError):self.run_turn()
        self.assertEqual(self.tts_calls,0)

    def test_spoken_llm_uses_actual_user_and_complete_reply(self):
        llm=SpokenLLM('http://127.0.0.1:8000/v1/chat/completions','test')
        response={'choices':[{'finish_reason':'stop','message':{'content':'Hi!'}}]}
        with patch('research.h3_personal.e2e.request',return_value=(json.dumps(response).encode(),{})) as send:
            self.assertEqual(llm.reply('actual user',[]),'Hi!')
            payload=json.loads(send.call_args.args[1]);self.assertEqual(payload['messages'][-1]['content'],'actual user')
        with self.assertRaises(ValueError):SpokenLLM('https://example.com','test')

    def test_requested_router_defaults_and_safe_cached_auth(self):
        from research.h3_personal.conversation_producer import HF_ROUTER_ENDPOINT,HF_ROUTER_MODEL,hf_router_token
        self.assertEqual(HF_ROUTER_ENDPOINT,'https://router.huggingface.co/v1/chat/completions')
        self.assertEqual(HF_ROUTER_MODEL,'Qwen/Qwen3.8-27B:cerebras')
        with patch.dict('os.environ',{'HF_TOKEN':'synthetic-test-token'},clear=True):
            self.assertEqual(hf_router_token(),'synthetic-test-token')
        with patch.dict('os.environ',{},clear=True), \
             patch('research.provision_modal_identity_secret.cached_token',return_value='synthetic-cache-token') as read:
            self.assertEqual(hf_router_token(),'synthetic-cache-token');read.assert_called_once_with()

    def test_hf_router_uses_verified_sdk_without_provider_fallback(self):
        from research.h3_personal.conversation_producer import HF_ROUTER_ENDPOINT,HF_ROUTER_MODEL
        result=N(choices=[N(finish_reason='stop',message=N(content='こんにちは！'))])
        client=N(chat=N(completions=N(create=lambda **kw:result)))
        from unittest.mock import Mock
        constructor=Mock(return_value=client)
        with patch.dict('sys.modules',huggingface_hub=N(InferenceClient=constructor)):
            llm=SpokenLLM(HF_ROUTER_ENDPOINT,HF_ROUTER_MODEL,allow_remote=True,key='synthetic')
            self.assertEqual(llm.reply('hello',[]),'こんにちは！')
        self.assertEqual(constructor.call_args.kwargs['base_url'],'https://router.huggingface.co/v1')

    def test_no_tts_model_download_or_install(self):
        with self.assertRaises(ValueError):LocalZeroShotTTS('OpenBMB/VoxCPM2','missing.wav','a'*64)


if __name__=='__main__':unittest.main()
