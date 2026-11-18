"""Text -> complete LLM utterance -> zero-shot TTS -> pinned H3 -> ready bundle.

Injected adapters are not automatically started. No ASR/VAD, tool agent, networking,
paid Job or playback at import. Publish/history commit only after complete A/V.
"""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import wave
from .component_cache import canonical, sha256, digest_ok
from .generated_transfer import H3_REVISION
from .buffered_stream import Clip, validate_clip


class ConversationInterrupted(RuntimeError):pass


HF_ROUTER_ENDPOINT='https://router.huggingface.co/v1/chat/completions'
HF_ROUTER_MODEL='Qwen/Qwen3.8-27B:cerebras'


def hf_router_token():
    # Never echo/export into logs; HF_TOKEN from user's shell takes precedence.
    if os.environ.get('HF_TOKEN'):return os.environ['HF_TOKEN']
    from research.provision_modal_identity_secret import cached_token
    return cached_token()


class SpokenLLM:
    def __init__(self, endpoint, model, *, allow_remote=False, key='', timeout=120):
        from .e2e import check_endpoint
        check_endpoint(endpoint,allow_remote)
        self.endpoint=endpoint;self.model=model;self.allow_remote=allow_remote
        self.key=key;self.timeout=timeout

    def reply(self, text, history):
        from .e2e import request
        messages=[{'role':'system','content':
            'You are an explicitly AI avatar, not the real person. Reply to the user in one '
            'brief natural spoken sentence, about3-4seconds. No stage directions or markup.'}]
        messages+=copy.deepcopy(history)
        messages.append({'role':'user','content':text})
        if self.endpoint==HF_ROUTER_ENDPOINT:
            # Verified official SDK path; urllib POST returned403 despite valid auth/model.
            from huggingface_hub import InferenceClient
            import sys
            hub_offline=(os.environ.get('HF_HUB_OFFLINE')=='1' or
                os.environ.get('TRANSFORMERS_OFFLINE')=='1' or
                getattr(sys.modules.get('huggingface_hub.constants'),'HF_HUB_OFFLINE',False))
            if hub_offline:
                # Keep model-loader process strictly offline. ONLY chat SDK gets online
                # access in a separate short-lived process, no global offline mutation.
                import subprocess
                env=dict(os.environ,HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_DEBUG='0',HF_TOKEN=self.key)
                payload=json.dumps({'model':self.model,'messages':messages,'timeout':self.timeout})
                completed=subprocess.run([sys.executable,'-m','research.hf_router_spoken_reply'],
                    input=payload,capture_output=True,text=True,env=env,timeout=self.timeout+5)
                if completed.returncode:raise RuntimeError('HF spoken chat process failed; stderr withheld')
                if len(completed.stdout)>16384:raise ValueError('bounded chat result required')
                result=json.loads(completed.stdout);content=result['content'];finish=result['finish_reason']
            else:
                client=InferenceClient(base_url='https://router.huggingface.co/v1',
                                       api_key=self.key,timeout=self.timeout)
                result=client.chat.completions.create(model=self.model,messages=messages,
                    max_tokens=160,temperature=.8,stream=False).choices[0]
                content=result.message.content;finish=result.finish_reason
        else:
            raw,_=request(self.endpoint,json.dumps({'model':self.model,'messages':messages,
                'max_tokens':160,'temperature':.8,'stream':False}).encode(),'application/json',
                key=self.key,timeout=self.timeout,allow_remote=self.allow_remote)
            result=json.loads(raw)['choices'][0];content=result['message']['content']
            finish=result.get('finish_reason')
        if finish!='stop' or not isinstance(content,str) or not content.strip():
            raise ValueError('complete spoken LLM response required')
        return content.strip()


class LocalZeroShotTTS:
    def __init__(self, model_path, reference, reference_sha256):
        if (not Path(model_path).is_dir() or sha256(reference)!=reference_sha256 or
                os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1'):
            raise ValueError('preprovisioned offline TTS model and pinned local voice required')
        self.model_path=model_path;self.reference=reference;self.reference_sha256=reference_sha256

    def synthesize(self,text,output):
        # Do not assume VoxCPM2 and the H3 DiT can coexist on one GPU.
        import gc
        import sys
        import soundfile as sf
        from voxcpm import VoxCPM
        if sha256(self.reference)!=self.reference_sha256:raise ValueError('voice reference changed')
        # Reuse research/hf_infer_personal.py's fixed zero-shot recipe, no badcase retry.
        backend=VoxCPM.from_pretrained(self.model_path,load_denoiser=False,
                                       device='cuda:0',optimize=False)
        try:
            wav=backend.generate(text=text,reference_wav_path=str(self.reference),cfg_value=2.0,
                inference_timesteps=10,normalize=False,denoise=False,max_len=93,
                retry_badcase=False,seed=42)
            sf.write(output,wav,backend.tts_model.sample_rate,subtype='PCM_16')
        finally:
            del backend;gc.collect()
            if 'torch' in sys.modules:sys.modules['torch'].cuda.empty_cache()


class PinnedH3Backend:
    def __init__(self,request,checkpoint,case,adapter_sha256,*,session=None):
        self.session=session
        self.request=copy.deepcopy(request);self.checkpoint=Path(checkpoint)
        self.case=copy.deepcopy(case);self.adapter_sha256=adapter_sha256
        pin=self.request.get('pins',{}).get('models',{}).get('transformer_ref',{})
        if pin!={'repo_id':'MiniMaxAI/MiniMax-H3','revision':H3_REVISION}:
            raise ValueError('original pinned H3 Ref2VA base required')
        if sha256(self.checkpoint/'adapter_model.safetensors')!=adapter_sha256:
            raise ValueError('actual identity adapter differs')
        self.max_audio_seconds=124/24

    def prepare_for_tts(self):
        # Vox peak not yet measured: no speculative Vox+DiT co-residency.
        if self.session is not None and self.session.resident:self.session.release_model()

    def generate(self,audio,output,directory,is_current):
        from research.hf_identity_preview import generate,decode,TurnInterrupted
        if sha256(self.checkpoint/'adapter_model.safetensors')!=self.adapter_sha256:
            raise ValueError('identity adapter changed')
        case=copy.deepcopy(self.case);case.update(output=str(directory),
            audio_path=str(audio),audio_sha256=sha256(audio),
            driving_audio={'path':str(audio),'sha256':sha256(audio)})
        if case.get('schedule_points')!=50:raise ValueError('baseline50points required')
        try:generate(self.request,self.checkpoint,case,session=self.session,is_current=is_current)
        except TurnInterrupted as exc:raise ConversationInterrupted('H3 turn interrupted') from exc
        if not is_current():raise ConversationInterrupted('cancel before VAE')
        if self.session is not None and not self.session.allow_measured_auxiliary:
            self.session.release_model()
        try:decode(self.request,case,session=self.session)
        except BaseException:
            if self.session is not None:self.session.release_model()
            raise
        shutil.copyfile(directory/'video-only.mp4',output)
        return {'h3_revision':H3_REVISION,'adapter_sha256':self.adapter_sha256,
                'driving_audio_sha256':sha256(audio)}


class ConversationProducer:
    def __init__(self,llm,tts,h3,adapter_sha256):
        if not digest_ok(adapter_sha256):raise ValueError('expected adapter SHA required')
        self.llm=llm;self.tts=tts;self.h3=h3;self.adapter_sha256=adapter_sha256
        self.history=[];self._lock=threading.Lock()

    def produce(self,user_text,destination,*,is_current=lambda:True,validator=validate_clip):
        if not isinstance(user_text,str) or not user_text.strip() or len(user_text)>4096:
            raise ValueError('bounded text input required')
        if not self._lock.acquire(blocking=False):raise RuntimeError('overlapping conversation turns refused')
        staging=None
        def check():
            if not is_current():raise ConversationInterrupted('old conversation epoch discarded')
        try:
            check();destination=Path(destination)
            if destination.exists() or destination.is_symlink():raise ValueError('never overwrite a conversation bundle')
            destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            if destination.parent.is_symlink() or destination.parent.stat().st_mode & 0o077:
                raise ValueError('private output parent required')
            staging=Path(tempfile.mkdtemp(prefix='.conversation-',dir=destination.parent))
            utterance=self.llm.reply(user_text,copy.deepcopy(self.history));check()
            if not isinstance(utterance,str) or not utterance.strip() or len(utterance)>240:
                raise ValueError('short complete utterance required; no silent truncation')
            audio=staging/'tts.wav'
            if hasattr(self.h3,'prepare_for_tts'):self.h3.prepare_for_tts()
            self.tts.synthesize(utterance,audio);check()
            if audio.is_symlink():raise ValueError('TTS must be a regular local artifact')
            with wave.open(str(audio),'rb') as wav:
                duration=wav.getnframes()/wav.getframerate()
                if wav.getsampwidth()!=2 or wav.getnchannels() not in (1,2) or not 0<duration<=self.h3.max_audio_seconds:
                    raise ValueError('PCM16 TTS must fit fixed H3 geometry; shorten utterance, never truncate')
            audio_sha=sha256(audio);video=staging/'video.mp4'
            proof=self.h3.generate(audio,video,staging,is_current);check()
            if (proof!={'h3_revision':H3_REVISION,'adapter_sha256':self.adapter_sha256,
                        'driving_audio_sha256':audio_sha} or sha256(audio)!=audio_sha or video.is_symlink()):
                raise ValueError('H3 must use expected identity and actual generated TTS')
            clip=Clip(video.name,audio.name,sha256(video),audio_sha)
            validator(staging,clip);check()
            manifest={'schema':'h3-generated-playback-transfer-v1','ai_generated':True,
                'audio_source':'voxcpm2_zero_shot_generated','video_source':'h3_ref2va_identity',
                'h3_revision':H3_REVISION,'adapter_sha256':self.adapter_sha256,
                'assets':{k:{'file':p.name,'sha256':sha256(p),'bytes':p.stat().st_size}
                          for k,p in (('video',video),('original_tts',audio))}}
            (staging/'manifest.json').write_bytes(canonical(manifest))
            (staging/'clips.json').write_bytes(canonical({'schema':'h3-buffered-playback-v1','clips':[clip.__dict__]}))
            # Publish only generated playback files, not backend intermediates/transcript.
            ready=staging/'ready';ready.mkdir(mode=0o700)
            for name in ('video.mp4','tts.wav','manifest.json','clips.json'):
                shutil.move(staging/name,ready/name);(ready/name).chmod(0o600)
            check()
            if destination.exists() or destination.is_symlink():raise ValueError('destination appeared')
            os.rename(ready,destination)
            self.history=(self.history+[{'role':'user','content':user_text},
                {'role':'assistant','content':utterance}])[-16:]
            return destination/'manifest.json'
        finally:
            if staging is not None and staging.exists():shutil.rmtree(staging)
            self._lock.release()
