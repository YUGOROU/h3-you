"""One finite resident voice+video session. Fresh text requests only after READY.

GPU child owns pinned offline Irodori; main owns resident pinned H3. No devices.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time


def emit(value):print(json.dumps(value),flush=True)

def native(root,source,paths):
    from irodori_tts.inference_runtime import InferenceRuntime,RuntimeKey,SamplingRequest,save_wav
    from research.h3_personal.director_irodori import IrodoriResident,SOURCE,MODEL,CODEC
    with contextlib.redirect_stdout(sys.stderr):
        runtime=InferenceRuntime.from_key(RuntimeKey(checkpoint=paths['model'],model_device='cuda:0',codec_repo=paths['codec'],model_precision='fp32',codec_device='cuda:0',codec_precision='fp32',compile_model=False))
        owner=IrodoriResident(runtime,SamplingRequest,save_wav,paths['reference'],paths['reference_sha256'],source_revision=SOURCE,model_revision=MODEL,codec_revision=CODEC)
    measured = os.environ.get('H3_TTS_MEASURE_MEMORY') == '1'
    def metrics():
        import torch
        model_ok = all(p.device.type == 'cuda' and p.dtype == torch.float32
                       for p in runtime.model.parameters() if p.is_floating_point())
        codec_ok = all(p.device.type == 'cuda' and p.dtype == torch.float32
                       for p in runtime.codec.model.parameters() if p.is_floating_point())
        if not model_ok or not codec_ok:
            raise RuntimeError('measured native model/codec device precision drift')
        return dict(model_all_cuda_fp32=model_ok, codec_all_cuda_fp32=codec_ok,
                    device_name=torch.cuda.get_device_name(0), device_count=torch.cuda.device_count(),
                    device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(0),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(0),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    emit(dict(type='tts_ready', **({'metrics': metrics()} if measured else {})))
    for line in sys.stdin:
        row=json.loads(line)
        if row.get('op')=='stop':break
        if row.get('op')!='synthesize' or not isinstance(row.get('text'),str) or not 1<=len(row['text'])<=160:raise ValueError('bounded fresh text required')
        out=Path(row['directory']);out.mkdir(mode=0o700,parents=True,exist_ok=False)
        started = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):receipt=owner.synthesize(row['text'],out/'reply.wav')
        extra = dict(metrics=dict(metrics(), synthesis_seconds=time.monotonic()-started)) if measured else {}
        emit(dict(type='tts_result',turn_id=row['turn_id'],receipt=receipt,wav=str(out/'reply.wav'), **extra))


class TTS:
    def __init__(self,process):self.process=process;self.buffer=b''
    def read(self,kind,deadline):
        with selectors.DefaultSelector() as select:
            select.register(self.process.stdout,selectors.EVENT_READ)
            while time.monotonic()<deadline:
                while b'\n' in self.buffer:
                    line,self.buffer=self.buffer.split(b'\n',1)
                    try:value=json.loads(line)
                    except (ValueError,UnicodeDecodeError):continue
                    if value.get('type')=='tts_native_failure':
                        self.last_failure=value
                        emit(dict(type='h3_irodori_native_failure', error_type=value.get('error_type', 'NativeFailure')))
                        raise RuntimeError('native TTS failed; no replay')
                    if value.get('type')==kind:return value
                if not select.select(min(1,max(0,deadline-time.monotonic()))):continue
                block=os.read(self.process.stdout.fileno(),65536)
                if not block:raise RuntimeError('native TTS child terminated')
                self.buffer+=block
        raise TimeoutError('finite native TTS deadline')
    def synthesize(self,text,turn_id,directory,deadline):
        self.process.stdin.write((json.dumps(dict(op='synthesize',text=text,turn_id=turn_id,directory=str(directory)))+'\n').encode());self.process.stdin.flush()
        result=self.read('tts_result',deadline)
        if result['turn_id']!=turn_id:raise ValueError('stale native TTS response')
        return result
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()


def prepare_tts(root,spec,*,workspace=None):
    from huggingface_hub import HfApi,hf_hub_download
    from research.hf_irodori_tts_diagnose import checked_snapshot
    from research.h3_personal.director_tts_job import REFERENCE_REPO,REFERENCE_REV,REFERENCE_SHA,LOCK_SHA,digest
    token=os.environ['HF_TOKEN'];api=HfApi(token=token)
    workspace=Path('/work') if workspace is None else Path(workspace)
    workspace.mkdir(mode=0o700,parents=True,exist_ok=True)
    manifest=json.loads((root/'tts-manifest.json').read_text())
    reference=Path(hf_hub_download(REFERENCE_REPO,'reference.wav',repo_type='dataset',revision=REFERENCE_REV,token=token))
    if digest(reference)!=REFERENCE_SHA:raise ValueError('speaker reference differs')
    source=workspace/'h3-native-live-tts'
    subprocess.run(['git','clone','--no-checkout','https://github.com/Aratako/Irodori-TTS.git',str(source)],check=True)
    subprocess.run(['git','checkout','--detach',manifest['source_revision']],cwd=source,check=True)
    if digest(source/'uv.lock')!=LOCK_SHA:raise ValueError('native lock differs')
    tools=workspace/'h3-live-tools'
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(tools),'uv==0.13.1'],check=True)
    uv_env=dict(os.environ,PYTHONPATH=str(tools)+os.pathsep+str(root))
    uv=[sys.executable,'-m','uv']
    if subprocess.check_output([*uv,'--version'],env=uv_env,text=True).split()[1]!='0.13.1':raise ValueError('uv pin differs')
    subprocess.run([*uv,'sync','--frozen','--extra','cu128'],cwd=source,env=uv_env,check=True)
    roots={m['repo']:checked_snapshot(api,m,token) for m in manifest['models']}
    paths=dict(model=str(roots['Aratako/Irodori-TTS-v4-Large']/'model.safetensors'),codec=str(roots['Aratako/Semantic-DACVAE-Japanese-32dim']/'weights.pth'),reference=str(reference),reference_sha256=REFERENCE_SHA)
    p=workspace/'h3-live-tts-paths.json';p.write_text(json.dumps(paths))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(root)+os.pathsep+str(source),PYTHONDONTWRITEBYTECODE='1')
    env.pop('HF_TOKEN',None)
    env.pop('H3_TTS_KEY',None)
    child=subprocess.Popen([str(source/'.venv/bin/python'),str(root/'research/hf_h3_live_voice.py'),'--native',str(root),str(source),str(p)],cwd=source,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if (spec.get('live_relay_session') is not None or spec.get('live_direct_session') is not None) else None,bufsize=0)
    return TTS(child)


def validate_request(row,session,sequence,asr_policy='apple-final'):
    if row.get('schema')!='h3-live-voice-input-v1' or row.get('session_id')!=session or row.get('sequence')!=sequence:raise ValueError('session sequence differs')
    if asr_policy=='endpoint-native-v1':
        if (row.get('origin')!='comet_process_audio_endpoint_asr' or row.get('asr_final') is not True or row.get('asr_stopped') is not True or row.get('asr_on_device') is not False or row.get('endpoint_source')!='native_completed' or 'asr_endpoint_source' in row or 'asr_revision' in row):raise ValueError('truthful native Endpoint metadata required')
    elif row.get('origin')!='discord_received_audio_ondevice_asr' or row.get('asr_stopped') is not True:raise ValueError('real stopped call ASR required')
    if asr_policy=='endpoint-native-v1':pass
    elif asr_policy=='apple-final':
        if row.get('asr_final') is not True or 'asr_endpoint_source' in row or 'asr_revision' in row:raise ValueError('real finalized call ASR required; stream tags forbidden')
    elif asr_policy=='application-stream-v1':
        if row.get('asr_final') is not False or row.get('asr_endpoint_source')!='application_stable_partial' or type(row.get('asr_revision')) is not int or row['asr_revision']<=0:raise ValueError('application streaming endpoint required')
    else:raise ValueError('unsupported ASR policy')
    if row.get('director_model')!='Qwen/Qwen3.8-27B:cerebras':raise ValueError('director fallback forbidden')
    if not isinstance(row.get('reply'),str) or not 1<=len(row['reply'])<=160:raise ValueError('bounded reply required')
    if not time.time()-5<=row.get('expires_at',0)<=time.time()+65:raise ValueError('expired/unbounded input')
    return row


def close_tts_future(future,owner=None):
    """Close only this session's child, even if initialization finishes late."""
    if owner is not None:
        owner.close()
        return
    if future.cancel():return
    def close_returned(done):
        if done.cancelled():return
        try:late_owner=done.result()
        except BaseException:return
        try:late_owner.close()
        except ProcessLookupError:pass
    future.add_done_callback(close_returned)


def run_session(root,paths,spec,repo,tts_future,deadline):
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.director_tts_job import digest
    api=HfApi(token=os.environ['HF_TOKEN']);session=spec['live_voice_session']['id']
    runtime=LiveVoiceRuntime(model=paths['base'],model_manifest=root/'model-manifest.json',identity=paths['personal']/spec['identity_prefix']/'adapter_model.safetensors',identity_config=paths['personal']/spec['identity_prefix']/'adapter_config.json',smd=paths['smd']/'lora/adapter_model.safetensors',reference=paths['reference']/'production-inputs/references/secondary.jpeg',archive=root/'runtime/FastVideo.tar.gz',output=root.with_name('h3-live-resident-generated'))
    tts=None
    def publish(value,files=None):
        from io import BytesIO
        ops=[CommitOperationAdd(path_in_repo='live/state.json',path_or_fileobj=BytesIO(json.dumps(value).encode()))]
        if 'sequence' in value:
            ops.append(CommitOperationAdd(path_in_repo=f"live/turn-{value['sequence']:03d}/result.json",path_or_fileobj=BytesIO(json.dumps(value).encode())))
        for name,path in (files or {}).items():ops.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)))
        return api.create_commit(repo,repo_type='dataset',operations=ops,commit_message='Finite live session generated state/output only').oid
    try:
        runtime.start()
        # Child workers inherited offline mode; owner continues only approved Hub IO.
        os.environ['HF_HUB_OFFLINE']='0'
        tts=tts_future.result(timeout=max(1,deadline-time.monotonic()))
        tts.read('tts_ready',deadline)
        if deadline-time.monotonic()<45:raise TimeoutError('no useful live-input window after preparation')
        publish(dict(schema='h3-live-voice-state-v1',session_id=session,status='ready',expires_at=time.time()+deadline-time.monotonic(),previous_audio_reused=False))
        for sequence in range(1,spec['live_voice_session']['max_turns']+1):
            row=None
            while deadline-time.monotonic()>25:
                info=api.repo_info(repo,repo_type='dataset')
                name=f'live/input-{sequence:03d}.json'
                if name in api.list_repo_files(repo,repo_type='dataset',revision=info.sha):
                    row=json.loads(Path(hf_hub_download(repo,name,repo_type='dataset',revision=info.sha,token=os.environ['HF_TOKEN'])).read_text());break
                time.sleep(1)
            if row is None:break
            validate_request(row,session,sequence,spec.get('asr_policy','apple-final'))
            turn_id=f'turn-{sequence:03d}';out=root.with_name('h3-live-turns')/turn_id
            started=time.monotonic();speech=tts.synthesize(row['reply'],turn_id,out/'tts',deadline)
            receipt=speech['receipt'];wav=Path(speech['wav'])
            if digest(wav)!=receipt['sha256'] or receipt.get('watermark_applied') is not True:raise ValueError('native generated speech binding differs')
            result=runtime.render(wav,receipt['sha256'],out/'video')
            files={f'live/{turn_id}/reply.wav':wav,f'live/{turn_id}/profile.json':result.profile,f'live/{turn_id}/review.mp4':result.review_with_speech}
            state=dict(schema='h3-live-voice-state-v1',session_id=session,status='output',sequence=sequence,files={n:digest(p) for n,p in files.items()},generated_seconds=time.monotonic()-started,expires_at=time.time()+deadline-time.monotonic(),previous_audio_reused=False,reply_sha256=hashlib.sha256(row['reply'].encode()).hexdigest(),tts_receipt=receipt)
            publish(state,files)
        publish(dict(schema='h3-live-voice-state-v1',session_id=session,status='finished',previous_audio_reused=False))
    finally:
        try:runtime.close()
        finally:close_tts_future(tts_future,tts)

if __name__=='__main__':
    if sys.argv[1]=='--native':
        try:
            native(Path(sys.argv[2]),Path(sys.argv[3]),json.loads(Path(sys.argv[4]).read_text()))
        except BaseException as error:
            if os.environ.get('H3_TTS_MEASURE_MEMORY') == '1':
                try: emit(dict(type='tts_native_failure', error_type=type(error).__name__))
                except BaseException: pass
            raise
