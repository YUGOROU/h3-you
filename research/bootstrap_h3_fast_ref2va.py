"""Private pinned SMD Ref2VA GPU bootstrap; no installs/retries/fallback/uploads of inputs."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def session_tts_ownership(future, pool):
    """Own preparation until run_session takes responsibility for closing it."""
    owner = {'transferred': False}
    try:
        yield owner
    finally:
        try:
            if future is not None and not owner['transferred']:
                def close_prepared(done):
                    if not done.cancelled():
                        try:
                            prepared = done.result()
                        except BaseException:
                            return
                        prepared.close()
                # Register before cancellation: a running prepare may finish later.
                future.add_done_callback(close_prepared)
                future.cancel()
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)


def prepare_session_tts(root, spec):
    """Select direct backend before importing or scheduling local TTS."""
    pool = None

    def submit_local():
        nonlocal pool
        from research.hf_h3_live_voice import prepare_tts
        pool = ThreadPoolExecutor(max_workers=1)
        return pool.submit(prepare_tts, root, spec)

    try:
        if spec.get('live_direct_session') is not None:
            from research.h3_personal.native_tts_backend import prepare_direct_tts
            future = prepare_direct_tts(root, spec, submit_local)
        else:
            future = submit_local()
        return future, pool
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise


def output_directory(root):
    output=root.with_name(root.name+'-generated')
    if output.is_relative_to(root):raise ValueError('generated output must be outside deployed repository')
    return output


def locked_speech_plan(spec):
    """Validate the metadata-only binding without reading speech or contacting Hub."""
    import re
    from research.h3_personal.external_audio import plan_windows,window_manifest,window_bounds,HOP_LENGTH,AUDIO_SAMPLE_RATE
    speech=spec.get('locked_speech')
    if speech is None:return None
    if not isinstance(speech,dict) or set(speech)!={'repo','revision','path','sha256','sample_rate','samples'}:
        raise ValueError('exact locked speech binding required')
    for key,length in (('revision',40),('sha256',64)):
        if not isinstance(speech[key],str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',speech[key]):raise ValueError('immutable speech pin required')
    if not isinstance(speech['repo'],str) or not re.fullmatch(r'[\w.-]+/[\w.-]+',speech['repo']):raise ValueError('speech repo required')
    name=speech['path']
    if not isinstance(name,str) or not name or PurePosixPath(name).is_absolute() or any(p in ('','..','.') for p in name.split('/')) or '\\' in name or not name.endswith('.wav'):raise ValueError('safe speech WAV path required')
    if any(type(speech[k]) is not int or speech[k]<=0 for k in ('sample_rate','samples')):raise ValueError('positive integer WAV metadata required')
    if spec.get('layout')!='portrait' or spec.get('runtime_policy') not in ('resident-visual','resident-visual-cached') or spec.get('face_crop_box') is not None or spec.get('regional_compile',False):raise ValueError('locked speech requires portrait resident no-crop eager runtime')
    plan=plan_windows(speech['samples']/speech['sample_rate'],wav=name,sha256_value=speech['sha256'])
    resampled=(speech['samples']*AUDIO_SAMPLE_RATE+speech['sample_rate']-1)//speech['sample_rate']
    for row in plan['windows']:
        binding=window_manifest(plan,row['index'])
        window_bounds(resampled,binding['latent_start'],binding['latent_count'],binding['planned_latent_total'])
    return plan


def locked_speech_args(spec,paths):
    plan=locked_speech_plan(spec)
    if plan is None:return [],4
    import wave
    from research.h3_personal.external_audio import sha256
    speech=spec['locked_speech'];path=paths['speech']/speech['path']
    if sha256(path)!=speech['sha256']:raise ValueError('locked speech SHA differs')
    with wave.open(str(path),'rb') as wav:
        if (wav.getframerate(),wav.getnframes())!=(speech['sample_rate'],speech['samples']) or wav.getcomptype()!='NONE':raise ValueError('locked speech WAV metadata differs')
        if len(wav.readframes(speech['samples']))!=speech['samples']*wav.getnchannels()*wav.getsampwidth():raise ValueError('truncated speech WAV')
    return ['--locked-speech',str(path),'--locked-speech-sha256',speech['sha256']],plan['windows_required']


def dispatch_continuation(root,spec,deployment,boot_started):
    """Fresh GPU owner; one finite process group, never retry or reload."""
    import signal
    import time
    from research.hf_h3_continuation_diagnostic import remaining
    deadline=boot_started+spec['config']['timeout_seconds']-25
    if os.environ.get('H3_SESSION_NOT_AFTER'):
        deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
    remaining(deadline)
    command=[sys.executable,'-m','research.hf_h3_continuation_diagnostic',
             str(root),deployment,str(deadline)]
    process=subprocess.Popen(command,cwd=root,start_new_session=True)
    try:
        code=process.wait(timeout=remaining(deadline))
        if code:raise RuntimeError(f'continuation diagnostic failed with exit{code}')
    finally:
        # Include spawned ranks/FFmpeg, even when the owner exited on failure.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def main(deployment,revision,allow_sha):
    import time
    boot_started=time.monotonic()
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    token=os.environ['HF_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(deployment,repo_type='dataset',revision=revision).private:raise ValueError('private deployment required')
    def fetch(name):return Path(hf_hub_download(deployment,name,repo_type='dataset',revision=revision,token=token))
    body=fetch('upload-allowlist.json').read_bytes()
    if hashlib.sha256(body).hexdigest()!=allow_sha:raise ValueError('allowlist digest differs')
    root=Path('/work/fast-ref2va');root.mkdir(parents=True,exist_ok=False)
    for item in json.loads(body)['files']:
        name=item['path'];p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe deployment path')
        data=fetch(name).read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('deployment hash differs')
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    sys.path.insert(0,str(root))
    from research.run_h3_realtime_first import encoder_preflight
    encoder_preflight(root/'encoder-preflight')
    for package,version in {'torch':'2.12.0','fastvideo-kernel':'0.3.5','quack-kernels':'0.5.3'}.items():
        if importlib.metadata.version(package).split('+')[0]!=version:raise ValueError('pinned dependency differs')
    import torch
    if torch.cuda.device_count()!=2 or any('H200' not in torch.cuda.get_device_name(i) for i in range(2)):raise ValueError('exactly2H200 required')
    for i in (0,1):torch.zeros(16,device=f'cuda:{i}').add_(1)
    print('FAST_REF2VA_BOOT_PREFLIGHT',torch.__version__,torch.version.cuda,flush=True)
    spec=json.loads((root/'spec.json').read_text());paths={}
    if 'idle_diagnostic' in spec:
        from research.hf_h3_idle_diagnostic import validate_spec
        validate_spec(spec)
    if 'visual_ancestry_comparison' in spec:
        from research.hf_h3_visual_ancestry_diagnostic import validate_spec
        from research.h3_personal.visual_ancestry_diagnostic import speech
        speech(root,validate_spec(spec))
    locked_speech_plan(spec)
    if spec.get('live_direct_session') is not None:
        from research.hf_h3_direct_conversation import validate_spec
        validate_spec(spec)
    if spec.get('live_relay_session') is not None:
        from research.hf_h3_relay_conversation import validate_spec
        validate_spec(spec)
        if spec['live_relay_session']['not_after']-time.time()<450:
            raise TimeoutError('expired/insufficient relay lease before model preparation')
    if 'continuation_diagnostic' in spec:
        from research.hf_h3_continuation_diagnostic import validate_spec
        validate_spec(spec,json.loads((root/'asset-manifest.json').read_text())['assets'])
    tts_future=None
    tts_pool=None
    if spec.get('live_voice_session') is not None:
        tts_future,tts_pool=prepare_session_tts(root,spec)
    with session_tts_ownership(tts_future,tts_pool) as tts_owner:
        return finish_bootstrap(root,spec,paths,api,token,deployment,boot_started,
                                torch,hf_hub_download,CommitOperationAdd,tts_future,tts_owner)


def finish_bootstrap(root,spec,paths,api,token,deployment,boot_started,
                     torch,hf_hub_download,CommitOperationAdd,tts_future,tts_owner):
    """All post-preparation work runs under the bootstrap TTS owner."""
    import time
    if spec.get('attention_mode','fa2')!='fa2':
        # Check the installed CuTe entrypoints before paying for model downloads.
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
        if not callable(_flash_attn_bwd) or not callable(_flash_attn_fwd):raise RuntimeError('FA4 entrypoints unavailable')
        if any(torch.cuda.get_device_capability(i)!=(9,0) for i in (0,1)):raise RuntimeError('FA4 experiment requires SM90')
        try:dsl=importlib.metadata.version('nvidia-cutlass-dsl')
        except importlib.metadata.PackageNotFoundError:dsl='unreported'
        print('FAST_REF2VA_FA4_IMPORT_PREFLIGHT',json.dumps(dict(cutlass_dsl=dsl,capability='sm90',installs=False)),flush=True)
    for asset in json.loads((root/'asset-manifest.json').read_text())['assets']:
        kind=asset.get('repo_type','model')
        if (asset['role'] in ('personal','reference') or asset['role']=='speech') and not api.repo_info(asset['repo'],repo_type=kind,revision=asset['revision']).private:
            raise ValueError('personal source must remain private')
        directory=root/asset['role'];directory.mkdir()
        def download_verified(row):
            p=Path(hf_hub_download(asset['repo'],row['path'],repo_type=kind,revision=asset['revision'],token=token,local_dir=directory))
            size=p.stat().st_size;h=hashlib.sha256() if row['sha256'] else hashlib.sha1(b'blob '+str(size).encode()+b'\0')
            with p.open('rb') as f:
                while block:=f.read(16*1024*1024):h.update(block)
            if size!=row['bytes'] or h.hexdigest()!=(row['sha256'] or row['git_blob']):raise ValueError('input asset hash differs')
        # Bounded IO-only parallelism; does not alter inference or measured warm time.
        with ThreadPoolExecutor(max_workers=4 if asset['role']=='base' else 1) as pool:
            list(pool.map(download_verified,asset['files']))
        paths[asset['role']]=directory
    os.environ['PYTHONPATH']=str(root)
    for key,value in dict(FASTVIDEO_ATTENTION_BACKEND='FLASH_ATTN',FASTVIDEO_FA4='0',
        FASTVIDEO_MINIMAX_H3_FUSIONS='all',FASTVIDEO_ENABLE_STARTUP_WARMUP='0',
        HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false').items():os.environ[key]=value
    from research.h3_personal.ref2va_attention import configure_environment
    configure_environment(spec.get('attention_mode','fa2'))
    if 'visual_ancestry_comparison' in spec:
        from research.hf_h3_visual_ancestry_diagnostic import run_session
        deadline=boot_started+spec['config']['timeout_seconds']-25
        if os.environ.get('H3_SESSION_NOT_AFTER'):
            deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
        run_session(root,paths,spec,deployment,deadline)
        return
    if 'idle_diagnostic' in spec:
        from research.hf_h3_idle_diagnostic import run_session
        deadline=boot_started+spec['config']['timeout_seconds']-25
        if os.environ.get('H3_SESSION_NOT_AFTER'):
            deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
        run_session(root,paths,spec,deployment,deadline)
        return
    if spec.get('live_direct_session') is not None:
        from research.hf_h3_direct_conversation import run_session,validate_spec
        validate_spec(spec)
        deadline=boot_started+spec['config']['timeout_seconds']-25
        if os.environ.get('H3_SESSION_NOT_AFTER'):
            deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
        tts_owner['transferred']=True
        run_session(root,paths,spec,deployment,tts_future,deadline)
        return
    if spec.get('live_relay_session') is not None:
        from research.hf_h3_relay_conversation import run_session,validate_spec
        validate_spec(spec)
        deadline=min(boot_started+spec['config']['timeout_seconds']-25,
                     time.monotonic()+spec['live_relay_session']['not_after']-time.time()-10)
        tts_owner['transferred']=True
        run_session(root,paths,spec,deployment,tts_future,deadline)
        return
    if spec.get('streaming_trial') is not None:
        from research.hf_h3_streaming_trial import run_session, validate_spec
        validate_spec(spec)
        deadline=boot_started+spec['config']['timeout_seconds']-25
        if os.environ.get('H3_SESSION_NOT_AFTER'):
            deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
        run_session(root,paths,spec,deployment,deadline)
        return
    if spec.get('conversation_scenario') is not None:
        from research.hf_h3_conversation_scenario import run_session, validate_spec
        validate_spec(spec)
        deadline=boot_started+spec['config']['timeout_seconds']-25
        if os.environ.get('H3_SESSION_NOT_AFTER'):
            deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
        run_session(root,paths,spec,deployment,deadline)
        return
    if 'continuation_diagnostic' in spec:
        dispatch_continuation(root,spec,deployment,boot_started)
        return
    if spec.get('live_voice_session') is not None:
        from research.hf_h3_live_voice import run_session
        deadline=boot_started+spec['config']['timeout_seconds']-25
        if os.environ.get('H3_SESSION_NOT_AFTER'):
            deadline=min(deadline,time.monotonic()+float(os.environ['H3_SESSION_NOT_AFTER'])-time.time()-25)
        tts_owner['transferred']=True
        run_session(root,paths,spec,deployment,tts_future,deadline)
        return
    from research.h3_personal.ref2va_runtime import IDENTITY_SHA
    identity=paths['personal']/spec['identity_prefix'];output=output_directory(root)
    command=[sys.executable,'-m','research.run_h3_ref2va','--model',str(paths['base']),
        '--model-manifest',str(root/'model-manifest.json'),'--identity',str(identity/'adapter_model.safetensors'),
        '--identity-config',str(identity/'adapter_config.json'),
        '--reference',str(paths['reference']/'production-inputs/references/secondary.jpeg'),
        '--reference-sha256',spec['reference_sha256'],'--runtime-archive',str(root/'runtime/FastVideo.tar.gz'),
        '--output',str(output),'--fast-ref-smd','--smd',str(paths['smd']/'lora/adapter_model.safetensors'),'--run']
    command.extend(['--attention-mode',spec.get('attention_mode','fa2'),
                    '--layout',spec.get('layout','landscape'),'--runtime-policy',spec.get('runtime_policy','baseline')])
    speech_args,segments=locked_speech_args(spec,paths)
    command.extend(speech_args)
    if spec.get('incremental_transfer',False):command.extend(['--live-publication-repo',deployment])
    if spec.get('face_crop_box') is not None:
        from research.h3_personal.ref2va_reference_assets import prepare
        crop_dir=root.with_name(root.name+'-face-crop')
        prepare(paths['reference']/'production-inputs/references/secondary.jpeg',crop_dir,spec['reference_sha256'],spec['face_crop_box'],preserve_resolution=True)
        command.extend(['--face-crop',str(crop_dir/'face-crop.png'),'--face-crop-receipt',str(crop_dir/'crop-receipt.json')])
    if spec.get('warm_stage_diagnostic',False):command.append('--profile-warm-stages')
    if spec.get('regional_compile',False):command.append('--regional-compile')
    result=subprocess.run(command,cwd=root)
    if result.returncode:raise RuntimeError(f'isolated SMD Ref2VA runner failed with exit{result.returncode}')
    # Explicit generated-output-only publication. Never traverse runtime/coverage/derived/input trees.
    names=['profile.json']+[f'segment-{i:03d}.mp4' for i in range(4)]
    if spec.get('locked_speech') is not None:
        names=['profile.json']+[f'segment-{i:03d}.mp4' for i in range(segments)]
    operations=[]
    for name in names:
        path=output/name
        if not path.is_file() or path.is_symlink():raise ValueError('missing generated output')
        if path.suffix=='.mp4':
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,timeout=120)
        operations.append(CommitOperationAdd(path_in_repo='outputs/'+name,path_or_fileobj=str(path)))
    if not api.repo_info(deployment,repo_type='dataset').private:raise ValueError('publication must remain private')
    receipt=api.create_commit(deployment,repo_type='dataset',operations=operations,commit_message='Generated Ref2VA SMD4 validation outputs only; identity and realtime remain separate gates')
    print('FAST_REF2VA_PUBLISHED',receipt.oid,flush=True)

if __name__=='__main__':main(*sys.argv[1:])
