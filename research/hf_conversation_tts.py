"""Explicit cloud-only Vox preparation/TTS phase for an approved finite Job.

Separate venv keeps the verified H3 runtime untouched. Voice already exists in a
private pinned dataset; only GENERATED TTS is returned for preview/publication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from research.h3_personal.component_cache import sha256

VOX_SOURCE='cce58a5b59303c9bd63d12f23afe6a49a4a80c59'
VOX_PIN='32279effe8c19989596f05d353d1447f51d9e915'
VOX_ARCHIVE_SHA256='a2b9d9f3076d9d6eaaf5fdb6b92e1bc7c3b4643bc9a489bace7622e04bcdbd2c'


def install_vox(python,constraints):
    # Commit archive + byte pin avoids requiring git in the minimal Job image.
    # SCM cannot infer a version from archives; this is metadata, NOT a model pin.
    env=dict(os.environ,SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM='0.0.0+source.'+VOX_SOURCE[:12])
    archive='https://codeload.github.com/OpenBMB/VoxCPM/tar.gz/'+VOX_SOURCE
    subprocess.run([str(python),'-m','pip','install','-q','-c',str(constraints),
        'voxcpm @ '+archive+'#sha256='+VOX_ARCHIVE_SHA256],env=env,check=True)


def prepare(request,root):
    from huggingface_hub import HfApi,hf_hub_download
    token=os.environ['HF_TOKEN'];api=HfApi(token=token)
    voice=request['voice'];info=api.repo_info(voice['repo'],repo_type='dataset',revision=voice['revision'])
    if info.private is not True:raise ValueError('voice source must stay private')
    voice_path=hf_hub_download(voice['repo'],voice['path'],repo_type='dataset',revision=voice['revision'],token=token)
    if sha256(voice_path)!=voice['sha256']:raise ValueError('existing voice byte pin differs')
    import shutil
    shutil.copyfile(voice_path,root/'voice-reference.wav')
    vox=request['vox'];model=root/'vox-model';model.mkdir(mode=0o700)
    if vox['repo']!='openbmb/VoxCPM2' or vox['revision']!=VOX_PIN:raise ValueError('Vox pin differs')
    for name,meta in sorted(vox['files'].items()):
        if '/' in name or name in ('.','..'):raise ValueError('flat pinned Vox asset expected')
        cached=hf_hub_download(vox['repo'],name,revision=VOX_PIN,token=token)
        p=model/name;shutil.copyfile(cached,p)
        if p.stat().st_size!=meta['size']:raise ValueError('Vox size mismatch')
        if meta['lfs_sha256']:
            if sha256(p)!=meta['lfs_sha256']:raise ValueError('Vox LFS bytes differ')
        elif hashlib.sha1(b'blob '+str(meta['size']).encode()+b'\0'+p.read_bytes()).hexdigest()!=meta['blob_id']:
            raise ValueError('Vox Git bytes differ')
    env=root/'tts-env'
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
    constraints=root/'constraints.txt'
    constraints.write_text('torch==2.8.0\ntorchaudio==2.8.0\ntorchcodec==0.7.0\ntransformers==5.15.0\nhuggingface_hub==1.33.0\n')
    python=env/'bin/python'
    install_vox(python,constraints)
    return python


def synthesize(request,root):
    from research.h3_personal.conversation_producer import LocalZeroShotTTS
    tts=LocalZeroShotTTS(str(root/'vox-model'),str(root/'voice-reference.wav'),request['voice']['sha256'])
    tts.synthesize(request['text'],root/'tts.wav')
    import wave
    with wave.open(str(root/'tts.wav'),'rb') as w:seconds=w.getnframes()/w.getframerate()
    if not 0<seconds<=124/24:raise ValueError('TTS exceeds fixed124frame geometry; never truncate')
    (root/'tts-receipt.json').write_text(json.dumps({'tts_model_revision':VOX_PIN,
        'source_revision':VOX_SOURCE,'voice_sha256':request['voice']['sha256'],
        'text_sha256':hashlib.sha256(request['text'].encode()).hexdigest(),
        'llm_model':request['llm_model'],'llm_response_sha256':request['llm_response_sha256'],
        'tts_sha256':sha256(root/'tts.wav'),'spoken_seconds':seconds,'sample_complete_verified':False,
        'ai_generated':True,'voice_quality_verified':False},sort_keys=True))
    print('CONVERSATION_TTS_READY',json.dumps({'spoken_seconds':seconds}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','synthesize'])
    p.add_argument('--request',required=True);p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    request=json.loads(Path(args.request).read_text())
    if args.stage=='prepare':
        python=prepare(request,root)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([str(python),'-m','research.hf_conversation_tts','synthesize',
            '--request',args.request,'--root',args.root],env=env,check=True)
    else:synthesize(request,root)


if __name__=='__main__':main()
