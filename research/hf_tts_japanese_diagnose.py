"""Bounded audio-only diagnosis. Never loads H3 or silently substitutes a voice.

A no-reference voice is DIAGNOSTIC ONLY, not an acceptable personalized output.
Keep model/steps/seed fixed; compare absence of reference and style guidance to
already retained reference-only baseline. Human pronunciation acceptance required.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess


def case_specs(text):
    return [('no-reference',text,False),
            ('reference-japanese-style','(in Japanese, clear natural pronunciation)'+text,True)]


def synthesize(request,root):
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM
    from research.h3_personal.component_cache import sha256
    from research.hf_conversation_tts import VOX_PIN,VOX_SOURCE
    model=VoxCPM.from_pretrained(str(root/'vox-model'),load_denoiser=False,
        device='cuda:0',optimize=False,local_files_only=True)
    results=[]
    try:
        for name,text,use_ref in case_specs(request['text']):
            wav=model.generate(text=text,reference_wav_path=str(root/'voice-reference.wav') if use_ref else None,
                cfg_value=2.0,inference_timesteps=10,normalize=False,denoise=False,
                max_len=93,retry_badcase=False,seed=42)
            path=root/(name+'.wav');sf.write(path,wav,model.tts_model.sample_rate,subtype='PCM_16')
            seconds=len(wav)/model.tts_model.sample_rate
            if not 0<seconds<=15:raise ValueError('bounded diagnostic audio required; no crop/retry')
            results.append({'case':name,'file':path.name,'sha256':sha256(path),'bytes':path.stat().st_size,
                'seconds':seconds,'reference_used':use_ref,'model_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'fits_h3_geometry':seconds<=124/24,'pronunciation_verified':False,
                'personalized_acceptance':False})
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    report={'schema':'h3-tts-japanese-diagnostic-v1','model_revision':VOX_PIN,'source_revision':VOX_SOURCE,
        'voice_sha256':request['voice']['sha256'],'llm_response_sha256':request['llm_response_sha256'],
        'spoken_content_sha256':hashlib.sha256(request['text'].encode()).hexdigest(),
        'cfg':2.0,'steps':10,'seed':42,'normalize':False,'denoise':False,
        'h3_executed':False,'ai_generated':True,'cases':results}
    (root/'diagnostic-report.json').write_text(json.dumps(report,sort_keys=True))
    print('TTS_DIAGNOSTICS_READY',flush=True)


def publish(request,root):
    from huggingface_hub import HfApi,CommitOperationAdd
    results=json.loads((root/'diagnostic-report.json').read_text())['cases']
    api=HfApi(token=os.environ['HF_TOKEN']);repo=request['diagnostic_output_repo']
    if api.repo_info(repo,repo_type='dataset').private is not True:raise ValueError('private diagnostic repo required')
    files=[root/'diagnostic-report.json']+[root/x['file'] for x in results]
    commit=api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(
        path_in_repo='generated-audio/'+p.name,path_or_fileobj=str(p)) for p in files],
        commit_message='Private generated Japanese diagnostic audio; no reference recording')
    print('TTS_DIAGNOSTICS_UPLOADED',json.dumps({'revision':commit.oid,'cases':results}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','synthesize'])
    p.add_argument('--request',required=True);p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    request=json.loads(Path(args.request).read_text())
    if args.phase=='prepare':
        from research.hf_conversation_tts import prepare
        python=prepare(request,root)
        subprocess.run([str(python),str(Path(__file__).resolve()),'synthesize','--request',args.request,
            '--root',args.root],check=True,env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        publish(request,root)
    else:synthesize(request,root)


if __name__=='__main__':main()
