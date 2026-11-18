"""Finite new-text Irodori synthesis, native offline child, generated-only publication."""
import json
import os
from pathlib import Path
import subprocess
import sys
from research.h3_personal.director_tts_job import validate,digest,REFERENCE_REPO,REFERENCE_REV,REFERENCE_SHA


def native(deployment,source,paths):
    from irodori_tts.inference_runtime import InferenceRuntime,RuntimeKey,SamplingRequest,save_wav
    from research.h3_personal.director_irodori import IrodoriResident,SOURCE,MODEL,CODEC
    request=validate(json.loads((deployment/'request.json').read_text()))
    runtime=InferenceRuntime.from_key(RuntimeKey(checkpoint=paths['model'],model_device='cuda:0',codec_repo=paths['codec'],
        model_precision='fp32',codec_device='cuda:0',codec_precision='fp32',compile_model=False))
    adapter=IrodoriResident(runtime,SamplingRequest,save_wav,paths['reference'],REFERENCE_SHA,
        source_revision=SOURCE,model_revision=MODEL,codec_revision=CODEC)
    out=Path('/tmp/irodori-turn-generated');out.mkdir(mode=0o700)
    receipt=adapter.synthesize(request['text'],out/'reply.wav')
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out/'reply.wav'),'-f','null','-'],check=True)
    receipt.update(schema='h3-director-new-irodori-turn-v1',synthesis_count=1,h3_executed=False,
                   director_model=request['director_model'],input_origin=request['input_origin'],generated_only=True)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))


def main(deployment):
    os.umask(0o077)
    request=validate(json.loads((deployment/'request.json').read_text()))
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    from research.hf_irodori_tts_diagnose import checked_snapshot
    token=os.environ['HF_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(request['output_repo'],repo_type='dataset').private:raise ValueError('private output required')
    if not api.repo_info(REFERENCE_REPO,repo_type='dataset',revision=REFERENCE_REV).private:raise ValueError('private reference required')
    reference=Path(hf_hub_download(REFERENCE_REPO,'reference.wav',repo_type='dataset',revision=REFERENCE_REV,token=token))
    if digest(reference)!=REFERENCE_SHA:raise ValueError('reference bytes differ')
    source=Path('/tmp/irodori-turn-native')
    subprocess.run(['git','clone','--no-checkout','https://github.com/Aratako/Irodori-TTS.git',str(source)],check=True)
    subprocess.run(['git','checkout','--detach',request['source_revision']],cwd=source,check=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=request['source_revision'] or digest(source/'uv.lock')!=request['uv_lock_sha256']:raise ValueError('native source/lock drift')
    if subprocess.check_output(['uv','--version'],text=True).split()[1]!=request['uv_version']:raise ValueError('uv version differs')
    subprocess.run(['uv','sync','--frozen','--extra','cu128'],cwd=source,check=True)
    roots={m['repo']:checked_snapshot(api,m,token) for m in request['models']}
    paths=dict(model=str(roots['Aratako/Irodori-TTS-v4-Large']/'model.safetensors'),codec=str(roots['Aratako/Semantic-DACVAE-Japanese-32dim']/'weights.pth'),reference=str(reference))
    paths_file=Path('/tmp/irodori-turn-paths.json');paths_file.write_text(json.dumps(paths))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DEBUG='0',HF_HUB_DISABLE_TELEMETRY='1',
             PYTHONPATH=str(deployment)+os.pathsep+str(source))
    env.pop('HF_TOKEN',None)
    subprocess.run([str(source/'.venv/bin/python'),str(deployment/'research/hf_irodori_turn.py'),'--native',str(deployment),str(source),str(paths_file)],cwd=source,env=env,check=True)
    out=Path('/tmp/irodori-turn-generated')
    if {p.name for p in out.iterdir()}!={'reply.wav','receipt.json'}:raise ValueError('generated allowlist differs')
    receipt=json.loads((out/'receipt.json').read_text())
    if digest(out/'reply.wav')!=receipt['sha256'] or not receipt['watermark_applied'] or receipt['synthesis_count']!=1:raise ValueError('generated receipt mismatch')
    commit=api.create_commit(request['output_repo'],repo_type='dataset',operations=[CommitOperationAdd(path_in_repo='generated/'+name,path_or_fileobj=str(out/name)) for name in ('reply.wav','receipt.json')],commit_message='One fresh Pi-directed Irodori utterance; no raw inputs; no H3')
    print(json.dumps(dict(completed=True,output_revision=commit.oid,h3_executed=False)),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]),json.loads(Path(sys.argv[4]).read_text()))
    else:main(Path(sys.argv[1]))
