"""Private, finite audio-only Irodori evaluation. No H3 or automatic retries.

Online preparation/publication surrounds strictly offline native inference children.
Reference upload is user-authorized; publication includes generated WAV/report only.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cases():
    return [('no-reference','こんにちは！',False),
            ('reference-30s','こんにちは！',True),
            ('previous-text-reference-30s','おはようございます。声と口の動きを確認します。',True)]


def checked_snapshot(api,spec,token):
    from huggingface_hub import snapshot_download
    # Sony's native caller uses main. Verify the resolved immutable revision and
    # every file before offline execution; never manufacture cache refs/fallback.
    revision='main' if spec['repo']=='sony/silentcipher' else spec['revision']
    root=Path(snapshot_download(spec['repo'],revision=revision,token=token))
    if root.name!=spec['revision']:raise ValueError('snapshot revision drift')
    for name,meta in spec['files'].items():
        p=root/name;b=p.read_bytes()
        if len(b)!=meta['bytes']:raise ValueError('asset size drift')
        actual=hashlib.sha256(b).hexdigest() if meta['lfs_sha256'] else hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if actual!=(meta['lfs_sha256'] or meta['git_blob_id']):raise ValueError('asset hash drift')
    return root


def main():
    os.umask(0o077)
    deployment=Path(sys.argv[1]);request=json.loads((deployment/'request.json').read_text())
    if digest(deployment/'reference.wav')!=request['reference_sha256']:raise ValueError('reference drift')
    token=os.environ['HF_TOKEN']
    from huggingface_hub import HfApi,CommitOperationAdd
    api=HfApi(token=token)
    if not api.repo_info(request['output_repo'],repo_type='dataset').private:raise ValueError('private output required')
    source=Path('/tmp/irodori-native')
    subprocess.run(['git','clone','--no-checkout','https://github.com/Aratako/Irodori-TTS.git',str(source)],check=True)
    subprocess.run(['git','checkout','--detach',request['source_revision']],cwd=source,check=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=request['source_revision']:raise ValueError('source drift')
    if digest(source/'uv.lock')!=request['uv_lock_sha256']:raise ValueError('dependency lock drift')
    subprocess.run(['uv','sync','--frozen','--extra','cu128'],cwd=source,check=True)
    roots={s['repo']:checked_snapshot(api,s,token) for s in request['models']}
    model=roots['Aratako/Irodori-TTS-v4-Large']/'model.safetensors'
    codec=roots['Aratako/Semantic-DACVAE-Japanese-32dim']/'weights.pth'
    out=Path('/tmp/irodori-generated');out.mkdir(mode=0o700)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DEBUG='0',HF_HUB_DISABLE_TELEMETRY='1')
    # Enforce native watermark availability instead of its unwatermarked fallback.
    wrapper="""import runpy,sys
from irodori_tts.watermark import SilentCipherWatermarker
original=SilentCipherWatermarker.__init__
def checked(self,*args,**kwargs):
    original(self,*args,**kwargs)
    if not self.ready:raise RuntimeError('mandatory watermark unavailable')
SilentCipherWatermarker.__init__=checked
sys.argv=['infer.py']+sys.argv[1:]
runpy.run_path('infer.py',run_name='__main__')
"""
    results=[]
    for name,text,reference in cases():
        path=out/(name+'.wav')
        args=[str(source/'.venv/bin/python'),'-c',wrapper,'--checkpoint',str(model),
              '--codec-repo',str(codec),'--model-device','cuda:0','--codec-device','cuda:0',
              '--model-precision','fp32','--codec-precision','fp32','--num-steps','40',
              '--cfg-scale-text','3','--cfg-scale-speaker','5','--seed','42',
              '--text',text,'--output-wav',str(path)]
        args+=['--ref-wav',str(deployment/'reference.wav')] if reference else ['--no-ref']
        subprocess.run(args,cwd=source,env=env,check=True)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
             'format=duration:stream=sample_rate,channels','-of','json',str(path)],text=True))
        seconds=float(probe['format']['duration'])
        if not 0<seconds<=30:raise ValueError('bounded diagnostic output; never crop')
        results.append({'case':name,'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path),
                        'seconds':seconds,'fits_h3_geometry':seconds<=124/24,'reference_used':reference,
                        'probe':probe,'pronunciation_verified':False,'speaker_verified':False})
    report={'schema':'h3-irodori-japanese-diagnostic-v1','model_revision':request['models'][0]['revision'],
            'source_revision':request['source_revision'],'reference_sha256':request['reference_sha256'],
            'reference_seconds':request['reference_seconds'],'num_steps':40,'precision':'fp32',
            'text_cfg':3,'speaker_cfg':5,'seed':42,'watermark_required':True,'h3_executed':False,
            'ai_generated':True,'cases':results}
    (out/'diagnostic-report.json').write_text(json.dumps(report,sort_keys=True))
    names=[r['file'] for r in results]+['diagnostic-report.json']
    commit=api.create_commit(request['output_repo'],repo_type='dataset',operations=[
        CommitOperationAdd(path_in_repo='generated/'+n,path_or_fileobj=str(out/n)) for n in names],
        commit_message='Private generated Irodori Japanese/reference30s comparison; no raw reference publication')
    print(json.dumps({'completed':True,'output_revision':commit.oid,'generated_files':names,'h3_executed':False}),flush=True)


if __name__=='__main__':main()
