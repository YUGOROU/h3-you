"""Pinned isolated GPU bootstrap. No package installation or source auto-updates."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path,PurePosixPath
import sys
import tarfile


def main(deployment,revision,allow_sha):
    from huggingface_hub import hf_hub_download,HfApi
    token=os.environ['HF_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(deployment,repo_type='dataset',revision=revision).private:raise ValueError('private deployment required')
    def download(name):return Path(hf_hub_download(deployment,name,repo_type='dataset',revision=revision,token=token))
    inventory=download('upload-allowlist.json').read_bytes()
    if hashlib.sha256(inventory).hexdigest()!=allow_sha:raise ValueError('deployment allowlist SHA mismatch')
    root=Path('/work/realtime-first');root.mkdir(parents=True,exist_ok=False)
    for item in json.loads(inventory)['files']:
        name=item['path'];p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe deployment path')
        data=download(name).read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('deployment file differs')
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    spec=json.loads((root/'spec.json').read_text())
    sys.path.insert(0,str(root))
    from research.run_h3_realtime_first import encoder_preflight
    encoder_preflight(root/'encoder-preflight')
    for package,expected in {'torch':'2.12.0','fastvideo-kernel':'0.3.5','quack-kernels':'0.5.3'}.items():
        if importlib.metadata.version(package).split('+')[0]!=expected:raise ValueError('runtime dependency mismatch: '+package)
    import torch
    if torch.cuda.device_count()!=2 or any('H200' not in torch.cuda.get_device_name(i) for i in range(2)):raise ValueError('requires2H200')
    torch.zeros(16,device='cuda:0').add_(1);torch.zeros(16,device='cuda:1').add_(1)
    print('REALTIME_RUNTIME_PREFLIGHT',torch.__version__,torch.version.cuda,flush=True)
    archive=root/spec['source']['path']
    with tarfile.open(archive,'r:gz') as source:source.extractall(root/'source',filter='data')
    src=root/'source'/('FastVideo-'+spec['config']['runtime_revision'])
    if spec['config'].get('personalized'):
        from research.build_h3_personal_fast_adapter import validate_personal_spec
        from research.h3_personal.personal_fast_adapter import install_coverage_guard
        validate_personal_spec(spec)
        overlay=install_coverage_guard(src/'fastvideo/pipelines/lora_pipeline.py')
        (root/'personal-loader-overlay.json').write_text(json.dumps(overlay)+'\n')
        os.environ['H3_PERSONAL_COVERAGE_DIR']=str(root/'personal-coverage')
    os.environ['PYTHONPATH']=str(root)+':'+str(src)
    for key,value in dict(FASTVIDEO_ATTENTION_BACKEND='VIDEO_SPARSE_ATTN_H3',FASTVIDEO_FA4='1',
        FASTVIDEO_MINIMAX_H3_FUSIONS='all',FASTVIDEO_VSA_SM100A='0',FASTVIDEO_ENABLE_STARTUP_WARMUP='0',
        HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false').items():os.environ[key]=value
    manifest=json.loads((root/spec['asset_manifest']).read_text());paths={}
    for asset in manifest['assets']:
        if asset['role']=='personal' and not api.repo_info(asset['repo'],revision=asset['revision']).private:
            raise ValueError('personal adapter source must remain private')
        directory=root/asset['role'];directory.mkdir()
        for item in asset['files']:
            file=Path(hf_hub_download(asset['repo'],item['path'],revision=asset['revision'],token=token,local_dir=directory))
            size=file.stat().st_size
            h=hashlib.sha256() if item['sha256'] else hashlib.sha1(b'blob '+str(size).encode()+b'\0')
            with file.open('rb') as stream:
                while block:=stream.read(16*1024*1024):h.update(block)
            if size!=item['bytes'] or h.hexdigest()!=(item['sha256'] or item['git_blob']):raise ValueError('model asset hash differs')
        paths[asset['role']]=directory
    adapter=paths['adapter']/'vsa-datafree/adapter_model.safetensors'
    if spec['config'].get('personalized'):
        from research.build_h3_personal_fast_adapter import build
        personal=paths['personal']/spec['personal_adapter']['prefix']
        derived=root/'derived-private/adapter_model.safetensors'
        build(adapter,personal/'adapter_model.safetensors',personal/'adapter_config.json',derived)
        adapter=derived
    os.chdir(root)
    command=[sys.executable,'-m','research.run_h3_realtime_first','--spec',str(root/'spec.json'),
        '--base',str(paths['base']),'--adapter',str(adapter),
        '--output',str(root/'generated')]
    os.execv(sys.executable,command)

if __name__=='__main__':main(*sys.argv[1:])
