"""Isolated native FastVideo Ref2VA contract; no FastH3 T2VA adapter or uploads."""
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

BASE_REPO='MiniMaxAI/MiniMax-H3'
BASE_REV='42ed227ee7df40d41602854ae760620d6eb651fe'
PIPELINE='MiniMaxH3Ref2VAModularPipeline'
IDENTITY_SHA='43fd1f8c6e4097611ba027dcf42503b4c889cfd5ef5e45d91abcb08c482ef151'
SCOPE='h3-native-fastvideo-ref2va-identity-v1'
PROMPT='The person in reference image 1, wearing a plain white shirt, calmly speaking to the camera. Preserve the appearance of the person in the reference. Stable lighting and background.'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(16*1024*1024):h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Ref2VAConfig:
    height:int=256
    width:int=448
    num_frames:int=124
    fps:int=24
    sigma_grid_points:int=51
    seed:int=314159
    prompt:str=PROMPT

    def validate(self):
        if asdict(self)!=asdict(Ref2VAConfig()):raise ValueError('initial Ref2VA pilot contract is fixed')
        if any(type(getattr(self,k)) is not int for k in ('height','width','num_frames','fps','sigma_grid_points','seed')):
            raise ValueError('integer parameters required')
        return self


def validate_identity_header(header):
    from types import SimpleNamespace
    from research.h3_personal.personal_fast_adapter import pairs
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    if any(v['dtype']!='BF16' for v in tensors.values()):raise ValueError('original BF16 identity required')
    state={k:SimpleNamespace(shape=v['shape'],ndim=len(v['shape'])) for k,v in tensors.items()}
    modules,other=pairs(state,allow_dense=False)
    if len(modules)!=208 or other or any(p['A'][1].shape[0]!=16 for p in modules.values()):
        raise ValueError('original208 rank16 pairs required')
    return len(modules)


def relative(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:raise ValueError('unsafe model path')
    return p


def validate_model_manifest(manifest):
    if manifest.get('repo')!=BASE_REPO or manifest.get('revision')!=BASE_REV:raise ValueError('wrong native base pin')
    files=manifest['files'];names=[r['path'] for r in files]
    if len(names)!=len(set(names)):raise ValueError('duplicate model files')
    for row in files:
        relative(row['path'])
        if row['path'].startswith('transformer/') or 'adapter' in row['path'].lower():raise ValueError('T2VA/adapter payload forbidden')
        if not isinstance(row['bytes'],int) or row['bytes']<=0:raise ValueError('model size missing')
        import re
        digest=row.get('sha256') or row.get('git_blob')
        if not digest or not re.fullmatch('[0-9a-f]{64}' if row.get('sha256') else '[0-9a-f]{40}',digest):
            raise ValueError('valid model content digest required')
    required={'model_index.json','scheduler/scheduler_config.json','audio_scheduler/scheduler_config.json',
        'transformer_ref/config.json','transformer_ref/diffusion_pytorch_model.safetensors.index.json'}
    if not required.issubset(names):raise ValueError('missing native Ref2VA metadata')
    if not {'transformer_ref','text_encoder','tokenizer','processor','vae','audio_vae','scheduler','audio_scheduler'}.issubset({n.split('/')[0] for n in names}):
        raise ValueError('missing Ref2VA model components')
    shards=[n for n in names if n.startswith('transformer_ref/') and n.endswith('.safetensors')]
    if len(shards)!=14:raise ValueError('require original 14-shard transformer_ref, not curated 13-shard Diffusers layout')
    return files


def verify_model(base,manifest):
    base=Path(base).resolve();files=validate_model_manifest(manifest)
    # Ignore no unexpected alternate checkpoint/schedule in the loader's search roots.
    expected={r['path'] for r in files}
    actual={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file() and '.cache' not in p.relative_to(base).parts}
    if actual!=expected:raise ValueError('local model inventory differs; use a dedicated Ref2VA-only snapshot')
    for row in files:
        p=base/row['path']
        if p.stat().st_size!=row['bytes']:raise ValueError(f'model size differs: {row["path"]}')
        if row.get('sha256'):
            valid=sha(p)==row['sha256']
        else:
            body=p.read_bytes();valid=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==row['git_blob']
        if not valid:raise ValueError(f'model digest differs: {row["path"]}')
    index=json.loads((base/'transformer_ref/diffusion_pytorch_model.safetensors.index.json').read_text())
    referenced={'transformer_ref/'+str(relative(n)) for n in index['weight_map'].values()}
    present={n for n in expected if n.startswith('transformer_ref/') and n.endswith('.safetensors')}
    if referenced!=present:raise ValueError('Ref2VA index and verified shards differ')
    arch=json.loads((base/'transformer_ref/config.json').read_text())
    if 'pdd_steps' in arch:raise ValueError('PDD student is not the original native Ref2VA model')
    return dict(repo=BASE_REPO,revision=BASE_REV,partition='transformer_ref',shards=len(present),full_hashes_verified=True)


def request(api,reference_type,reference,config,*,face_crop=None,ordered_references=None):
    config.validate()
    if ordered_references is not None:
        from .ordered_references import verify_local_files
        if face_crop is not None:
            raise ValueError('ordered surroundings references cannot use legacy face crop')
        selected=verify_local_files(ordered_references)
        if str(reference)!=selected['references'][0]['path']:
            raise ValueError('primary reference must match ordered person source')
        references=[reference_type(source=r['path'],media_type='image')
                    for r in selected['references']]
    else:
        references=[reference_type(source=str(reference),media_type='image')]
        if face_crop is not None:references.append(reference_type(source=str(face_crop),media_type='image'))
    return api.GenerationRequest(prompt=config.prompt,negative_prompt='',
        inputs=api.InputConfig(references=references),
        sampling=api.SamplingConfig(height=config.height,width=config.width,num_frames=config.num_frames,
            fps=config.fps,num_inference_steps=config.sigma_grid_points,guidance_scale=1.0,batch_cfg=False,seed=config.seed),
        output=api.OutputConfig(save_video=False,return_frames=True))


def generator_config(api,base,identity):
    return api.GeneratorConfig(model_path=str(base),
        pipeline=api.PipelineSelection(components=api.ComponentConfig(
            override_pipeline_cls_name=PIPELINE,lora_path=str(identity),lora_strength=1.0),
            experimental={'attention_backend':'FLASH_ATTN','inference_torch_compile':False,
                'vae_parallel_decode':True,'vae_parallel_decode_strategy':'gather'}),
        engine=api.EngineConfig(num_gpus=2,parallelism=api.ParallelismConfig(tp_size=1,sp_size=2,dist_timeout=300),
            offload=api.OffloadConfig(dit=False,dit_layerwise=False,text_encoder=True,image_encoder=True,vae=True,pin_cpu_memory=True),
            compile=api.CompileConfig(enabled=False,vae_enabled=False,audio_vae_enabled=False)))


def assert_ref_loaded_coverage(pipeline,state,consumed,adapted_count):
    if not getattr(pipeline,'_ref2va',False) or pipeline._extra_config_module_map.get('transformer')!='transformer_ref':
        raise RuntimeError('actual pipeline is not Ref2VA/transformer_ref')
    a=[k for k in state if k.endswith('.lora_A')]
    complete=len(a)==208 and len(state)==416
    for k in a:
        b=k[:-len('.lora_A')]+'.lora_B'
        complete=complete and len(state[k].shape)==2 and state[k].shape[0]==16 and b in state
        if b in state:complete=complete and len(state[b].shape)==2 and state[b].shape[1]==16
    if not complete or set(state)-set(consumed) or adapted_count!=208:raise RuntimeError('native identity loaded coverage differs')
    report=dict(identity_modules=208,total_modules=208,unmatched=0,rank=16,partition='transformer_ref')
    folder=os.environ.get('H3_REF2VA_COVERAGE_DIR')
    if folder:
        import torch
        if not torch.distributed.is_initialized():raise RuntimeError('distributed Ref2VA coverage required')
        rank=torch.distributed.get_rank();report['worker_rank']=rank
        Path(folder).mkdir(parents=True,exist_ok=True)
        Path(folder,f'coverage-rank-{rank}.json').write_text(json.dumps(report)+'\n')
    print('REF2VA_IDENTITY_LOADED',json.dumps(report),flush=True)


def install_ref_guard(path):
    from research.h3_personal.personal_fast_adapter import install_coverage_guard
    path=Path(path);receipt=install_coverage_guard(path)
    old=('        from research.h3_personal.personal_fast_adapter import assert_loaded_coverage\n'
         '        assert_loaded_coverage(self.lora_adapters[lora_nickname], consumed, adapted_count)\n')
    new=('        from research.h3_personal.ref2va_runtime import assert_ref_loaded_coverage\n'
         '        assert_ref_loaded_coverage(self, self.lora_adapters[lora_nickname], consumed, adapted_count)\n')
    text=path.read_text()
    if text.count(old)!=1:raise ValueError('unexpected intermediate coverage overlay')
    path.write_text(text.replace(old,new));receipt['patched_sha256']=sha(path)
    return receipt
