"""Private, fail-closed derived adapter builder. Never writes originals or uploads."""
import hashlib
import json
from pathlib import Path
from research.h3_personal.personal_fast_adapter import compose,pairs,validate_identity_config

IDENTITY_SHA='43fd1f8c6e4097611ba027dcf42503b4c889cfd5ef5e45d91abcb08c482ef151'
FAST_SHA='42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470'
PERSONAL_SCOPE='h3-realtime-personal-fastvideo-v1'
PERSONAL_PROMPT='A realistic front-facing medium shot of a person wearing a plain white shirt, calmly speaking to the camera in a stable indoor room. Natural small head movements.'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(16*1024*1024):h.update(block)
    return h.hexdigest()


def validate_personal_spec(spec):
    item=spec.get('personal_adapter',{})
    if item.get('schema')!=PERSONAL_SCOPE or not spec['config'].get('personalized'):
        raise ValueError('explicit personal experiment required')
    expected=dict(repo='YUGOROU/h3-identity-production-20261012',revision='3bf9460304d524e7ae28ba3261336317cdea7b19',
        prefix='identity/h3-be4f7c7b8e9849bbba9fc6c6b65aa15d/escrow/checkpoint-0000064/',sha256=IDENTITY_SHA,
        identity_modules=208,fast_modules=362,dense_tensors=132,combined_rank=80,strength=1.0)
    if any(item.get(k)!=v for k,v in expected.items()):raise ValueError('personal pins/scale/count contract changed')
    if spec['config']['prompt']!=PERSONAL_PROMPT:raise ValueError('personal prompt differs')


def build(fast_path,identity_path,config_path,destination):
    from safetensors.torch import load_file,save_file
    from safetensors import safe_open
    import torch
    fast_path,identity_path,config_path,destination=map(Path,(fast_path,identity_path,config_path,destination))
    if destination.exists() or destination.resolve() in (fast_path.resolve(),identity_path.resolve()):raise ValueError('new derived path required')
    if sha(fast_path)!=FAST_SHA or sha(identity_path)!=IDENTITY_SHA:raise ValueError('source adapter digest differs')
    config=json.loads(config_path.read_text());validate_identity_config(config)
    torch.set_num_threads(4)
    fast=load_file(str(fast_path),device='cpu');identity=load_file(str(identity_path),device='cpu')
    result,report=compose(fast,identity,config)
    if (report['identity_modules'],report['total_modules'],report['dense_tensors_preserved'])!=(208,362,132):raise ValueError('adapter coverage differs')
    if set(report['combined_ranks'].values())!={80}:raise ValueError('expected rank80 on208targets')
    for tensor in result.values():
        if not torch.isfinite(tensor).all().item():raise ValueError('nonfinite derived tensor')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with safe_open(str(fast_path),framework='pt',device='cpu') as source:metadata=dict(source.metadata() or {})
    metadata.update(experimental='true',identity_sha256=IDENTITY_SHA,fast_sha256=FAST_SHA,
        rank='mixed64-80',composition='per-tensor rank authoritative; 208 rank80,154 rank64')
    save_file(result,str(destination),metadata=metadata)
    # Verify full serialization, not only state-dict counts; sources remain byte-identical.
    reloaded=load_file(str(destination),device='cpu')
    if set(reloaded)!=set(result) or any(not torch.equal(reloaded[k],v) for k,v in result.items()):raise ValueError('derived strict reload differs')
    if sha(fast_path)!=FAST_SHA or sha(identity_path)!=IDENTITY_SHA:raise ValueError('original source changed')
    report.update(derived_sha256=sha(destination),derived_bytes=destination.stat().st_size,strict_cpu_reload=True)
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PERSONAL_ADAPTER_BUILT',json.dumps({k:v for k,v in report.items() if k!='combined_ranks'}),flush=True)
    return report
