"""Metadata/header-only feasibility audit. No model download, GPU or publication."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request
from research.h3_personal.ref2va_runtime import BASE_REPO,BASE_REV,IDENTITY_SHA,sha,validate_identity_header
from research.h3_personal.fast_ref2va import REPO,REV,WEIGHT_SHA,TRAIN_BASE_REV,raw_sigmas,shifted_sigmas
from research.h3_personal.adapter_header import header_length
from research.h3_personal.personal_fast_adapter import pairs
from research.prepare_h3_realtime_first import metadata_entry,write_json


def component_inventory(info):
    prefixes={'transformer_ref','text_encoder','tokenizer','processor','vae','audio_vae','scheduler','audio_scheduler'}
    return {s.rfilename:(s.size,getattr(s.lfs,'sha256',None) if s.lfs else s.blob_id)
        for s in info.siblings if s.rfilename.split('/')[0] in prefixes or s.rfilename in ('model_index.json','modular_model_index.json')}


def compare_headers(student,identity):
    validate_identity_header(identity)
    def unpack(header):
        tensors={k:v for k,v in header.items() if k!='__metadata__'}
        if any(v['dtype']!='BF16' for v in tensors.values()):raise ValueError('BF16 required')
        return pairs({k:SimpleNamespace(shape=v['shape'],ndim=len(v['shape'])) for k,v in tensors.items()},allow_dense=False)[0]
    a,b=unpack(student),unpack(identity)
    if student.get('__metadata__',{}).get('alpha')!='128' or len(a)!=312:raise ValueError('SMD alpha/coverage differs')
    if any(v['A'][1].shape[0]!=128 for v in a.values()):raise ValueError('SMD rank differs')
    for name,v in b.items():
        if name not in a or a[name]['A'][1].shape[1]!=v['A'][1].shape[1] or a[name]['B'][1].shape[0]!=v['B'][1].shape[0]:
            raise ValueError('identity target mismatch: '+name)
    return dict(identity_modules=len(b),student_modules=len(a),combined_rank=144,other_rank=128,missing_targets=0)


def audit(api,identity):
    if sha(identity)!=IDENTITY_SHA:raise ValueError('original64 digest differs')
    with Path(identity).open('rb') as f:n=header_length(f.read(8));ih=json.loads(f.read(n))
    old=api.model_info(BASE_REPO,revision=TRAIN_BASE_REV,files_metadata=True)
    new=api.model_info(BASE_REPO,revision=BASE_REV,files_metadata=True)
    if old.sha!=TRAIN_BASE_REV or new.sha!=BASE_REV:raise ValueError('base pins differ')
    if not component_inventory(old) or component_inventory(old)!=component_inventory(new):raise ValueError('SMD trained base is not byte-equivalent to identity base')
    info=api.model_info(REPO,revision=REV,files_metadata=True)
    row=metadata_entry(next(s for s in info.siblings if s.rfilename=='lora/adapter_model.safetensors'))
    if info.sha!=REV or row['sha256']!=WEIGHT_SHA:raise ValueError('SMD source pin differs')
    url=f'https://huggingface.co/{REPO}/resolve/{REV}/lora/adapter_model.safetensors'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1048575'}),timeout=60) as response:
        length=header_length(response.read(8));header=json.loads(response.read(length))
    return dict(schema='fast-ref2va-feasibility-v1',repo=REPO,revision=REV,student_file=row,
        base_components_byte_equivalent=True,component_files=len(component_inventory(new)),
        shape_coverage=compare_headers(header,ih),raw_sigmas=raw_sigmas(),video_sigmas=shifted_sigmas(12),audio_sigmas=shifted_sigmas(3),
        constructor_composition='rank128+16, alpha/rank1',reference_conditioning_required=True,
        vsa_compatible=False,vsa_reason='no trained H3 VSA gates in SMD; do not stack T2VA gates',
        identity_quality_verified=False,gpu_parity_verified=False,realtime_verified=False,
        paid_job_ready=False,remaining_gates=['actual GPU load','identity review','warm server throughput','client continuity'],
        published_pdd_rejected='maxsssssss/MiniMax-H3-PDD-LoRA card explicitly trains t2va')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('new private output required')
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    result=audit(HfApi(token=cached_token()),a.identity);write_json(a.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
