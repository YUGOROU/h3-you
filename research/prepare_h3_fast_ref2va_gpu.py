"""Metadata-only preparation of one private SMD4 Ref2VA GPU validation."""
import argparse
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
from research.launch_h3_fast_ref2va import SCOPE,REFERENCE_SHA,IDENTITY_REPO,IDENTITY_REV,IDENTITY_PREFIX,inspect
from research.h3_personal.fast_ref2va import REPO,REV,FastRef2VAConfig,WEIGHT_SHA
from research.h3_personal.ref2va_budget import budget_bound
from research.prepare_h3_realtime_first import ROOT,IMAGE,metadata_entry,write_json,check_frozen
from research.prepare_h3_ref2va import manifest as base_manifest


def prepare(output,template,ledger,api,publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-20261016-v1',timeout_seconds=1500,profile_warm_stages=False,regional_compile=False,attention_mode='fa2',layout='landscape',runtime_policy='baseline',face_crop_box=None,incremental_transfer=False,locked_speech=None,direct_session=None):
    if direct_session is not None and (type(direct_session) is not dict
            or direct_session.get('long_session') is not True or timeout_seconds != 10800):
        raise ValueError('explicit immutable three-hour direct session required')
    if attention_mode not in ('fa2','fa4','fa4-packed') or (attention_mode!='fa2' and regional_compile):raise ValueError('isolated eager FA4 or unchanged FA2 required')
    from research.h3_personal.fast_ref2va import config_for_layout
    fast_config=config_for_layout(layout)
    if runtime_policy not in ('baseline','resident-visual','resident-visual-cached') or (regional_compile and (layout!='landscape' or runtime_policy!='baseline')):raise ValueError('isolated eager runtime/layout experiment required')
    if face_crop_box is not None:
        from research.h3_personal.ref2va_reference_assets import validate_box
        face_crop_box=list(validate_box(face_crop_box,1737,3088))
    from research.bootstrap_h3_fast_ref2va import locked_speech_plan
    locked_speech_plan(dict(locked_speech=locked_speech,layout=layout,runtime_policy=runtime_policy,face_crop_box=face_crop_box,regional_compile=regional_compile))
    output,template=Path(output).resolve(),Path(template).resolve()
    if output.exists() or output.is_relative_to(ROOT):raise ValueError('new private output required')
    from research.launch_h3_realtime_first import inspect as inspect_template
    old,_=inspect_template(template);check_frozen()
    l=json.loads(Path(ledger).read_text());total=Decimal(str(l['spent_usd']))+sum(Decimal(str(v['upper_bound_usd'])) for v in l['reservations'].values())
    bound=budget_bound(timeout_seconds=timeout_seconds,ledger_bound=total)
    base=base_manifest(api)
    def asset(role,repo,revision,names,kind='model'):
        info=api.repo_info(repo,repo_type=kind,revision=revision,files_metadata=True)
        if info.sha!=revision or (role in ('personal','reference','speech') and not info.private):raise ValueError('private pinned source required')
        rows={r.rfilename:r for r in info.siblings}
        return dict(role=role,repo=repo,revision=revision,repo_type=kind,files=[metadata_entry(rows[n]) for n in names])
    assets=[dict(role='base',repo=base['repo'],revision=base['revision'],repo_type='model',files=base['files']),
        asset('personal',IDENTITY_REPO,IDENTITY_REV,[IDENTITY_PREFIX+n for n in ('adapter_model.safetensors','adapter_config.json')]),
        asset('smd',REPO,REV,['lora/adapter_model.safetensors']),
        asset('reference','YUGOROU/h3-production-715-raw-20261012','60124b7a79c4711a14995594f8c95d54c6cc8924',
            ['production-inputs/references/secondary.jpeg'],'dataset')]
    if locked_speech is not None:
        speech=asset('speech',locked_speech['repo'],locked_speech['revision'],[locked_speech['path']],'dataset')
        if speech['files'][0]['sha256']!=locked_speech['sha256']:raise ValueError('speech content pin differs')
        assets.append(speech)
    output.mkdir(parents=True,mode=0o700)
    files=['research/bootstrap_h3_fast_ref2va.py','research/run_h3_ref2va.py','research/run_h3_realtime_first.py',
        'research/h3_personal/ref2va_runtime.py','research/h3_personal/fast_ref2va.py','research/h3_personal/personal_fast_adapter.py',
        'research/h3_personal/external_audio.py','research/h3_personal/ref2va_helpers.py','research/h3_personal/ref2va_reference_assets.py','research/h3_personal/ref2va_attention.py','research/h3_personal/ref2va_runtime_options.py','research/h3_personal/ref2va_live_transfer.py',
        'research/h3_personal/realtime_first.py','research/h3_personal/realtime_segments.py','research/h3_personal/adapter_header.py',
        'research/prepare_h3_realtime_first.py','research/provision_modal_identity_secret.py']
    for name in files:
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for package in ('research/__init__.py','research/h3_personal/__init__.py'):(output/package).write_text('')
    source=old['source'];target=output/source['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(template/source['path'],target)
    shutil.copytree(template/'documents',output/'documents')
    spec=dict(schema=SCOPE,approval_scope=SCOPE,image=IMAGE,config={'timeout_seconds':timeout_seconds},fast_config=asdict(fast_config),layout=layout,runtime_policy=runtime_policy,face_crop_box=face_crop_box,incremental_transfer=incremental_transfer,
        execution=dict(timeout_seconds=timeout_seconds,max_jobs=1,job_bound_usd=str(bound['job_bound'])),
        warm_stage_diagnostic=profile_warm_stages,regional_compile=regional_compile,attention_mode=attention_mode,
        license_evidence=old['license_evidence'],source=source,identity_prefix=IDENTITY_PREFIX,reference_sha256=REFERENCE_SHA,
        publication_repo=publication_repo)
    if locked_speech is not None:spec['locked_speech']=dict(locked_speech)
    if direct_session is not None:
        spec['live_direct_session']=dict(direct_session)
        spec['live_voice_session']=dict(id=direct_session['session'],max_turns=1000,fresh_input_required=True)
        from research.hf_h3_direct_conversation import validate_spec as validate_direct_spec
        validate_direct_spec(spec)
    write_json(output/'spec.json',spec);write_json(output/'model-manifest.json',base)
    write_json(output/'asset-manifest.json',dict(schema=SCOPE,assets=assets))
    inventory=[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'upload-allowlist.json',dict(schema=SCOPE,private_only=True,files=inventory))
    inspect(output)
    return {k:str(v) for k,v in bound.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','template','ledger'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--publication-repo',default='YUGOROU/h3-fast-ref2va-smd4-gpu-20261016-v1')
    p.add_argument('--timeout-seconds',type=int,default=1500)
    p.add_argument('--profile-warm-stages',action='store_true')
    p.add_argument('--regional-compile',action='store_true')
    p.add_argument('--attention-mode',choices=('fa2','fa4','fa4-packed'),default='fa2')
    p.add_argument('--layout',choices=('landscape','portrait'),default='landscape')
    p.add_argument('--runtime-policy',choices=('baseline','resident-visual','resident-visual-cached'),default='baseline')
    p.add_argument('--incremental-transfer',action='store_true')
    p.add_argument('--face-crop-box',type=int,nargs=4)
    p.add_argument('--locked-speech',type=json.loads,help='JSON binding: repo, revision, path, sha256, sample_rate, samples (private dataset WAV)')
    a=p.parse_args()
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    print(json.dumps(prepare(a.output,a.template,a.ledger,HfApi(token=cached_token()),a.publication_repo,a.timeout_seconds,a.profile_warm_stages,a.regional_compile,a.attention_mode,a.layout,a.runtime_policy,a.face_crop_box,a.incremental_transfer,a.locked_speech)))

if __name__=='__main__':main()
