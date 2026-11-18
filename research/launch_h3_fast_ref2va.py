"""One USER-authorized SMD4+identity Ref2VA GPU validation; defaults offline."""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from research.launch_h3_realtime_first import main as launch
from research.prepare_h3_realtime_first import IMAGE,check_frozen,safe_path,metadata_entry
from research.h3_personal.ref2va_runtime import validate_model_manifest
from research.h3_personal.fast_ref2va import FastRef2VAConfig,config_for_layout,REPO,REV,WEIGHT_SHA

SCOPE='h3-fast-ref2va-smd4-gpu-validation-v1'
REFERENCE_SHA='6789db42c765d2611a640bc9b47934b4912584e45c800d2700b1fddf9902faeb'
IDENTITY_REPO='YUGOROU/h3-identity-production-20261012'
IDENTITY_REV='3bf9460304d524e7ae28ba3261336317cdea7b19'
IDENTITY_PREFIX='identity/h3-be4f7c7b8e9849bbba9fc6c6b65aa15d/escrow/checkpoint-0000064/'


def inspect(stage):
    inventory=json.loads((stage/'upload-allowlist.json').read_text());rows={r['path']:r for r in inventory['files']}
    if len(rows)!=len(inventory['files']):raise ValueError('duplicate deployment file')
    actual={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    if actual!=set(rows)|{'upload-allowlist.json'}:raise ValueError('unreviewed deployment files')
    for name,row in rows.items():
        safe_path(name);p=stage/name
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('deployment bytes differ')
    spec=json.loads((stage/'spec.json').read_text());check_frozen()
    if spec['schema']!=SCOPE or spec['approval_scope']!=SCOPE or spec['image']!=IMAGE:raise ValueError('wrong paid scope')
    seconds=spec['config'].get('timeout_seconds')
    modal_live=spec.get('modal_live_approval')
    if modal_live is not None and (modal_live.get('schema')!='h3-modal-live-approval-v1' or modal_live.get('provider')!='modal' or modal_live.get('region') not in ('ap','jp','au') or modal_live.get('allowed_runtime_regions')!=['jp','au','ca','mx']):raise ValueError('explicit Modal live authority required')
    long_row=spec.get('live_direct_session',{})
    long_session=(long_row.get('long_session') is True
        and long_row.get('attended_visual') is True
        and long_row.get('tts_mode')=='native_voxcpm'
        and long_row.get('lease_seconds')==10800 and modal_live is None)
    if long_session:
        from research.hf_h3_direct_conversation import validate_spec as validate_direct_spec
        validate_direct_spec(spec)
    max_seconds=10800 if long_session else 1800 if modal_live is not None else 1500
    if set(spec['config'])!={'timeout_seconds'} or type(seconds) is not int or not 1<=seconds<=max_seconds or spec['fast_config']!=asdict(config_for_layout(spec.get('layout','landscape'))):raise ValueError('fixed four-request candidate only')
    if spec['execution']['timeout_seconds']!=seconds or spec['execution']['max_jobs']!=1:raise ValueError('one finite <=25minJob only')
    for flag in ('warm_stage_diagnostic','regional_compile'):
        if type(spec.get(flag,False)) is not bool:raise ValueError('explicit boolean experiment flags required')
    if spec.get('regional_compile',False) and spec.get('warm_stage_diagnostic',False):raise ValueError('do not mix compile speed gate with diagnostic sync')
    attention=spec.get('attention_mode','fa2')
    if attention not in ('fa2','fa4','fa4-packed') or (attention!='fa2' and spec.get('regional_compile',False)):raise ValueError('isolated eager FA4 or unchanged FA2 required')
    policy=spec.get('runtime_policy','baseline')
    if policy not in ('baseline','resident-visual','resident-visual-cached') or (spec.get('regional_compile',False) and (policy!='baseline' or spec.get('layout','landscape')!='landscape')):raise ValueError('isolated eager runtime/layout experiment required')
    if spec.get('face_crop_box') is not None:
        from research.h3_personal.ref2va_reference_assets import validate_box
        validate_box(spec['face_crop_box'],1737,3088)
    from research.bootstrap_h3_fast_ref2va import locked_speech_plan
    locked=locked_speech_plan(spec)
    if 'research/h3_personal/external_audio.py' not in rows:raise ValueError('external audio helper must be staged')
    evidence=spec['license_evidence']
    trust_jp=modal_live is not None and modal_live.get('trust_explicit_jp_contract') is True
    if modal_live is not None and 'trust_explicit_jp_contract' in modal_live and type(modal_live['trust_explicit_jp_contract']) is not bool:raise ValueError('explicit boolean Japan contract approval required')
    if trust_jp and modal_live['region']!='jp':raise ValueError('Japan contract trust cannot authorize other selectors')
    expected=(('operator_modal_jp_contract' if trust_jp else 'runtime_modal_region_gate'),'modal') if modal_live is not None else ('operator_hf_acceptance','huggingface_jobs')
    if (evidence['mode'],evidence['provider'])!=expected or evidence['country_verified']:raise ValueError('provider-specific location authority differs')
    if spec['identity_prefix']!=IDENTITY_PREFIX or spec['reference_sha256']!=REFERENCE_SHA:raise ValueError('private input selector differs')
    if not spec['publication_repo'].startswith('YUGOROU/h3-fast-ref2va-smd4-gpu-'):raise ValueError('wrong publication namespace')
    manifest=json.loads((stage/'asset-manifest.json').read_text())
    assets={a['role']:a for a in manifest['assets']}
    if set(assets)!=({'base','personal','smd','reference'}|({'speech'} if locked is not None else set())) or len(assets)!=len(manifest['assets']):raise ValueError('unexpected asset roles')
    if locked is not None:
        speech=spec['locked_speech'];asset=assets['speech']
        if (asset['repo'],asset['revision'],asset.get('repo_type'))!=(speech['repo'],speech['revision'],'dataset') or len(asset['files'])!=1:raise ValueError('speech asset binding differs')
        row=asset['files'][0]
        if row['path']!=speech['path'] or row['sha256']!=speech['sha256']:raise ValueError('speech content binding differs')
    validate_model_manifest(json.loads((stage/'model-manifest.json').read_text()))
    from research.h3_personal.ref2va_runtime import IDENTITY_SHA,BASE_REPO,BASE_REV
    pins={'base':(BASE_REPO,BASE_REV),'personal':(IDENTITY_REPO,IDENTITY_REV),'smd':(REPO,REV),
        'reference':('YUGOROU/h3-production-715-raw-20261012','60124b7a79c4711a14995594f8c95d54c6cc8924')}
    for role,(repo,revision) in pins.items():
        if (assets[role]['repo'],assets[role]['revision'])!=(repo,revision):raise ValueError('asset pin differs')
    if assets['base']['files']!=json.loads((stage/'model-manifest.json').read_text())['files']:raise ValueError('base manifest differs')
    if {r['path'] for r in assets['personal']['files']}!={IDENTITY_PREFIX+'adapter_model.safetensors',IDENTITY_PREFIX+'adapter_config.json'}:raise ValueError('wrong original files')
    for role,path,digest in [('personal',IDENTITY_PREFIX+'adapter_model.safetensors',IDENTITY_SHA),
        ('smd','lora/adapter_model.safetensors',WEIGHT_SHA),('reference','production-inputs/references/secondary.jpeg',REFERENCE_SHA)]:
        matches=[r for r in assets[role]['files'] if r['path']==path and r['sha256']==digest]
        if len(matches)!=1:raise ValueError('input content pin differs')
    if len(assets['smd']['files'])!=1 or len(assets['reference']['files'])!=1:raise ValueError('unapproved extras')
    return spec,inventory


def pre_submit(api,spec,stage):
    if spec.get('modal_live_approval') is not None:raise ValueError('Modal live deployment cannot be submitted to HF Jobs')
    checked,_=inspect(stage)
    if checked!=spec:raise ValueError('submission spec differs')
    for asset in json.loads((stage/'asset-manifest.json').read_text())['assets']:
        kind=asset.get('repo_type','model');info=api.repo_info(asset['repo'],repo_type=kind,revision=asset['revision'],files_metadata=True)
        if info.sha!=asset['revision']:raise ValueError('remote revision differs')
        if asset['role'] in ('personal','reference','speech') and not info.private:raise ValueError('personal source must be private')
        api.auth_check(asset['repo'],repo_type=kind,write=False)
        rows={s.rfilename:s for s in info.siblings}
        for row in asset['files']:
            if metadata_entry(rows[row['path']])!=row:raise ValueError('live input metadata differs')


def main(argv=None):
    from research.h3_personal.ref2va_budget import budget_bound
    return launch(argv,budget_fn=budget_bound,inspect_fn=inspect,bootstrap_entry='research/bootstrap_h3_fast_ref2va.py',pre_submit=pre_submit,
        approval_request='USER gpu検証までgo; latest USER 映像も行きましょう: fresh Irodori locked-speech SMD4 Ref2VA validation toward Discord. Current HF cap190.35 from169.35+21; Modal separate. One finite Job, original64/v12 portrait and BF16 SMD4 preserved; no T2VA fallback/campaign')

if __name__=='__main__':raise SystemExit(main())
