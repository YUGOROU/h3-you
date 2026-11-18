"""Fresh immutable Modal live deployment, preserving original64/v12/native voice.
Offline by default; --upload publishes code/manifests only, never starts a GPU.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from research.launch_h3_fast_ref2va import inspect as inspect_assets
from research.h3_personal.fast_ref2va import config_for_layout
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(stage,deployment=None,allowlist_sha=None):
    stage=Path(stage)
    if allowlist_sha is not None and digest(stage/'upload-allowlist.json')!=allowlist_sha:raise ValueError('immutable inventory differs')
    spec,inventory=inspect_assets(stage)
    if deployment is not None and spec['publication_repo']!=deployment:raise ValueError('deployment binding differs')
    if spec.get('locked_speech') is not None or spec['config']['timeout_seconds'] not in (600,720,900,1200,1500,1800) or spec['live_voice_session']['max_turns']!=16 or spec['live_voice_session']['fresh_input_required'] is not True:raise ValueError('fresh finite conversation required')
    if spec['layout']!='portrait' or spec['runtime_policy']!='resident-visual-cached' or spec['attention_mode']!='fa4-packed' or spec.get('regional_compile') or spec.get('face_crop_box') is not None or spec['fast_config']!=asdict(config_for_layout('portrait')):raise ValueError('v12 quality policy differs')
    original=PRIVATE/'gpu-hf-irodori-diagnostic-v1/request.json'
    if digest(original)!='b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':raise ValueError('original native approval manifest drift')
    expected=json.loads(original.read_text())
    selected={k:expected[k] for k in ('source_revision','uv_lock_sha256','uv_version','models','license')}
    if json.loads((stage/'tts-manifest.json').read_text())!=selected:raise ValueError('native TTS pins differ')
    # Executable bytes must match this reviewed working tree, not historical submitters.
    for row in inventory['files']:
        if row['path'] in ('research/__init__.py','research/h3_personal/__init__.py') and row['bytes']==0:continue
        if row['path'].startswith('research/') and digest(ROOT/row['path'])!=row['sha256']:raise ValueError('current reviewed runtime differs: '+row['path'])
    return spec,inventory


def prepare(output,template,region='ap',timeout_seconds=1800,trust_jp_contract=False):
    if trust_jp_contract and region!='jp':raise ValueError('Japan contract trust requires explicit jp')
    if timeout_seconds not in (600,720,900,1200,1500,1800):raise ValueError('finite approved duration required')
    if region not in ('ap','jp','au'):raise ValueError('explicit approved placement selector required')
    from research.launch_h3_live_voice import inspect as old_inspect
    old_inspect(template)
    output=Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT):raise ValueError('fresh private stage required')
    shutil.copytree(template,output)
    for path in output.rglob('*.py'):
        name=path.relative_to(output)
        if name.as_posix() in ('research/__init__.py','research/h3_personal/__init__.py') and path.stat().st_size==0:continue
        shutil.copyfile(ROOT/name,path)
    spec=json.loads((output/'spec.json').read_text())
    spec['publication_repo']='YUGOROU/h3-fast-ref2va-smd4-gpu-modal-live-'+uuid.uuid4().hex[:10]
    spec['config']['timeout_seconds']=timeout_seconds
    spec['execution'].update(timeout_seconds=timeout_seconds,max_jobs=1)
    spec['execution'].pop('job_bound_usd',None) # HF rate is NOT a Modal quote.
    spec['live_voice_session']=dict(id=uuid.uuid4().hex,max_turns=16,fresh_input_required=True)
    spec['modal_live_approval']=dict(schema='h3-modal-live-approval-v1',provider='modal',region=region,allowed_runtime_regions=['jp','au','ca','mx'],approval='USER go 2-4; all subsequent GPU on Modal H200x2, shared20.44 pool; runtime location gate before model load; new actual call speech only')
    spec['license_evidence']=dict(mode='runtime_modal_region_gate',provider='modal',country_verified=False,excluded=['EU','GB','KR','US'],allowed_runtime_regions=['jp','au','ca','mx'])
    if trust_jp_contract:
        spec['modal_live_approval']['trust_explicit_jp_contract']=True
        spec['license_evidence'].update(mode='operator_modal_jp_contract',requested_region='jp',runtime_country_verified=False,approval_text='USER: 日本を指定して確保できているなら、表示がより広い範囲でもjpと信頼してOKです。Modalを信じましょう')
    (output/'spec.json').write_text(json.dumps(spec,indent=2))
    rows=[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='upload-allowlist.json']
    (output/'upload-allowlist.json').write_text(json.dumps(dict(files=rows),indent=2))
    inspect(output)
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--region',choices=['ap','jp','au'],default='ap');p.add_argument('--session-seconds',type=int,choices=[600,720,900,1200,1500,1800],default=1800);p.add_argument('--trust-jp-contract',action='store_true');p.add_argument('--upload',action='store_true');a=p.parse_args()
    spec=prepare(a.output,PRIVATE/'live-voice-session-20261017-a/deployment',a.region,a.session_seconds,a.trust_jp_contract)
    result=dict(deployment=spec['publication_repo'],stage=str(a.output),allowlist_sha256=digest(a.output/'upload-allowlist.json'))
    if a.upload:
        from huggingface_hub import HfApi
        from research.provision_modal_identity_secret import cached_token
        from research.prepare_h3_realtime_first import metadata_entry
        api=HfApi(token=cached_token())
        for asset in json.loads((a.output/'asset-manifest.json').read_text())['assets']:
            kind=asset.get('repo_type','model');api.auth_check(asset['repo'],repo_type=kind,write=False)
            info=api.repo_info(asset['repo'],repo_type=kind,revision=asset['revision'],files_metadata=True)
            if info.sha!=asset['revision'] or (asset['role'] in ('personal','reference') and not info.private):raise ValueError('live asset rights/privacy differs')
            rows={r.rfilename:r for r in info.siblings}
            if any(metadata_entry(rows[r['path']])!=r for r in asset['files']):raise ValueError('live pinned metadata differs')
        api.create_repo(spec['publication_repo'],repo_type='dataset',private=True)
        if not api.repo_info(spec['publication_repo'],repo_type='dataset').private:raise ValueError('private publication required')
        commit=api.upload_folder(repo_id=spec['publication_repo'],repo_type='dataset',folder_path=str(a.output),commit_message='Fresh Modal resident conversation code; runtime territory gate; no raw media')
        result['revision']=commit.oid
        with (a.output.parent/'deployment-upload.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
