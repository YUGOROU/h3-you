"""One explicitly requested realtime-first Job. No retry, pair, or general cap waiver.

Defaults to offline verification. --submit verifies live idleness/price and reserves
full timeout under shared ledger lock before exactly one submission call.
"""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from research.h3_personal.realtime_first import RealtimeFirstConfig,budget_bound
from research.prepare_h3_realtime_first import IMAGE,SCOPE,check_frozen,safe_path

TERMINAL={'COMPLETED','CANCELED','ERROR','DELETED'}


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def inspect(stage):
    inventory=json.loads((stage/'upload-allowlist.json').read_text());expected={i['path']:i for i in inventory['files']}
    actual={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    if actual!=set(expected)|{'upload-allowlist.json'}:raise ValueError('deployment contains unreviewed/missing files')
    for name,item in expected.items():
        safe_path(name);p=stage/name
        if p.is_symlink() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('deployment file changed: '+name)
    spec=json.loads((stage/'spec.json').read_text());RealtimeFirstConfig.from_dict(spec['config']);check_frozen()
    scope=SCOPE
    if spec['config'].get('personalized'):
        from research.build_h3_personal_fast_adapter import PERSONAL_SCOPE,validate_personal_spec
        validate_personal_spec(spec);scope=PERSONAL_SCOPE
    if spec['image']!=IMAGE or spec['schema']!=scope or spec['approval_scope']!=scope:raise ValueError('unapproved execution scope')
    if spec['execution']['timeout_seconds']!=spec['config']['timeout_seconds'] or not 1<=spec['execution']['timeout_seconds']<=3000 or spec['execution']['max_jobs']!=1:raise ValueError('single finite Job only')
    evidence=spec['license_evidence']
    if evidence['mode']!='operator_hf_acceptance' or evidence['provider']!='huggingface_jobs' or evidence['country_verified']:
        raise ValueError('preserve existing HF acceptance, not verified geography')
    return spec,inventory


def main(argv=None, *, inspect_fn=None, bootstrap_entry='research/bootstrap_h3_realtime_first.py',
         approval_request=None, pre_submit=None, budget_fn=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--submit',action='store_true');a=p.parse_args(argv)
    inspector=inspect_fn or inspect
    bound_fn=budget_fn or budget_bound
    stage=a.stage.resolve();spec,inventory=inspector(stage);ledger_path=a.ledger.resolve()
    if not a.submit:
        l=json.loads(ledger_path.read_text());total=Decimal(str(l['spent_usd']))+sum(Decimal(str(v['upper_bound_usd'])) for v in l['reservations'].values())
        print(json.dumps({k:str(v) for k,v in bound_fn(ledger_bound=total,timeout_seconds=spec['config']['timeout_seconds']).items()}));return 0
    from huggingface_hub import HfApi,CommitOperationAdd
    try:from huggingface_hub import Volume
    except ImportError:Volume=None
    from research.provision_modal_identity_secret import cached_token
    api=HfApi(token=cached_token());repo=spec['publication_repo'];owner=repo.split('/')[0];scope=spec['schema']
    if spec['config'].get('personalized'):
        person=spec['personal_adapter']
        source=api.model_info(person['repo'],revision=person['revision'],files_metadata=True)
        asset=next(s for s in source.siblings if s.rfilename==person['prefix']+'adapter_model.safetensors')
        if not source.private or source.sha!=person['revision'] or not asset.lfs or asset.lfs.sha256!=person['sha256']:
            raise ValueError('private pinned original adapter differs')
        api.auth_check(person['repo'],write=False)
    if pre_submit is not None:pre_submit(api,spec,stage)
    who=api.whoami()
    if owner!='YUGOROU' or who['name']!=owner or who.get('isPro') is not True:raise ValueError('Jobs-enabled owner required')
    def idle():
        if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace=owner)):raise ValueError('HF not idle')
        binary=os.environ.get('H3_MODAL_CLI') or shutil.which('modal')
        if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise ValueError('modal CLI not found; set H3_MODAL_CLI')
        envs=json.loads(subprocess.check_output([binary,'environment','list','--json'],timeout=30))
        if not envs:raise ValueError('Modal inventory unavailable')
        for env in envs:
            apps=json.loads(subprocess.check_output([binary,'app','list','--env',env['name'],'--json'],timeout=30))
            if any(int(app['tasks'])!=0 for app in apps):raise ValueError('Modal not idle')
    idle();hardware=next(h for h in api.list_jobs_hardware() if h.name=='h200x2')
    units={'hour':Decimal(1),'minute':Decimal(60)}
    if hardware.unit_label not in units or not hardware.accelerator:raise ValueError('unverified hardware pricing')
    rate=Decimal(str(hardware.unit_cost_usd))*units[hardware.unit_label]
    seconds=spec['config']['timeout_seconds']
    if budget_fn is None:budget_bound(hourly_rate=rate,timeout_seconds=seconds,ledger_bound=Decimal('0'))
    else:
        current=json.loads(ledger_path.read_text())
        total=Decimal(str(current['spent_usd']))+sum(Decimal(str(v['upper_bound_usd'])) for v in current['reservations'].values())
        bound_fn(hourly_rate=rate,timeout_seconds=seconds,ledger_bound=total)
    marker=stage.parent/'submission_guard.json'
    if marker.exists():raise ValueError('NEVER replay a submitted/unknown-outcome deployment')
    api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
    api.auth_check(repo,repo_type='dataset',write=True)
    names=[i['path'] for i in inventory['files']]+['upload-allowlist.json']
    commit=api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(stage/n)) for n in names],commit_message='Isolated realtime-first visual PoC; one finite Job; no personal media')
    remote=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True)
    rows={s.rfilename:s for s in remote.siblings}
    if not remote.private or set(rows)-{'.gitattributes'}!=set(names):raise ValueError('remote deployment differs')
    for name in names:
        data=(stage/name).read_bytes();s=rows[name]
        if s.size!=len(data):raise ValueError('remote bytes differ')
        expected=hashlib.sha256(data).hexdigest() if s.lfs else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if expected!=(s.lfs.sha256 if s.lfs else s.blob_id):raise ValueError('remote digest differs')
    write(stage.parent/'deployment-upload.json',dict(repo=repo,revision=commit.oid,image=IMAGE))
    with ledger_path.with_suffix(ledger_path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);idle();inspector(stage)
        ledger=json.loads(ledger_path.read_text());total=Decimal(str(ledger['spent_usd']))+sum(Decimal(str(v['upper_bound_usd'])) for v in ledger['reservations'].values())
        bound=bound_fn(ledger_bound=total,hourly_rate=rate,timeout_seconds=seconds);ident='realtime-first-'+commit.oid[:24]
        if marker.exists() or ident in ledger['reservations']:raise ValueError('duplicate submission')
        proof=dict(schema='h3-explicit-single-realtime-first-approval-v1',approved_by='USER',scope=scope,
            request=approval_request or ('USER go: implement original personal LoRA on isolated realtime branch; preserve originals and success' if spec['config'].get('personalized') else 'Preserve baseline; quality may break; combine compatible accelerations into ONE realtime streaming experimental Job'),
            cumulative_cap_usd=float(bound.get('authorized_cap',170)),max_concurrent_jobs=1,max_timeout_seconds=seconds,max_rate_usd_hour=10.00002,
            max_job_usd=float(bound['job_bound']),provider='huggingface_jobs',flavor='h200x2',general_authority_extension=False)
        write(stage.parent/'ledger-before-submission.json',ledger)
        ledger['reservations'][ident]=dict(upper_bound_usd=float(bound['job_bound']),state='reserved_unknown_no_retry',
            provider='huggingface_jobs',flavor='h200x2',scope=scope,rate_usd_hour=float(rate),timeout_seconds=seconds,
            deployment_repo=repo,deployment_revision=commit.oid,approval=proof)
        write(ledger_path,ledger)
        with marker.open('x') as f:json.dump(dict(reservation=ident,state='reserved_unknown_no_retry'),f);f.flush();os.fsync(f.fileno())
        bootstrap=next(i['sha256'] for i in inventory['files'] if i['path']==bootstrap_entry)
        allow_sha=hashlib.sha256((stage/'upload-allowlist.json').read_bytes()).hexdigest()
        script="import hashlib,runpy; p='/deployment/"+str(safe_path(bootstrap_entry))+"'; assert hashlib.sha256(open(p,'rb').read()).hexdigest()=='"+bootstrap+"'; runpy.run_path(p,run_name='__main__')"
        mount_args={}
        if Volume is not None:
            mount_args['volumes']=[Volume(type='dataset',source=repo,revision=commit.oid,mount_path='/deployment',read_only=True)]
        else:
            # Older local SDK: fetch only the SHA-pinned bootstrap, which then
            # verifies the full private immutable deployment allowlist itself.
            script=("import hashlib,runpy,os; from huggingface_hub import hf_hub_download; p=hf_hub_download("
                +repr(repo)+","+repr(str(safe_path(bootstrap_entry)))+",repo_type='dataset',revision="+repr(commit.oid)
                +",token=os.environ['HF_TOKEN']); assert hashlib.sha256(open(p,'rb').read()).hexdigest()=="
                +repr(bootstrap)+"; runpy.run_path(p,run_name='__main__')")
        job=api.run_job(image=IMAGE,command=['/opt/venv/bin/python','-c',script,repo,commit.oid,allow_sha],
            flavor='h200x2',namespace=owner,timeout=seconds,secrets={'HF_TOKEN':cached_token()},
            env={'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1'},
            **mount_args,
            labels={'project':'h3-realtime-first','realtime_deploy':commit.oid[:32]})
        ledger['reservations'][ident].update(job_id=job.id,state='submitted_no_retry');write(ledger_path,ledger)
        receipt=dict(job_id=job.id,scope=scope,deployment_revision=commit.oid,
            reserved_usd=str(bound['job_bound']),cumulative_bound_usd=str(bound['cumulative_bound']))
        write(stage.parent/'submission.json',receipt);print(json.dumps(receipt),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
