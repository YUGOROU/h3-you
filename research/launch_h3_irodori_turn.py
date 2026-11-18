"""Prepare/submit ONE fresh audio-only conversational utterance. No polling/devices.

--proposal is the saved JSON turn emitted by the Pi director. Default is offline.
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
import sys
from research.h3_personal.director_tts_job import IMAGE,SCOPE,FILES,REFERENCE_REPO,REFERENCE_REV,inspect,digest,DEFAULT_TIMEOUT_SECONDS,INPUT_ORIGINS,validate_options
from research.h3_personal.director_runtime import atomic,MODEL
from research.h3_personal.ref2va_budget import budget_bound
from research.h3_personal.private_root import root as _private_root

TERMINAL={'COMPLETED','CANCELED','ERROR','DELETED'}
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()


def prepare(directory,proposal,timeout_seconds=DEFAULT_TIMEOUT_SECONDS,input_origin=INPUT_ORIGINS[0]):
    validate_options(timeout_seconds,input_origin)
    p=json.loads(proposal.read_text())
    if p.get('type')!='turn' or p.get('model')!=MODEL or p.get('status')!='proposal_only':raise ValueError('fresh Pi proposal required')
    original=PRIVATE/'gpu-hf-irodori-diagnostic-v1/request.json'
    if digest(original)!='b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':raise ValueError('original native manifest drift')
    directory.mkdir(mode=0o700);stage=directory/'deployment';stage.mkdir(mode=0o700)
    request=json.loads(original.read_text());request.update(scope=SCOPE,text=p['reply'],director_model=MODEL,input_origin=input_origin,
        timeout_seconds=timeout_seconds,max_syntheses=1,output_repo='YUGOROU/h3-director-irodori-turn-'+directory.name)
    atomic(stage/'request.json',request)
    for name in FILES-{'request.json'}:
        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    inventory={'files':[{'path':name,'bytes':(stage/name).stat().st_size,'sha256':digest(stage/name)} for name in sorted(FILES)]}
    atomic(stage/'upload-allowlist.json',inventory);inspect(stage);return stage


def total(ledger):return Decimal(str(ledger['spent_usd']))+sum(Decimal(str(r['upper_bound_usd'])) for r in ledger['reservations'].values())


def submit(stage,ledger_path):
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    from research.provision_modal_identity_secret import cached_token
    api=HfApi(token=cached_token());request,inventory=inspect(stage);owner='YUGOROU'
    who=api.whoami()
    if who['name']!=owner or who.get('isPro') is not True:raise ValueError('Jobs-enabled owner required')
    def idle():
        if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace=owner)):raise ValueError('HF GPU not idle')
        environments=json.loads(subprocess.check_output([str(Path.home()/'.local/bin/modal'),'environment','list','--json'],timeout=30))
        if not environments:raise ValueError('Modal inventory unavailable')
        for env in environments:
            apps=json.loads(subprocess.check_output([str(Path.home()/'.local/bin/modal'),'app','list','--env',env['name'],'--json'],timeout=30))
            if any(int(a['tasks']) for a in apps):raise ValueError('Modal GPU not idle')
    idle()
    api.auth_check(REFERENCE_REPO,repo_type='dataset',write=False)
    source=api.repo_info(REFERENCE_REPO,repo_type='dataset',revision=REFERENCE_REV)
    if not source.private or source.sha!=REFERENCE_REV:raise ValueError('private immutable reference missing')
    original=Path(hf_hub_download(REFERENCE_REPO,'request.json',repo_type='dataset',revision=REFERENCE_REV,token=cached_token()))
    if digest(original)!='b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':raise ValueError('remote original manifest drift')
    if request['models']!=json.loads(original.read_text())['models']:raise ValueError('model inventory differs from accepted native deployment')
    for m in request['models']:
        api.auth_check(m['repo'],write=False)
        if api.model_info(m['repo'],revision=m['revision']).sha!=m['revision']:raise ValueError('model pin unavailable')
        if m['repo']=='sony/silentcipher' and api.model_info(m['repo'],revision='main').sha!=m['revision']:raise ValueError('native watermark main drift')
    hardware=next(h for h in api.list_jobs_hardware() if h.name=='h200')
    units={'minute':Decimal(60),'hour':Decimal(1)}
    if hardware.unit_label not in units or not hardware.accelerator:raise ValueError('unverified H200 pricing')
    rate=Decimal(str(hardware.unit_cost_usd))*units[hardware.unit_label]
    budget_bound(ledger_bound=total(json.loads(ledger_path.read_text())),hourly_rate=rate,timeout_seconds=request['timeout_seconds'])
    marker=stage.parent/'submission_guard.json'
    if marker.exists():raise ValueError('never replay any submitted/unknown deployment')
    repo=request['output_repo'];api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False);api.auth_check(repo,repo_type='dataset',write=True)
    names=sorted(FILES|{'upload-allowlist.json'})
    commit=api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(stage/n)) for n in names],commit_message='Immutable one fresh Pi-Irodori audio turn; no new raw upload')
    remote=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True);siblings={r.rfilename:r for r in remote.siblings}
    if not remote.private or set(siblings)-{'.gitattributes'}!=set(names):raise ValueError('remote deployment differs')
    for name in names:
        b=(stage/name).read_bytes();s=siblings[name]
        if s.size!=len(b) or (s.lfs.sha256 if s.lfs else s.blob_id)!=(hashlib.sha256(b).hexdigest() if s.lfs else hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()):raise ValueError('remote deployment digest differs')
    atomic(stage.parent/'deployment-upload.json',dict(repo=repo,revision=commit.oid,image=IMAGE))
    allow_sha=digest(stage/'upload-allowlist.json')
    # Bootstrap is inline/hash anchored; no mutable mount or historical worker execution.
    bootstrap='''import hashlib,json,os,pathlib,shutil,subprocess,sys,runpy
sys.dont_write_bytecode=True
subprocess.run(['python3','-m','pip','install','--quiet','huggingface_hub==1.7.2','uv==0.13.1'],check=True)
if any(shutil.which(name) is None for name in ('ffmpeg','ffprobe','git')):
 subprocess.run(['apt-get','update','-qq'],check=True)
 subprocess.run(['apt-get','install','-y','-qq','ffmpeg','libsndfile1','git'],check=True)
for name in ('ffmpeg','ffprobe','git','uv'):
 assert shutil.which(name) is not None, 'required runtime executable unavailable: '+name
from huggingface_hub import hf_hub_download
repo,rev,sha=sys.argv[1:];token=os.environ['HF_TOKEN'];root=pathlib.Path('/tmp/director-turn-deployment');root.mkdir(mode=0o700)
f=pathlib.Path(hf_hub_download(repo,'upload-allowlist.json',repo_type='dataset',revision=rev,token=token))
assert hashlib.sha256(f.read_bytes()).hexdigest()==sha
for row in json.loads(f.read_text())['files']:
 name=row['path'];p=pathlib.PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in name
 src=pathlib.Path(hf_hub_download(repo,name,repo_type='dataset',revision=rev,token=token));b=src.read_bytes()
 assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
shutil.copyfile(f,root/'upload-allowlist.json');os.chdir(root);sys.path.insert(0,str(root))
from research.h3_personal.director_tts_job import inspect
from research.h3_personal.private_root import root as _private_root
inspect(root);sys.argv=['hf_irodori_turn.py',str(root)];runpy.run_path(str(root/'research/hf_irodori_turn.py'),run_name='__main__')
'''
    with ledger_path.with_suffix(ledger_path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);idle();inspect(stage)
        ledger=json.loads(ledger_path.read_text());bound=budget_bound(ledger_bound=total(ledger),hourly_rate=rate,timeout_seconds=request['timeout_seconds'])
        key='irodori-turn-'+commit.oid[:24]
        ledger['reservations'][key]=dict(provider='huggingface_jobs',scope=SCOPE,flavor='h200',state='reserved_unknown_no_retry',upper_bound_usd=float(bound['job_bound']),rate_usd_hour=float(rate),timeout_seconds=request['timeout_seconds'],deployment_repo=repo,deployment_revision=commit.oid)
        atomic(ledger_path,ledger)
        with marker.open('x') as stream:json.dump(dict(reservation=key,state='reserved_unknown_no_retry'),stream);stream.flush();os.fsync(stream.fileno())
        job=api.run_job(image=IMAGE,command=['python3','-c',bootstrap,repo,commit.oid,allow_sha],flavor='h200',namespace=owner,timeout=request['timeout_seconds'],
                        secrets={'HF_TOKEN':cached_token()},env={'HF_HUB_DISABLE_TELEMETRY':'1','HF_DEBUG':'0','UV_NO_PROGRESS':'1','PYTHONDONTWRITEBYTECODE':'1'},labels={'project':'h3-irodori-turn','deploy':commit.oid[:24]})
        ledger['reservations'][key].update(job_id=job.id,state='submitted_no_retry');atomic(ledger_path,ledger)
        receipt=dict(job_id=job.id,reservation=key,repo=repo,deployment_revision=commit.oid,reserved_usd=str(bound['job_bound']),hourly_rate=str(rate),timeout_seconds=request['timeout_seconds'],cumulative_bound=str(bound['cumulative_bound']))
        atomic(stage.parent/'submission.json',receipt);print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--proposal',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--timeout-seconds',type=int,default=DEFAULT_TIMEOUT_SECONDS);p.add_argument('--input-origin',choices=INPUT_ORIGINS,default=INPUT_ORIGINS[0]);a=p.parse_args()
    try:validate_options(a.timeout_seconds,a.input_origin)
    except ValueError as error:p.error(str(error))
    stage=a.directory/'deployment'
    if not stage.exists():
        if a.proposal is None:p.error('new stage requires --proposal')
        stage=prepare(a.directory,a.proposal,timeout_seconds=a.timeout_seconds,input_origin=a.input_origin)
    request,_=inspect(stage)
    if a.submit:submit(stage,PRIVATE/'compute-budget.json')
    else:print(json.dumps(dict(verified=True,scope=request['scope'],image=IMAGE,syntheses=1,timeout_seconds=request['timeout_seconds'])))

if __name__=='__main__':main()
