"""Finite two-CPU-Job Group/TLS connectivity probe; offline by default."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import ssl
import subprocess
import tempfile
import time
import uuid
import zipfile
from research.run_h3_native_coupled import PRIVATE,ROOT,FILES,persist
from research.run_h3_network_pair import publish,replace_json,terminal
from research.run_h3_native_tts_probe import bootstrap
from research.run_h3_asr_endpoint import BASE
from research.h3_personal.job_network import network

FENCE=PRIVATE/'.network-group-probe-unresolved.json'
TERMINAL={'CANCELED','COMPLETED','ERROR','DELETED'}


def log_event(api,job,kind,deadline):
    while time.monotonic()<deadline:
        for n,line in enumerate(api.fetch_job_logs(job_id=job,namespace='YUGOROU')):
            if n>=10000 or time.monotonic()>=deadline:raise TimeoutError('CPU probe logs bound')
            if len(line)>16384:raise ValueError('CPU log line bound')
            try:row=json.loads(line)
            except (ValueError,TypeError):continue
            if type(row) is dict and row.get('type')==kind:return row
            if type(row) is dict and row.get('type')=='h3_network_cpu_failed':raise RuntimeError('CPU probe worker failed')
        state=api.inspect_job(job_id=job,namespace='YUGOROU').status.stage
        if state in TERMINAL:raise RuntimeError('CPU worker ended before objective')
        time.sleep(.2)
    raise TimeoutError('finite CPU probe readiness')


def run(directory):
    import httpx
    from huggingface_hub import HfApi,set_client_factory
    from research.provision_modal_identity_secret import cached_token
    root=Path(os.path.abspath(directory));fd=os.open(PRIVATE/'.network-group-probe-owner.lock',os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if FENCE.exists() or FENCE.is_symlink():raise ValueError('unresolved CPU creation; never replay')
        if root.exists() or not root.is_relative_to(PRIVATE) or any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('fresh private root')
        root.mkdir(mode=0o700)
        set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(20,connect=5),trust_env=False))
        token=cached_token();api=HfApi(token=token)
        if api.whoami().get('name')!='YUGOROU':raise ValueError('owned namespace required')
        active=[j for j in api.list_jobs(namespace='YUGOROU') if j.status.stage not in TERMINAL]
        if any(str(j.flavor)!='cpu-basic' for j in active):raise ValueError('active non-CPU Job')
        persist(root/'preexisting-jobs.json',dict(jobs=[dict(id=j.id,status=j.status.stage,flavor=str(j.flavor)) for j in active]))
        with httpx.Client(timeout=httpx.Timeout(20,connect=5),follow_redirects=False,trust_env=False) as client:
            return owned(api,client,token,root)
    finally:os.close(fd)


def owned(api,client,token,root):
    jobs={};pending=None;errors=[];result=dict(scope='synthetic CPU only',gpu=False,devices=False,turn_used=False)
    try:
        with tempfile.TemporaryDirectory(prefix='h3-net-cpu-') as tmp:
            cert=Path(tmp)/'cert.pem';keyfile=Path(tmp)/'key.pem'
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=h3-network-cpu',
                '-out',str(cert),'-keyout',str(keyfile)],timeout=10,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            certificate=cert.read_text();private=keyfile.read_text();key=secrets.token_urlsafe(40)
            group='h3-cpu-'+uuid.uuid4().hex[:24];session=uuid.uuid4().hex;boot=uuid.uuid4().hex;expires=int(time.time())+600
            spec=dict(connection=dict(group=group,alias='voxcpm',port=17861,
                certificate_sha256=hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest()),session=session,boot=boot,expires_at=expires)
            archive=root/'runtime.zip'
            extra=('research/hf_h3_network_group_probe.py','research/h3_personal/network_voxcpm.py',
                'research/h3_personal/network_voxcpm_service.py','research/h3_personal/tts_preroll.py')
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                for name in dict.fromkeys((*FILES,*extra)):z.writestr(name,(ROOT/name).read_bytes())
                for name in ('research/__init__.py','research/h3_personal/__init__.py'):z.writestr(name,'')
                z.writestr('request.json',json.dumps(spec))
            repo,revision,digest=publish(api,root,archive)
            script=bootstrap(repo,revision,digest).replace('runtime.zip','runtime.source').replace('research/hf_h3_native_tts_probe.py','research/hf_h3_network_group_probe.py')
            command=['python','-c',script]
            def create(role):
                nonlocal pending
                pending=role
                state=dict(directory=str(root),jobs=dict(jobs),pending=role)
                if FENCE.exists():replace_json(FENCE,state)
                else:persist(FENCE,state)
                environment=dict(H3_PROBE_ROLE=role)
                if role=='client':environment['H3_PROBE_SERVER_JOB_ID']=jobs['server']
                source_secrets=dict(HF_TOKEN=token,H3_PROBE_KEY=key)
                if role=='server':source_secrets.update(H3_NATIVE_TLS_CERT=certificate,H3_NATIVE_TLS_KEY=private)
                body=dict(dockerImage=BASE,command=command,arguments=[],environment=environment,secrets=source_secrets,
                    flavor='cpu-basic',timeoutSeconds=600,attempts=1,network=network(group,['voxcpm'] if role=='server' else []),
                    labels=dict(scope='h3-network-cpu',deployrev=revision))
                response=client.post('https://huggingface.co/api/jobs/YUGOROU',headers={'Authorization':'Bearer '+token},json=body)
                if response.status_code not in (200,201):
                    if response.status_code in (400,401,403,422):pending=None
                    raise RuntimeError('CPU Job create rejected or unknown')
                job=response.json().get('id')
                if type(job) is not str or not re.fullmatch('[a-f0-9]{24}',job):raise RuntimeError('unknown CPU Job outcome')
                jobs[role]=job;pending=None
                replace_json(FENCE,dict(directory=str(root),jobs=dict(jobs),pending=None))
                persist(root/(role+'-submission.json'),dict(job_id=job,network=body['network'],flavor='cpu-basic'))
                print(json.dumps(dict(type='h3_network_cpu_submitted',role=role,job_id=job)),flush=True)
                return job
            server=create('server');log_event(api,server,'h3_network_cpu_ready',time.monotonic()+180)
            consumer=create('client');proof=log_event(api,consumer,'h3_network_cpu_result',time.monotonic()+180)
            if proof.get('exact_match') is not True or proof.get('tls_pinned') is not True or proof.get('turns')!=2 or proof.get('gpu') is not False:
                raise ValueError('CPU TLS proof incomplete')
            result['proof']=proof;persist(root/'connection.json',proof)
            return result
    finally:
        primary=__import__('sys').exc_info()[1]
        for role,job in reversed(tuple(jobs.items())):
            try:terminal(api,job,root/(role+'-terminal.json'))
            except BaseException as error:errors.append(error)
        if pending is not None:errors.append(RuntimeError('unknown CPU create outcome; fence retained'))
        if not errors and FENCE.exists() and json.loads(FENCE.read_text()).get('directory')==str(root):FENCE.unlink()
        result['cleanup_confirmed']=not errors;result['jobs']=dict(jobs);persist(root/'result.json',result)
        if errors:
            if primary is not None:primary.add_note('CPU cleanup unconfirmed; fence retained')
            else:raise errors[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--directory',type=Path)
    a=p.parse_args()
    if not a.run:print(json.dumps(dict(scope='offline',gpu=False,devices=False,jobs=0)));return 0
    if a.directory is None:p.error('fresh private directory required')
    import signal
    def interrupted(*_):raise InterruptedError('finite CPU owner stopped')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(660)
    try:run(a.directory)
    except BaseException as error:
        print(json.dumps(dict(type='h3_network_cpu_owner_failed',error_type=type(error).__name__)),flush=True);return 1
    finally:signal.alarm(0)
    return 0

if __name__=='__main__':raise SystemExit(main())
