"""Finite paired L4/H3 owner. Offline CLI; live API requires placement gates.

No app/capture operations are implicit. Caller supplies an explicitly approved
conversation callback and independent source-placement validation.
"""
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
from types import SimpleNamespace
import uuid

from research.run_h3_native_coupled import PRIVATE,_model_preflight,persist,control
from research.prepare_h3_network_tts_job import bundle,payload
from research.h3_personal.network_voxcpm import validate_config,SCHEMA
from research.h3_personal.job_network import assert_source
from research.h3_personal.voxcpm_endpoint_protocol import PINS

TERMINAL={'CANCELED','COMPLETED','ERROR','DELETED'}
FENCE=PRIVATE/'.network-tts-unresolved.json'
LEASE_SECONDS=1200
SCHEDULING_WAIT_MAX=4*60*60


def window(scheduling_wait):
    """Submission deadline = queue allowance + unchanged runtime lease.

    Hardware may take a long time to be scheduled; that wait must never eat the
    source/conversation lease, and it must not silently become unlimited either.
    The worker clamps its effective lease to LEASE_SECONDS from container start,
    which is what READY reports back and what the H3 side validates.
    """
    if type(scheduling_wait) is not int or not 0<=scheduling_wait<=SCHEDULING_WAIT_MAX:
        raise ValueError('explicit bounded scheduling wait required')
    return int(time.time())+scheduling_wait+LEASE_SECONDS,scheduling_wait


def replace_json(path,row):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    persist(temporary,row);os.replace(temporary,path)


def terminal(api,job_id,root):
    """Inspect failure never prevents cancellation; persist only confirmed end."""
    try:state=api.inspect_job(job_id=job_id,namespace='YUGOROU').status.stage
    except Exception:state=None
    if state not in TERMINAL:api.cancel_job(job_id=job_id,namespace='YUGOROU')
    for _ in range(8):
        state=api.inspect_job(job_id=job_id,namespace='YUGOROU').status.stage
        if state in TERMINAL:
            persist(Path(root),dict(job_id=job_id,status=state));return
        time.sleep(1)
    raise RuntimeError('owned Job terminal status unconfirmed')


def publish(api,root,artifact):
    """Fresh private source, uncertain writes reconciled GET-only, no replay."""
    import httpx
    repo='YUGOROU/h3-network-tts-'+uuid.uuid4().hex[:16]
    raw=artifact.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    persist(root/'publication.json',dict(repo=repo,state='unknown-no-replay',source_sha256=digest))
    try:api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
    except httpx.TransportError:
        info=api.repo_info(repo,repo_type='dataset')
        if info.id!=repo or info.private is not True or any(s.rfilename!='.gitattributes' for s in info.siblings):raise
    try:
        commit=api.upload_file(path_or_fileobj=str(artifact),path_in_repo='runtime.source',repo_id=repo,
            repo_type='dataset',commit_message='Immutable finite native TLS TTS Job; source only')
    except httpx.TransportError:
        info=api.repo_info(repo,repo_type='dataset',files_metadata=True)
        commit=SimpleNamespace(oid=info.sha)
    info=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True)
    entries={s.rfilename:s for s in info.siblings};entry=entries.get('runtime.source')
    expected=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if (info.id!=repo or info.private is not True or info.sha!=commit.oid
            or set(entries)-{'.gitattributes'}!={'runtime.source'} or entry.size!=len(raw)
            or (entry.lfs.sha256 if entry.lfs else entry.blob_id)!=(digest if entry.lfs else expected)):
        raise ValueError('private immutable source mismatch')
    replace_json(root/'publication.json',dict(repo=repo,revision=commit.oid,source_sha256=digest,state='verified'))
    return repo,commit.oid,digest


def metadata(api,job_id,deadline,kind):
    rows=[]
    for count,line in enumerate(api.fetch_job_logs(job_id=job_id,namespace='YUGOROU')):
        if count>=10000 or time.monotonic()>=deadline:raise TimeoutError('bounded source metadata')
        if len(line)>65536:raise ValueError('source metadata byte bound')
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if type(row) is dict and row.get('type')==kind:
            rows.append(row)
            if kind=='h3_voxcpm_network_ready' or len(rows)==2:return rows
    return rows


def run(directory,*,conversation,source_placement_gate,scheduling_wait=600,h3_first=False):
    """conversation(root,config) uses the existing direct RTC/ASR owner.

    placement gate must return exactly True both for a sanitized pre-create
    plan (before reference access on a worker), and raw running Job metadata
    before H3 starts. It must independently validate the approved placement
    contract; absent sufficient evidence it must reject.

    USER 2026-10-29 scoped the TTS source policy: voxcpm2 TTS has no region or
    territory restriction, so country evidence is not required for the L4
    source and an unverifiable region must stay labelled unverified rather than
    inferred from network membership or an L4 device name. This does NOT relax
    H3's own territory/license placement, which keeps its separate gates.
    """
    if type(h3_first) is not bool:raise ValueError('explicit h3_first bool required')
    if not callable(conversation) or not callable(source_placement_gate):raise ValueError('explicit conversation and placement gates required')
    root=Path(os.path.abspath(directory));locks=[]
    if root.exists() or not root.is_relative_to(PRIVATE) or any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('fresh private pair root')
    try:
        for name in ('.voxcpm-endpoint-owner.lock','.irodori-endpoint-owner.lock','.native-streaming-tts-owner.lock','.network-tts-owner.lock'):
            fd=os.open(PRIVATE/name,os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW,0o600);locks.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in (FENCE.name,'.voxcpm-endpoint-unresolved.json','.irodori-endpoint-unresolved.json','.native-streaming-tts-unresolved.json'):
            if (PRIVATE/name).exists() or (PRIVATE/name).is_symlink():raise RuntimeError('unresolved owned resource')
        if any(name in os.environ for name in ('H3_NATIVE_TTS_KEY','H3_NATIVE_TTS_HF_TOKEN')):raise ValueError('preexisting native secrets forbidden')
        root.mkdir(mode=0o700)
        import httpx
        from huggingface_hub import HfApi,set_client_factory
        from research.provision_modal_identity_secret import cached_token
        set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(20,connect=5),trust_env=False))
        token=cached_token();api=HfApi(token=token)
        if api.whoami().get('name')!='YUGOROU':raise ValueError('owned namespace')
        if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):raise ValueError('Jobs not idle; leave others untouched')
        endpoints=control(token,'/v2/endpoint/YUGOROU')
        if type(endpoints.get('items')) is not list or any(e.get('status',{}).get('state') not in ('paused','scaledToZero') for e in endpoints['items']):
            raise ValueError('preexisting Endpoint allocation; leave untouched')
        hardware=[h for h in api.list_jobs_hardware() if h.name=='l4x1']
        if len(hardware)!=1 or hardware[0].accelerator is None or str(hardware[0].accelerator.quantity)!='1':raise ValueError('one L4 required')
        spec=_model_preflight(api)
        with httpx.Client(timeout=httpx.Timeout(20,connect=5),follow_redirects=False,trust_env=False) as client:
            return owned(api,client,token,root,spec,conversation,source_placement_gate,scheduling_wait,h3_first=h3_first)
    finally:
        for fd in reversed(locks):os.close(fd)


def owned(api,client,token,root,spec,conversation,source_placement_gate,scheduling_wait=600,*,h3_first=False):
    source=None;rejected=False;cleanup_errors=[];deferred=None;source_attempted=False
    result=dict(schema='h3-network-pair-v1',complete=False,all_media_webrtc=False)
    session=uuid.uuid4().hex;boot=uuid.uuid4().hex;group='h3-pair-'+uuid.uuid4().hex[:24]
    key=secrets.token_urlsafe(40)
    expires,wait=window(scheduling_wait)
    try:
        with tempfile.TemporaryDirectory(prefix='h3-pair-tls-') as temporary:
            temporary=Path(temporary);cert=temporary/'cert.pem';private=temporary/'key.pem'
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=h3-native-pair',
                '-out',str(cert),'-keyout',str(private)],check=True,timeout=10,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            cert.chmod(0o600);private.chmod(0o600)
            certificate=cert.read_text();secret=private.read_text()
            fingerprint=hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest()
            spec=dict(spec,expires_at=expires,queue_budget_seconds=wait,native_pins=dict(PINS),network_service=dict(
                group=group,alias='voxcpm',port=17861,certificate_sha256=fingerprint))
            from research.h3_personal.director_tts_job import IMAGE
            plan=dict(phase='before_source_create',namespace='YUGOROU',flavor='l4x1',image=IMAGE,group=group)
            if source_placement_gate(plan,spec) is not True:raise ValueError('source placement preauthorization required')
            artifact=bundle(root/'artifact',spec);repo,revision,digest=publish(api,root,artifact)
            deployment=payload(repo,revision,digest,group=group,session=session,boot=boot,
                token=token,key=key,certificate=certificate,private_key=secret)
            deployment['command'][2]=deployment['command'][2].replace('runtime.zip','runtime.source')
            persist(root/'source-deployment.json',dict(repo=repo,revision=revision,group=group,expires_at=expires,
                scheduling_wait_seconds=wait,run_lease_seconds=LEASE_SECONDS,
                command_sha256=hashlib.sha256(json.dumps(deployment['command']).encode()).hexdigest(),timeout_seconds=1200,attempts=1))
            if h3_first:
                from research.h3_personal.h3_first_owner import DeferredH3Owner
                deferred=DeferredH3Owner(api,client,token,root,group)
                persist(FENCE,dict(directory=str(root),state='unknown-h3-first-create-no-replay'))
                deferred.start()
                replace_json(FENCE,dict(directory=str(root),h3_job_id=deferred.job.id,state='h3-first-owned'))
            else:
                persist(FENCE,dict(directory=str(root),state='unknown-source-create-no-replay'))
            source_attempted=True
            response=client.post('https://huggingface.co/api/jobs/YUGOROU',headers={'Authorization':'Bearer '+token},json=deployment)
            if response.status_code not in (200,201):
                rejected=response.status_code in (400,401,403,422)
                detail=str(getattr(response,'text',''))[:8192]
                for value in deployment.get('secrets',{}).values():
                    if type(value) is str and value:detail=detail.replace(value,'[REDACTED]')
                persist(root/'source-rejection.json',dict(status=response.status_code,detail=detail[:2048]))
                raise RuntimeError('source Job creation rejected or uncertain (HTTP '+str(response.status_code)+')')
            row=response.json();job_id=row.get('id')
            if type(job_id) is not str or not re.fullmatch('[a-f0-9]{24}',job_id):raise RuntimeError('unknown source Job creation')
            source=job_id
            replace_json(FENCE,dict(directory=str(root),source_job_id=source,state='source-owned'))
            persist(root/'source-submission.json',dict(job_id=source))
            deadline=time.monotonic()+wait
            if deferred is not None:deadline=min(deadline,time.monotonic()+deferred.remaining()-540)
            ready=None;interval=.2
            now=time.monotonic()
            while now<deadline:
                state=api.inspect_job(job_id=source,namespace='YUGOROU').status.stage
                if state in TERMINAL:raise RuntimeError('source ended before ready')
                # Skip log polling while queued: keeps a long scheduling wait to
                # roughly one control-plane call per interval instead of two.
                if state!='SCHEDULING':
                    rows=metadata(api,source,deadline,'h3_voxcpm_network_ready')
                    if rows:ready=rows[0];break
                time.sleep(interval);interval=min(15.0,interval*1.5);now=time.monotonic()
            if ready is None:raise TimeoutError('source readiness lease margin')
            config=validate_config(ready['config'])
            expected=dict(spec['network_service'],server_job_id=source)
            effective=config['expires_at']
            if (config['connection']!=expected or config['session']!=session or config['boot']!=boot
                    or not time.time()+540<effective<=expires):raise ValueError('source READY mismatch')
            response=client.get('https://huggingface.co/api/jobs/YUGOROU/'+source,headers={'Authorization':'Bearer '+token})
            if response.status_code!=200:raise RuntimeError('source inspection unavailable')
            raw=response.json();assert_source(raw,config)
            if raw.get('command')!=deployment['command']:raise ValueError('source immutable command mismatch')
            running_spec=dict(spec,network_service=expected)
            if source_placement_gate(raw,running_spec) is not True:raise ValueError('source placement not accepted')
            runtime=ready['runtime']
            if runtime.get('device_name')!='NVIDIA L4' or runtime.get('device_count')!=1 or runtime.get('optimize') is not False or runtime.get('sample_rate')!=48000:
                raise ValueError('source runtime proof')
            persist(root/'source-ready.json',ready);persist(root/'native-config.json',config)
            if deferred is not None:deferred.bind_source(source,deployment['command'],revision)
            os.environ['H3_NATIVE_TTS_KEY']=key
            try:result['h3_result']=conversation(root,config,**({'adoption':deferred} if deferred is not None else {}))
            finally:os.environ.pop('H3_NATIVE_TTS_KEY',None)
            completions=metadata(api,source,time.monotonic()+15,'candidate.completion')
            evidence=dict(binding=dict(PINS,session=session,boot=boot,expires_at=effective,epoch=0),
                          evidence=dict(runtime=runtime,completions=completions))
            persist(root/'source-evidence.json',evidence)
            from research.run_h3_rtc_conversation import validate_evidence
            result['validation']=validate_evidence(root,config,evidence,result['h3_result'])
            result['complete']=result['h3_result']['media_complete']
            return result
    finally:
        primary=__import__('sys').exc_info()[1]
        h3=root/'h3/submission.json';unknown_h3=(root/'h3/submission_guard.json').exists() and not h3.exists()
        jobs=[]
        if deferred is not None and deferred.job is not None:
            jobs.append(('h3',deferred.job.id))
        early_guard=(root/'h3-first-guard.json').exists()
        early_rejected=False
        if (root/'h3-first-rejection.json').exists():
            try:early_rejected=json.loads((root/'h3-first-rejection.json').read_text()).get('status') in (400,401,403,422)
            except (ValueError,OSError):pass
        if deferred is not None and deferred.job is None and early_guard and not early_rejected:unknown_h3=True
        if h3.exists():
            try:
                h3_id=json.loads(h3.read_text())['job_id']
                if type(h3_id) is not str or not re.fullmatch('[a-f0-9]{24}',h3_id):raise ValueError('H3 receipt identity')
                if deferred is not None and deferred.job is not None and h3_id!=deferred.job.id:
                    raise ValueError('adopted H3 identity differs')
                if ('h3',h3_id) not in jobs:jobs.append(('h3',h3_id))
            except BaseException as error:cleanup_errors.append(error)
        if source is not None:jobs.append(('source',source))
        for name,job_id in jobs:
            try:terminal(api,job_id,root/(name+'-terminal.json'))
            except BaseException as error:cleanup_errors.append(error)
        if deferred is not None and deferred.job is not None:
            # Existing reservation is owned even if capsule response was lost;
            # no second H3 POST is possible in the adoption path.
            unknown_h3=False
        if source_attempted and source is None and not rejected:
            cleanup_errors.append(RuntimeError('source create outcome unknown; fence retained'))
        if unknown_h3:cleanup_errors.append(RuntimeError('H3 create outcome unknown; fence retained'))
        if not cleanup_errors and (source is not None or rejected or early_rejected or deferred is not None and deferred.job is not None):
            if FENCE.exists() and json.loads(FENCE.read_text()).get('directory')==str(root):FENCE.unlink()
        result['cleanup_confirmed']=not cleanup_errors and (source is not None or deferred is not None and deferred.job is not None)
        persist(root/'result.json',result)
        if cleanup_errors:
            if primary is not None:primary.add_note('paired Job cleanup unconfirmed; fence retained')
            else:raise cleanup_errors[0]

if __name__=='__main__':
    print(json.dumps(dict(scope='offline',gpu_started=False,capture_started=False,
        live_api_requires='explicit conversation and source placement gates',source='L4 Job',h3='original H200x2')))
