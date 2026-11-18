"""One finite eager Vox L4 Endpoint + original H3 Job, two headless fresh turns.

No devices/capture/calls. Explicit --run only; two resource cleanup authorities.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time
import uuid

from research.run_h3_irodori_endpoint import control,pause_owned,persist,ControlError
from research.run_h3_asr_endpoint import assert_deployment
from research.h3_personal.director_tts_job import IMAGE
from research.h3_personal.voxcpm_endpoint_protocol import PINS
from research.h3_personal.native_tts_backend import validate_native_config
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()
GUARD=PRIVATE/'.voxcpm-endpoint-unresolved.json'
TERMINAL={'CANCELED','COMPLETED','ERROR','DELETED'}
FILES=('research/hf_h3_native_tts_probe.py',
 'research/h3_personal/voxcpm2_streaming_candidate.py',
 'research/h3_personal/voxcpm_endpoint_service.py',
 'research/h3_personal/voxcpm_endpoint_protocol.py',
 'research/h3_personal/irodori_endpoint_service.py',
 'research/h3_personal/director_irodori.py',
 'research/h3_personal/director_tts_job.py',
 'research/tts_streaming_experiment/voxcpm2/pyproject.toml',
 'research/tts_streaming_experiment/voxcpm2/uv.lock')


def copy_runtime(artifact,spec):
    artifact.mkdir(mode=0o700)
    for name in FILES:
        target=artifact/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    for name in ('research/__init__.py','research/h3_personal/__init__.py'):
        (artifact/name).write_text('')
    shutil.copyfile(ROOT/'research/endpoints/voxcpm_l4/entrypoint.py',artifact/'entrypoint.py')
    persist(artifact/'request.json',spec)
    (artifact/'README.md').write_text('# Private finite native eager VoxCPM2 L4 service\nApache-2.0 model/source; no weight, reference media, reply text or credential upload.\n')
    (artifact/'NOTICE').write_text('VoxCPM2: OpenBMB, Apache-2.0. Source cce58a5b59303c9bd63d12f23afe6a49a4a80c59, model32279effe8c19989596f05d353d1447f51d9e915. Native reference-only BF16/eager, no Irodori watermark.\n')


def upload_verified(api,artifact,repo):
    api.create_repo(repo,repo_type='model',private=True,exist_ok=False)
    commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=artifact,
        commit_message='Immutable finite eager Vox L4 native stream runtime')
    info=api.repo_info(repo,repo_type='model',revision=commit.oid,files_metadata=True)
    local={str(p.relative_to(artifact)):p for p in artifact.rglob('*') if p.is_file()}
    remote={e.rfilename:e for e in info.siblings if e.rfilename!='.gitattributes'}
    if not info.private or info.sha!=commit.oid or set(local)!=set(remote):raise ValueError('runtime inventory drift')
    for name,path in local.items():
        body=path.read_bytes();entry=remote[name]
        digest=hashlib.sha256(body).hexdigest() if entry.lfs else hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if entry.size!=len(body) or digest!=(entry.lfs.sha256 if entry.lfs else entry.blob_id):raise ValueError('runtime bytes drift')
    return commit.oid


def deployment(repo,revision,name,session,boot,expires):
    return dict(name=name,type='authenticated',provider=dict(vendor='aws',region='us-east-1'),
        compute=dict(accelerator='gpu',instanceType='nvidia-l4',instanceSize='x1',scaling=dict(minReplica=1,maxReplica=1)),
        model=dict(repository=repo,revision=revision,framework='custom',task='custom',
            image=dict(custom=dict(url=IMAGE,port=7860,healthRoute='/health')),
            command=['python','/repository/entrypoint.py','/repository'],args=[],
            env=dict(H3_TTS_SESSION=session,H3_TTS_BOOT=boot,H3_TTS_EXPIRES_AT=str(expires))),
        cacheHttpResponses=False,tags=['h3-native-voxcpm-l4','finite-coupled-headless'])


def preflight(api,token):
    if api.whoami().get('name')!='YUGOROU':raise ValueError('owned namespace required')
    if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):raise ValueError('Jobs not idle')
    endpoints=control(token,'/v2/endpoint/YUGOROU')
    if type(endpoints.get('items')) is not list:raise ValueError('Endpoint inventory required')
    if any(e.get('status',{}).get('state') not in ('paused','scaledToZero') for e in endpoints['items']):raise ValueError('existing Endpoint not paused')
    providers=control(token,'/v2/provider/YUGOROU')
    choices=[c for v in providers['vendors'] if v['name']=='aws' for r in v['regions'] if r['name']=='us-east-1'
        for c in r['computes'] if c['instanceType']=='nvidia-l4' and c['instanceSize']=='x1']
    if len(choices)!=1:raise ValueError('exact approved L4 catalog entry needed')
    c=choices[0]
    if (c['status']!='available' or c.get('numAccelerators')!=1 or c.get('gpuMemoryGb',0)<24
            or c['quota']['usedAccelerators']+1>c['quota']['maxAccelerators']
            or Decimal(str(c['pricePerHour']))>Decimal('10')):raise ValueError('L4 hardware/quota/rate scope')
    return _model_preflight(api),{k:c[k] for k in ('instanceType','instanceSize','numAccelerators','gpuMemoryGb','memoryGb','numCpus','status')}


def _model_preflight(api):
    spec=json.loads((PRIVATE/'tts-voxcpm2-streaming-20261028-v3/request.json').read_text())
    if (spec['model']['repo']!='openbmb/VoxCPM2' or spec['model']['revision']!=PINS['model_revision']
            or spec['reference']['sha256']!=PINS['reference_sha256']):raise ValueError('approved source manifest required')
    info=api.model_info(spec['model']['repo'],revision=spec['model']['revision'],files_metadata=True)
    if info.sha!=spec['model']['revision'] or info.card_data.get('license')!='apache-2.0':raise ValueError('model pin/license')
    actual=[dict(path=s.rfilename,bytes=s.size,digest=s.lfs.sha256 if s.lfs else s.blob_id,lfs=bool(s.lfs)) for s in info.siblings]
    if sorted(actual,key=lambda x:x['path'])!=sorted(spec['model']['files'],key=lambda x:x['path']):raise ValueError('pinned model inventory')
    ref=spec['reference'];api.auth_check(ref['repo'],repo_type='dataset',write=False)
    if api.repo_info(ref['repo'],repo_type='dataset',revision=ref['revision']).private is not True:raise ValueError('private original reference')
    for name,digest in (('pyproject.toml','a916c30585052139cf162288639c6b2652ca90ca8dca42cd4ea6e84d2e6980ee'),
            ('uv.lock','a278ea799f8a8862ffb0740672248c29ce47cb199f139be0a81f86b0c0631142')):
        if hashlib.sha256((ROOT/'research/tts_streaming_experiment/voxcpm2'/name).read_bytes()).hexdigest()!=digest:raise ValueError('approved isolated lock drift')
    if spec['uv']['sha256']!='fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946':raise ValueError('uv wheel pin drift')
    spec.pop('texts',None);spec.pop('qwen_profile',None)
    return spec


def collect_h3(api,root):
    """Independent stop/confirmation must run even if log harvesting fails."""
    try:
        return _collect_h3_logs(api,root)
    finally:
        submission=root/'h3/submission.json'
        if submission.exists():
            job_id=json.loads(submission.read_text())['job_id']
            state=None
            for _ in range(10):
                try:state=api.inspect_job(job_id=job_id,namespace='YUGOROU').status.stage
                except Exception:state=None
                if state in TERMINAL:break
                api.cancel_job(job_id=job_id,namespace='YUGOROU');time.sleep(1)
            if state not in TERMINAL:raise RuntimeError('H3 termination unconfirmed')
            active=[dict(id=j.id,status=j.status.stage) for j in api.list_jobs(namespace='YUGOROU') if j.status.stage not in TERMINAL]
            persist(root/'h3-terminal.json',dict(job_id=job_id,status=state,active_jobs=active))
            if active:raise RuntimeError('unexpected active Jobs after cleanup')


def _collect_h3_logs(api,root):
    submission=root/'h3/submission.json'
    if not submission.exists():return
    job_id=json.loads(submission.read_text())['job_id']
    allowed={'h3_window_substages','h3_worker_activity','h3_window_profile_end','h3_producer_timing',
        'h3_producer_failure','h3_http_server_timing','h3_stage_timing','h3_native_tts_receipts','h3_direct_ready'}
    rows=[]
    for index,line in enumerate(api.fetch_job_logs(job_id=job_id,namespace='YUGOROU')):
        if index>10000 or len(line)>65536:raise ValueError('Job event bound')
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if type(row) is dict and row.get('type') in allowed:
            if len(rows)>=1500:raise ValueError('event count bound')
            rows.append(row)
    persist(root/'gpu-events.json',dict(job_id=job_id,records=rows,raw_logs_retained=False))


def validate_evidence(root,config,evidence,h3_result):
    if h3_result.get('complete') is not True or h3_result.get('final') is not True:raise ValueError('headless final sink acceptance required')
    if evidence.get('binding')!=dict(config['pins'],session=config['session'],boot=config['boot'],expires_at=config['expires_at'],epoch=0):raise ValueError('endpoint evidence binding')
    proof=evidence['evidence'];runtime=proof['runtime']
    if (runtime['device_name']!='NVIDIA L4' or runtime['device_count']!=1 or runtime['optimize'] is not False
            or runtime['sample_rate']!=48000 or not runtime['torch'].startswith('2.8.0')):raise ValueError('native runtime proof')
    remote=proof['completions']
    local=json.loads((root/'h3/native-worker-evidence.json').read_text())['receipts']
    if len(remote)!=2 or len(local)!=2:raise ValueError('exactly two native completions')
    actual={1:bytearray(),2:bytearray()};last=None
    for path in sorted((root/'packets').glob('*.json')):
        row=json.loads(path.read_text());pcm=path.with_suffix('.pcm').read_bytes();last=row
        for region in row['regions']:
            if region['kind']=='speech':
                epoch=region['source_epoch'];lo=(region['sample_start']-row['sample_start'])*2
                hi=(region['sample_stop']-row['sample_start'])*2
                if epoch not in actual or not 0<=lo<hi<=len(pcm):raise ValueError('captured speech region')
                actual[epoch].extend(pcm[lo:hi])
    if last is None or last.get('conversation_final') is not True:raise ValueError('final AV tail missing')
    for epoch,(native,receipt) in enumerate(zip(remote,local),1):
        c=native['completion']
        if (native['epoch']!=epoch or receipt['epoch']!=epoch or receipt['pins']!=PINS
                or native['native_float_sha256']!=receipt['native_sha256']
                or native['samples']!=receipt['native_samples']):raise ValueError('native provenance mismatch')
        if (c['reason']!='natural_stop' or c['natural_stop'] is not True or c['stream_exhausted'] is not True
                or type(c['stop_index']) is not int or type(c['effective_max_len']) is not int
                or not 0<=c['stop_index']<c['effective_max_len']-1 or c['decision_monotonic_ns'] is None):raise ValueError('native natural stop not proved')
        if (len(actual[epoch])!=receipt['converted_samples']*2
                or hashlib.sha256(actual[epoch]).hexdigest()!=receipt['converted_sha256']):raise ValueError('sink original speech PCM mismatch')
    return dict(native_natural_stop_verified=True,exact_two_turn_pcm=True,final_tail=True,voice_identity_verified=False,spoken_words_verified=False)


def headless(root,config):
    from research.launch_h3_direct_conversation import run
    from research.h3_personal.headless_av_sink import HeadlessAVSink
    class CaptureSink(HeadlessAVSink):
        def __init__(self,*args,**kwargs):
            if config['expires_at']-time.time()<180:raise TimeoutError('insufficient two-turn native lease')
            super().__init__(*args,paced=True,**kwargs)
            self.capture=root/'packets';self.capture.mkdir(mode=0o700);self.total=0
        def accept(self,path,lease):
            e=lease.segment;video=(self.root/path).read_bytes();pcm=lease.audio_pcm
            self.total+=len(video)+len(pcm)
            if self.total>32*1024*1024:raise ValueError('private capture bound')
            for suffix,body in (('mp4',video),('pcm',pcm)):
                fd=os.open(self.capture/f'{e.index:03d}.{suffix}',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'wb') as f:f.write(body)
            persist(self.capture/f'{e.index:03d}.json',asdict(e))
            return super().accept(path,lease)
    turns=['朝の休憩に温かい飲み物を用意しました。気持ちをゆっくり切り替える話を、自然な会話として五十字以内で返してください。',
        '少し窓を開けて空気を入れ替えようと思います。今日を無理なく始める話を、自然な会話として五十字以内で返してください。']
    return run(root/'h3',turns,tts_mode='native_voxcpm',native_tts=config,
        conversation_anchor=True,prefetch_receive=True,overlap_send=True,
        http_keepalive=True,continuous_av=True,local_sink_factory=CaptureSink)


def run(root,*,rtc_conversation=None):
    if rtc_conversation is not None and not callable(rtc_conversation):raise ValueError('explicit RTC runner required')
    os.umask(0o077);locks=[]
    try:
        for name in ('.voxcpm-endpoint-owner.lock','.irodori-endpoint-owner.lock','.native-streaming-tts-owner.lock'):
            fd=os.open(PRIVATE/name,os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600);locks.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in (GUARD.name,'.irodori-endpoint-unresolved.json','.native-streaming-tts-unresolved.json'):
            p=PRIVATE/name
            if p.exists() or p.is_symlink():raise RuntimeError('unresolved resource fence')
        return _owned(root,rtc_conversation=rtc_conversation)
    finally:
        for fd in reversed(locks):os.close(fd)


def _owned(root,*,rtc_conversation=None):
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
    root=Path(os.path.abspath(root))
    if root.exists() or not root.is_relative_to(PRIVATE) or any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('fresh private root needed')
    root.mkdir(mode=0o700)
    token=cached_token();api=HfApi(token=token)
    spec,hardware=preflight(api,token);persist(root/'hardware.json',hardware)
    expires=int(time.time())+1200;spec['expires_at']=expires
    artifact=root/'deployment';copy_runtime(artifact,spec)
    identity=uuid.uuid4().hex[:12];name='h3-voxcpm-l4-'+identity
    repo='YUGOROU/h3-voxcpm-l4-runtime-'+identity
    revision=upload_verified(api,artifact,repo)
    session=uuid.uuid4().hex;boot=uuid.uuid4().hex;key=secrets.token_urlsafe(40)
    intent=deployment(repo,revision,name,session,boot,expires);persist(root/'endpoint-intent.json',intent)
    intent['model']['secrets']=dict(HF_TOKEN=token,H3_TTS_KEY=key)
    path='/v2/endpoint/YUGOROU/'+name;rejected=False;client=None
    result=dict(schema='h3-native-coupled-headless-v1',endpoint=name,repository=repo,revision=revision,
        expires_at=expires,devices_used=False,discord_used=False,playback_verified=False)
    persist(GUARD,dict(name=name,namespace='YUGOROU',directory=str(root),expires_at=expires))
    try:
        try:response=control(token,'/v2/endpoint/YUGOROU',intent,'POST')
        except ControlError as error:
            if error.status in (400,401,403,422):rejected=True;result['create_rejected_status']=error.status
            raise
        persist(root/'endpoint-submission.json',dict(name=name,revision=revision,expires_at=expires))
        print(json.dumps(dict(type='h3_voxcpm_created',name=name,state=response.get('status',{}).get('state'))),flush=True)
        previous=None
        while time.time()<expires-420:
            current=control(token,path);assert_deployment(current,intent);state=current.get('status',{}).get('state')
            if state!=previous:print(json.dumps(dict(type='h3_voxcpm_state',name=name,state=state)),flush=True);previous=state
            if state=='running':break
            if state in ('failed','paused'):raise RuntimeError('native Endpoint startup failed')
            time.sleep(5)
        else:raise TimeoutError('insufficient native lease for coupled startup')
        config=validate_native_config(dict(schema='h3-native-voxcpm-backend-v1',url=current['status']['url'].rstrip('/'),
            session=session,boot=boot,expires_at=expires,pins=dict(PINS),
            preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0)))
        persist(root/'native-config.json',config)
        client=RemoteVoxCPM(config['url'],token=token,key=key,session=session,boot=boot,expires_at=expires)
        result['ready']=client.read('tts_ready',time.monotonic()+30)
        os.environ.update(H3_NATIVE_TTS_HF_TOKEN=token,H3_NATIVE_TTS_KEY=key)
        if expires-time.time()<600:raise TimeoutError('native lease insufficient before H3 staging')
        if rtc_conversation is not None:
            result.update(execution_mode='rtc-browser',devices_used=True)
        result['h3_result']=(headless if rtc_conversation is None else rtc_conversation)(root,config)
        evidence=client._request('GET','/evidence',None,time.monotonic()+15)
        persist(root/'endpoint-evidence.json',evidence)
        if rtc_conversation is None:
            result['validation']=validate_evidence(root,config,evidence,result['h3_result'])
        else:
            from research.run_h3_rtc_conversation import validate_evidence as validate_rtc
            result['validation']=validate_rtc(root,config,evidence,result['h3_result'])
        result['complete']=True if rtc_conversation is None else result['h3_result']['media_complete']
        if rtc_conversation is not None:
            result['source_complete']=result['h3_result']['source_complete']
            result['closing_tail_unverified']=result['h3_result']['closing_tail_unverified']
    except BaseException as error:
        result['error_type']=type(error).__name__;raise
    finally:
        primary=sys.exc_info()[1];errors=[]
        os.environ.pop('H3_NATIVE_TTS_HF_TOKEN',None);os.environ.pop('H3_NATIVE_TTS_KEY',None)
        try:
            if client is not None:client.close()
        except BaseException as error:errors.append(error)
        # Stop Endpoint before potentially slow diagnostics. Unknown create is owned too.
        try:
            result['endpoint_terminal']='not-created-explicit-rejection' if rejected else pause_owned(token,path)
            if json.loads(GUARD.read_text()).get('name')!=name:raise RuntimeError('Endpoint fence mismatch')
            GUARD.unlink()
        except BaseException as error:
            errors.append(error);result['endpoint_terminal']='pause-unconfirmed'
        try:collect_h3(api,root)
        except BaseException as error:errors.append(error)
        result['cleanup_error_types']=[type(e).__name__ for e in errors]
        def safe(v):
            if isinstance(v,Path):return str(v)
            raise TypeError('nonmetadata result')
        persist(root/'result.json',json.loads(json.dumps(result,default=safe,allow_nan=False)))
        print(json.dumps(dict(type='h3_native_coupled_terminal',name=name,complete=result.get('complete',False),
            endpoint_terminal=result.get('endpoint_terminal'),cleanup_error_types=result['cleanup_error_types'])),flush=True)
        if errors:
            if primary is not None:primary.add_note('cleanup unconfirmed: '+','.join(result['cleanup_error_types']))
            else:raise errors[0]
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--directory',type=Path)
    a=p.parse_args(argv)
    if not a.run:
        print(json.dumps(dict(scope='offline-only',tts='native eager VoxCPM2 L4 x1',h3='original h200x2 Job',max_seconds=1200,devices=False)))
        return 0
    if a.directory is None:p.error('--directory required')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('owner terminated')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite owner deadline')))
    signal.setitimer(signal.ITIMER_REAL,1500)
    try:run(a.directory)
    except BaseException as error:
        print(json.dumps(dict(type='h3_native_coupled_failed',error_type=type(error).__name__)),flush=True);return 1
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0


if __name__=='__main__':raise SystemExit(main())
