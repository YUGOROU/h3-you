"""Offline default; finite sequential L4 candidate Jobs, no devices/media upload."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import time
import urllib.request
import uuid
import zipfile

from research.h3_personal.director_tts_job import IMAGE as VOX_IMAGE,REFERENCE_REPO,REFERENCE_REV,REFERENCE_SHA
from research.h3_personal.voxcpm2_streaming_candidate import MODEL_ID as VOX_MODEL, MODEL_REVISION as VOX_REV
from research.h3_personal.qwen3_streaming_candidate import MODEL_ID as QWEN_MODEL, MODEL_REVISION as QWEN_REV
from research.h3_personal.qwen3_streaming_runtime import IMAGE as QWEN_IMAGE
from research.run_h3_irodori_endpoint import persist, control
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()
LOCK=PRIVATE/'.native-streaming-tts-owner.lock'
FENCE=PRIVATE/'.native-streaming-tts-unresolved.json'
TERMINAL={'COMPLETED','CANCELED','ERROR','DELETED'}
FILES=('research/hf_h3_native_tts_probe.py','research/h3_personal/tts_streaming_measure.py',
       'research/h3_personal/voxcpm2_streaming_candidate.py','research/h3_personal/qwen3_streaming_candidate.py',
       'research/h3_personal/qwen3_streaming_runtime.py','research/h3_personal/voxcpm_supply_diagnostic.py','research/h3_personal/voxcpm_stage_diagnostic.py',
       'research/tts_streaming_experiment/voxcpm2/pyproject.toml','research/tts_streaming_experiment/voxcpm2/uv.lock')
TEXTS=['こんにちは。今日はどんなことを話そうか。気楽に話しかけてね。',
       'なるほど、そういうことだったんだね。それで、そのあとどうなったの？',
       '今回はL4を一台使って、四十八キロヘルツの音声を生成します。次は、Discordで映像と声のタイミングを確認しましょう。',
       'こんにちは。今日はどんなことを話そうか。気楽に話しかけてね。']


def uv_pin():
    with urllib.request.urlopen('https://pypi.org/pypi/uv/0.13.5/json',timeout=20) as r:rows=json.load(r)['urls']
    rows=[r for r in rows if r['filename'].endswith('manylinux_2_17_x86_64.manylinux2014_x86_64.whl')]
    if len(rows)!=1:raise ValueError('unique pinned uv linux wheel required')
    return dict(url=rows[0]['url'],sha256=rows[0]['digests']['sha256'])


def model_manifest(api,repo,revision):
    info=api.model_info(repo,revision=revision,files_metadata=True)
    if info.sha!=revision or info.card_data.get('license')!='apache-2.0':raise ValueError('model license/pin gate')
    return dict(repo=repo,revision=revision,license='apache-2.0',files=[dict(path=s.rfilename,bytes=s.size,
        digest=s.lfs.sha256 if s.lfs else s.blob_id,lfs=bool(s.lfs)) for s in info.siblings])


def bundle(root,spec):
    path=root/'runtime.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            source=ROOT/f
            if source.is_symlink() or not source.is_file():raise ValueError('first party runtime source required')
            z.writestr(f,source.read_bytes())
        for f in ('research/__init__.py','research/h3_personal/__init__.py'):z.writestr(f,'')
        z.writestr('request.json',json.dumps(spec,allow_nan=False))
    if path.stat().st_size>2*1024*1024:raise ValueError('bounded source zip')
    return path


def bootstrap(repo,revision,digest):
    # Hash-pinned source zip; HF CDN redirects never receive the HF bearer token.
    return f'''import hashlib,io,json,os,pathlib,runpy,urllib.request,urllib.parse,zipfile
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args):return None
url='https://huggingface.co/datasets/{repo}/resolve/{revision}/runtime.zip'
opener=urllib.request.build_opener(NoRedirect())
for _ in range(5):
 try:
  host=urllib.parse.urlsplit(url).hostname
  headers={{'Authorization':'Bearer '+os.environ['HF_TOKEN']}} if host=='huggingface.co' else {{}}
  response=opener.open(urllib.request.Request(url,headers=headers),timeout=30)
  body=response.read(2097153);break
 except urllib.error.HTTPError as e:
  if e.code not in (301,302,303,307,308):raise
  url=urllib.parse.urljoin(url,e.headers['Location'])
  u=urllib.parse.urlsplit(url)
  assert u.scheme=='https' and (u.hostname=='huggingface.co' or u.hostname.endswith('.hf.co')) and not u.username
else:raise RuntimeError('source redirect limit')
assert len(body)<=2097152 and hashlib.sha256(body).hexdigest()=={digest!r}
root=pathlib.Path('/tmp/h3-native-tts-probe');root.mkdir(mode=0o700)
with zipfile.ZipFile(io.BytesIO(body)) as z:
 assert len(z.infolist())<32
 for row in z.infolist():
  p=pathlib.PurePosixPath(row.filename)
  assert not p.is_absolute() and '..' not in p.parts and row.file_size<2097152
  target=root/row.filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(row))
os.chdir(root);os.environ['PYTHONPATH']=str(root);os.environ['HF_HOME']='/tmp/h3-probe-hf-cache'
os.execv(__import__('sys').executable,[__import__('sys').executable,str(root/'research/hf_h3_native_tts_probe.py'),str(root)])
'''


def run(candidate,directory,voxcpm_optimize=False,voxcpm_trace_abba=False,voxcpm_supply_diagnostic=False,voxcpm_stage_diagnostic=False,voxcpm_lm_compile=False):
    validate_diagnostic(candidate,voxcpm_optimize,voxcpm_trace_abba,voxcpm_supply_diagnostic,voxcpm_stage_diagnostic,voxcpm_lm_compile)
    import httpx
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    fd=os.open(LOCK,os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if FENCE.exists() or FENCE.is_symlink():raise ValueError('unresolved prior Job; no new launch')
        return owned(candidate,Path(directory),httpx,HfApi,cached_token(),voxcpm_optimize,voxcpm_trace_abba,voxcpm_supply_diagnostic,voxcpm_stage_diagnostic,voxcpm_lm_compile)
    finally:os.close(fd)


def validate_profile(candidate,voxcpm_optimize,voxcpm_trace_abba=False):
    if type(voxcpm_optimize) is not bool or (voxcpm_optimize and candidate!='voxcpm2'):
        raise ValueError('--voxcpm-optimize requires VoxCPM2 and a boolean profile')
    if type(voxcpm_trace_abba) is not bool or (voxcpm_trace_abba and (candidate!='voxcpm2' or not voxcpm_optimize)):
        raise ValueError('--voxcpm-trace-abba requires VoxCPM2 and --voxcpm-optimize')


def validate_diagnostic(candidate,optimize,abba,diagnostic,stage=False,lm=False):
    validate_profile(candidate,optimize,abba)
    if type(diagnostic) is not bool or (diagnostic and (candidate!='voxcpm2' or optimize or abba)):
        raise ValueError('supply diagnostic requires eager VoxCPM2 only')
    if type(stage) is not bool or (stage and (candidate!='voxcpm2' or optimize or abba or diagnostic)):
        raise ValueError('stage diagnostic requires eager VoxCPM2 without other experiments')
    if type(lm) is not bool or (lm and not stage):
        raise ValueError('LM-only compile requires stage diagnostic')


def owned(candidate,root,httpx,HfApi,token,voxcpm_optimize=False,voxcpm_trace_abba=False,voxcpm_supply_diagnostic=False,voxcpm_stage_diagnostic=False,voxcpm_lm_compile=False):
    validate_diagnostic(candidate,voxcpm_optimize,voxcpm_trace_abba,voxcpm_supply_diagnostic,voxcpm_stage_diagnostic,voxcpm_lm_compile)
    os.umask(0o077)
    root=Path(os.path.abspath(root))
    if root.exists() or root.is_relative_to(ROOT) or any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('fresh external private directory required')
    root.mkdir(mode=0o700)
    api=HfApi(token=token)
    if api.whoami().get('name')!='YUGOROU':raise ValueError('owned namespace required')
    if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):raise ValueError('GPU Jobs not idle')
    hardware=[h for h in api.list_jobs_hardware() if h.name=='l4x1']
    if len(hardware)!=1 or hardware[0].accelerator.model!='L4' or hardware[0].accelerator.quantity!='1' or hardware[0].unit_label!='minute' or hardware[0].unit_cost_usd*60>10:
        raise ValueError('single L4 existing rate/hardware gate')
    persist(root/'hardware.json',dict(flavor='l4x1',cpu=hardware[0].cpu,ram=hardware[0].ram,vram=hardware[0].accelerator.vram))
    endpoints=control(token,'/v2/endpoint/YUGOROU')
    if type(endpoints.get('items')) is not list:raise ValueError('endpoint inventory schema')
    if any(e.get('name','').startswith('h3-irodori-l4-') and e.get('status',{}).get('state') not in ('paused','scaledToZero') for e in endpoints['items']):raise ValueError('TTS Endpoint not paused')
    if not api.repo_info(REFERENCE_REPO,repo_type='dataset',revision=REFERENCE_REV).private:raise ValueError('private voice required')
    repo_model,rev=(VOX_MODEL,VOX_REV) if candidate=='voxcpm2' else (QWEN_MODEL,QWEN_REV)
    spec=dict(schema='h3-native-streaming-tts-probe-v1',candidate=candidate,model=model_manifest(api,repo_model,rev),
        reference=dict(repo=REFERENCE_REPO,revision=REFERENCE_REV,sha256=REFERENCE_SHA),texts=TEXTS,uv=uv_pin(),
        expires_at=int(time.time())+1200,voxcpm_optimize=voxcpm_optimize,voxcpm_trace_abba=voxcpm_trace_abba,voxcpm_supply_diagnostic=voxcpm_supply_diagnostic,voxcpm_stage_diagnostic=voxcpm_stage_diagnostic,voxcpm_lm_compile=voxcpm_lm_compile,
        precision='native BF16 model defaults; no quantization',
        qwen_profile='official0.30.0/sourcea8576ccb; CUDA-default V2/eager/single sequence/BF16/x-vector-only; official ramp[4,4,8,16,25]; hash-identical dedicated local-file reference (not oversized base64)',
        privacy='existing private voice download; no generated media upload; no devices',
        placement='USER authorized non-JP/AU TTS-only HF compute; Apache2 no H3 territory transfer')
    persist(root/'request.json',spec)
    z=bundle(root,spec)
    repo='YUGOROU/h3-native-tts-'+candidate+'-'+uuid.uuid4().hex[:10]
    api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
    commit=api.upload_file(path_or_fileobj=str(z),path_in_repo='runtime.zip',repo_id=repo,repo_type='dataset',commit_message='Immutable isolated finite L4 streaming candidate; no audio')
    remote=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True)
    if not remote.private or remote.sha!=commit.oid:raise ValueError('private immutable deployment required')
    entry=next(s for s in remote.siblings if s.rfilename=='runtime.zip')
    raw=z.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if entry.size!=len(raw) or (entry.lfs.sha256!=digest if entry.lfs else entry.blob_id!=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()):raise ValueError('deployment hash')
    key=secrets.token_hex(32)
    payload=dict(dockerImage=VOX_IMAGE if candidate=='voxcpm2' else QWEN_IMAGE,
        command=['python','-c',bootstrap(repo,commit.oid,digest)],arguments=[],
        environment={'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1','HF_DEBUG':'0'},
        secrets={'HF_TOKEN':token,'H3_PROBE_KEY':key},flavor='l4x1',timeoutSeconds=1200,attempts=1,
        expose={'ports':[7860],'portsPublic':[]},labels={'scope':'h3-native-tts','candidate':candidate,'deployrev':commit.oid})
    persist(root/'deployment.json',dict(repo=repo,revision=commit.oid,image=payload['dockerImage'],flavor='l4x1',timeout_seconds=1200))
    if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):raise ValueError('pre-submit not idle')
    persist(FENCE,dict(directory=str(root),state='unknown-no-replay',revision=commit.oid))
    job=None;boot=None
    try:
        with httpx.Client(timeout=30,follow_redirects=False,trust_env=False) as client:
            response=client.post('https://huggingface.co/api/jobs/YUGOROU',headers={'Authorization':'Bearer '+token},json=payload)
            if response.status_code not in (200,201):
                if response.status_code in (400,401,403,422):FENCE.unlink()
                raise RuntimeError('Job creation rejected: HTTP '+str(response.status_code))
            value=response.json();job=value.get('id')
            import re
            if not re.fullmatch('[0-9a-f]{24}',str(job)):raise RuntimeError('unknown create outcome')
            persist(root/'submission.json',dict(job_id=job,repo=repo,revision=commit.oid))
            print(json.dumps(dict(type='tts_probe_submitted',candidate=candidate,job_id=job)),flush=True)
            base='https://'+job+'--7860.hf.jobs'
            headers={'Authorization':'Bearer '+token,'X-H3-Probe-Key':key,'Accept-Encoding':'identity'}
            end=time.monotonic()+1100
            while time.monotonic()<end:
                state=api.inspect_job(job_id=job,namespace='YUGOROU').status.stage
                if state in TERMINAL:raise RuntimeError('Job terminated before result: '+state)
                try:r=client.get(base+'/status',headers=headers,timeout=10)
                except httpx.TransportError:time.sleep(3);continue
                if r.status_code in (404,502,503,504):time.sleep(3);continue
                if r.status_code!=200:raise RuntimeError('probe access rejected')
                row=r.json()
                if boot is None:boot=row['boot']
                if row.get('boot')!=boot:raise RuntimeError('probe restarted')
                if row['state']=='failed':
                    persist(root/'worker-failure.json',row)
                    client.post(base+'/ack',headers=headers,content=b'')
                    raise RuntimeError('candidate worker failed: '+row.get('error_type','unknown'))
                if row['state']=='complete':
                    persist(root/'artifact-manifest.json',row)
                    total=0
                    for a in row['artifacts']:
                        name=a['path']
                        p=Path(name.removeprefix('/artifacts/'))
                        if not name.startswith('/artifacts/') or p.is_absolute() or '..' in p.parts or not 0<a['bytes']<=8*1024*1024:raise ValueError('artifact path/bounds')
                        total+=a['bytes']
                        if total>32*1024*1024:raise ValueError('artifact aggregate bound')
                        with client.stream('GET',base+name,headers=headers,timeout=30) as response:
                            if response.status_code!=200 or response.headers.get('X-H3-Probe-Boot')!=boot:raise ValueError('artifact response/boot')
                            body=bytearray()
                            for chunk in response.iter_bytes(65536):
                                body.extend(chunk)
                                if len(body)>a['bytes']:raise ValueError('artifact size overflow')
                        if len(body)!=a['bytes'] or hashlib.sha256(body).hexdigest()!=a['sha256']:raise ValueError('artifact digest')
                        target=root/'outputs'/p;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                        target.write_bytes(body)
                    response=client.post(base+'/ack',headers=headers,content=b'')
                    if response.status_code!=200:raise RuntimeError('result ACK unknown; no replay')
                    print(json.dumps(dict(type='tts_probe_collected',candidate=candidate,job_id=job)),flush=True)
                    break
                time.sleep(3)
            else:raise TimeoutError('finite probe result wait')
    finally:
        if job is not None:
            info=api.inspect_job(job_id=job,namespace='YUGOROU')
            if info.status.stage not in TERMINAL:api.cancel_job(job_id=job,namespace='YUGOROU')
            for _ in range(15):
                info=api.inspect_job(job_id=job,namespace='YUGOROU')
                if info.status.stage in TERMINAL:break
                time.sleep(2)
            persist(root/'provider-terminal.json',dict(job_id=job,state=info.status.stage))
            if info.status.stage not in TERMINAL:raise RuntimeError('Job termination unconfirmed')
            FENCE.unlink(missing_ok=True)
            print(json.dumps(dict(type='tts_probe_terminal',candidate=candidate,job_id=job,state=info.status.stage)),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--candidate',choices=('voxcpm2','qwen3'));p.add_argument('--directory',type=Path);p.add_argument('--voxcpm-optimize',action='store_true',help='explicit VoxCPM2 native optimize experiment; default false');p.add_argument('--voxcpm-trace-abba',action='store_true');p.add_argument('--voxcpm-supply-diagnostic',action='store_true');p.add_argument('--voxcpm-stage-diagnostic',action='store_true');p.add_argument('--voxcpm-lm-compile',action='store_true');a=p.parse_args(argv)
    try:validate_diagnostic(a.candidate,a.voxcpm_optimize,a.voxcpm_trace_abba,a.voxcpm_supply_diagnostic,a.voxcpm_stage_diagnostic,a.voxcpm_lm_compile)
    except ValueError as error:p.error(str(error))
    if a.voxcpm_optimize and a.candidate!='voxcpm2':p.error('--voxcpm-optimize requires --candidate voxcpm2')
    if a.voxcpm_trace_abba and (a.candidate!='voxcpm2' or not a.voxcpm_optimize):p.error('--voxcpm-trace-abba requires VoxCPM2 and --voxcpm-optimize')
    if not a.run:print(json.dumps(dict(offline=True,candidates=[VOX_MODEL,QWEN_MODEL],hardware='L4x1,sequential',max_seconds=1200,voxcpm_optimize=a.voxcpm_optimize,voxcpm_trace_abba=a.voxcpm_trace_abba,voxcpm_supply_diagnostic=a.voxcpm_supply_diagnostic,voxcpm_stage_diagnostic=a.voxcpm_stage_diagnostic,voxcpm_lm_compile=a.voxcpm_lm_compile)));return 0
    if not a.candidate or not a.directory:p.error('--run requires --candidate --directory')
    def stop(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    run(a.candidate,a.directory,a.voxcpm_optimize,a.voxcpm_trace_abba,a.voxcpm_supply_diagnostic,a.voxcpm_stage_diagnostic,a.voxcpm_lm_compile);return 0


if __name__=='__main__':raise SystemExit(main())
