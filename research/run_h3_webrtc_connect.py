"""Finite real HF↔Mac RTC connection setup; CPU only, no personal media/devices.

The same authenticated offer/answer and RTC peer serve the live-media path.
Host-only by default; explicit approved Cloudflare STUN or HF TURN/TLS opt-in.
No signaling SDP persistence, no HTTP media fallback.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import uuid
import zipfile

from research.h3_personal.director_tts_job import IMAGE
from research.h3_personal.webrtc_ice import HOST_ONLY, CLOUDFLARE_STUN, HF_CLOUDFLARE_TURN_TLS, ice_urls
from research.h3_personal.webrtc_turn import fetch_turn_credentials
from research.run_h3_irodori_endpoint import persist
from research.run_h3_native_tts_probe import bootstrap as zip_bootstrap
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()
FENCE=PRIVATE/'.webrtc-network-unresolved.json'
TERMINAL={'COMPLETED','CANCELED','ERROR','DELETED'}
FILES=('research/hf_h3_webrtc_worker.py','research/h3_personal/webrtc_live_peer.py',
 'research/h3_personal/webrtc_signal_server.py','research/h3_personal/webrtc_ice.py',
 'research/h3_personal/webrtc_turn.py','research/webrtc_experiment/pyproject.toml',
 'research/webrtc_experiment/uv.lock')


def stage(root,expires,session,ice_profile=HOST_ONLY):
    ice_urls(ice_profile)
    previous=json.loads((PRIVATE/'tts-voxcpm2-streaming-20261028-v3/request.json').read_text())
    if previous['uv']['sha256']!='fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946':raise ValueError('uv pin')
    spec=dict(schema='h3-rtc-cpu-network-v1',expires_at=expires,session=session,uv=previous['uv'],ice_profile=ice_profile)
    archive=root/'runtime.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p=ROOT/name
            if p.is_symlink():raise ValueError('runtime symlink')
            z.writestr(name,p.read_bytes())
        for name in ('research/__init__.py','research/h3_personal/__init__.py'):z.writestr(name,'')
        z.writestr('request.json',json.dumps(spec,allow_nan=False))
    if archive.stat().st_size>2*1024*1024:raise ValueError('runtime archive bound')
    return archive


def attest(api,client,job,image,command):
    info=api.inspect_job(job_id=job,namespace='YUGOROU')
    if (info.id!=job or info.owner.name!='YUGOROU' or info.docker_image!=image
            or info.command!=command or str(info.flavor)!='cpu-basic' or info.status.stage!='RUNNING'):
        raise ValueError('owned immutable CPU Job required')
    base='https://'+job+'--7860.hf.jobs'
    anonymous=client.get(base+'/health',headers={'Accept-Encoding':'identity'},timeout=10)
    if anonymous.status_code in (502,503,504):raise RuntimeError('gateway not ready')
    if anonymous.status_code not in (401,403,404):raise ValueError('anonymous access not denied')
    return base


def cleanup(api,job,root):
    state=None
    for _ in range(8):
        try:state=api.inspect_job(job_id=job,namespace='YUGOROU').status.stage
        except Exception:state=None
        if state in TERMINAL:break
        try:api.cancel_job(job_id=job,namespace='YUGOROU')
        except Exception:pass
        time.sleep(1)
    if state not in TERMINAL:raise RuntimeError('CPU Job termination unconfirmed')
    active=[dict(id=j.id,status=j.status.stage) for j in api.list_jobs(namespace='YUGOROU') if j.status.stage not in TERMINAL]
    persist(root/'provider-terminal.json',dict(job_id=job,status=state,active_jobs=active))
    if json.loads(FENCE.read_text()).get('job_id')!=job:raise RuntimeError('network fence mismatch')
    FENCE.unlink()


def run(directory,ice_profile=HOST_ONLY,*,browser=False,on_browser=None):
    ice_urls(ice_profile)
    fd=os.open(PRIVATE/'.webrtc-network-owner.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if FENCE.exists() or FENCE.is_symlink():raise ValueError('unresolved CPU RTC create; reconcile first')
        return owned(directory,ice_profile,browser=browser,on_browser=on_browser)
    finally:os.close(fd)


def preexisting_cpu_jobs(api):
    active=[j for j in api.list_jobs(namespace='YUGOROU') if j.status.stage not in TERMINAL]
    if any(str(j.flavor)!='cpu-basic' for j in active):raise ValueError('existing GPU/non-basic Job active')
    return [dict(id=j.id,status=j.status.stage,flavor=str(j.flavor)) for j in active]


def browser_probe(config,token,key,turn_secret,*,on_browser=None,wait_seconds=90):
    """Attested CPU Job only. Operator opens URL; never opens apps itself.

    Events establish renderer progress/explicit unmute request, not audibility.
    URL is an ephemeral loopback capability; deliver privately, never persist it.
    """
    from research.h3_personal.webrtc_browser_proxy import BrowserProxy
    deadline=min(time.monotonic()+wait_seconds,time.monotonic()+config['expires_at']-time.time())
    turn=None if turn_secret is None else json.loads(turn_secret)
    with BrowserProxy(config,hf_token=token,app_key=key,owner_attested=True,turn_config=turn) as proxy:
        if on_browser is None:
            print(json.dumps(dict(type='h3_rtc_private_operator_url',url=proxy.url)),flush=True)
        else:on_browser(proxy.url)
        while time.monotonic()<deadline:
            events={e['event'] for e in proxy.receipts}
            if 'error' in events:raise RuntimeError('browser receiver failed; no replay')
            if 'stop' in events:
                if not {'connected','frame-presented'}<=events:
                    raise RuntimeError('browser stopped before receiver progress')
                return dict(stage='browser-stopped',connected=True,video_frame_presented=True,
                    audio_enable_requested='unmuted' in events,audibility_verified=False,
                    exact_media_verified=False,perceptual_acceptance=False)
            time.sleep(.05)
        raise TimeoutError('finite browser preflight expired')


def owned(directory,ice_profile=HOST_ONLY,*,browser=False,on_browser=None):
    urls=ice_urls(ice_profile)
    import httpx
    from huggingface_hub import HfApi,set_client_factory
    from research.provision_modal_identity_secret import cached_token
    os.umask(0o077);root=Path(os.path.abspath(directory))
    if root.exists() or not root.is_relative_to(PRIVATE) or any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('fresh private root')
    rtc_python=ROOT/'research/webrtc_experiment/.venv/bin/python'
    if not rtc_python.exists():raise RuntimeError('approved isolated RTC env required')
    root.mkdir(mode=0o700)
    set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(20,connect=5),trust_env=False))
    token=cached_token();api=HfApi(token=token)
    if api.whoami().get('name')!='YUGOROU':raise ValueError('owned namespace')
    # Our lock/fence serializes this CPU-only probe; don't cancel unrelated CPU.
    persist(root/'preexisting-jobs.json',dict(jobs=preexisting_cpu_jobs(api),policy='leave-unrelated-CPU-Jobs-untouched'))
    hw=[h for h in api.list_jobs_hardware() if h.name=='cpu-basic']
    if len(hw)!=1 or hw[0].accelerator is not None:raise ValueError('CPU only hardware required')
    session=uuid.uuid4().hex;expires=int(time.time())+600
    archive=stage(root,expires,session,ice_profile);raw=archive.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    repo='YUGOROU/h3-rtc-connect-'+uuid.uuid4().hex[:12]
    persist(root/'staging-intent.json',dict(repo=repo,source_sha256=digest,bytes=len(raw),job_creation_attempted=False))
    try:
        api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
    except httpx.TransportError:
        # Creation can commit before its response times out. Adopt only this
        # exact fresh random empty private repo; never repeat a create POST.
        info=api.repo_info(repo,repo_type='dataset')
        if (info.id!=repo or info.private is not True
                or any(s.rfilename!='.gitattributes' for s in info.siblings)):
            raise RuntimeError('uncertain source repository creation') from None
    persist(root/'repository-created.json',dict(repo=repo,private=True))
    # Small source archive: ordinary Git blob avoids needless LFS/Xet upload
    # negotiation. Content remains a hash-verified ZIP, not a media payload.
    try:
        commit=api.upload_file(path_or_fileobj=str(archive),path_in_repo='runtime.source',repo_id=repo,repo_type='dataset',commit_message='Immutable CPU-only RTC signaling; synthetic media, no private source')
    except httpx.TransportError:
        from types import SimpleNamespace
        info=api.repo_info(repo,repo_type='dataset',files_metadata=True)
        entries={s.rfilename:s for s in info.siblings}
        e=entries.get('runtime.source')
        if (info.private is not True or set(entries)-{'.gitattributes'}!={'runtime.source'} or e.size!=len(raw)
                or (e.lfs.sha256!=digest if e.lfs else e.blob_id!=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())):
            raise RuntimeError('uncertain source commit; no replay') from None
        commit=SimpleNamespace(oid=info.sha)
    remote=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True)
    entry=next(s for s in remote.siblings if s.rfilename=='runtime.source')
    if (not remote.private or remote.sha!=commit.oid or entry.size!=len(raw)
            or (entry.lfs.sha256!=digest if entry.lfs else entry.blob_id!=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())):raise ValueError('immutable runtime mismatch')
    script=zip_bootstrap(repo,commit.oid,digest).replace('runtime.zip','runtime.source').replace('research/hf_h3_native_tts_probe.py','research/hf_h3_webrtc_worker.py').replace('/tmp/h3-native-tts-probe','/tmp/h3-rtc-peer').replace('/tmp/h3-probe-hf-cache','/tmp/h3-rtc-hf-cache')
    command=['python','-c',script];key=secrets.token_hex(32)
    # Fetch only after source publication. Credentials never enter staged JSON,
    # runtime ZIP, argv or receipts; only Job secrets/local child environment.
    turn_secret=(json.dumps(fetch_turn_credentials(token,ttl=600),separators=(',',':'))
                 if ice_profile==HF_CLOUDFLARE_TURN_TLS else None)
    job_secrets={'HF_TOKEN':token,'H3_RTC_KEY':key}
    if turn_secret is not None:job_secrets['H3_RTC_ICE_CREDENTIALS']=turn_secret
    payload=dict(dockerImage=IMAGE,command=command,arguments=[],environment={'HF_DEBUG':'0','PYTHONDONTWRITEBYTECODE':'1'},
        secrets=job_secrets,flavor='cpu-basic',timeoutSeconds=600,attempts=1,
        expose={'ports':[7860],'portsPublic':[]},labels={'scope':'h3-rtc-connect','deployrev':commit.oid})
    persist(root/'deployment.json',dict(repo=repo,revision=commit.oid,image=IMAGE,flavor='cpu-basic',seconds=600,expires_at=expires,ice_profile=ice_profile))
    job=None;result=dict(schema='h3-rtc-cpu-owner-v1',gpu_used=False,private_media_used=False,devices_used=False,
        ice_profile=ice_profile,external_ice_servers=urls)
    persist(FENCE,dict(directory=str(root),revision=commit.oid,state='unknown-no-replay'))
    try:
        with httpx.Client(timeout=httpx.Timeout(30,connect=5),follow_redirects=False,trust_env=False) as client:
            response=client.post('https://huggingface.co/api/jobs/YUGOROU',headers={'Authorization':'Bearer '+token},json=payload)
            if response.status_code not in (200,201):
                if response.status_code in (400,401,403,422):FENCE.unlink()
                raise RuntimeError('CPU Job rejected')
            job=response.json().get('id')
            if not re.fullmatch('[a-f0-9]{24}',str(job)):job=None;raise RuntimeError('unknown CPU Job response')
            # Own ID before all other receipt I/O. Rewrite durable fence with the known ID.
            temp=FENCE.with_name(FENCE.name+'.tmp')
            persist(temp,dict(directory=str(root),revision=commit.oid,job_id=job));os.replace(temp,FENCE)
            persist(root/'submission.json',dict(job_id=job,repo=repo,revision=commit.oid))
            print(json.dumps(dict(type='h3_rtc_cpu_submitted',job_id=job)),flush=True)
            base=None;row=None
            while time.time()<expires-90:
                state=api.inspect_job(job_id=job,namespace='YUGOROU').status.stage
                if state in TERMINAL:raise RuntimeError('CPU peer startup failed')
                if state!='RUNNING':time.sleep(2);continue
                if base is None:
                    try:
                        base=attest(api,client,job,IMAGE,command)
                    except httpx.TransportError:
                        time.sleep(2);continue
                    except RuntimeError:
                        time.sleep(2);continue
                try:r=client.get(base+'/status',headers={'Authorization':'Bearer '+token,'X-H3-RTC-Key':key},timeout=10)
                except httpx.TransportError:time.sleep(2);continue
                if r.status_code in (404,502,503,504):time.sleep(2);continue
                if r.status_code!=200:raise RuntimeError('CPU signaling rejected')
                row=r.json()
                if (row.get('schema')!='h3-rtc-signal-v1' or row.get('session')!=session or row.get('ice_profile')!=ice_profile
                        or not re.fullmatch('[a-f0-9]{32}',str(row.get('boot')))
                        or not time.time()<row.get('expires_at',0)<=expires):raise ValueError('peer binding')
                break
            else:raise TimeoutError('CPU signaling startup deadline')
            config=dict(job_id=job,session=session,boot=row['boot'],expires_at=row['expires_at'],ice_profile=ice_profile);persist(root/'client.json',config)
            env=dict(os.environ,H3_RTC_HF_TOKEN=token,H3_RTC_KEY=key)
            env.pop('H3_RTC_ICE_CREDENTIALS',None)
            if turn_secret is not None:env['H3_RTC_ICE_CREDENTIALS']=turn_secret
            if browser:
                result['devices_used']=True  # Attended browser output, never capture/routing.
                result['browser_playback_requested']=True
                result['connection']=browser_probe(config,token,key,turn_secret,on_browser=on_browser)
                return result
            child=subprocess.run([str(rtc_python),'-m','research.h3_personal.webrtc_connect_client',str(root/'client.json')],
                cwd=ROOT,env=env,timeout=70,capture_output=True,text=True)
            result['client_exit_code']=child.returncode
            result['connection']=json.loads((root/'connection.json').read_text())
            print(json.dumps(dict(type='h3_rtc_connection_result',**result['connection'])),flush=True)
            proof=result['connection']
            if (child.returncode!=0 or proof.get('connected') is not True
                    or proof.get('media_received') is not True or proof.get('stage')!='complete'):
                raise RuntimeError('direct RTC connection not established')
    except BaseException as error:
        result['error_type']=type(error).__name__;raise
    finally:
        primary=sys.exc_info()[1];cleanup_error=None
        if job is not None:
            try:cleanup(api,job,root)
            except BaseException as error:cleanup_error=error;result['cleanup_error_type']=type(error).__name__
            # No log-stream diagnostics in the resource owner: cleanup cannot
            # become blocked waiting for an SSE event. Inspect terminal logs
            # separately only when diagnosis is necessary; never persist SDP.
        persist(root/'result.json',result)
        print(json.dumps(dict(type='h3_rtc_cpu_terminal',job_id=job,cleanup_confirmed=job is not None and cleanup_error is None)),flush=True)
        if cleanup_error is not None:
            if primary is not None:primary.add_note('CPU Job cleanup unconfirmed')
            else:raise cleanup_error
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--directory',type=Path)
    ice=p.add_mutually_exclusive_group()
    ice.add_argument('--cloudflare-stun',action='store_true',help='Explicit USER-approved public STUN only')
    ice.add_argument('--hf-turn',action='store_true',help='Explicit USER-approved HF Cloudflare TURN over TLS443')
    p.add_argument('--browser',action='store_true',help='Attended browser preflight; explicit playback approval required. No automatic app launch.')
    p.add_argument('--open-browser',action='store_true',help='Explicit approved Mac browser launch; requires --browser')
    a=p.parse_args(argv)
    if a.open_browser and (not a.browser or sys.platform!='darwin'):p.error('--open-browser requires --browser on Mac')
    profile=HF_CLOUDFLARE_TURN_TLS if a.hf_turn else (CLOUDFLARE_STUN if a.cloudflare_stun else HOST_ONLY)
    if not a.run:
        print(json.dumps(dict(scope='offline',gpu=False,ice_profile=profile,external_ice_servers=ice_urls(profile),media='synthetic H264/Opus')));return 0
    if a.directory is None:p.error('--directory required')
    def interrupted(*_):raise InterruptedError('finite RTC owner interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(660)
    def open_browser(url):
        subprocess.run(['/usr/bin/open',url],timeout=10,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:run(a.directory,ice_profile=profile,browser=a.browser,on_browser=open_browser if a.open_browser else None)
    except BaseException as error:
        import traceback
        frames=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno)
                for f in (traceback.extract_tb(error.__traceback__)[:5]+traceback.extract_tb(error.__traceback__)[-5:])]
        print(json.dumps(dict(type='h3_rtc_cpu_owner_failed',error_type=type(error).__name__,frames=frames)),flush=True);return 1
    finally:signal.alarm(0)
    return 0


if __name__=='__main__':raise SystemExit(main())
