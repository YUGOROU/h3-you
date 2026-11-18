"""Offline stage by default; --run finite PRIVATE CPU-Basic synthetic relay test.

No GPU/model/personal input/device/OBS/Discord operations. Two local peers route
synthetic generated AV THROUGH the HF TLS relay, not a GPU producer. The fresh
Space is deleted in finally; keys stay only in RAM/provider secret configuration.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_FILES={'research/live_relay_server.py','research/h3_personal/live_relay.py',
    'research/__init__.py','research/h3_personal/__init__.py','Dockerfile','README.md'}
IMAGE='python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70'


def _write(path,body):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(body)
    path.chmod(0o400)


def prepare(directory):
    directory=Path(directory).expanduser().resolve()
    if directory.is_relative_to(ROOT):raise ValueError('research outputs must remain outside repository')
    directory.mkdir(mode=0o700,exist_ok=False)
    stage=directory/'deployment';stage.mkdir(mode=0o700)
    files={p:(ROOT/p).read_bytes() for p in ('research/live_relay_server.py','research/h3_personal/live_relay.py')}
    files.update({'research/__init__.py':b'','research/h3_personal/__init__.py':b'',
        'Dockerfile':(f'FROM {IMAGE}\nENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n'
            'WORKDIR /app\nCOPY --chown=1000:1000 research /app/research\nUSER 1000:1000\n'
            'CMD ["python", "-m", "research.live_relay_server"]\n').encode(),
        'README.md':b'---\ntitle: Private AV relay transport only\nsdk: docker\napp_port: 7860\n---\nFinite RAM-only transport; no model, media store or capture.\n'})
    inventory=[]
    for name,body in sorted(files.items()):
        path=stage/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);_write(path,body)
        inventory.append(dict(path=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
    _write(stage/'source-manifest.json',json.dumps(inventory,sort_keys=True).encode())
    plan=dict(schema='h3-relay-smoke-v1',space_id='YUGOROU/h3-live-relay-'+uuid.uuid4().hex[:12],
              files=inventory,hardware='cpu-basic',private=True,model_loaded=False,gpu=False,
              source_manifest_sha256=hashlib.sha256((stage/'source-manifest.json').read_bytes()).hexdigest())
    _write(directory/'plan.json',json.dumps(plan,sort_keys=True).encode())
    return plan


def _trial(producer,consumer,directory):
    from fractions import Fraction
    import subprocess
    from research.h3_personal.paired_av import LiveAVBinding,AudioRegion,encode
    from research.h3_personal.live_conversation import Packet
    from research.h3_personal.paired_av_pump import ProducerPump,ReceiverPump
    from research.h3_personal.headless_av_sink import HeadlessAVSink
    from research.h3_personal.conversation_scenario import segment_timing
    from research.run_h3_realtime_first import encode_rgb_segment
    from research.h3_personal.live_relay_commands import Commands
    session=producer._session
    if consumer._session!=session:raise ValueError('carrier room mismatch')
    tx=Commands(consumer,session);rx=Commands(producer,session)
    tx.send('turn',1,{'probe':'synthetic-only'})
    if rx.receive()['payload']!={'probe':'synthetic-only'}:raise ValueError('control lane differs')
    tx.send('stop',1,{})
    if rx.receive()['kind']!='stop':raise ValueError('control framing differs')
    binding=LiveAVBinding(session,'c'*64,1,max_segments=8)
    packets=[];start=0;directory=Path(directory)
    for index,count in enumerate((5,7)):
        path=directory/f'synthetic-segment-{index}.mp4'
        encode_rgb_segment(bytes([40+index*30,20,90])*(252*448*count),path,
            segment_timing(index,start,start+count),width=252,height=448)
        path.chmod(0o400);video=path.read_bytes();pcm=(index+1).to_bytes(2,'little')*(count*2000)
        region=AudioRegion('speech',start*2000,(start+count)*2000,'synthetic',1,start*2000,hashlib.sha256(pcm).hexdigest())
        e=encode(binding,index,start,start+count,video,pcm,(region,),index==1)
        packets.append(Packet(e,video,pcm));start+=count
    client=directory/'receiver';client.mkdir(mode=0o700)
    result=[];errors=[];deadline=time.monotonic()+120;begin=time.monotonic()
    with HeadlessAVSink(client,binding,max_frames=124,timeout=30) as sink:
        receiver=ReceiverPump(consumer,binding,client,sink,deadline=deadline)
        def receive():
            try:result.append(receiver.run())
            except BaseException as exc:errors.append(type(exc).__name__)
        thread=threading.Thread(target=receive,name='h3-relay-smoke-receiver');thread.start()
        sender=ProducerPump(producer,binding,deadline=deadline)
        try:sent=sender.run(iter(packets))
        finally:
            thread.join(125)
            if thread.is_alive():consumer.abort();sink.cancel();thread.join(10)
        if errors or thread.is_alive() or not result:raise RuntimeError('carrier receive failed')
        out=result[0]['output']
        pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-map','0:a','-f','s16le','pipe:1'],timeout=30)
        if pcm!=b''.join(p.pcm for p in packets):raise ValueError('original synthetic PCM differs')
        frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_streams','-show_frames','-of','json',str(out)],timeout=30))
        if len(frames['frames'])!=12 or any(Fraction(f['pts'])*Fraction(frames['streams'][0]['time_base'])!=Fraction(i,24)
                                             for i,f in enumerate(frames['frames'])):
            raise ValueError('carrier PTS differs')
        _write(directory/'synthetic-output.nut',Path(out).read_bytes())
    return dict(completed=True,model_loaded=False,gpu=False,new_live_input=False,device_playback=False,
        discord_reception_verified=False,frames=12,width=252,height=448,original_pcm_exact=True,
        rational_pts_exact=True,sink_acks=sent['acked'],queue_highwater=sent['queue_highwater'],
        input_lane_framing=True,wall_seconds=time.monotonic()-begin)


def run(directory,*,timeout=600):
    if type(timeout) is not int or not 60<=timeout<=900:raise ValueError('finite deployment timeout required')
    if os.environ.get('HF_ENDPOINT','https://huggingface.co')!='https://huggingface.co' or os.environ.get('HF_HUB_STAGING'):
        raise ValueError('canonical HF API endpoint required; environment override refused')
    from huggingface_hub import HfApi,CommitOperationAdd,set_client_factory
    import httpx
    set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(20,connect=5),follow_redirects=False,trust_env=False))
    from research.provision_modal_identity_secret import cached_token
    from research.h3_personal.live_relay_endpoint import attest
    from research.h3_personal.live_relay_client import SocketAdapter
    import secrets
    directory=Path(directory).expanduser().resolve();plan=json.loads((directory/'plan.json').read_bytes())
    stage=directory/'deployment';repo=plan['space_id']
    import re
    if (directory.is_relative_to(ROOT) or plan['schema']!='h3-relay-smoke-v1'
            or not re.fullmatch(r'YUGOROU/h3-live-relay-[a-z0-9]{12}',repo)
            or len(plan['files'])!=len(EXPECTED_FILES) or {r['path'] for r in plan['files']}!=EXPECTED_FILES):
        raise ValueError('owned isolated relay plan required')
    if (directory/'submission-intent.json').exists():raise ValueError('do not replay submitted or unknown-outcome plan')
    for row in plan['files']:
        relative=Path(row['path'])
        if any((stage/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
            raise ValueError('staged symlink refused')
        body=(stage/row['path']).read_bytes()
        if len(body)!=row['bytes'] or hashlib.sha256(body).hexdigest()!=row['sha256']:raise ValueError('staged code drift')
        if row['path'].endswith('.py'):compile(body,row['path'],'exec')
    if hashlib.sha256((stage/'source-manifest.json').read_bytes()).hexdigest()!=plan['source_manifest_sha256']:
        raise ValueError('manifest drift')
    token=cached_token();api=HfApi(endpoint='https://huggingface.co',token=token)
    if api.whoami()['name']!='YUGOROU':raise ValueError('explicit private owner required')
    if api.repo_exists(repo,repo_type='space'):raise ValueError('fresh Space required')
    session=uuid.uuid4().hex;producer_key=secrets.token_hex(32);consumer_key=secrets.token_hex(32)
    not_after=time.time()+1800
    _write(directory/'submission-intent.json',json.dumps(dict(space_id=repo,session=session,
        not_after=not_after,private=True,hardware='cpu-basic',submitted_utc=datetime.now(timezone.utc).isoformat())).encode())
    created=False;clients=[];report=dict(space_id=repo,completed=False,cleanup_confirmed=False)
    started=time.monotonic()
    try:
        # One documented API request, NOT SDK create_repo's unbounded409 loop.
        with httpx.Client(timeout=httpx.Timeout(20,connect=5),follow_redirects=False,trust_env=False) as control:
            response=control.post('https://huggingface.co/api/repos/create',
                headers={'Authorization':'Bearer '+token},json=dict(name=repo.split('/')[1],organization='YUGOROU',
                    type='space',private=True,sdk='docker',hardware='cpu-basic',
                    secrets=[dict(key='H3_RELAY_PRODUCER_KEY',value=producer_key),dict(key='H3_RELAY_CONSUMER_KEY',value=consumer_key)],
                    variables=[dict(key='H3_RELAY_SESSION',value=session),dict(key='H3_RELAY_NOT_AFTER',value=str(not_after)),
                               dict(key='H3_RELAY_RUN',value='1')]))
            if response.status_code not in (200,201):raise RuntimeError('single relay creation request rejected')
        created=True
        names=[r['path'] for r in plan['files']]+['source-manifest.json']
        commit=api.create_commit(repo,repo_type='space',operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(stage/n)) for n in names],
                                 commit_message='Private finite RAM-only synthetic carrier smoke, CPU Basic only')
        report['revision']=commit.oid
        while True:
            if time.monotonic()-started>=timeout:raise TimeoutError('finite relay build wait')
            info=api.space_info(repo)
            if info.private is not True or info.sha!=commit.oid:raise ValueError('private pinned Space changed')
            runtime=api.get_space_runtime(repo)
            for hw in (runtime.hardware,runtime.requested_hardware):
                if hw is not None and str(hw)!='cpu-basic':raise ValueError('unexpected relay hardware')
            if str(runtime.stage)=='RUNNING' and str(runtime.hardware)=='cpu-basic':break
            if str(runtime.stage) in ('BUILD_ERROR','RUNTIME_ERROR','CONFIG_ERROR'):raise RuntimeError('relay service failed to build/start')
            time.sleep(2)
        permit=attest(api,space_id=repo,revision=commit.oid)
        report.update(private_gateway_anonymous_denied=True,hardware=str(runtime.hardware))
        # Space RUNNING is not an application HTTP readiness acknowledgement.
        import http.client,re
        from urllib.parse import urlsplit
        probe_end=time.monotonic()+min(60,timeout-(time.monotonic()-started))
        while True:
            probe=http.client.HTTPSConnection(urlsplit(permit.base_url).hostname,timeout=10)
            try:
                probe.request('GET','/health',headers={'Authorization':'Bearer '+token,
                    'Cache-Control':'no-store','Accept-Encoding':'identity'})
                response=probe.getresponse()
                check=dict(http_status=response.status,
                    boot_header=bool(re.fullmatch('[0-9a-f]{32}',response.getheader('X-H3-Boot',''))),
                    cache_no_store='no-store' in response.getheader('Cache-Control','').lower(),
                    encoding_present=response.getheader('Content-Encoding') is not None,
                    reject_stage=response.getheader('X-H3-Reject-Stage')
                        if response.getheader('X-H3-Reject-Stage') in ('path','headers-duplicate','headers-framing','length','route-auth') else None,
                    reject_flags=int(response.getheader('X-H3-Reject-Flags','0'))
                        if response.getheader('X-H3-Reject-Flags','0').isdigit() and len(response.getheader('X-H3-Reject-Flags','0'))<=2 else None)
                report['gateway_probe']=check
            finally:probe.close()
            if check['http_status']==200 and check['boot_header']:
                if not check['cache_no_store'] or check['encoding_present']:
                    raise RuntimeError('gateway response contract differs')
                break
            if check['boot_header'] and check['http_status']>=400:
                raise RuntimeError('application rejected gateway request')
            if time.monotonic()>=probe_end:raise TimeoutError('finite application HTTP readiness')
            time.sleep(2)
        for role,key in (('producer',producer_key),('consumer',consumer_key)):
            c=SocketAdapter(permit.base_url,session=session,not_after=not_after,role=role,role_key=key,
                            hf_token=token,endpoint_permit=permit)
            clients.append(c);c.open_claim()
        report.update(_trial(clients[0],clients[1],directory))
    except BaseException as exc:
        report['failure_type']=type(exc).__name__
        report['client_diagnostics']=[dict(role=c._role,failure=c.last_failure) for c in clients]
        for c in clients:
            try:c.abort()
            except BaseException:pass
    finally:
        for c in clients:
            try:c.close()
            except BaseException:pass
        if not created:
            # Only recover an unknown create outcome if our unpredictable
            # session variable proves ownership. Never delete another Space.
            try:
                variables=api.get_space_variables(repo)
                created=variables['H3_RELAY_SESSION'].value==session and api.space_info(repo).private is True
            except Exception:pass
        if created:
            try:
                api.delete_repo(repo,repo_type='space')
                report['cleanup_confirmed']=not api.repo_exists(repo,repo_type='space')
            except BaseException:report['cleanup_confirmed']=False
        report['total_seconds']=time.monotonic()-started
        _write(directory/'smoke-report.json',json.dumps(report,sort_keys=True).encode())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.directory.exists():plan=prepare(a.directory)
    else:plan=json.loads((a.directory/'plan.json').read_bytes())
    report=run(a.directory) if a.run else plan
    print(json.dumps(report,sort_keys=True))
    if a.run and (not report.get('completed') or not report.get('cleanup_confirmed')):raise SystemExit(1)


if __name__=='__main__':main()
