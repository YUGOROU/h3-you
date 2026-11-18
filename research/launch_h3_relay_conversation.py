"""One finite private typed Pi/H3 relay session. Offline by default; no ASR.

--allow-obs approves existing OBS playback only, not launch/routing/Discord join.
TTS currently waits for a complete watermarked WAV; this is not streaming TTS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import time
from types import FunctionType
import uuid

from research.launch_h3_conversation_scenario import ROOT, PRIVATE, _bound, closure, rewrite_allowlist

TERMINAL = {'COMPLETED', 'CANCELED', 'ERROR', 'DELETED'}


def canonical():
    if os.environ.get('HF_ENDPOINT', 'https://huggingface.co') != 'https://huggingface.co' or os.environ.get('HF_HUB_STAGING'):
        raise ValueError('canonical HF endpoint required')


def sdk_redirect_guard(response):
    if response.is_redirect:
        target=response.url.join(response.headers.get('location',''))
        if (response.url.host!='huggingface.co' or target.host!='huggingface.co'
                or target.scheme!='https' or target.port not in (None,443)):
            raise ValueError('unexpected SDK redirect origin')


def prepare(directory, row, api):
    from research import prepare_h3_fast_ref2va_gpu as base
    from research.launch_h3_live_voice import EXTRA, inspect
    from research.hf_h3_relay_conversation import validate_spec
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('private staging outside repository required')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stage = directory / 'deployment'
    if stage.exists():
        raise ValueError('fresh deployment required')
    dependencies = dict(base.prepare.__globals__, budget_bound=_bound)
    preparer = FunctionType(base.prepare.__code__, dependencies, base.prepare.__name__, base.prepare.__defaults__)
    preparer(stage, PRIVATE/'gpu-hf-realtime-first-v2/deployment', PRIVATE/'compute-budget.json', api,
             publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-relay-'+uuid.uuid4().hex[:12],
             timeout_seconds=1200, attention_mode='fa4-packed', layout='portrait',
             runtime_policy='resident-visual-cached', locked_speech=None)
    source = (PRIVATE/'gpu-hf-irodori-diagnostic-v1/request.json').read_bytes()
    if hashlib.sha256(source).hexdigest() != 'b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':
        raise ValueError('TTS manifest drift')
    manifest = json.loads(source)
    (stage/'tts-manifest.json').write_text(json.dumps({k:manifest[k] for k in ('source_revision','uv_lock_sha256','uv_version','models','license')}))
    # Seed every current Python module: includes dynamic imports and both worker
    # and streaming closures, not a historical deployment's stale Python files.
    for path in sorted((ROOT/'research').rglob('*.py')):
        target = stage/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    closure(stage, [p[:-3].replace('/', '.') for p in EXTRA])
    spec = json.loads((stage/'spec.json').read_text())
    spec['live_voice_session'] = dict(id=row['session'], max_turns=2, fresh_input_required=True)
    spec['live_relay_session'] = row
    (stage/'spec.json').write_text(json.dumps(spec))
    rewrite_allowlist(stage)
    validate_spec(spec)
    inspect(stage)
    return stage


def submit(api, stage, token, producer_key):
    from huggingface_hub import CommitOperationAdd
    from research.launch_h3_live_voice import inspect, pre_submit
    spec, inventory = inspect(stage)
    guard = stage.parent/'submission_guard.json'
    if guard.exists():
        raise ValueError('unknown/submitted outcome: NEVER replay')
    if api.whoami().get('name') != 'YUGOROU' or api.whoami().get('isPro') is not True:
        raise ValueError('Jobs-enabled owner required')
    if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):
        raise ValueError('HF not idle')
    pre_submit(api, spec, stage)
    repo = spec['publication_repo']
    # Non-idempotent creation is one request; an empty/non-JSON success body
    # need not be parsed. Subsequent authenticated metadata proves visibility.
    import httpx
    with httpx.Client(timeout=20,follow_redirects=False,trust_env=False) as control:
        response=control.post('https://huggingface.co/api/repos/create',
            headers={'Authorization':'Bearer '+token},json=dict(name=repo.split('/')[1],organization='YUGOROU',type='dataset',private=True))
        if response.status_code not in (200,201):raise RuntimeError('dataset creation status '+str(response.status_code))
    if api.repo_info(repo,repo_type='dataset').private is not True:raise ValueError('private deployment required')
    names = [r['path'] for r in inventory['files']] + ['upload-allowlist.json']
    commit = api.create_commit(repo, repo_type='dataset', operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(stage/n)) for n in names], commit_message='Immutable finite live relay runtime')
    remote = api.repo_info(repo, repo_type='dataset', revision=commit.oid, files_metadata=True)
    entries = {s.rfilename:s for s in remote.siblings}
    if not remote.private or set(entries)-{'.gitattributes'} != set(names):
        raise ValueError('remote inventory differs')
    for n in names:
        body = (stage/n).read_bytes(); entry = entries[n]
        digest = hashlib.sha256(body).hexdigest() if entry.lfs else hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if entry.size != len(body) or digest != (entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError('remote digest differs')
    bootstrap = 'research/bootstrap_h3_fast_ref2va.py'
    script = ("import hashlib,os,runpy;from huggingface_hub import hf_hub_download;"
              f"p=hf_hub_download({repo!r},{bootstrap!r},repo_type='dataset',revision={commit.oid!r},token=os.environ['HF_TOKEN']);"
              f"assert hashlib.sha256(open(p,'rb').read()).hexdigest()=={hashlib.sha256((stage/bootstrap).read_bytes()).hexdigest()!r};runpy.run_path(p,run_name='__main__')")
    # Intent is durable BEFORE the non-idempotent request, including exceptions.
    with guard.open('x') as f:
        json.dump(dict(state='unknown_no_replay', revision=commit.oid), f)
        f.flush(); os.fsync(f.fileno())
    job=api.run_job(image=spec['image'], command=['/opt/venv/bin/python','-c',script,repo,commit.oid,hashlib.sha256((stage/'upload-allowlist.json').read_bytes()).hexdigest()],
                       flavor='h200x2', namespace='YUGOROU', timeout=1200,
                       secrets={'HF_TOKEN':token,'H3_RELAY_PRODUCER_KEY':producer_key},
                       env={'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1'})
    receipt=dict(job_id=job.id,repo=repo,revision=commit.oid,timeout_seconds=1200,flavor='h200x2')
    (stage.parent/'submission.json').write_text(json.dumps(receipt))
    print(json.dumps(dict(type='h3_job_submitted',**receipt)),flush=True)
    return job


def await_ready(api, job, session, deadline):
    """Finite foreground polling; retain only the validated scalar READY row."""
    while time.monotonic() < deadline:
        for log in api.fetch_job_logs(job_id=job.id, follow=False):
            text = log if isinstance(log, str) else getattr(log, 'data', '')
            if not isinstance(text, str) or len(text) > 4096:
                continue
            try:
                row = json.loads(text)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('type') == 'h3_relay_ready':
                keys = {'type','status','session','profile_sha256','epoch','max_segments','max_total_frames','expires_at','origin','asr_used','tts_streaming'}
                import re
                if (set(row) != keys or row['session'] != session or row['status'] != 'ready'
                        or not re.fullmatch('[a-f0-9]{64}', str(row['profile_sha256']))
                        or row['epoch'] != 1 or row['max_segments'] != 16 or row['max_total_frames'] != 1654
                        or row['origin'] != 'fresh_typed_text' or row['asr_used'] is not False
                        or row['tts_streaming'] is not False or not time.time() < row['expires_at'] <= time.time()+1800):
                    raise ValueError('invalid READY')
                return row
        if api.inspect_job(job_id=job.id).status.stage in TERMINAL:
            raise RuntimeError('Job ended before READY')
        time.sleep(2)
    raise TimeoutError('finite READY wait')


def run(directory, turns, *, allow_obs=False, obs_port=None):
    canonical()
    if obs_port is not None and not allow_obs:
        raise PermissionError('OBS approval required')
    if not 1 <= len(turns) <= 2 or any(not isinstance(t,str) or not t.strip() or len(t.encode()) > 2000 for t in turns):
        raise ValueError('one or two fresh bounded typed turns required')
    from huggingface_hub import HfApi, CommitOperationAdd, set_client_factory
    import httpx
    from research import run_h3_relay_smoke as smoke
    from research.provision_modal_identity_secret import cached_token
    from research.run_h3_relay_conversation import PiDirector, run_conversation
    from research.h3_personal.live_relay_endpoint import attest
    from research.h3_personal.live_relay_client import SocketAdapter
    from research.h3_personal.paired_av import LiveAVBinding
    if not all(shutil.which(p) for p in ('node','ffmpeg','ffprobe')):
        raise RuntimeError('existing node/ffmpeg/ffprobe required; no installs')
    if allow_obs:
        from research.h3_personal.obs_av_sink import output_arguments
        output_arguments(obs_port)
    set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(20,connect=5), follow_redirects=True,
        event_hooks={'response':[sdk_redirect_guard]},trust_env=False))
    token = cached_token(); api = HfApi(endpoint='https://huggingface.co',token=token)
    directory = Path(directory).resolve()
    plan = smoke.prepare(directory); repo = plan['space_id']; stage = directory/'deployment'
    # Smoke staging is the relay source only; GPU deployment uses a separate root.
    session = uuid.uuid4().hex; pk = secrets.token_hex(32); ck = secrets.token_hex(32)
    expires = time.time()+1800; job = None; consumer = None; director = None
    try:
        director = PiDirector(token, deadline=time.monotonic()+1750, max_turns=2)
        with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as control:
            response = control.post('https://huggingface.co/api/repos/create', headers={'Authorization':'Bearer '+token}, json=dict(name=repo.split('/')[1], organization='YUGOROU', type='space', private=True, sdk='docker', hardware='cpu-basic', secrets=[dict(key='H3_RELAY_PRODUCER_KEY',value=pk),dict(key='H3_RELAY_CONSUMER_KEY',value=ck)], variables=[dict(key='H3_RELAY_SESSION',value=session),dict(key='H3_RELAY_NOT_AFTER',value=str(expires)),dict(key='H3_RELAY_RUN',value='1')]))
            if response.status_code not in (200,201):
                raise RuntimeError('single Space create failed; no retry')
        names = [r['path'] for r in plan['files']]+['source-manifest.json']
        commit = api.create_commit(repo, repo_type='space', operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(stage/n)) for n in names], commit_message='Finite private live relay')
        end = time.monotonic()+450
        while True:
            info = api.space_info(repo); runtime = api.get_space_runtime(repo)
            if not info.private or info.sha != commit.oid or any(h is not None and str(h) != 'cpu-basic' for h in (runtime.hardware,runtime.requested_hardware)):
                raise ValueError('Space privacy/revision/hardware drift')
            if str(runtime.stage) == 'RUNNING' and str(runtime.hardware)=='cpu-basic':
                try:
                    permit = attest(api,space_id=repo,revision=commit.oid)
                    with httpx.Client(timeout=10,follow_redirects=False,trust_env=False) as c:
                        health = c.get(permit.base_url+'/health',headers={'Authorization':'Bearer '+token,'Cache-Control':'no-store','Accept-Encoding':'identity'})
                    import re
                    if health.status_code == 200 and re.fullmatch('[a-f0-9]{32}',health.headers.get('X-H3-Boot','')) and 'no-store' in health.headers.get('Cache-Control','') and 'Content-Encoding' not in health.headers:
                        break
                except (OSError, httpx.HTTPError):
                    pass
            if time.monotonic() >= end or str(runtime.stage) in ('BUILD_ERROR','RUNTIME_ERROR','CONFIG_ERROR'):
                raise TimeoutError('finite authenticated relay readiness')
            time.sleep(2)
        consumer = SocketAdapter(permit.base_url,session=session,not_after=expires,role='consumer',role_key=ck,hf_token=token,endpoint_permit=permit)
        consumer.open_claim()
        gpu = directory/'gpu'; gpu.mkdir(mode=0o700)
        row = dict(schema='h3-live-relay-session-v1',session=session,space_id=repo,revision=commit.oid,not_after=expires,max_windows=16,max_frames=1654)
        deployment = prepare(gpu,row,api)
        job = submit(api,deployment,token,pk)
        ready = await_ready(api,job,session,min(time.monotonic()+1050,time.monotonic()+expires-time.time()-30))
        binding = LiveAVBinding(session,ready['profile_sha256'],ready['epoch'],max_segments=16,max_total_frames=1654)
        return run_conversation(consumer,binding,directory/'receiver',turns,director,ready=ready,max_seconds=min(600,ready['expires_at']-time.time()),max_turns=2,allow_obs=allow_obs,obs_port=obs_port)
    finally:
        if consumer is not None:
            consumer.close()
        if director is not None:
            director.close()
        try:
            if job is not None and api.inspect_job(job_id=job.id).status.stage not in TERMINAL:
                api.cancel_job(job_id=job.id)
        finally:
            # Random name plus session proof also handles unknown create outcome.
            try:
                owned = api.get_space_variables(repo)['H3_RELAY_SESSION'].value == session
            except Exception:
                owned = False
            if owned:
                api.delete_repo(repo,repo_type='space')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--allow-obs',action='store_true')
    p.add_argument('--obs-port',type=int)
    a = p.parse_args(argv)
    if not a.run:
        print('Offline: --run reads 1–2 fresh typed stdin lines, then starts one finite private relay/H200x2 session. No ASR; no Discord join.')
        return
    if a.obs_port is not None and not a.allow_obs:
        p.error('--obs-port requires --allow-obs')
    turns = []
    for _ in range(2):
        try:
            line = input('Fresh typed turn (blank ends): ')
        except EOFError:
            break
        if not line.strip():
            break
        turns.append(line)
    run(a.directory,turns,allow_obs=a.allow_obs,obs_port=a.obs_port)
    print('Session ended; Discord reception is not independently verified.')


if __name__ == '__main__':
    main()
