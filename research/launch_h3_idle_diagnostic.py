"""Offline-first finite idle diagnostic staging; --submit creates one autonomous Job.

No ports, input services, playback, watcher, retries, or budget ledger writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from types import FunctionType, SimpleNamespace
import uuid

from research.launch_h3_conversation_scenario import ROOT, PRIVATE, _bound, closure, rewrite_allowlist
from research.launch_h3_relay_conversation import canonical, sdk_redirect_guard, TERMINAL
from research.launch_h3_direct_conversation import durable_json

SECTION = {'schema': 'h3-idle-drift-diagnostic-v1', 'windows': 14}
WORKER = 'research.hf_h3_idle_diagnostic'


def inspect(stage):
    from research.launch_h3_fast_ref2va import inspect as base_inspect
    from research.hf_h3_idle_diagnostic import validate_spec
    stage = Path(stage)
    spec, inventory = base_inspect(stage)
    validate_spec(spec)
    modes = ('conversation_scenario', 'streaming_trial', 'late_listening_diagnostic',
             'live_voice_session', 'live_direct_session', 'live_relay_session',
             'modal_live_approval', 'incremental_transfer', 'regional_compile',
             'warm_stage_diagnostic')
    if any(spec.get(key) for key in modes) or spec.get('face_crop_box') is not None:
        raise ValueError('exclusive idle diagnostic only')
    if not re.fullmatch(r'YUGOROU/h3-fast-ref2va-smd4-gpu-idle-[a-f0-9]{32}',
                        spec.get('publication_repo', '')):
        raise ValueError('fresh random idle publication selector required')
    if (spec['config']['timeout_seconds'] != 1200
            or spec.get('locked_speech') is not None
            or spec.get('attention_mode') != 'fa4-packed'
            or spec.get('layout') != 'portrait'
            or spec.get('runtime_policy') != 'resident-visual-cached'):
        raise ValueError('fixed finite accepted idle runtime required')
    names = {r['path'] for r in inventory['files']}
    if WORKER.replace('.', '/') + '.py' not in names:
        raise ValueError('idle worker missing')
    for name in names:
        if name.endswith('.py'):
            compile((stage/name).read_text(), name, 'exec')
    return spec, inventory


def prepare(directory, api, idle_anchor=False):
    from research import prepare_h3_fast_ref2va_gpu as base
    from research.hf_h3_idle_diagnostic import ANCHOR_SECTION
    if type(idle_anchor) is not bool:
        raise ValueError('boolean idle anchor selector required')
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT) or directory.exists():
        raise ValueError('fresh private directory outside repository required')
    directory.mkdir(mode=0o700, parents=True)
    stage = directory/'deployment'
    dependencies = dict(base.prepare.__globals__, budget_bound=_bound)
    preparer = FunctionType(base.prepare.__code__, dependencies,
                            base.prepare.__name__, base.prepare.__defaults__)
    preparer(stage, PRIVATE/'gpu-hf-realtime-first-v2/deployment',
             PRIVATE/'compute-budget.json', api,
             publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-idle-'+uuid.uuid4().hex,
             timeout_seconds=1200, attention_mode='fa4-packed', layout='portrait',
             runtime_policy='resident-visual-cached', locked_speech=None)
    closure(stage, [WORKER])
    spec = json.loads((stage/'spec.json').read_text())
    spec['idle_diagnostic'] = dict(ANCHOR_SECTION if idle_anchor else SECTION)
    (stage/'spec.json').write_text(json.dumps(spec))
    rewrite_allowlist(stage)
    inspect(stage)
    return stage


def submit_job(client, token, payload, on_created):
    """One canonical POST; callback owns the validated ID before receipt I/O."""
    if (payload.get('flavor') != 'h200x2'
            or type(payload.get('timeoutSeconds')) is not int
            or payload['timeoutSeconds'] != 1200
            or type(payload.get('attempts')) is not int or payload['attempts'] != 1
            or 'expose' in payload
            or type(token) is not str or not token
            or any(ord(c) < 33 or ord(c) > 126 for c in token)):
        raise ValueError('finite unexposed Job required')
    response = client.post('https://huggingface.co/api/jobs/YUGOROU',
        headers={'Authorization': 'Bearer '+token}, json=payload, follow_redirects=False)
    if response.status_code not in (200, 201):
        raise RuntimeError('Job creation rejected; never replay')
    row = response.json()
    if type(row) is not dict or type(row.get('id')) is not str or not re.fullmatch('[a-f0-9]{24}', row['id']):
        raise RuntimeError('unknown Job creation outcome; reconcile without replay')
    job = SimpleNamespace(id=row['id'])
    on_created(job)
    return job


def submit(api, stage, token):
    canonical()
    from huggingface_hub import CommitOperationAdd
    from research.launch_h3_fast_ref2va import pre_submit
    
    import httpx
    spec, inventory = inspect(stage)
    guard = stage.parent/'submission_guard.json'
    if guard.exists():
        raise ValueError('unknown/submitted outcome: NEVER replay')
    owner = api.whoami()
    if owner.get('name') != 'YUGOROU' or owner.get('isPro') is not True:
        raise ValueError('Jobs-enabled owner required')
    if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):
        raise ValueError('HF not idle')
    pre_submit(api, spec, stage)
    repo = spec['publication_repo']
    with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as control:
        try:
            response = control.post('https://huggingface.co/api/repos/create',
                headers={'Authorization':'Bearer '+token}, json=dict(name=repo.split('/')[1], organization='YUGOROU', type='dataset', private=True))
        except httpx.TransportError:
            # The POST may have succeeded. Never replay it; only adopt the exact
            # fresh random deployment repo when it is still empty and private.
            try:
                remote = api.repo_info(repo, repo_type='dataset')
                verified = (remote.id == repo and remote.id.split('/')[0] == owner['name']
                            and remote.private is True
                            and all(s.rfilename == '.gitattributes' for s in remote.siblings))
            except Exception:
                verified = False
            if not verified:
                raise
        else:
            if response.status_code not in (200, 201):
                raise RuntimeError('dataset creation failed; no retry')
    if api.repo_info(repo, repo_type='dataset').private is not True:
        raise ValueError('private deployment required')
    names = [r['path'] for r in inventory['files']] + ['upload-allowlist.json']
    commit = api.create_commit(repo, repo_type='dataset', operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(stage/n)) for n in names], commit_message='Immutable finite idle diagnostic runtime')
    remote = api.repo_info(repo, repo_type='dataset', revision=commit.oid, files_metadata=True)
    entries = {s.rfilename:s for s in remote.siblings}
    if remote.sha != commit.oid or remote.private is not True or set(entries)-{'.gitattributes'} != set(names):
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
    command = ['/opt/venv/bin/python','-c',script,repo,commit.oid,hashlib.sha256((stage/'upload-allowlist.json').read_bytes()).hexdigest()]
    payload = dict(dockerImage=spec['image'], command=command, arguments=[],
        environment={'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1'},
        secrets={'HF_TOKEN':token},
        flavor='h200x2', timeoutSeconds=1200, attempts=1,
        labels={'deployrev':commit.oid})
    # Recheck idleness/pins after upload, immediately before the one POST.
    pre_submit(api, spec, stage)
    if any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace='YUGOROU')):
        raise ValueError('HF not idle')
    durable_json(guard, dict(state='unknown_no_replay', revision=commit.oid))
    job = None
    def on_created(created):
        nonlocal job
        # Own the ID before receipt I/O, which may itself be interrupted.
        job = created
        durable_json(stage.parent/'submission.json', dict(job_id=job.id, repo=repo, revision=commit.oid))
    try:
        with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as control:
            job = submit_job(control, token, payload, on_created=on_created)
        print(json.dumps(dict(type='h3_job_submitted', job_id=job.id)), flush=True)
        return job, spec['image'], command
    except BaseException as original:
        if job is not None:
            try:
                api.cancel_job(job_id=job.id)
            except BaseException as error:
                original.add_note('Job cancellation failed: '+type(error).__name__)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--idle-anchor', action='store_true')
    args = parser.parse_args(argv)
    if args.idle_anchor and not args.prepare:
        parser.error('--idle-anchor requires --prepare')
    if args.prepare or args.submit:
        canonical()
        import httpx
        from huggingface_hub import HfApi, set_client_factory
        from research.provision_modal_identity_secret import cached_token
        set_client_factory(lambda: httpx.Client(timeout=20, trust_env=False,
            follow_redirects=True, event_hooks={'response': [sdk_redirect_guard]}))
        token = cached_token()
        api = HfApi(endpoint='https://huggingface.co', token=token)
    stage = prepare(args.directory, api, idle_anchor=args.idle_anchor) if args.prepare else args.directory/'deployment'
    if args.submit:
        job, _, _ = submit(api, stage, token)
        print(json.dumps({'job_id': job.id, 'receipt': str(stage.parent/'submission.json')}))
    else:
        spec, _ = inspect(stage)
        print(json.dumps({'offline': True, 'repo': spec['publication_repo'],
                          'timeout_seconds': 1200, 'flavor': 'h200x2'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
