"""Finite saved-PCM native H3 interruption probe; offline description by default.

--run stages immutable original HF H200x2 assets, submits once, waits finitely,
collects metadata, then independently confirms termination. No ASR/TTS Endpoint,
model/placement change, device, browser, playback, Discord or ledger mutation.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import time

from research.hf_h3_interrupt_probe import SECTION, REPORT, validate_spec
from research.launch_h3_conversation_scenario import PRIVATE, ROOT

TERMINAL = {'COMPLETED', 'CANCELED', 'ERROR', 'DELETED'}
FENCE = '.interrupt-probe-unresolved.json'
OLD_IMPORT = 'from research.hf_h3_streaming_trial import run_session, validate_spec'
NEW_IMPORT = 'from research.hf_h3_interrupt_probe import run_session, validate_spec'


def inspect(stage):
    from research.launch_h3_conversation_scenario import inspect as previous
    spec, inventory = previous(stage)
    validate_spec(spec)
    names = {row['path'] for row in inventory['files']}
    if 'research/hf_h3_interrupt_probe.py' not in names:
        raise ValueError('interrupt worker missing')
    text = (Path(stage)/'research/bootstrap_h3_fast_ref2va.py').read_text()
    if text.count(NEW_IMPORT) != 1 or OLD_IMPORT in text:
        raise ValueError('explicit interrupt bootstrap dispatch required')
    return spec, inventory


def prepare(directory):
    """Existing read-only-metadata staging gates; no GPU or provider mutation."""
    from research.launch_h3_conversation_scenario import prepare as previous
    directory = Path(directory).resolve()
    previous(directory, streaming_trial=True)
    stage = directory/'deployment'
    freeze_probe(stage)
    return stage


def freeze_probe(stage):
    """Freeze explicit staged dispatch and revalidate its immutable inventory.

    Operates only on a fresh stage. No provider API call or source-tree bootstrap
    mutation occurs. Kept separate so original pinned metadata can be checked
    offline before main launches the finite owner.
    """
    from research.launch_h3_conversation_scenario import rewrite_allowlist
    from research.launch_h3_direct_conversation import first_party_sources
    stage = Path(stage)
    # Freeze current first-party sources, excluding nested environments/caches.
    for source in first_party_sources():
        target = stage/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    bootstrap = stage/'research/bootstrap_h3_fast_ref2va.py'
    text = bootstrap.read_text()
    if text.count(OLD_IMPORT) != 1:
        raise ValueError('reviewed streaming bootstrap dispatch missing')
    bootstrap.write_text(text.replace(OLD_IMPORT, NEW_IMPORT))
    spec = json.loads((stage/'spec.json').read_text())
    spec['interrupt_probe'] = dict(SECTION)
    (stage/'spec.json').write_text(json.dumps(spec, indent=2))
    rewrite_allowlist(stage)
    inspect(stage)
    return stage


def create_publication_once(api, repo):
    """Never repeat a repository POST after an uncertain successful create."""
    import httpx
    try:
        api.create_repo(repo, repo_type='dataset', private=True, exist_ok=False)
    except httpx.TransportError:
        # Exact fresh random repo only; no historical owner adoption or POST retry.
        remote = api.repo_info(repo, repo_type='dataset')
        if (remote.id != repo or remote.private is not True
                or any(row.rfilename != '.gitattributes' for row in remote.siblings)):
            raise RuntimeError('uncertain publication is not the exact empty private repository') from None


def submit(api, stage, token, on_created):
    """One guarded raw POST using the existing no-retry Jobs API adapter."""
    from huggingface_hub import CommitOperationAdd
    import httpx
    from research.launch_h3_direct_conversation import durable_json
    from research.launch_h3_fast_ref2va import pre_submit
    from research.h3_personal.job_network import assert_idle
    from research.h3_personal.direct_job_api import submit_job
    spec, inventory = inspect(stage)
    guard = stage.parent/'submission_guard.json'
    if guard.exists():
        raise ValueError('NEVER replay submitted or unknown deployment')
    who = api.whoami()
    if who.get('name') != 'YUGOROU' or who.get('isPro') is not True:
        raise ValueError('Jobs-enabled original owner required')
    assert_idle(api, spec, token); pre_submit(api, spec, stage)
    repo = spec['publication_repo']
    create_publication_once(api, repo)
    names = [row['path'] for row in inventory['files']] + ['upload-allowlist.json']
    commit = api.create_commit(repo, repo_type='dataset', operations=[
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(stage/name)) for name in names],
        commit_message='Immutable finite saved-PCM H3 native interruption probe')
    remote = api.repo_info(repo, repo_type='dataset', revision=commit.oid, files_metadata=True)
    entries = {row.rfilename: row for row in remote.siblings}
    if remote.sha != commit.oid or remote.private is not True or set(entries)-{'.gitattributes'} != set(names):
        raise ValueError('private immutable deployment inventory differs')
    for name in names:
        body = (stage/name).read_bytes(); entry = entries[name]
        digest = (hashlib.sha256(body).hexdigest() if entry.lfs else
                  hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest())
        if entry.size != len(body) or digest != (entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError('remote deployment bytes differ')
    bootstrap = 'research/bootstrap_h3_fast_ref2va.py'
    bootstrap_sha = hashlib.sha256((stage/bootstrap).read_bytes()).hexdigest()
    script = ("import hashlib,os,runpy;from huggingface_hub import hf_hub_download;"
              f"p=hf_hub_download({repo!r},{bootstrap!r},repo_type='dataset',revision={commit.oid!r},token=os.environ['HF_TOKEN']);"
              f"assert hashlib.sha256(open(p,'rb').read()).hexdigest()=={bootstrap_sha!r};runpy.run_path(p,run_name='__main__')")
    payload = dict(dockerImage=spec['image'], command=['/opt/venv/bin/python', '-c', script, repo,
        commit.oid, hashlib.sha256((stage/'upload-allowlist.json').read_bytes()).hexdigest()],
        arguments=[], environment={'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
        secrets={'HF_TOKEN': token}, flavor='h200x2', timeoutSeconds=1200, attempts=1,
        expose={'ports': [7860], 'portsPublic': []}, labels={'deployrev': commit.oid})
    # Keep the exact original license/location/model metadata checks; no new gate
    # or geographic assumption is substituted by this device-free harness.
    pre_submit(api, spec, stage); assert_idle(api, spec, token)
    durable_json(guard, dict(state='unknown_no_replay', revision=commit.oid, repo=repo))
    durable_json(PRIVATE/FENCE,
                 dict(directory=str(stage.parent), revision=commit.oid, repo=repo))
    def created(job):
        on_created(job)  # owner has the ID before any potentially failing receipt IO
        durable_json(stage.parent/'submission.json', dict(job_id=job.id, repo=repo, revision=commit.oid))
    with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as client:
        return submit_job(client, token, payload, on_created=created)


def confirm_terminal(api, job, *, seconds=90):
    deadline = time.monotonic()+seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            state = api.inspect_job(job_id=job.id, namespace='YUGOROU').status.stage
            if state in TERMINAL:
                return state
        except Exception as error:
            last_error = type(error).__name__
        time.sleep(2)
    raise RuntimeError('owned Job termination unconfirmed'+(':'+last_error if last_error else ''))


def run(directory):
    directory = Path(os.path.abspath(directory))
    if (directory.exists() or not directory.is_relative_to(PRIVATE)
            or any(path.is_symlink() for path in (directory, *directory.parents))):
        raise ValueError('fresh private campaign directory required')
    from huggingface_hub import HfApi, hf_hub_download, set_client_factory
    import httpx
    from research.provision_modal_identity_secret import cached_token
    from research.launch_h3_relay_conversation import canonical, sdk_redirect_guard
    from research.launch_h3_direct_conversation import durable_json
    from research.h3_personal.direct_job_api import JobCreationRejected
    os.umask(0o077)
    fd = os.open(PRIVATE/'.interrupt-probe-owner.lock', os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fence = PRIVATE/FENCE
        if fence.exists() or fence.is_symlink():
            raise RuntimeError('unresolved owned interrupt probe')
        canonical()
        set_client_factory(lambda: httpx.Client(timeout=120, follow_redirects=False,
            event_hooks={'response': [sdk_redirect_guard]}, trust_env=False))
        token = cached_token(); api = HfApi(token=token)
        stage = prepare(directory)
        owned = [None]; rejected = False; complete = False
        result = dict(schema=SECTION['schema'], devices_used=False, discord_used=False,
                      playback_verified=False, cleanup_confirmed=False)
        try:
            def created(job):
                owned[0] = job
            try:
                job = submit(api, stage, token, created)
            except JobCreationRejected as error:
                # A server-side 5xx can be an uncertain create, not proof that
                # nothing was leased. Retain global fence unless explicit reject.
                rejected = error.status in (400, 401, 403, 422)
                durable_json(directory/'submission-rejection.json',
                             dict(status=error.status, no_create_confirmed=rejected))
                raise
            print(json.dumps(dict(type='h3_interrupt_probe_submitted', job_id=job.id)), flush=True)
            end = time.monotonic()+1350
            while time.monotonic() < end:
                state = api.inspect_job(job_id=job.id, namespace='YUGOROU').status.stage
                if state in TERMINAL:
                    break
                time.sleep(5)
            else:
                raise TimeoutError('finite interrupt probe owner deadline')
            if state != 'COMPLETED':
                raise RuntimeError('native probe Job did not complete')
            path = hf_hub_download(json.loads((directory/'submission.json').read_text())['repo'],
                                   REPORT, repo_type='dataset', token=token)
            report = json.loads(Path(path).read_text())
            if (report.get('schema') != SECTION['schema'] or report.get('complete') is not True
                    or report.get('native_backend_connected') is not True
                    or report.get('both_rank_evidence') is not True):
                raise ValueError('native report did not prove requested boundary')
            durable_json(directory/'native-report.json', report)
            complete = True
        except BaseException as error:
            import traceback
            result['error_type'] = type(error).__name__
            result['failure_frames'] = [dict(file=Path(frame.filename).name,
                function=frame.name, line=frame.lineno)
                for frame in traceback.extract_tb(error.__traceback__)[-12:]]
            raise
        finally:
            primary = __import__('sys').exc_info()[1]
            result['complete'] = complete
            job = owned[0]
            try:
                if job is not None:
                    try:
                        state = api.inspect_job(job_id=job.id, namespace='YUGOROU').status.stage
                    except Exception:
                        state = None  # uncertain inspection must not suppress owned cancellation
                    if state not in TERMINAL:
                        try:
                            api.cancel_job(job_id=job.id, namespace='YUGOROU')
                        except Exception as error:
                            result['cancel_error_type'] = type(error).__name__
                    result.update(job_id=job.id, terminal=confirm_terminal(api, job), cleanup_confirmed=True)
                elif rejected:
                    result.update(terminal='not-created-explicit-rejection', cleanup_confirmed=True)
                if result['cleanup_confirmed'] and fence.exists():
                    if json.loads(fence.read_text()).get('directory') != str(directory):
                        raise RuntimeError('foreign interrupt fence')
                    fence.unlink()
            except BaseException as error:
                result['cleanup_error_type'] = type(error).__name__
                if primary is not None:
                    primary.add_note('interrupt probe cleanup unconfirmed')
                else:
                    raise
            finally:
                durable_json(directory/'result.json', result)
        return result
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(scope='offline-only', section=SECTION, hardware='original h200x2',
                              timeout_seconds=1200, attempts=1, devices=False,
                              source='approved legacy Irodori saved PCM; not current Vox or fresh conversation')))
        return 0
    if args.directory is None:
        parser.error('--run requires --directory')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('owner terminated')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('finite owner deadline')))
    signal.setitimer(signal.ITIMER_REAL, 1530)
    try:
        run(args.directory)
    except BaseException as error:
        print(json.dumps(dict(type='h3_interrupt_probe_failed', error_type=type(error).__name__)), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
