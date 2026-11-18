"""Finite foreground HF READY observation. Offline unless --run --directory."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

TERMINAL = {'COMPLETED', 'CANCELED', 'CANCELLED', 'ERROR', 'DELETED'}


def private_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def load(directory):
    paths = [directory / 'submission.json', directory / 'deployment/spec.json']
    raw = [p.read_bytes() for p in paths]
    submission, spec = map(json.loads, raw)
    if (submission.get('provider', 'huggingface_jobs') != 'huggingface_jobs' or
            submission.get('namespace', 'YUGOROU') != 'YUGOROU' or
            not isinstance(submission.get('job_id'), str) or not submission['job_id']):
        raise ValueError('own HF submission required')
    repo = spec['publication_repo']
    if not isinstance(repo, str) or len(repo.split('/')) != 2 or not repo.startswith('YUGOROU/') or not repo.split('/')[1]:
        raise ValueError('owned dataset required')
    if (spec['config']['timeout_seconds'] not in (660,1200,1500) or
            spec.get('execution', {}).get('timeout_seconds', spec['config']['timeout_seconds']) != spec['config']['timeout_seconds']):
        raise ValueError('HF server timeout must remain finite660/1500s')
    session = spec['live_voice_session']['id']
    if not isinstance(session, str) or not session:
        raise ValueError('session required')
    return paths, raw, submission, spec


def wait(directory, api, fetch, *, monotonic=time.monotonic, wall=time.time,
         sleep=time.sleep, scheduling_seconds=240, running_seconds=660, poll_seconds=2,
         cancel_on_failure=True):
    """Injected API/download and clocks permit fully offline deterministic tests.

    fetch(repo, name, revision) returns decoded JSON at exactly that revision.
    No job enumeration, launch, or background work is performed.
    """
    if not (0 < scheduling_seconds <= 240 and 0 < running_seconds <= 1500 and
            0 < poll_seconds <= 10):
        raise ValueError('finite wait bounds required')
    directory = Path(directory)
    paths, raw, submission, spec = load(directory)
    if running_seconds > spec['config']['timeout_seconds']:
        raise ValueError('wait cannot exceed immutable server timeout')
    job_id = submission['job_id']
    repo = spec['publication_repo']
    session = spec['live_voice_session']['id']

    def unchanged():
        try:
            return all(p.read_bytes() == b for p, b in zip(paths, raw))
        except OSError:
            return False

    def stage():
        return api.inspect_job(job_id=job_id, namespace='YUGOROU').status.stage

    started = monotonic()
    first_running = None
    with (directory / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if api.whoami()['name'] != 'YUGOROU':
                raise ValueError('authenticated owner required')
            while True:
                if not unchanged():
                    raise ValueError('submission/spec changed')
                current = stage()
                now = monotonic()
                if current in TERMINAL:
                    raise RuntimeError('own job terminal')
                if current == 'RUNNING' and first_running is None:
                    first_running = now
                if first_running is None:
                    if now - started >= scheduling_seconds:
                        raise TimeoutError('scheduling deadline')
                    if current not in {'SCHEDULING', 'PENDING'}:
                        raise ValueError('unknown job state')
                    remaining_wait = scheduling_seconds - (now - started)
                else:
                    remaining_wait = running_seconds - (now - first_running)
                    if remaining_wait <= 0:
                        raise TimeoutError('running deadline')
                    if current != 'RUNNING':
                        raise ValueError('running job state regressed')
                    info = api.repo_info(repo, repo_type='dataset', timeout=15)
                    if info.private is not True or not info.sha:
                        raise ValueError('private pinned dataset required')
                    names = api.list_repo_files(repo, repo_type='dataset', revision=info.sha)
                    if 'live/state.json' in names:
                        state = fetch(repo, 'live/state.json', info.sha)
                        if state.get('session_id') != session or state.get('previous_audio_reused') is not False:
                            raise ValueError('stale/reused session state')
                        status = state.get('status')
                        if status == 'ready':
                            expires = state.get('expires_at')
                            if type(expires) not in (int, float) or not math.isfinite(expires):
                                raise ValueError('finite expiration required')
                            remaining = min(expires - wall(), running_seconds - (monotonic() - first_running))
                            if not 165 <= remaining <= spec['config']['timeout_seconds']:
                                raise TimeoutError('insufficient READY window')
                            if not unchanged() or stage() != 'RUNNING':
                                raise ValueError('submission no longer current/running')
                            remaining = min(expires - wall(), running_seconds - (monotonic() - first_running))
                            if remaining < 165:
                                raise TimeoutError('READY window closed')
                            receipt = dict(job_id=job_id, namespace='YUGOROU', session_id=session,
                                           repository=repo, revision=info.sha, status='ready',
                                           previous_audio_reused=False, observed_at=wall(),
                                           expires_at=expires, remaining_seconds=remaining,
                                           submission_sha256=hashlib.sha256(raw[0]).hexdigest())
                            private_json(directory / 'ready-observed.json', receipt)
                            return receipt
                        if status in {'error', 'failed', 'expired', 'stopped', 'output'}:
                            raise RuntimeError('live session not usable')
                sleep(min(poll_seconds, remaining_wait))
        except BaseException as error:
            # No exception messages persisted: remote errors can contain credentials.
            receipt = dict(job_id=job_id, namespace='YUGOROU', reason=type(error).__name__,
                           cancel_attempts=0, confirmed_terminal=False, stages=[])
            if not cancel_on_failure:
                receipt['skipped']='resident_operator_policy_keep_warm'
                private_json(directory / 'ready-error.json', receipt)
                raise
            for attempt in range(3):
                if not unchanged():
                    receipt['skipped'] = 'submission_or_spec_changed'
                    break
                try:
                    current = stage()
                    receipt['stages'].append(current)
                    if current in TERMINAL:
                        receipt['confirmed_terminal'] = True
                        break
                    if unchanged():
                        receipt['cancel_attempts'] += 1
                        api.cancel_job(job_id=job_id, namespace='YUGOROU')
                except Exception:
                    pass
                sleep(1)
            if unchanged() and not receipt['confirmed_terminal']:
                try:
                    current = stage()
                    receipt['stages'].append(current)
                    receipt['confirmed_terminal'] = current in TERMINAL
                except Exception:
                    pass
            private_json(directory / 'ready-cancellation.json', receipt)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    load(args.directory)
    if not args.run:
        print(json.dumps(dict(mode='offline', network=False, scheduling_max_seconds=240,
                              running_max_seconds=660, minimum_ready_seconds=165)))
        return 0
    from huggingface_hub import HfApi, hf_hub_download
    from research.provision_modal_identity_secret import cached_token
    token = cached_token()
    api = HfApi(token=token)

    def fetch(repo, name, revision):
        path = hf_hub_download(repo, name, repo_type='dataset', revision=revision,
                               token=token, etag_timeout=15)
        return json.loads(Path(path).read_text())

    try:
        _, _, _, spec = load(args.directory)
        wait(args.directory, api, fetch, running_seconds=spec['config']['timeout_seconds'])
    except Exception:
        print(json.dumps(dict(status='failed', detail='see private cancellation receipt')))
        return 1
    print(json.dumps(dict(status='ready', receipt='ready-observed.json')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
