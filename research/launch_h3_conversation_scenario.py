"""Stage and (with --submit) run ONE finite HF conversation-scenario Job.

Offline by default: it stages an immutable private deployment and prints the
plan. No ledger writes, no retries, no second Job, no device access. Budget
management stays with the USER; this tool only performs provider idleness and
immutability checks before a single submission.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
from research.h3_personal.private_root import root as _private_root

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = _private_root()
SPEECH = dict(repo='YUGOROU/h3-fast-ref2va-smd4-gpu-live-4182459cb5',
              revision='ed5df76d9efac41f53cf4ee7dc879b71af66324b',
              path='live/turn-001/reply.wav', sample_rate=48000, samples=577920,
              sha256='7fd491b50ac16f7f9c0644795632101613a5b576fbd1e8551f9c80a564f6a585')
TIMEOUT_SECONDS = 1200
CONVERSATION = dict(schema='h3-conversation-scenario-v1', profile='reference1024',
                    listening_seconds=1.0, expected_frames=532, paced=True)
REQUIRED = ('research/hf_h3_conversation_scenario.py',
            'research/h3_personal/late_listening_diagnostic.py',
            'research/h3_personal/conversation_av.py',
            'research/h3_personal/conversation_scenario.py',
            'research/h3_personal/conversation_sequence.py',
            'research/h3_personal/conversation_timeline.py',
            'research/h3_personal/conversation_continuation.py',
            'research/h3_personal/conversation_decode.py',
            'research/h3_personal/conversation_stages.py',
            'research/h3_personal/conversation_control.py',
            'research/h3_personal/conversation_conditioning.py',
            'research/h3_personal/conversation_state.py',
            'research/h3_personal/live_voice_runtime.py',
            'research/h3_personal/headless_av_sink.py',
            'research/h3_personal/av_delivery.py',
            'research/h3_personal/buffered_stream.py',
            'research/h3_personal/accepted_profiles.py',
            'research/h3_personal/conditioning_experiment.py')

STREAMING_REQUIRED = ('research/hf_h3_streaming_trial.py',
    'research/hf_h3_late_listening_diagnostic.py',
    'research/h3_personal/streaming_trial.py', 'research/h3_personal/streaming_native_owner.py',
    'research/h3_personal/streaming_native_stages.py', 'research/h3_personal/streaming_pcm.py',
    'research/h3_personal/streaming_windows.py', 'research/h3_personal/streaming_waveform.py',
    'research/h3_personal/streaming_admission.py')

_IMPORT = re.compile(r'(?:from\s+(research[\w.]*)\s+import|^\s*import\s+(research[\w.]*))', re.M)


def _module_path(name):
    return name.replace('.', '/') + '.py'


def _references(text):
    found = set()
    for match in _IMPORT.finditer(text):
        name = match.group(1) or match.group(2)
        found.add(name)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            base = '.' * node.level + (node.module or '')
            if node.level or base == 'research' or base.startswith('research.'):
                if base.strip('.'):
                    found.add(base)
                for alias in node.names:
                    if alias.name != '*':
                        separator = '' if base.endswith('.') else '.'
                        found.add(base + separator + alias.name)
    return found


def closure(stage, seeds):
    """Add every locally referenced staged module, transitively."""
    stage = Path(stage)
    pending = list(seeds)
    added = []
    visited = set()
    while pending:
        name = pending.pop()
        if name.startswith('.') or name in visited:
            continue
        visited.add(name)
        path = ROOT / _module_path(name)
        target = stage / _module_path(name)
        if not target.is_file():
            if not path.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            added.append(_module_path(name))
        for ref in _references(target.read_text()):
            if ref.startswith('.'):
                level = len(ref) - len(ref.lstrip('.'))
                parts = name.split('.')[:-1]
                if level <= len(parts):
                    package = '.'.join(parts[:len(parts) - level + 1])
                    pending.append(package + '.' + ref[level:])
            else:
                pending.append(ref)
    for package in ('research/__init__.py', 'research/h3_personal/__init__.py'):
        if not (stage / package).exists():
            (stage / package).write_text('')
    return added


def rewrite_allowlist(stage):
    rows = [dict(path=p.relative_to(stage).as_posix(), bytes=p.stat().st_size,
                 sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(Path(stage).rglob('*'))
            if p.is_file() and p.name != 'upload-allowlist.json']
    (Path(stage) / 'upload-allowlist.json').write_text(json.dumps(dict(
        schema=json.loads((Path(stage) / 'upload-allowlist.json').read_text())['schema'],
        private_only=True, files=rows)))
    return rows


def inspect(stage):
    from research.launch_h3_fast_ref2va import inspect as base_inspect
    spec, inventory = base_inspect(Path(stage))
    section = spec.get('conversation_scenario')
    if section != CONVERSATION:
        raise ValueError('exact bounded conversation scenario section required')
    if spec['config']['timeout_seconds'] != TIMEOUT_SECONDS:
        raise ValueError('fixed finite 1200s conversation Job required')
    if spec.get('locked_speech') != SPEECH:
        raise ValueError('existing full 12.04s speech binding required')
    if spec.get('attention_mode') != 'fa4-packed' or spec.get('layout') != 'portrait':
        raise ValueError('accepted portrait fa4-packed runtime required')
    if spec.get('runtime_policy') != 'resident-visual-cached':
        raise ValueError('resident visual cached runtime required')
    if spec.get('incremental_transfer') or spec.get('face_crop_box') is not None:
        raise ValueError('no transfer/face-crop variants in this scope')
    names = {row['path'] for row in inventory['files']}
    required = REQUIRED
    if spec.get('streaming_trial') is not None:
        from research.hf_h3_streaming_trial import validate_spec
        validate_spec(spec)
        required += STREAMING_REQUIRED
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError('staged conversation modules missing: ' + ','.join(missing))
    for name in names:
        if name.endswith('.py'):
            source = (Path(stage) / name).read_text()
            compile(source, name, 'exec')
    return spec, inventory


def _bound(*, ledger_bound, timeout_seconds, hourly_rate=10.00002):
    """Arithmetic plan only. No ledger mutation and no agent-side spending gate."""
    from decimal import Decimal
    rate, total = Decimal(str(hourly_rate)), Decimal(str(ledger_bound))
    cost = rate * Decimal(timeout_seconds) / Decimal(3600)
    return dict(job_bound=cost, cumulative_bound=total + cost, rate_usd_hour=rate)


def prepare(directory, *, streaming_trial=False, late_listening_diagnostic=False):
    if late_listening_diagnostic:streaming_trial=True
    """Stage an immutable private deployment using the shared SMD4 preparer.

    The preparer is bound to the pinned realtime-first template and to an
    arithmetic-only bound; the USER's ledger is never written or enforced here.
    """
    from types import FunctionType
    import shutil
    from huggingface_hub import HfApi
    from research import prepare_h3_fast_ref2va_gpu as staging
    from research.provision_modal_identity_secret import cached_token
    directory = Path(directory).resolve()
    target = directory / 'deployment'
    if target.exists():
        raise ValueError('fresh private staging directory required')
    if directory.is_relative_to(ROOT):
        raise ValueError('private staging must stay outside the repository')
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            raise ValueError('runtime integration missing: ' + name)
    directory.mkdir(parents=True, mode=0o700)
    dependencies = dict(staging.prepare.__globals__, budget_bound=_bound)
    preparer = FunctionType(staging.prepare.__code__, dependencies,
                            staging.prepare.__name__, staging.prepare.__defaults__)
    publication = 'YUGOROU/h3-fast-ref2va-smd4-gpu-conversation-' + uuid.uuid4().hex[:10]
    report = preparer(target, PRIVATE / 'gpu-hf-realtime-first-v2/deployment',
                      PRIVATE / 'compute-budget.json', HfApi(token=cached_token()),
                      publication_repo=publication, timeout_seconds=TIMEOUT_SECONDS,
                      attention_mode='fa4-packed', layout='portrait',
                      runtime_policy='resident-visual-cached',
                      locked_speech=dict(SPEECH))
    required = REQUIRED + (STREAMING_REQUIRED if streaming_trial else ())
    added = closure(target, [p[:-3].replace('/', '.') for p in required])
    spec = json.loads((target / 'spec.json').read_text())
    spec['conversation_scenario'] = dict(CONVERSATION)
    if streaming_trial:
        from research.hf_h3_streaming_trial import SECTION
        spec['streaming_trial'] = dict(SECTION)
    if late_listening_diagnostic:
        from research.hf_h3_late_listening_diagnostic import SECTION as DIAGNOSTIC
        spec['late_listening_diagnostic'] = dict(DIAGNOSTIC)
    (target / 'spec.json').write_text(json.dumps(spec, indent=2))
    rows = rewrite_allowlist(target)
    spec, inventory = inspect(target)
    return dict(directory=str(directory), stage=str(target), plan={k: str(v) for k, v in report.items()},
                added=added, files=len(rows), publication_repo=spec['publication_repo'],
                allowlist_sha256=hashlib.sha256((target / 'upload-allowlist.json').read_bytes()).hexdigest())


def submit(directory):
    from huggingface_hub import CommitOperationAdd, HfApi
    from research.provision_modal_identity_secret import cached_token
    from research.prepare_h3_realtime_first import safe_path
    directory = Path(directory).resolve()
    target = directory / 'deployment'
    spec, inventory = inspect(target)
    marker = directory / 'submission_guard.json'
    if marker.exists():
        raise ValueError('NEVER replay a submitted/unknown-outcome deployment')
    token = cached_token()
    api = HfApi(token=token)
    who = api.whoami()
    owner = spec['publication_repo'].split('/')[0]
    if owner != 'YUGOROU' or who['name'] != owner or who.get('isPro') is not True:
        raise ValueError('Jobs-enabled owner required')
    terminal = {'COMPLETED', 'CANCELED', 'ERROR', 'DELETED'}
    if any(j.status.stage not in terminal for j in api.list_jobs(namespace=owner)):
        raise ValueError('HF is not idle')
    repo = spec['publication_repo']
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=False)
    api.auth_check(repo, repo_type='dataset', write=True)
    names = [row['path'] for row in inventory['files']] + ['upload-allowlist.json']
    commit = api.create_commit(repo, repo_type='dataset', operations=[
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(target / name)) for name in names],
        commit_message='Private immutable conversation-scenario deployment; one finite Job')
    remote = api.repo_info(repo, repo_type='dataset', revision=commit.oid, files_metadata=True)
    rows = {s.rfilename: s for s in remote.siblings}
    if not remote.private or set(rows) - {'.gitattributes'} != set(names):
        raise ValueError('remote deployment differs')
    for name in names:
        data = (target / name).read_bytes()
        entry = rows[name]
        if entry.size != len(data):
            raise ValueError('remote bytes differ')
        expected = (hashlib.sha256(data).hexdigest() if entry.lfs else
                    hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())
        if expected != (entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError('remote digest differs')
    bootstrap = 'research/bootstrap_h3_fast_ref2va.py'
    bootstrap_sha = next(r['sha256'] for r in inventory['files'] if r['path'] == bootstrap)
    allow_sha = hashlib.sha256((target / 'upload-allowlist.json').read_bytes()).hexdigest()
    script = ("import hashlib,os,runpy; from huggingface_hub import hf_hub_download; "
              "p=hf_hub_download(" + repr(repo) + "," + repr(str(safe_path(bootstrap))) +
              ",repo_type='dataset',revision=" + repr(commit.oid) + ",token=os.environ['HF_TOKEN']); "
              "assert hashlib.sha256(open(p,'rb').read()).hexdigest()=='" + bootstrap_sha +
              "'; runpy.run_path(p,run_name='__main__')")
    seconds = spec['config']['timeout_seconds']
    job = api.run_job(image=spec['image'],
        command=['/opt/venv/bin/python', '-c', script, repo, commit.oid, allow_sha],
        flavor='h200x2', namespace=owner, timeout=seconds,
        secrets={'HF_TOKEN': token},
        env={'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
        labels={'project': 'h3-conversation-scenario', 'deploy': commit.oid[:32]})
    receipt = dict(job_id=job.id, repo=repo, revision=commit.oid,
                   timeout_seconds=seconds, flavor='h200x2',
                   submitted_utc=datetime.now(timezone.utc).isoformat(),
                   url='https://huggingface.co/jobs/' + repo)
    (directory / 'submission.json').write_text(json.dumps(receipt, indent=2))
    with marker.open('x') as handle:
        json.dump(dict(job_id=job.id, state='submitted_no_retry'), handle)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--streaming-trial', action='store_true', help='isolated native PCM-prefix trial')
    parser.add_argument('--late-listening-diagnostic',action='store_true',help='matched global/local positions + native decode parity')
    args = parser.parse_args(argv)
    if not args.submit:
        info = prepare(args.directory, streaming_trial=args.streaming_trial,
                       late_listening_diagnostic=args.late_listening_diagnostic)
        print(json.dumps(info, indent=2))
        return 0
    if not (args.directory / 'deployment' / 'spec.json').exists():
        prepare(args.directory, streaming_trial=args.streaming_trial,
                late_listening_diagnostic=args.late_listening_diagnostic)
    print(json.dumps(submit(args.directory), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
