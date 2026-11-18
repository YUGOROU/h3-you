"""Stage small immutable inputs, or explicitly submit ONE bounded HF CPU job.
No local media encoding/copies; originals are uploaded directly from known paths.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil

from hf_prepare_identity_media import check_selection, dump, helpers, sha256, verify_sources

TERMINAL = {'COMPLETED', 'CANCELED', 'ERROR', 'DELETED'}


def stage(frozen_path, target, repo):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('personal private dataset repo ID required')
    frozen_path, target = Path(frozen_path).expanduser().resolve(), Path(target).expanduser().resolve()
    project = Path(__file__).resolve().parent
    freeze, _ = helpers(project / 'h3_personal')
    selection = json.loads(frozen_path.read_text())
    rows = check_selection(selection, freeze)
    local = {}
    for row in rows:
        digest, source = row['source_sha256'], str(Path(row['source']).expanduser().resolve())
        if digest in local and local[digest] != source:
            raise ValueError('one known source path per digest required')
        if Path(row['source']).expanduser().is_symlink():
            raise ValueError('no source symlinks')
        local[digest] = source
    verify_sources(rows, local)
    target.mkdir(parents=True, exist_ok=False)
    files = {'frozen_selection.json': frozen_path,
             'hf_prepare_identity_media.py': project / 'hf_prepare_identity_media.py',
             'freeze_dataset.py': project / 'h3_personal/freeze_dataset.py',
             'preprocess.py': project / 'h3_personal/preprocess.py'}
    for name, source in files.items():
        shutil.copyfile(source, target / name)  # only tiny freeze/code; NEVER originals
    request = {'schema': 'h3-cpu-raw-request-v1', 'repo': repo,
               'dataset_sha256': selection['dataset_sha256'],
               'freeze_sha256': sha256(target / 'frozen_selection.json'),
               'code_sha256': {n: sha256(target / n) for n in files if n.endswith('.py')},
               'source_map': {digest: '/inputs/sources/' + digest + '.mov' for digest in local}}
    dump(target / 'request.json', request)
    dump(target / 'local_sources.json', local)  # NEVER uploaded; direct upload whitelist
    dump(target / 'plan.json', {'state': 'staged_not_submitted', 'repo': repo,
         'dataset_sha256': selection['dataset_sha256'], 'samples': 715,
         'flavor': 'cpu-upgrade', 'hard_timeout': '3h', 'spend_cap_usd': .20,
         'maximum_reservation_usd': .15, 'cumulative_cap_usd': 150,
         'original_bytes': sum(Path(p).stat().st_size for p in local.values()),
         'local_original_copies': 0, 'automatic_retry': False, 'automatic_next_gpu': False})
    return target


def cpu_rate(api):
    hardware = next(h for h in api.list_jobs_hardware() if h.name == 'cpu-upgrade')
    unit = {'hour': 1, 'minute': 60}.get(hardware.unit_label)
    rate = float(hardware.unit_cost_usd) * unit if unit else float('nan')
    if hardware.accelerator or not math.isfinite(rate) or not 0 < rate <= .05:
        raise ValueError('CPU-only rate <= $0.05/hour could not be verified')
    if rate * 3 > .20:
        raise ValueError('CPU spend cap exceeded')
    return rate


def reserve(ledger, ident, rate):
    spent = ledger.get('spent_usd')
    reservations = ledger.get('reservations')
    if type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0 or not isinstance(reservations, dict):
        raise ValueError('reviewed cumulative ledger required')
    if ident in reservations:
        raise ValueError('dataset already reserved; no automatic retry')
    total = spent
    for value in reservations.values():
        amount = value['upper_bound_usd']
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError('invalid reservation')
        total += amount
    upper = rate * 3
    if total + upper > 150:
        raise ValueError('cumulative $150 cap exceeded')
    reservations[ident] = {'upper_bound_usd': upper, 'state': 'reserved_no_retry',
                           'flavor': 'cpu-upgrade', 'hard_timeout': '3h'}
    return upper


def submit(target, ledger_path, image, api, volume_class):
    target, ledger_path = Path(target), Path(ledger_path).expanduser()
    if not re.fullmatch(r'[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('reviewed immutable CPU image digest required')
    request = json.loads((target / 'request.json').read_text())
    local = json.loads((target / 'local_sources.json').read_text())
    freeze, _ = helpers(target)
    selection = json.loads((target / 'frozen_selection.json').read_text())
    rows = check_selection(selection, freeze)
    if sha256(target / 'frozen_selection.json') != request['freeze_sha256'] or selection['dataset_sha256'] != request['dataset_sha256']:
        raise ValueError('staged freeze changed')
    if request['schema'] != 'h3-cpu-raw-request-v1' or set(request['code_sha256']) != {'hf_prepare_identity_media.py', 'freeze_dataset.py', 'preprocess.py'}:
        raise ValueError('bad code allowlist')
    for name, digest in request['code_sha256'].items():
        if sha256(target / name) != digest:
            raise ValueError('staged code changed')
    if request['source_map'] != {d: '/inputs/sources/' + d + '.mov' for d in local}:
        raise ValueError('mounted source map changed')
    verify_sources(rows, local)
    who = api.whoami()
    owner = who['name']
    # Fail closed for unrecognized fine-grained permissions; never print whoami/token.
    if who.get('auth', {}).get('accessToken', {}).get('role') != 'write' or who.get('isPro') is not True:
        raise ValueError('verified personal PRO account and write token required')
    if request['repo'].split('/')[0] != owner:
        raise ValueError('personal account namespace mismatch')
    jobs = list(api.list_jobs(namespace=owner))  # permissions check; all previous jobs
    for job in jobs:
        status = getattr(getattr(job, 'status', None), 'stage', None)
        if status not in TERMINAL:
            raise ValueError('prior active/unknown-status job; reconcile before submission')
        if getattr(job, 'labels', {}).get('dataset_sha256') in {request['dataset_sha256'], request['dataset_sha256'][:32]}:
            raise ValueError('dataset job already exists; no duplicate submission')
    rate = cpu_rate(api)
    # Lock is local/account ledger scope. Reservation survives all ambiguous errors.
    with ledger_path.with_suffix(ledger_path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text())
        upper = reserve(ledger, request['dataset_sha256'], rate)
        marker = target / 'submission_guard.json'
        with marker.open('x') as stream:
            json.dump({'state': 'reserved_no_retry', 'upper_bound_usd': upper}, stream)
        dump(ledger_path, ledger)
        repo = request['repo']
        api.create_repo(repo, repo_type='dataset', private=True, exist_ok=False)
        if api.repo_info(repo, repo_type='dataset').private is not True:
            raise ValueError('private dataset required')
        plan = json.loads((target / 'plan.json').read_text())
        plan.update(state='uploading_inputs',sources_uploaded=0,source_count=len(local),
                    usd_per_hour=rate,reserved_upper_bound_usd=upper)
        dump(target / 'plan.json',plan)
        # No upload_folder: no adjacent metadata, photos, or credentials can leak.
        for digest, filename in local.items():
            api.upload_file(repo_id=repo, repo_type='dataset', path_or_fileobj=filename,
                            path_in_repo='sources/' + digest + '.mov',
                            commit_message='Known unchanged original MOV')
            plan['sources_uploaded'] += 1
            dump(target / 'plan.json',plan)
        commit = None
        for name in ('frozen_selection.json', 'hf_prepare_identity_media.py', 'freeze_dataset.py', 'preprocess.py', 'request.json'):
            commit = api.upload_file(repo_id=repo, repo_type='dataset', path_or_fileobj=str(target / name),
                                     path_in_repo=name, commit_message='Immutable CPU preparation inputs')
        revision = commit.oid
        plan.update(state='inputs_uploaded_not_submitted',input_revision=revision)
        dump(target / 'plan.json',plan)
        if api.repo_info(repo, repo_type='dataset', revision=revision).private is not True:
            raise ValueError('private immutable input mount required')
        command = '''set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update -qq
apt-get install -y -qq ffmpeg
python -m pip install -q huggingface_hub==1.12.0
python /inputs/hf_prepare_identity_media.py
'''
        # Finite ONE submission. No retry/watcher/GPU, even on uncertain API failure.
        # Keep digest pin; normalize official DockerHub spelling and bound label/name lengths.
        job = api.run_job(image=image.removeprefix('docker.io/library/'), name='h3-production-715-raw',
            command=['bash', '-lc', command], flavor='cpu-upgrade', timeout='3h',
            namespace=owner, secrets={'HF_TOKEN': os.environ['HF_TOKEN']},
            env={'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'HF_HUB_DISABLE_TELEMETRY': '1',
                 'HF_INPUT_REVISION': revision},
            volumes=[volume_class(type='dataset', source=repo, revision=revision, mount_path='/inputs', read_only=True)],
            labels={'project': 'h3-raw-media', 'dataset_sha256': request['dataset_sha256'][:32]})
        ledger['reservations'][request['dataset_sha256']].update(state='submitted', job_id=job.id,
                                                                repo=repo, input_revision=revision)
        dump(ledger_path, ledger)
        report = {'state': 'submitted', 'job_id': job.id, 'repo': repo, 'input_revision': revision,
                  'usd_per_hour': rate, 'reserved_upper_bound_usd': upper, 'timeout': '3h', 'cpu_only': True}
        dump(marker, report)
        plan.update(report)
        dump(target / 'plan.json',plan)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-selection', type=Path)
    p.add_argument('--stage-dir', type=Path, required=True)
    p.add_argument('--repo')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--budget-ledger', type=Path)
    p.add_argument('--image', help='Reviewed Debian/Python CPU image pinned by sha256 digest')
    a = p.parse_args()
    try:
        if not a.stage_dir.exists():
            if not a.frozen_selection or not a.repo:
                p.error('new stage requires --frozen-selection and --repo')
            stage(a.frozen_selection, a.stage_dir, a.repo)
        if not a.submit:
            print(json.dumps(json.loads((a.stage_dir / 'plan.json').read_text()), indent=2))
            return
        if not a.budget_ledger or not a.image:
            p.error('--submit requires --budget-ledger and --image')
        # Intentional cached-token override, consistent with this project's account.
        token = (Path.home() / '.cache/huggingface/token').read_text().strip()
        if not token:
            raise ValueError('cached token missing')
        os.environ['HF_TOKEN'] = token
        from huggingface_hub import HfApi, Volume
        print(json.dumps(submit(a.stage_dir, a.budget_ledger, a.image, HfApi(), Volume), indent=2))
    except Exception as error:
        print(json.dumps({'state': 'failed_closed', 'failure_type': type(error).__name__}))
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
