"""Finite foreground Modal Sandbox carrier. Offline plan is side-effect free.

Caller must independently confirm HF jobs terminal and license authorization.
Only generic metadata enters a Sandbox before its actual country/GPU gate.
"""
import argparse
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import hashlib
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import fcntl
import inspect
import json
import os
from pathlib import Path
import re
import uuid

from research.modal_gpu_plan import GPU_HOURLY, multiplier
from research.probe_modal_ref2va_location import classify

CAP = Decimal('20.44')
OLD_SPENT = Decimal('1')
RESERVATION = Decimal('6')
LIVE_FLOOR = Decimal('10')
# modal_gpu_plan official pricing, but charge TWO GPUs and the entire hard host.
CPU_HOURLY = Decimal('.141912')
GIB_HOURLY = Decimal('.024012')
CPU = 4
MEMORY_MIB = 262144
RATE = (2 * Decimal(str(GPU_HOURLY['H200'])) + CPU * CPU_HOURLY
        + 256 * GIB_HOURLY) * Decimal(str(multiplier('ap')))
TIMEOUT = 660
STARTUP = 180
TEARDOWN = 60
BUILD_MARGIN = Decimal('1')
BOUND = RATE * (TIMEOUT + STARTUP + TEARDOWN) / 3600 + BUILD_MARGIN
IMAGE = ('ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:'
         '34ef8907bf0c4329d81be5fd4a6e3056b806e65fa0bba439733ab8a45bef31cf')
TAGS = {'h3-carrier': 'live-voice'}


def plan(deployment=None, revision=None, allowlist_sha=None):
    """Pure metadata plan. Pins are all-or-none; no local files are consulted."""
    if any(x is not None for x in (deployment, revision, allowlist_sha)):
        if (not isinstance(deployment, str)
                or not re.fullmatch(r'[\w.-]+/[\w.-]+', deployment)
                or any(part in ('.', '..') for part in deployment.split('/'))
                or not isinstance(revision, str)
                or not re.fullmatch(r'[0-9a-f]{40}', revision)
                or not isinstance(allowlist_sha, str)
                or not re.fullmatch(r'[0-9a-f]{64}', allowlist_sha)):
            raise ValueError('private deployment and immutable revision/allowlist pins required')
    if BOUND > RESERVATION or CAP - OLD_SPENT - RESERVATION < LIVE_FLOOR:
        raise ValueError('phase bound exceeds shared Modal pool')
    return dict(schema='h3-modal-live-voice-plan-v1', executable=True, image=IMAGE,
                deployment=deployment, revision=revision, allowlist_sha256=allowlist_sha,
                region='ap', gpu='H200:2', cpu=[CPU, CPU],
                memory_mib=[MEMORY_MIB, MEMORY_MIB], max_containers=1, retries=0,
                timeout_seconds=TIMEOUT, startup_seconds=STARTUP,
                teardown_seconds=TEARDOWN, build_margin_usd=str(BUILD_MARGIN),
                rate_usd_hour=str(RATE), regional_multiplier='1.15',
                calculated_bound_usd=str(BOUND), reservation_usd=str(RESERVATION),
                modal_cap_usd=str(CAP), old_spent_usd=str(OLD_SPENT),
                remaining_before_usd='19.44', remaining_after_usd='13.44',
                minimum_live_balance_usd=str(LIVE_FLOOR),
                bootstrap='research.bootstrap_h3_fast_ref2va.main',
                quality_policy='original64/v12/BF16 H3; native FP32 Irodori; no fallback',
                detached=False, watchers=False, platform_billing_cap=False)


def placement_gate(metadata):
    """Use actual GPU-container environment, never requested region or host env."""
    if not isinstance(metadata, dict):
        raise ValueError('runtime metadata required')
    result = classify(metadata)
    if result['decision'] != 'allowed_candidate':
        raise ValueError('runtime placement denied: ' + result['decision'])
    return result


def metadata_script():
    """Only generic stdlib source may enter an unapproved GPU Sandbox.

    Performs the existing exact-country gate *inside* that actual container.
    Output has no credentials, source paths, model imports or personal assets.
    Host must validate again before using any post-gate staging capability.
    """
    return ('import os, json\n' + inspect.getsource(classify) + '\n'
            "metadata = {k: os.environ.get(k, '') for k in "
            "('MODAL_REGION', 'MODAL_CLOUD_PROVIDER')}\n"
            "result = classify(metadata)\n"
            "print(json.dumps(dict(metadata=metadata, classification=result)), flush=True)\n"
            "raise SystemExit(0 if result['decision'] == 'allowed_candidate' else 78)\n")


def _amount(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError('finite nonnegative ledger amount required')
    return number


def _save(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Ledger:
    """Existing separate Modal ledger; no implicit reset or HF budget transfer.

    Reserve before allocation. Ambiguous create/termination retains the full
    reservation, preventing a retry lottery. Confirmed termination uses measured
    conservative charge, or the full $6 if unmeasured; not a provider invoice.
    """
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            if (data.get('schema') != 'h3-ref2va-modal-budget-v1'
                    or _amount(data['cap_usd']) != CAP
                    or _amount(data['spent_bound_usd']) < OLD_SPENT
                    or not isinstance(data.get('reservations'), dict)):
                raise ValueError('existing separate Modal authority required')
            yield data

    def reserve(self):
        with self.locked() as data:
            spent = _amount(data['spent_bound_usd'])
            if data['reservations'] or CAP - spent - RESERVATION < LIVE_FLOOR:
                raise ValueError('active reservation or insufficient live balance')
            key = uuid.uuid4().hex
            data['reservations'][key] = dict(upper_bound_usd=str(RESERVATION),
                scope='modal-live-voice', state='reserved_unknown_no_retry')
            _save(self.path, data)
            return key

    def terminated(self, key, charge=None):
        with self.locked() as data:
            row = data['reservations'][key]
            if row.get('scope') != 'modal-live-voice' or _amount(row['upper_bound_usd']) != RESERVATION:
                raise ValueError('reservation mismatch')
            del data['reservations'][key]
            charge = RESERVATION if charge is None else _amount(charge)
            if charge > RESERVATION:
                raise ValueError('charge exceeds reservation')
            data['spent_bound_usd'] = str(_amount(data['spent_bound_usd']) + charge)
            _save(self.path, data)


def exercise_lifecycle(adapter, ledger, approved_side):
    """Dependency-injected contract harness, NOT a production Modal launcher.

    adapter.create receives only the generic plan/script: no secrets, payload,
    source directory, HF credentials or model image. metadata() must report
    the very same GPU instance's script output. Only after its gate may the
    approved_side callback stage pinned sources and arrange managed credentials
    on that same instance, then invoke bootstrap.main(repo, revision, allow_sha).
    terminate() returning exactly True means confirmed terminal; anything else
    leaves the reservation locked. There is no retry, spawn, detach or watcher.
    """
    key = ledger.reserve()
    sandbox = None
    try:
        sandbox = adapter.create(plan(), metadata_script())
        placement_gate(sandbox.metadata())
        return approved_side(sandbox)
    finally:
        if sandbox is not None and sandbox.terminate() is True:
            ledger.terminated(key)


def validate_stage(stage, deployment, allow_sha):
    from research.launch_h3_live_voice import inspect as inspect_stage
    stage = Path(stage)
    if stage.is_symlink() or any(p.is_symlink() for p in stage.rglob('*')):
        raise ValueError('stage symlinks forbidden')
    spec, inventory = inspect_stage(stage)
    if spec['publication_repo'] != deployment:
        raise ValueError('deployment binding differs')
    if hashlib.sha256((stage / 'upload-allowlist.json').read_bytes()).hexdigest() != allow_sha:
        raise ValueError('local allowlist pin differs')
    return spec, inventory


def bootstrap_script(deployment, revision, allow_sha):
    """Generic starter: credential is exec Secret environment, never source."""
    plan(deployment, revision, allow_sha)
    return '''import os, json, hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo, revision, digest = ''' + repr((deployment, revision, allow_sha)) + '''
token = os.environ['HF_TOKEN']
api = HfApi(token=token)
if not api.repo_info(repo, repo_type='dataset', revision=revision).private:
    raise ValueError('private deployment required')
def fetch(name):
    return Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision, token=token)).read_bytes()
body = fetch('upload-allowlist.json')
if hashlib.sha256(body).hexdigest() != digest:
    raise ValueError('allowlist digest differs')
rows = json.loads(body)['files']
seen = set()
for row in rows:
    name = row['path']
    if not isinstance(name, str) or not name or name.startswith('/') or '\\\\' in name or any(p in ('', '.', '..') for p in name.split('/')) or name in seen or name == 'upload-allowlist.json':
        raise ValueError('unsafe inventory path')
    seen.add(name)
    if type(row['bytes']) is not int or row['bytes'] < 0:
        raise ValueError('invalid inventory size')
root = Path('/work/h3-modal-carrier-source')
root.mkdir(parents=True, exist_ok=False)
for row in rows:
    data = fetch(row['path'])
    if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('deployment bytes differ')
    dest = root / row['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
os.environ['PYTHONPATH'] = str(root)
sys.path.insert(0, str(root))
from research.bootstrap_h3_fast_ref2va import main
main(repo, revision, digest)
'''


def gpu_script():
    return ("import subprocess,json\n"
            "p=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],"
            "capture_output=True,text=True,check=True,timeout=15)\n"
            "print(json.dumps(p.stdout.splitlines()))\n")


def redact(text, token=''):
    if token:
        text = text.replace(token, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED]', text)


def conservative_charge(elapsed):
    return min(RESERVATION, (Decimal(str(max(0, elapsed))) * RATE / 3600
                            + BUILD_MARGIN).quantize(Decimal('.01'), rounding=ROUND_CEILING))


def run_production(args, sdk=None):
    """No retries; ambiguous allocation/termination preserves ledger lock.

    Host HF terminal/license checks remain mandatory caller preconditions.
    """
    validate_stage(args.stage, args.deployment, args.allowlist_sha256)
    if sdk is None:
        import modal as sdk
    # SDK list returns active Sandbox objects, not raw API responses.
    if any(sb.poll() is None for sb in sdk.Sandbox.list(tags=TAGS)):
        raise ValueError('active carrier; no concurrent allocation')
    output = Path(args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    ledger = Ledger(args.budget)
    key = ledger.reserve()
    receipt = dict(schema='h3-modal-live-voice-receipt-v1', reservation=key,
                   state='reserved_unknown_no_retry', sandbox_id=None,
                   process_exitcode=None, termination_exitcode=None, charge_bound_usd=None)
    path = output / 'receipt.json'
    sb = None
    started = time.monotonic()
    try:
        _save(path, receipt)
        with sdk.App('h3-live-voice-carrier').run(detach=False) as app:
            try:
                sb = sdk.Sandbox.create('sleep', str(TIMEOUT), app=app,
                    image=sdk.Image.from_registry(IMAGE), timeout=TIMEOUT,
                    idle_timeout=TIMEOUT, gpu='H200:2', region='ap',
                    cpu=(CPU, CPU), memory=(MEMORY_MIB, MEMORY_MIB), tags=TAGS)
                receipt.update(sandbox_id=sb.object_id, state='created_unapproved')
                _save(path, receipt)
                proc = sb.exec('python3', '-c', metadata_script(), timeout=30)
                metadata = json.loads(proc.stdout.read())['metadata']
                if proc.wait() != 0:
                    raise ValueError('container country gate failed')
                placement_gate(metadata)
                proc = sb.exec('python3', '-c', gpu_script(), timeout=30)
                names = json.loads(proc.stdout.read())
                if proc.wait() != 0 or names != ['NVIDIA H200', 'NVIDIA H200']:
                    raise ValueError('exact two NVIDIA H200 required')
                receipt.update(state='approved', metadata=metadata, gpu_names=names)
                _save(path, receipt)
                # No credential cache access or Secret construction before both gates.
                from research.provision_modal_identity_secret import cached_token
                token = cached_token()
                secret = sdk.Secret.from_dict({'HF_TOKEN': token})
                proc = sb.exec('python3', '-u', '-c', bootstrap_script(
                    args.deployment, args.revision, args.allowlist_sha256),
                    timeout=TIMEOUT, secrets=[secret])
                def drain(stream, destination):
                    for line in stream:
                        clean = redact(line, token)
                        destination.write(clean)
                        destination.flush()
                with open(output / 'stdout.log', 'x', opener=lambda p, f: os.open(p, f, 0o600)) as log:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(drain, proc.stdout, log),
                                   pool.submit(drain, proc.stderr, log)]
                        for future in futures:
                            future.result()
                receipt['process_exitcode'] = proc.wait()
                receipt['state'] = 'process_finished'
            finally:
                if sb is not None:
                    code = sb.terminate(wait=True)
                    if type(code) is int:
                        receipt['termination_exitcode'] = code
                        charge = conservative_charge(time.monotonic() - started)
                        ledger.terminated(key, charge)
                        receipt.update(state='terminated', charge_bound_usd=str(charge))
    except Exception:
        # Never serialize SDK exceptions or raw API objects (may contain secrets).
        receipt['failed'] = True
        raise RuntimeError('carrier failed; inspect private receipt; no automatic retry') from None
    finally:
        _save(path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment')
    parser.add_argument('--revision')
    parser.add_argument('--allowlist-sha256')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--budget', type=Path)
    args = parser.parse_args(argv)
    value = plan(args.deployment, args.revision, args.allowlist_sha256)
    if args.run:
        if not all((args.stage, args.output, args.budget, args.deployment,
                    args.revision, args.allowlist_sha256)):
            parser.error('--run requires --stage --output --budget and all immutable pins')
        value = run_production(args)
    elif args.stage:
        if not args.deployment:
            parser.error('--stage requires all immutable pins')
        validate_stage(args.stage, args.deployment, args.allowlist_sha256)
        value['stage_validated'] = True
    print(json.dumps(value, indent=2))
    if args.run and (value['process_exitcode'] != 0 or value['termination_exitcode'] is None):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
