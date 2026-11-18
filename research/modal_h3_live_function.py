"""Foreground, single-attempt Function carrier. Dry run is entirely offline.

Caller must confirm HF termination/license and inspect the immutable uploaded
1800-second, <=16 fresh-turn stage. No old voice inputs are accepted by policy.
"""
import argparse
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from research.modal_h3_live_voice import IMAGE, bootstrap_script, _amount, _save
from research.probe_modal_ref2va_location import classify

# USER reports $20.40 available after the prior $16.59477055573329922705920 bound.
# Preserve historical spend; this is an explicit allowance update, not a reset.
CAP = Decimal('20.44')
USER_UPDATED_CAP = Decimal('36.99477055573329922705920')
# 192 GiB trial was SIGKILL137 during LoRA conversion (provider reports likely OOM).
# Increase host RAM only; no precision/offload/model fallback. 256 GiB fit unproven.
BASE_RATE = 2 * Decimal('4.5396') + 4 * Decimal('.04716') + 256 * Decimal('.007992')
TIMEOUT, STARTUP, TEARDOWN = 1800, 180, 60
# Existing pinned base + Python 3.11 control-plane shim; no new build or weights.
RUNTIME_IMAGE_ID = 'im-EHjNujv578xET3b6yvFTS2'
BUILD_MARGIN = Decimal('.10')


def plan(deployment=None, revision=None, allowlist_sha=None, region='ap', timeout_seconds=1800):
    if type(timeout_seconds) is not int or timeout_seconds not in (600, 720, 900, 1200, 1500, 1800):
        raise ValueError('timeout must be 600, 720, 900, 1200, 1500 or 1800 seconds')
    if region not in ('ap', 'jp', 'au'):
        raise ValueError('explicit ap, jp or au required')
    if any(v is not None for v in (deployment, revision, allowlist_sha)):
        bootstrap_script(deployment, revision, allowlist_sha)  # pure pin validation
    rate = BASE_RATE * Decimal('1.15' if region == 'ap' else '1.75')
    bound = rate * (timeout_seconds + STARTUP + TEARDOWN) / 3600 + BUILD_MARGIN
    reservation = bound.quantize(Decimal('.01'), rounding=ROUND_CEILING)
    return dict(image=IMAGE, runtime_image_id=RUNTIME_IMAGE_ID,
                runtime_image_provenance='existing pinned base + Python 3.11 shim; host RAM 256 GiB after SIGKILL137; model/precision unchanged',
                build_margin_usd=str(BUILD_MARGIN), region=region, gpu='H200:2', cpu=[4, 4],
                memory=[262144, 262144], timeout=timeout_seconds, startup_timeout=STARTUP,
                rate_usd_hour=str(rate), reservation_usd=str(reservation),
                calculated_bound_usd=str(bound),
                remaining_after_usd_assumes_old_spent1=str(CAP - 1 - reservation),
                deployment=deployment, revision=revision, allowlist_sha256=allowlist_sha,
                max_containers=1, retries=0, scaledown_window=2,
                single_use_containers=True, serialized=True, include_source=False,
                max_fresh_turns=16, session_seconds=timeout_seconds, no_old_voice=True)


def worker_source():
    # Only stdlib/generic gate source is serialized; no repo modules captured.
    return inspect.getsource(classify) + 'MIN_USEFUL_SECONDS = 900\n' + '''
def worker(bootstrap, not_after_epoch, requested_region='', trust_jp_contract=False):
    import os, json, subprocess, sys, re, time
    if time.time() >= not_after_epoch:
        return dict(classification='expired')
    def log(value):
        text = re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED]', str(value))
        print(text, flush=True)
    metadata = {k: os.environ.get(k, '') for k in ('MODAL_REGION', 'MODAL_CLOUD_PROVIDER')}
    decision = classify(metadata)
    evidence_mode = 'runtime_region_classifier'
    if (trust_jp_contract is True and requested_region == 'jp'
            and metadata['MODAL_REGION'] in ('ap-northeast', 'ap', 'jp')
            and metadata['MODAL_CLOUD_PROVIDER'] in (
                'CLOUD_PROVIDER_AWS', 'CLOUD_PROVIDER_GCP', 'CLOUD_PROVIDER_OCI')):
        evidence_mode = 'operator_accepted_modal_jp_contract'
        decision = dict(decision='allowed_candidate', country='JP',
                        evidence=evidence_mode, runtime_country_verified=False)
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10)
        names = p.stdout.strip().splitlines() if p.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        names = []
    valid = len(names) == 2 and all('H200' in n for n in names)
    placement = dict(kind='placement', metadata=metadata, classification=decision,
                     requested_region=requested_region, evidence_mode=evidence_mode,
                     gpu_names=names, gpu_valid=valid)
    log(json.dumps(placement))
    if decision['decision'] != 'allowed_candidate' or not valid:
        return dict(classification='gate_denied', placement=placement)
    if not_after_epoch - time.time() < MIN_USEFUL_SECONDS:
        return dict(classification='expired', placement=placement)
    os.environ['H3_SESSION_NOT_AFTER'] = str(not_after_epoch)
    os.environ['H3_PROVIDER'] = 'modal'
    # Modal's injected Python serves the control plane. Execute H3 using the
    # EXISTING pinned image interpreter/dependencies, not a replacement ML stack.
    import shutil
    candidates = [sys.executable, shutil.which('python'), shutil.which('python3'),
                  '/opt/venv/bin/python', '/opt/conda/bin/python',
                  '/usr/local/bin/python3.12', '/usr/bin/python3.12',
                  '/usr/local/bin/python3.11', '/usr/local/bin/python3.13',
                  '/app/FastVideo/.venv/bin/python', '/workspace/FastVideo/.venv/bin/python']
    check = "import importlib.metadata as m; assert m.version('torch').split('+')[0]=='2.12.0'; assert m.version('fastvideo-kernel')=='0.3.5'; assert m.version('quack-kernels')=='0.5.3'; import huggingface_hub"
    runtime = None
    for candidate in dict.fromkeys(candidates):
        if not candidate or not os.path.isfile(candidate): continue
        probe = subprocess.run([candidate,'-c',check],capture_output=True,text=True,timeout=15)
        if probe.returncode == 0: runtime = candidate; break
    if runtime is None: raise RuntimeError('pinned_existing_image_python_unavailable')
    log(json.dumps(dict(kind='runtime_interpreter',path=runtime)))
    # HF_TOKEN is first read by this post-gate child, never by the gate.
    child = subprocess.Popen([runtime, '-u', '-c', bootstrap],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in child.stdout:
            log(line.rstrip())
        code = child.wait()
        if code != 0:
            raise RuntimeError('bootstrap_failed')
        return dict(classification='completed', returncode=code)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
'''


def build_function(secret, deployment, revision, allowlist_sha, region='ap', sdk=None, timeout_seconds=1800, trust_jp_contract=False):
    """No allocation until caller enters app.run and consumes remote_gen."""
    if sdk is None:
        import modal as sdk
    p = plan(deployment, revision, allowlist_sha, region, timeout_seconds)
    if trust_jp_contract is True and region != 'jp':
        raise ValueError('trusted Japan contract requires jp region')
    requested_region = p['region']
    trust_jp_contract = trust_jp_contract is True
    # Fixed admission deadline: queue allowance is already budgeted. Never refresh
    # this in the worker; Modal's root timeout still limits execution separately.
    not_after_epoch = time.time() + timeout_seconds + STARTUP
    source = worker_source().replace('MIN_USEFUL_SECONDS = 900', 'MIN_USEFUL_SECONDS = ' + str(min(900, timeout_seconds - 60)))
    bootstrap = bootstrap_script(deployment, revision, allowlist_sha)
    app = sdk.App('h3-live-function-' + uuid.uuid4().hex[:10])

    # Official lazy hydration must succeed before Function spawn; no build fallback.
    @app.function(image=sdk.Image.from_id(p['runtime_image_id']), secrets=[secret],
                  region=region, gpu='H200:2', cpu=(4, 4), memory=(262144, 262144),
                  max_containers=1, retries=0, scaledown_window=2,
                  single_use_containers=True, serialized=True, include_source=False,
                  timeout=p['timeout'], startup_timeout=180)
    def carrier():
        namespace = {}
        exec(source, namespace)
        return namespace['worker'](bootstrap, not_after_epoch, requested_region, trust_jp_contract)
    return app, carrier


class Ledger:
    """Shared schema, distinct scope; uncertain termination retains reservation."""
    def __init__(self, path, region='ap', timeout_seconds=1800):
        self.path = Path(path)
        self.p = plan(region=region, timeout_seconds=timeout_seconds)
        self.reservation = Decimal(self.p['reservation_usd'])

    @contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            if (data.get('schema') != 'h3-ref2va-modal-budget-v1'
                    or _amount(data['cap_usd']) not in (CAP, USER_UPDATED_CAP)
                    or _amount(data['spent_bound_usd']) < 1
                    or not isinstance(data.get('reservations'), dict)):
                raise ValueError('existing shared Modal ledger required')
            yield data

    def reserve(self):
        with self.locked() as data:
            if data['reservations'] or _amount(data['cap_usd']) - _amount(data['spent_bound_usd']) < self.reservation:
                raise ValueError('active reservation or insufficient funds')
            key = uuid.uuid4().hex
            data['reservations'][key] = dict(scope='modal-live-function',
                upper_bound_usd=str(self.reservation), state='reserved_unknown_no_retry')
            _save(self.path, data)
            return key

    def terminated(self, key, seconds, *, no_function_call_proven=False):
        # Pre-spawn image initialization cannot consume Function GPU runtime.
        # Cached-artifact allowance only; closed historical receipts are untouched.
        margin = Decimal(self.p['build_margin_usd'])
        charge = margin if no_function_call_proven else _amount(seconds) * Decimal(self.p['rate_usd_hour']) / 3600 + margin
        with self.locked() as data:
            row = data['reservations'][key]
            if row['scope'] != 'modal-live-function' or _amount(row['upper_bound_usd']) != self.reservation:
                raise ValueError('reservation mismatch')
            charge = min(charge, self.reservation)
            data['spent_bound_usd'] = str(_amount(data['spent_bound_usd']) + charge)
            del data['reservations'][key]
            _save(self.path, data)
        return str(charge)


def sanitize(value, token):
    text = json.dumps(value)
    if token:
        text = text.replace(token, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED]', text)


def run(stage, output, budget, deployment, revision, allowlist_sha, region='ap', inspector=None):
    """Explicit-run API. Inspector must validate the caller's 1800s immutable spec.

    Deliberately no fallback to the older 660-second stage inspector.
    """
    p = plan(deployment, revision, allowlist_sha, region)
    if inspector is None:
        raise ValueError('caller must supply the 1800-second immutable stage inspector')
    inspector(Path(stage), deployment, allowlist_sha)
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_relative_to(root) or Path(budget).resolve().is_relative_to(root):
        raise ValueError('fresh private output and external ledger required')
    output.mkdir(mode=0o700, parents=True)
    import modal
    token = (Path.home() / '.cache/huggingface/token').read_text().strip()
    if not token:
        raise ValueError('cached HF credential required')
    secret = modal.Secret.from_dict({'HF_TOKEN': token})
    app, function = build_function(secret, deployment, revision, allowlist_sha, region, modal)
    ledger = Ledger(budget, region)
    key = ledger.reserve()
    started = time.time()
    terminal = False
    result = 'unknown'
    app_id = None
    try:
        with (output / 'events.jsonl').open('x') as stream:
            stream.write(sanitize(dict(kind='header', code_sha256=hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(), pins=p,
                log_policy='opaque_untrusted_bootstrap_output', started=started), token) + '\n')
            with app.run():
                app_id = app.app_id
                stream.write(sanitize(dict(kind='call_started', app_id=app_id, time=time.time()), token) + '\n')
                event = function.remote()
                result = event['classification']
                stream.write(sanitize(event, token) + '\n')
                stream.flush()
            terminal = True  # normal app context exit, never detach
    finally:
        elapsed = time.time() - started
        charge = ledger.terminated(key, elapsed) if terminal else None
        _save(output / 'terminal.json', dict(app_id=app_id, started=started,
            finished=time.time(), elapsed_wall_seconds=elapsed, classification=result,
            app_context_exited=terminal, charge_bound_usd=charge,
            full_reservation_retained=not terminal, invoice_verified=False))
    return result


def start(stage, output, budget, repo, revision, allow_sha, region='ap', inspector=None):
    """Spawn exactly one finite provider call; never poll or launch a watcher."""
    if inspector is None:
        raise ValueError('immutable stage inspector required')
    inspected = inspector(Path(stage), repo, allow_sha)
    spec = inspected[0] if isinstance(inspected, tuple) else json.loads((Path(stage) / 'spec.json').read_text())
    approval = spec.get('modal_live_approval', {})
    trust_jp_contract = approval.get('trust_explicit_jp_contract') is True
    if trust_jp_contract and not (region == approval.get('region') == 'jp'):
        raise ValueError('trusted Japan contract requires matching inspected jp region')
    timeout_seconds = spec['config']['timeout_seconds']
    p = plan(repo, revision, allow_sha, region, timeout_seconds)
    output = Path(output).resolve()
    budget = Path(budget).resolve()
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_relative_to(root) or budget.is_relative_to(root):
        raise ValueError('fresh private output and external ledger required')
    import modal
    token = (Path.home() / '.cache/huggingface/token').read_text().strip()
    if not token:
        raise ValueError('cached HF credential required')
    app, function = build_function(modal.Secret.from_dict({'HF_TOKEN': token}),
                                   repo, revision, allow_sha, region, modal, timeout_seconds,
                                   trust_jp_contract=trust_jp_contract)
    output.mkdir(mode=0o700, parents=True)
    key = Ledger(budget, region, timeout_seconds).reserve()
    state = dict(app_id=None, call_id=None, app_name=app.name, phase='initializing',reservation=key, budget=str(budget),
                 region=region, started=time.time(), status='UNKNOWN',
                 pins=p, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 full_reservation_retained=True, invoice_verified=False)
    _save(output / 'launch.json', state)
    try:
        with modal.enable_output(), app.run(detach=True):
            state['app_id'] = app.app_id
            state.update(phase='spawn_requested', spawn_requested_at=time.time())
            _save(output / 'launch.json', state)
            call = function.spawn()
            state.update(call_id=call.object_id, status='SUBMITTED')
            _save(output / 'launch.json', state)
    except Exception as exc:
        state.update(status='UNKNOWN', error_class=type(exc).__name__,error_detail=sanitize(str(exc),token))
        _save(output / 'launch.json', state)
    return state


def _confirm_stop(app_id):
    """CLI is a host control-plane operation, not a GPU allocation."""
    subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', '--yes',app_id],
                   check=False, capture_output=True, text=True, timeout=60)
    # CLI returns nonzero for already-stopped Apps; authoritative list decides.
    result = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--json'],
                            check=True, capture_output=True, text=True, timeout=60)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        return False
    return any(isinstance(row, dict) and row.get('app_id',row.get('App ID')) == app_id
               and str(row.get('state',row.get('State',''))).lower() == 'stopped' for row in rows)


def status(directory, *, stopping=False):
    directory = Path(directory)
    state = json.loads((directory / 'launch.json').read_text())
    if not state['full_reservation_retained']:
        return state
    terminal = stopping
    if not state.get('app_id') and state.get('app_name'):
        listing=subprocess.run([sys.executable,'-m','modal','app','list','--json'],check=True,capture_output=True,text=True,timeout=45)
        matches=[r for r in json.loads(listing.stdout) if r.get('description')==state['app_name']]
        if len(matches)==1:state['app_id']=matches[0]['app_id']
    if not stopping and state.get('call_id'):
        import modal
        try:
            result = modal.FunctionCall.from_id(state['call_id']).get(timeout=0)
            state['status'] = 'TERMINAL'
            if isinstance(result, dict) and isinstance(result.get('placement'),dict):
                placement=result['placement'];metadata=placement.get('metadata',{})
                state['placement']=dict(metadata={k:str(metadata.get(k,''))[:64] for k in ('MODAL_REGION','MODAL_CLOUD_PROVIDER')},classification=placement.get('classification'),gpu_names=placement.get('gpu_names'),gpu_valid=placement.get('gpu_valid'))
            if isinstance(result, dict):
                state['result'] = {k: v for k, v in result.items()
                    if (k == 'classification' and v in ('completed', 'gate_denied', 'expired'))
                    or (k == 'returncode' and type(v) is int)}
            terminal = True
        except (modal.exception.FunctionTimeoutError,modal.exception.ExecutionError,modal.exception.UserCodeException,modal.exception.RemoteError) as exc:
            state.update(status='ERROR',error_class=type(exc).__name__)
            terminal=True
        except (TimeoutError, modal.exception.TimeoutError):
            state['status'] = 'RUNNING'
        except Exception as exc:
            # A transport/credential failure is not proof that GPU execution ended.
            state.update(status='UNKNOWN', error_class=type(exc).__name__)
            terminal = isinstance(exc,RuntimeError) and str(exc)=='bootstrap_failed'
    if terminal and state.get('app_id'):
        try:
            confirmed = _confirm_stop(state['app_id'])
        except Exception as exc:
            confirmed = False
            state['stop_error_class'] = type(exc).__name__
        if confirmed:
            now = time.time()
            state['charge_bound_usd'] = Ledger(state['budget'], state['region'], state['pins']['timeout']).terminated(
                state['reservation'], max(0, now - state['started']),no_function_call_proven=state.get('phase')=='initializing' and state.get('call_id') is None)
            state.update(full_reservation_retained=False, stopped_confirmed=now)
            if stopping:
                state['status'] = 'STOPPED'
    _save(directory / 'launch.json', state)
    return state


def stop(directory):
    return status(directory, stopping=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--status', metavar='DIRECTORY')
    modes.add_argument('--stop', metavar='DIRECTORY')
    for name in ('stage', 'output', 'budget', 'deployment', 'revision', 'allowlist-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--region', choices=('ap', 'jp', 'au'), default='ap')
    args = parser.parse_args()
    if args.status or args.stop:
        print(json.dumps(stop(args.stop) if args.stop else status(args.status)))
        return
    p = plan(args.deployment, args.revision, args.allowlist_sha256, args.region)
    if not args.run:
        print(json.dumps(p))
        return
    if not all((args.stage, args.output, args.budget, args.deployment,
                args.revision, args.allowlist_sha256)):
        parser.error('--run requires stage, output, budget and immutable pins')
    from research.prepare_h3_modal_live import inspect as stage_inspect
    print(json.dumps(start(args.stage, args.output, args.budget, args.deployment,
                           args.revision, args.allowlist_sha256, args.region, stage_inspect)))


if __name__ == '__main__':
    main()
