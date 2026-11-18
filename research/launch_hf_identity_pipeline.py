"""Prepare small deployment artifacts; default plan-only, ONE gated HF H200 submission.
Never uploads, creates repos, changes auth, retries, or restarts canceled comparisons.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import hf_bootstrap_identity_pipeline as boot

TERMINAL = {'COMPLETED', 'CANCELED', 'ERROR', 'DELETED'}
CANCELED_COMPARISON = '6acc18b0095c5780893255c7'
DOCUMENT_KEYS = {'model_lock', 'runtime_lock', 'approval', 'country_evidence',
                 'license_approval_evidence', 'model_receipt', 'runtime_manifest'}


def number(value, low, high, name):
    boot.require(type(value) in (float, int) and math.isfinite(value) and low <= value <= high, name)
    return value


def repo(value):
    boot.require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', value),
                 'explicit repository ID required')
    return value


def revision(value):
    boot.require(isinstance(value, str) and re.fullmatch('[0-9a-f]{40}', value), 'full commit required')
    return value


def inspect_spec(spec, root):
    """Offline gates; no service call and no invented runtime/territory evidence."""
    gates = []
    def check(name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            gates.append(name)
    def documents():
        boot.require(set(spec['documents']) == DOCUMENT_KEYS, 'exact document set required')
        for entry in spec['documents'].values():
            boot.pinned(root, entry)
    check('independently pinned documents/model receipt', documents)
    def runtime():
        r = spec['runtime_bootstrap']
        repo(r['repo']); revision(r['revision'])
        boot.relative(r['manifest']['path']); boot.relative(r['entrypoint']['path'])
        for entry in (r['manifest'], r['entrypoint']):
            boot.require(re.fullmatch('[0-9a-f]{64}', entry['sha256']), 'runtime artifact SHA')
        boot.require(re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', r['module']) and
                     r['entrypoint']['path'] == r['module'].replace('.', '/') + '.py', 'runtime module')
        number(r['timeout_seconds'], 1, 1800, 'runtime bootstrap deadline')
        boot.require(type(r['timeout_seconds']) is int, 'integer deadline')
        manifest = boot.load(boot.pinned(root, spec['documents']['runtime_manifest']))
        boot.require(spec['documents']['runtime_manifest']['sha256'] == r['manifest']['sha256'] and
                     manifest['schema'] == 'h3-identity-gpu-runtime-bootstrap-v1' and
                     manifest['entrypoint'] == r['entrypoint'] and manifest['module'] == r['module'] and
                     manifest['image'] == spec['image'] and manifest['gpu_bootstrap_supported'] is True and
                     manifest['runtime_lock_sha256'] == spec['documents']['runtime_lock']['sha256'],
                     'independent GPU bootstrap manifest binding')
        boot.require(manifest['native_source_inventory'] == 'PINS23+LICENSE' and
                     manifest['source_revisions'] == spec['pipeline']['runtime_source_revisions'],
                     'native upstream subset audit required')
        boot.require(re.fullmatch('[0-9a-f]{64}', manifest['python_sha256']) and
                     manifest['bootstrap_files'].get(r['entrypoint']['path']) == r['entrypoint']['sha256'],
                     'independent interpreter/bootstrap code pins required')
        if spec['pipeline'].get('inference',{}).get('acceleration_experiment') is not None:
            required={'research/prepare_h3_fp8_runtime.py',
                      'research/h3_personal/acceleration_experiment.py',
                      'research/h3_personal/incremental_decode.py'}
            boot.require(required<=set(manifest['bootstrap_files']),
                         'experimental branch requires fresh complete runtime closure')
        for name, expected in manifest['bootstrap_files'].items():
            boot.relative(name)
            boot.require(re.fullmatch('[0-9a-f]{64}', expected), 'bootstrap helper SHA required')
    check('independent immutable runtime artifacts/bootstrap interface', runtime)
    def image():
        boot.require(re.fullmatch(r'[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}', spec['image']), 'image digest')
        e = boot.load(boot.pinned(root, spec['image_evidence']))
        boot.require(e['image'] == spec['image'] and e['os'] == 'linux' and
                     e['architecture'] == 'amd64' and e['digest_verified'] is True, 'image platform proof')
    check('Docker Linux/amd64 digest evidence', image)
    def pipeline():
        from research import hf_identity_pipeline as driver
        p = dict(spec['pipeline'])
        p.update(input_documents={k: {'path': str(boot.pinned(root, spec['documents']['model_lock'])),
                                       'sha256': spec['documents']['model_lock']['sha256']}
                                  for k in boot.DOCUMENTS})
        for key in ('model_lock', 'runtime_lock', 'approval'):
            e = spec['documents'][key]
            p[key] = {'path': str(boot.pinned(root, e)), 'sha256': e['sha256']}
        driver.validate(p)
        boot.require(p['overall_seconds'] + spec['runtime_bootstrap']['timeout_seconds'] <= 23400,
                     'bootstrap plus driver <=6.5h')
        receipt = boot.load(boot.pinned(root, spec['documents']['model_receipt']))
        boot.require(receipt['schema'] == 'h3-model-asset-source-receipt-v1' and
                     receipt['repo'] == 'MiniMaxAI/MiniMax-H3' and receipt['revision'] == boot.MODEL_PIN and
                     receipt['artifact_sha256']['model-lock.json'] == spec['documents']['model_lock']['sha256'] and
                     receipt['weights_downloaded'] is False and receipt['jobs_submitted'] is False,
                     'independent model source receipt')
        lock = boot.load(p['model_lock']['path'])
        expected = receipt['expected_files']
        boot.require(set(expected) == set(lock['files']) | set(p['transformer_files']),
                     'independent model/transformer inventory differs')
        for name, entry in expected.items():
            boot.relative(name)
            boot.require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) and
                         entry['method'] in {'revision-api-lfs-sha256', 'verified-git-blob-bytes'},
                         'independent model byte expectations required')
            if name in lock['files']:
                boot.require(entry['sha256'] == lock['files'][name], 'model lock/source receipt differs')
    check('pipeline bounds, model/source pins and independent locks', pipeline)
    def territory():
        from research import hf_identity_encode as encoder
        from research import hf_identity_train as trainer
        evidence = boot.load(boot.pinned(root, spec['documents']['approval']))
        original = boot.load(boot.pinned(root, spec['documents']['approval']))
        for key in ('country_evidence', 'license_approval_evidence'):
            e = spec['documents'][key]
            boot.require(evidence[key]['path'] == '/deployment/' + e['path'] and
                         evidence[key]['sha256'] == e['sha256'], 'explicit mounted proof pin')
            evidence[key]['path'] = str(boot.pinned(root, e))
        encoder.license_gate(evidence)
        t = dict(spec['pipeline']['territory_approval'])
        operator = original.get('evidence_mode') == 'operator_hf_acceptance'
        boot.require((t.get('evidence_kind') == 'operator_hf_acceptance') == operator,
                     'approval evidence modes differ')
        if operator:
            # Acceptance of unknown territory risk, never a location assertion.
            for document, country_key in ((original, 'execution_country'), (t, 'compute_country')):
                boot.require(country_key in document and document[country_key] is None and
                             document.get('provider') == 'huggingface_jobs' and
                             document.get('risk_acknowledged') is True and document.get('approved') is True and
                             isinstance(document.get('approved_by'), str) and document['approved_by'].strip(),
                             'explicit HF operator risk acceptance required')
        boot.require(t['compute_country'] == original['execution_country'] and
                     t['evidence_file'] == original['country_evidence']['path'] and
                     t['evidence_sha256'] == original['country_evidence']['sha256'], 'country evidence differs')
        if not operator:
            boot.require('hosting' not in original['country_evidence']['description'].lower(),
                         'hosting is not compute proof')
        t['evidence_file'] = evidence['country_evidence']['path']
        trainer.territory_gate(t)
    # Keep the established strict-path gate label; neither label certifies location.
    operator_requested = spec.get('pipeline', {}).get('territory_approval', {}).get('evidence_kind') == 'operator_hf_acceptance'
    check('explicit territory/license evidence or HF operator risk acceptance' if operator_requested else
          'explicit existing permitted execution-country/license evidence', territory)
    def inputs():
        boot.require(spec['input_repo'] == boot.INPUT_REPO and spec['input_revision'] == boot.INPUT_REVISION and
                     spec['raw_revision'] == boot.RAW_REVISION and
                     spec['original_revision'] == boot.ORIGINAL_REVISION, 'preserve immutable inputs')
        boot.require(set(spec['input_document_sha256']) == set(boot.DOCUMENTS), 'input document pins')
        for value in spec['input_document_sha256'].values():
            boot.require(re.fullmatch('[0-9a-f]{64}', value), 'sensitive document byte pin')
        boot.require(spec['input_document_sha256']['selection_path'] == boot.FREEZE_SHA, 'selection changed')
        for key in ('output_repo', 'cache_repo'):
            repo(spec['pipeline'][key])
        boot.require(spec['pipeline']['cache_repo'] != spec['pipeline']['output_repo'], 'separate private repos')
    check('completed raw/all715/mapping input pins and output repositories', inputs)
    return gates


def stage(spec_path, target):
    spec = boot.load(spec_path)
    boot.require(spec.get('schema') == 'h3-identity-launch-spec-v1', 'launch spec schema')
    boot.require(set(spec) <= {'schema', 'documents', 'image_evidence', 'image', 'pipeline',
                 'runtime_bootstrap', 'input_repo', 'input_revision', 'raw_revision',
                 'original_revision', 'input_document_sha256'}, 'unknown spec fields (no secrets)')
    boot.require(set(spec.get('pipeline', {})) <= {'schema', 'overall_seconds', 'rate_usd_hour',
                 'spent_and_reserved_usd', 'output_repo', 'cache_repo', 'flavor', 'max_updates',
                 'stage_seconds', 'source_revision', 'runtime_source_revisions', 'model_repo',
                 'transformer_files', 'runtime_expected_versions', 'runtime_provenance',
                 'territory_approval', 'checkpoint_interval', 'cache_reuse', 'inference', 'sampling_mode',
                 'checkpoint_escrow', 'warm_start_remote', 'runtime_overlay', 'preview_only', 'model_mount', 'operator_managed_stages',
                 'hf_price_precision_approval'}, 'unknown pipeline fields (no secrets)')
    def no_secrets(value):
        if isinstance(value, dict):
            for key, item in value.items():
                boot.require(key.lower() not in {'hf_token', 'token', 'secrets', 'password',
                             'authorization', 'access_token'}, 'secrets are not deployment fields')
                no_secrets(item)
        elif isinstance(value, list):
            for item in value:
                no_secrets(item)
    no_secrets(spec)
    target = Path(target)
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    documents = {}
    # Only explicitly supplied small, hash-pinned JSON/evidence; no weights or raw data.
    supplied = dict(spec.get('documents', {}))
    if spec.get('image_evidence'):
        supplied['image_evidence'] = spec['image_evidence']
    for key, entry in supplied.items():
        boot.require(key in DOCUMENT_KEYS | {'image_evidence'}, 'document allowlist')
        source = Path(entry['path']).expanduser()
        boot.require(source.is_file() and not source.is_symlink() and source.stat().st_size <= 4 * 1024**2 and
                     boot.sha256(source) == entry['sha256'], 'small pinned document required')
        destination = target / 'documents' / (key + '.json')
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        destination.write_bytes(source.read_bytes()); destination.chmod(0o600)
        documents[key] = {'path': 'documents/' + key + '.json', 'sha256': entry['sha256']}
    deployment = dict(spec, schema='h3-identity-deployment-v1', documents={
                      k: v for k, v in documents.items() if k != 'image_evidence'})
    if 'image_evidence' in documents:
        deployment['image_evidence'] = documents['image_evidence']
    boot.dump(target / 'deployment.json', deployment)
    bootstrap = Path(boot.__file__)
    (target / 'hf_bootstrap_identity_pipeline.py').write_bytes(bootstrap.read_bytes())
    (target / 'hf_bootstrap_identity_pipeline.py').chmod(0o600)
    allowlist = {str(p.relative_to(target)): boot.sha256(p) for p in target.rglob('*') if p.is_file()}
    boot.dump(target / 'upload-allowlist.json', allowlist)  # local only, not recursively uploaded
    report = {'state': 'prepared_not_submitted', 'plan_only': True,
              'remaining_gates': inspect_spec(deployment, target) +
                ['immutable deployment upload revision (by main, explicit allowlist)',
                 'live account-wide budget/jobs and private content-write checks'],
              'upload_files': sorted(allowlist), 'stage_dir': str(target),
              'maximum_job_seconds': 23400, 'automatic_retry': False, 'c_gate_enabled': False}
    boot.dump(target / 'plan.json', report)
    return report


def hardware_rate(api, flavor='h200', price_approval=None):
    boot.require(flavor in ('h200','h200x2'), 'explicit H200 comparison hardware only')
    h = next(h for h in api.list_jobs_hardware() if h.name == flavor)
    multiplier = {'hour': 1, 'minute': 60}.get(h.unit_label)
    boot.require(multiplier is not None and bool(h.accelerator), 'H200 price unit/accelerator unknown')
    limit=10
    if price_approval is not None:
        from research.h3_personal.compute_price_approval import validate
        boot.require(flavor=='h200x2','precision exception is only two H200')
        limit=validate(price_approval)
    from decimal import Decimal
    hourly=float(Decimal(str(h.unit_cost_usd))*multiplier)
    return number(hourly, .001, limit, 'H200 approved hourly cap')


def budget(ledger, jobs, rate, seconds, *, price_approval=None):
    """Never release reservations, even for terminal/unknown submissions without reconciliation."""
    from research.h3_personal.execution_budget import approval,cap
    pair=approval(price_approval);cumulative_cap=cap(price_approval)
    if pair:
        boot.require(ledger.get('approved_parallel_pair')==pair and ledger.get('cumulative_cap_usd')==170,
                     'ledger must explicitly bind approved pair and cumulative authority')
        prior=[v for v in ledger.get('reservations',{}).values() if v.get('pair_id')==pair['id']
               and not v.get('state','').startswith('terminal_')]
        boot.require(len(prior)<2 and all(v.get('scope')!=price_approval['scope'] for v in prior),
                     'at most one active corrective and one active experimental Job')
        boot.require(seconds<=pair['max_timeout_seconds'],'pair finite75minute cap')
    total = number(ledger.get('spent_usd'), 0, cumulative_cap, 'reviewed spent amount')
    reservations = ledger.get('reservations')
    boot.require(isinstance(reservations, dict), 'reviewed ledger reservations')
    indexed = {}
    cpu_reserve = False
    for v in reservations.values():
        total += number(v['upper_bound_usd'], 0, 150, 'reservation amount')
        if v.get('flavor') == 'cpu-upgrade' and v['upper_bound_usd'] >= .09:
            cpu_reserve = True
        if v.get('job_id'):
            boot.require(v['job_id'] not in indexed, 'duplicate ledger job')
            indexed[v['job_id']] = v
    boot.require(cpu_reserve, 'existing $0.09 CPU reserve must remain')
    concurrent = rate
    for job in jobs:
        status = getattr(getattr(job, 'status', None), 'stage', None)
        if status not in TERMINAL:
            # Unknown jobs/prices/timeouts cannot be optimistically excluded.
            v = indexed.get(job.id)
            boot.require(v is not None, 'unaccounted nonterminal job')
            other_rate = number(v.get('rate_usd_hour'), .001, 10.00002 if pair else 10, 'active job rate')
            if pair:
                boot.require(v.get('pair_id')==pair['id'] and v.get('flavor')=='h200x2'
                             and v.get('scope') in ('h3-corrected-native-continuation-v1','h3-fp8-streaming-experiment-v1')
                             and other_rate==10.00002, 'no unrelated active allocation in approved pair')
            cap = number(v.get('timeout_seconds'), 1, 23400, 'active job finite timeout')
            boot.require(v['upper_bound_usd'] >= other_rate * cap / 3600,
                         'nonterminal job underreserved')
            concurrent += other_rate
    limit=10
    if price_approval is not None:
        from research.h3_personal.compute_price_approval import validate,MAX_SECONDS,MAX_USD
        limit=validate(price_approval)
        boot.require(rate==limit and seconds<=MAX_SECONDS and rate*seconds/3600<=MAX_USD,
                     'precision exception exact rate/finite total cap')
        if pair:limit=pair['max_concurrent_usd_hour']
        else:
            boot.require(all(getattr(j.status,'stage',None) in TERMINAL for j in jobs),
                         'precision exception requires all other jobs terminal')
    boot.require(concurrent <= limit, 'concurrent compute exceeds approved hourly cap')
    upper = rate * seconds / 3600
    boot.require(total + upper <= cumulative_cap, 'cumulative budget exceeds explicit authority')
    return total, upper


def verify_uploaded(api, repo_id, commit, stage_dir, allowlist):
    info = api.repo_info(repo_id, repo_type='dataset', revision=commit, files_metadata=True)
    boot.require(info.private is True and info.sha == commit, 'existing private immutable dataset required')
    files = {s.rfilename: s for s in info.siblings}
    for name, sha in allowlist.items():
        p = boot.inside(stage_dir, name)
        boot.require(boot.sha256(p) == sha and name in files, 'deployment upload/local bytes differ')
        s = files[name]
        boot.require(s.size == p.stat().st_size, 'uploaded size differs')
        if s.lfs:
            boot.require(s.lfs.sha256 == sha, 'uploaded LFS hash differs')
        else:
            git = hashlib.sha1(('blob ' + str(s.size) + '\0').encode() + p.read_bytes()).hexdigest()
            boot.require(s.blob_id == git, 'uploaded Git hash differs')


def verify_runtime_artifacts(api, spec, root):
    runtime = spec['runtime_bootstrap']
    info = api.repo_info(runtime['repo'], repo_type='dataset', revision=runtime['revision'], files_metadata=True)
    boot.require(info.private is True and info.sha == runtime['revision'], 'immutable runtime artifacts required')
    files = {s.rfilename: s for s in info.siblings}
    manifest_path = boot.pinned(root, spec['documents']['runtime_manifest'])
    manifest = boot.load(manifest_path)
    name = runtime['manifest']['path']
    boot.require(name in files, 'runtime manifest missing from immutable mount')
    item = files[name]
    boot.require(item.size == manifest_path.stat().st_size, 'runtime manifest size differs')
    if item.lfs:
        boot.require(item.lfs.sha256 == runtime['manifest']['sha256'], 'runtime manifest LFS bytes differ')
    else:
        git = hashlib.sha1(('blob ' + str(item.size) + '\0').encode() + manifest_path.read_bytes()).hexdigest()
        boot.require(item.blob_id == git, 'runtime manifest Git bytes differ')
    for name, sha in manifest['bootstrap_files'].items():
        boot.require(name in files and 0 < files[name].size <= 4 * 1024**2,
                     'runtime bootstrap source missing/oversized')
        if files[name].lfs:
            boot.require(files[name].lfs.sha256 == sha, 'runtime bootstrap LFS bytes differ')
        # Git SHA256 is verified against actual bytes on the immutable remote mount.
        # Do not pretend a Git SHA1 proves the independently supplied SHA256.


def submit(target, ledger_path, deployment_repo, deployment_revision, api, volume_class, token):
    target, ledger_path = Path(target), Path(ledger_path)
    allowlist = boot.load(target / 'upload-allowlist.json')
    boot.require(set(allowlist) == {'deployment.json', 'hf_bootstrap_identity_pipeline.py'} |
                 {'documents/' + k + '.json' for k in DOCUMENT_KEYS | {'image_evidence'}}, 'upload allowlist differs')
    for name, sha in allowlist.items():
        boot.pinned(target, {'path': name, 'sha256': sha})
    spec = boot.load(target / 'deployment.json')
    boot.require(not inspect_spec(spec, target), 'deployment gates unresolved')
    repo(deployment_repo); revision(deployment_revision)
    verify_uploaded(api, deployment_repo, deployment_revision, target, allowlist)
    verify_runtime_artifacts(api, spec, target)
    who = api.whoami()
    owner = who['name']
    boot.require(who.get('isPro') is True, 'Jobs-enabled PRO account required')
    for name, kind in ((spec['pipeline']['output_repo'], 'model'), (spec['pipeline']['cache_repo'], 'dataset')):
        boot.require(api.repo_info(name, repo_type=kind).private is True, 'existing private output/cache required')
        api.auth_check(name, repo_type=kind, write=True)
    for name, commit in ((boot.INPUT_REPO, boot.INPUT_REVISION),
                         (spec['runtime_bootstrap']['repo'], spec['runtime_bootstrap']['revision'])):
        info = api.repo_info(name, repo_type='dataset', revision=commit)
        boot.require(info.private is True and info.sha == commit, 'private immutable runtime/input repo required')
    boot.require(bool(token), 'secret token required')
    flavor=spec['pipeline'].get('flavor','h200')
    proof=spec['pipeline'].get('hf_price_precision_approval')
    rate = hardware_rate(api) if flavor=='h200' else hardware_rate(api,flavor,proof)
    seconds = spec['pipeline']['overall_seconds'] + spec['runtime_bootstrap']['timeout_seconds']
    if proof is not None:
        from research.h3_personal.compute_price_approval import MAX_SECONDS,MAX_USD
        boot.require(seconds<=MAX_SECONDS and rate*seconds/3600<=MAX_USD,
                     'precision exception finite total timeout/budget cap')
    number(seconds, 1202, 23400, 'total job <=6.5h')
    ident = 'identity-' + allowlist['deployment.json']
    model_volume = None
    if 'model_mount' in spec['pipeline']:
        m = spec['pipeline']['model_mount']
        info = api.model_info(m['repo'], revision=m['revision'], files_metadata=True)
        boot.require(info.private is True and info.sha == m['revision'], 'private pinned model volume')
        expected = boot.load(boot.pinned(target, spec['documents']['model_receipt']))['expected_files']
        assets = {s.rfilename.removeprefix('assets/'): s for s in info.siblings
                  if s.rfilename.startswith('assets/')}
        boot.require(set(assets) == set(expected) and any(s.rfilename == 'CURATED.json' for s in info.siblings),
                     'complete selected model mount required before GPU reservation')
        original = api.model_info('MiniMaxAI/MiniMax-H3', revision=boot.MODEL_PIN, files_metadata=True)
        boot.require(original.sha == boot.MODEL_PIN, 'original model revision differs')
        source = {s.rfilename:s for s in original.siblings}
        for name, replica in assets.items():
            item = source[name]
            boot.require(replica.size == item.size and
                         (replica.lfs.sha256 == item.lfs.sha256 == expected[name]['sha256']
                          if replica.lfs and item.lfs else
                          not replica.lfs and not item.lfs and replica.blob_id == item.blob_id),
                         'model replica differs before GPU reservation')
        model_volume = volume_class(type='model', source=m['repo'], revision=m['revision'],
                                    mount_path=m['path'], read_only=True, path='assets')
    inference=spec['pipeline'].get('inference',{})
    native=inference.get('native_continuation')
    turbo=inference.get('context_parallel_matrix') or (native['turbo'] if native else None)
    if turbo:
        m=turbo
        info=api.model_info(m['repo'],revision=m['revision'],files_metadata=True)
        item=next(s for s in info.siblings if s.rfilename==m['file'])
        boot.require(info.sha==m['revision'] and item.size==m['bytes'] and item.lfs and
                     item.lfs.sha256==m['sha256'],'immutable selected Turbo payload inventory differs')
    if native:
        from research.h3_personal.continuation_contract import validate as check_native
        check_native(native)
        a=native['audio'];info=api.repo_info(a['repo'],repo_type='dataset',revision=a['revision'],files_metadata=True)
        item=next(s for s in info.siblings if s.rfilename==a['file'])
        boot.require(info.private is True and info.sha==a['revision'] and item.size==a['bytes'],
                     'private immutable recorded-audio volume required')
        from huggingface_hub import hf_hub_download
        audio=hf_hub_download(a['repo'],a['file'],repo_type='dataset',revision=a['revision'],token=token)
        boot.require(boot.sha256(audio)==a['sha256'],'original recorded WAV bytes differ')
    # Local account-wide lock; use the same ledger as CPU/other launchers.
    with ledger_path.with_suffix(ledger_path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = boot.load(ledger_path)
        marker = target / 'submission_guard.json'
        boot.require(not marker.exists() and ident not in ledger['reservations'], 'no retry/duplicate submission')
        jobs = list(api.list_jobs(namespace=owner))  # no status filter: all account jobs
        for j in jobs:
            labels = getattr(j, 'labels', {}) or {}
            boot.require(labels.get('identity_deploy') != allowlist['deployment.json'][:32], 'deployment already submitted')
            boot.require(not (j.id == CANCELED_COMPARISON and getattr(j.status, 'stage', None) not in TERMINAL),
                         'canceled comparison must never restart')
        spent, upper = (budget(ledger,jobs,rate,seconds) if proof is None else
                        budget(ledger,jobs,rate,seconds,price_approval=proof))
        ledger['reservations'][ident] = dict(upper_bound_usd=upper, state='reserved_unknown_no_retry',
            flavor=flavor, rate_usd_hour=rate, timeout_seconds=seconds,
            deployment_repo=deployment_repo, deployment_revision=deployment_revision)
        from research.h3_personal.execution_budget import approval
        pair=approval(proof)
        if pair:
            ledger['reservations'][ident].update(pair_id=pair['id'],scope=proof['scope'])
        # Durable reservation BEFORE the sole API call. Any outcome retains it.
        boot.dump(ledger_path, ledger)
        with marker.open('x') as f:
            import json
            json.dump({'state': 'reserved_unknown_no_retry', 'reservation': ident}, f)
            f.flush(); os.fsync(f.fileno())
        source_sha = allowlist['hf_bootstrap_identity_pipeline.py']
        script = ("import hashlib,runpy,sys; p='/deployment/hf_bootstrap_identity_pipeline.py'; "
                  f"assert hashlib.sha256(open(p,'rb').read()).hexdigest() == '{source_sha}'; "
                  "sys.argv=[p]+sys.argv[1:]; runpy.run_path(p,run_name='__main__')")
        command = ['python3', '-c', script, '--deployment-sha256', allowlist['deployment.json'],
                   '--rate', str(rate), '--spent-and-reserved', str(spent)]
        mounts = [(deployment_repo, deployment_revision, '/deployment'),
                  (boot.INPUT_REPO, boot.INPUT_REVISION, '/inputs'),
                  (spec['runtime_bootstrap']['repo'], spec['runtime_bootstrap']['revision'], '/runtime-inputs')]
        volumes = [volume_class(type='dataset', source=r, revision=c, mount_path=p, read_only=True)
                   for r, c, p in mounts]
        if model_volume is not None:
            volumes.append(model_volume)
        if turbo:
            m=turbo
            volumes.append(volume_class(type='model',source=m['repo'],revision=m['revision'],
                                        mount_path='/mounted-h3-turbo',read_only=True))
        if native:
            a=native['audio']
            volumes.append(volume_class(type='dataset',source=a['repo'],revision=a['revision'],
                                        mount_path='/mounted-h3-native-audio',read_only=True))
        job = api.run_job(image=spec['image'], command=command, flavor=flavor, timeout=seconds,
            namespace=owner, secrets={'HF_TOKEN': token}, env={'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
            volumes=volumes, labels={'project': 'h3-identity', 'identity_deploy': allowlist['deployment.json'][:32]})
        boot.require(isinstance(job.id, str) and job.id, 'submission outcome unknown')
        ledger['reservations'][ident].update(state='submitted_no_retry', job_id=job.id)
        boot.dump(ledger_path, ledger)
        report = dict(state='submitted', job_id=job.id, upper_bound_usd=upper,
                      rate_usd_hour=rate, timeout_seconds=seconds, automatic_retry=False)
        boot.dump(marker, report)
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path)
    p.add_argument('--stage-dir', type=Path, required=True)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--budget-ledger', type=Path)
    p.add_argument('--deployment-repo')
    p.add_argument('--deployment-revision')
    a = p.parse_args(argv)
    os.umask(0o077)
    if not a.stage_dir.exists():
        if not a.spec:
            p.error('new stage needs --spec')
        report = stage(a.spec, a.stage_dir)
    else:
        report = boot.load(a.stage_dir / 'plan.json')
    if a.submit:
        if not all((a.budget_ledger, a.deployment_repo, a.deployment_revision)):
            p.error('--submit requires ledger, existing deployment repo and full upload commit')
        from huggingface_hub import HfApi, Volume
        report = submit(a.stage_dir, a.budget_ledger, a.deployment_repo, a.deployment_revision,
                        HfApi(), Volume, os.environ.get('HF_TOKEN'))
    import json
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        import json
        print(json.dumps({'state': 'failed_closed_no_retry', 'failure_type': type(exc).__name__}))
        raise SystemExit(2) from None
