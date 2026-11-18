"""Concrete Modal cache-reuse bootstrap. No submission, fallback or retry loop.

Entrypoint consumes /deployment/BOOTSTRAP_REQUEST.json from a reviewed closure.
Provider-specific approval is mandatory; HF operator acceptance is not portable.
"""
import json
import os
from pathlib import Path
import re
import sys
import time
from research import hf_bootstrap_identity_pipeline as wire
from research import hf_identity_pipeline as pipeline
from research.modal_identity_worker import RATE
from research.modal_gpu_plan import hourly, validate_placement

_PHASE = 'validate'
STATUS_PATH = Path('/tmp/h3-bootstrap-status.json')


def status(phase, failure=None):
    global _PHASE
    _PHASE = phase
    print('H3_PHASE', phase, flush=True)
    wire.dump(STATUS_PATH, {'schema': 'h3-modal-bootstrap-status-v1', 'stage': phase,
                           'failure_type': failure, 'quality_verified': False})


FIELDS = {'schema', 'closure', 'runtime_manifest', 'input_repo', 'input_revision',
          'input_document_sha256', 'pipeline', 'runtime_seconds'}


def validate(root, request):
    root = Path(root).absolute()
    wire.require(set(request) - {'placement'} == FIELDS and request['schema'] == 'h3-modal-bootstrap-v1',
                 'literal Modal bootstrap request required')
    wire.require(request['input_repo'] == wire.INPUT_REPO and request['input_revision'] == wire.INPUT_REVISION,
                 'frozen private inputs required')
    wire.require(set(request['input_document_sha256']) == set(wire.DOCUMENTS) and
                 all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                     for v in request['input_document_sha256'].values()), 'complete input document pins')
    wire.require(type(request['runtime_seconds']) is int and 1 <= request['runtime_seconds'] <= 1800,
                 'bounded pristine runtime restoration required')
    closure = wire.load(wire.pinned(root, request['closure']))
    wire.require(closure['schema'] == 'h3-identity-redesign-closure-v1' and
                 closure['deployed'] is False and closure['actual_h3_gpu_verified'] is False,
                 'reviewed source closure required')
    for name, digest in closure['files'].items():
        wire.pinned(root, {'path': name, 'sha256': digest})
    manifest = wire.load(wire.pinned(root, request['runtime_manifest']))
    wire.require(request['runtime_manifest']['sha256'] == closure['runtime_manifest_sha256'] and
                 manifest['schema'] == 'h3-identity-gpu-runtime-bootstrap-v1', 'pristine runtime binding')
    r = dict(request['pipeline'])
    pipeline.trainer.finite_number(r.get('overall_seconds'), 'pipeline seconds', 1201, 21600)
    wire.require(r.get('rate_usd_hour') == hourly('H200', request.get('placement', {}).get('region')),
                 'reviewed regional resource rate required')
    # These checks precede even private input/model metadata retrieval.
    approval = r.get('territory_approval', {})
    wire.require(approval.get('provider') == 'modal' and
                 approval.get('evidence_mode') != 'operator_hf_acceptance',
                 'existing HF acceptance cannot authorize Modal execution')
    for key in ('model_lock', 'runtime_lock', 'approval'):
        value = r[key]
        r[key] = {'path': str(wire.pinned(root, value)), 'sha256': value['sha256']}
    evidence = wire.load(r['approval']['path'])
    wire.require(evidence.get('provider') == 'modal', 'Modal-bound approval evidence required')
    if 'placement' in request:
        placement = request['placement']
        pricing = wire.load(wire.pinned(root, placement['pricing_evidence']))
        validate_placement(placement, pricing)
        wire.require(evidence.get('execution_country') == placement['country'] == approval.get('compute_country'),
                     'requested placement and country approvals differ')
    pipeline.approvals(r)
    wire.require(r.get('checkpoint_escrow') is True and 'cache_reuse' in r,
                 'production requires recoverable checkpoints and completed cache reuse')
    wire.require(r.get('inference', {}).get('sample_ids') and
                 1 <= len(r['inference']['sample_ids']) <= 2, 'same-Job preview cases required')
    if 'runtime_overlay' in r:
        wire.require(r['runtime_overlay'] == {
            'stage': str(root / 'routing-overlay'), 'stage_sha256': closure['overlay_manifest_sha256'],
            'base_diffusers': '/work/runtime/diffusers-source/src/diffusers',
            'base_inventory_sha256': closure['base_inventory_sha256']}, 'active source overlay closure binding')
    r.update(source_root=str(root / manifest['source_root']), output='/work/identity')
    return r, manifest


def execute(root='/deployment'):
    root = Path(root)
    os.umask(0o077)
    status('validate')
    started = float(os.environ.get('H3_BOOTSTRAP_STARTED', time.monotonic()))
    request = wire.load(root / 'BOOTSTRAP_REQUEST.json')
    r, manifest = validate(root, request)
    placement = request.get('placement')
    wire.require(placement is not None and json.loads(os.environ.get('H3_REQUESTED_PLACEMENT', '{}')) == placement,
                 'bootstrap placement must equal bound carrier placement')
    carrier_seconds = int(os.environ.get('H3_CARRIER_SECONDS', '0'))
    wire.require(request['runtime_seconds'] + r['overall_seconds'] <= carrier_seconds - 20,
                 'bootstrap stages exceed enclosing carrier deadline')
    end = started + request['runtime_seconds'] + r['overall_seconds']
    os.umask(0o077)
    work = Path('/work')
    receipt = work / 'runtime-ready.json'
    environment = dict(os.environ, PYTHONPATH=str(root), H3_RUNTIME_IMAGE=manifest['image'],
                       H3_NATIVE_SOURCE_ROOT=r['source_root'])
    if os.environ.get('H3_RUNTIME_READY') != str(receipt):
        work.mkdir(mode=0o700, exist_ok=False)
        status('runtime')
        wire.bounded_command([sys.executable, '-m', manifest['module'], '--manifest',
                              str(root / request['runtime_manifest']['path']), '--manifest-sha256',
                              request['runtime_manifest']['sha256'], '--output', '/work/runtime',
                              '--receipt', str(receipt)], environment,
                             min(request['runtime_seconds'], max(1, int(end - time.monotonic()))))
        ready = wire.load(receipt)
        # No Hub imports in the stock PyTorch image. Re-exec in the verified venv.
        environment.update(H3_RUNTIME_READY=str(receipt), H3_BOOTSTRAP_STARTED=str(started))
        os.execve(ready['python'], [ready['python'], '-m', 'research.modal_bootstrap_identity'], environment)
    ready = wire.load(receipt)
    wire.require(ready.get('schema') == 'h3-identity-runtime-ready-v1' and
                 ready.get('gpu_imports_verified') is True and
                 sys.executable == ready['python'] and
                 Path(ready['python']).is_relative_to(work / 'runtime') and
                 ready['manifest_sha256'] == request['runtime_manifest']['sha256'] and
                 ready['runtime_lock_sha256'] == r['runtime_lock']['sha256'] and
                 ready['python_sha256'] == manifest['python_sha256'] == wire.sha256(ready['python']),
                 'independent restored interpreter/lock receipt')
    inputs = work / 'inputs'
    inputs.mkdir(mode=0o700)
    status('inputs')
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.repo_info(request['input_repo'], repo_type='dataset', revision=request['input_revision'])
    wire.require(info.private is True and info.sha == request['input_revision'], 'immutable private input repo')
    pipeline.private_repositories(r, api)
    r['input_documents'] = {}
    # Only six small JSON documents: no original videos, processed clips or audio.
    for key, name in wire.DOCUMENTS.items():
        path = hf_hub_download(request['input_repo'], name, repo_type='dataset',
                               revision=request['input_revision'], local_dir=str(inputs))
        digest = request['input_document_sha256'][key]
        wire.require(wire.sha256(path) == digest, 'input document byte mismatch')
        r['input_documents'][key] = {'path': path, 'sha256': digest}
    summary = wire.load(r['input_documents']['summary_path']['path'])
    wire.require(summary['input_revision'] == wire.ORIGINAL_REVISION and
                 summary['complete_raw_preprocessing'] is True and summary['processed_clips'] == 715,
                 'completed frozen preprocessing receipt')
    pipeline.validate(r)  # Reject invalid recipe/document/preview pins before runtime restoration.
    # Preview mux needs only the one/two selected short PCM files, remote only.
    media_root = inputs / 'raw' / wire.DATASET_SHA
    r['production_inputs'] = {'media_root': str(media_root)}
    raw = wire.load(r['input_documents']['manifest_path']['path'])
    by_id = {sample['id']: sample for sample in raw['samples']}
    if 'prompt' in r['inference']:
        mapping = wire.load(r['input_documents']['reference_mapping_path']['path'])['samples']
        r['production_inputs']['reference_images'] = {}
        for sid in r['inference']['sample_ids']:
            ref = mapping[sid]
            name = 'production-inputs/' + wire.relative(ref['path'])
            path = hf_hub_download(request['input_repo'], name, repo_type='dataset',
                                   revision=request['input_revision'], local_dir=str(inputs))
            wire.require(wire.sha256(path) == ref['sha256'], 'new preview reference image pin')
            r['production_inputs']['reference_images'][sid] = {'path': path, 'sha256': ref['sha256']}
    for sample_id in r['inference']['sample_ids']:
        sample = by_id[sample_id]
        name = 'raw/' + wire.DATASET_SHA + '/' + wire.relative(sample['audio'])
        path = hf_hub_download(request['input_repo'], name, repo_type='dataset',
                               revision=request['input_revision'], local_dir=str(inputs))
        wire.require(Path(path).stat().st_size <= 4 * 1024**2 and
                     wire.sha256(path) == sample['audio_sha256'], 'short pinned preview PCM required')
    remaining = min(r['overall_seconds'], int(end - time.monotonic()) - 120)
    reserve = (600 if r.get('preview_only', False) else 1200) + r['inference']['seconds']
    wire.require(remaining > reserve, 'preparation and final checks reserve exhausted')
    r['overall_seconds'] = remaining
    pipeline.validate(r)
    path = work / 'pipeline-request.json'
    wire.dump(path, r)
    status('pipeline')
    try:
        wire.bounded_command([ready['python'], '-m', 'research.hf_identity_pipeline', '--request', str(path),
                              '--request-sha256', wire.sha256(path), '--execute-remote'], environment,
                             remaining)
    finally:
        # Publication remains inside the enclosing finite carrier timeout; no retry.
        from research.modal_identity_receipt import publish
        from huggingface_hub import CommitOperationAdd
        report_path = Path(r['output']) / 'pipeline-report.json'
        report = wire.load(report_path) if report_path.exists() else {'stage': 'pipeline', 'readiness': False}
        references = {}
        train_request = Path(r['output']) / 'train-request.json'
        if train_request.exists():
            references['run_id'] = wire.load(train_request)['run_id']
        escrow = Path(r['output']) / 'training/latest-escrow.json'
        if escrow.exists():
            saved = wire.load(escrow)
            references.update(escrow_revision=saved['revision'], escrow_prefix=saved['prefix'],
                              escrow_exported_only=True)
        publish(api, repo=r['output_repo'], request_sha256=wire.sha256(path), report=report,
                references=references,
                placement={k: placement[k] for k in ('cloud', 'region', 'country')}, operation=CommitOperationAdd)
    status('complete')


if __name__ == '__main__':
    try:
        execute()
    except (Exception, KeyboardInterrupt) as exc:
        status(_PHASE, type(exc).__name__)
        print(json.dumps({'complete': False, 'failure_type': type(exc).__name__, 'quality_verified': False}))
        raise SystemExit(2) from None
