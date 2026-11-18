#!/usr/bin/env python3
"""Single remote H200 controller; default is an offline plan check, never submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from research import hf_identity_encode as encoder
from research import hf_identity_train as trainer
from research.h3_personal import production_inputs
from research.h3_personal.model_verification import hash_asset, encoder_inventory


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return trainer.sha256(path)


def write_json(path, value):
    encoder.dump(path, value)


def pinned(path, sha):
    return encoder.pinned_document(path, sha)


def validate(r):
    require(r.get('schema') == 'h3-identity-pipeline-v1', 'pipeline schema')
    if 'operator_managed_stages' in r:
        require(type(r['operator_managed_stages']) is bool and
                (not r['operator_managed_stages'] or r.get('preview_only') is True),
                'operator-managed stages are inference-only')
    if 'model_mount' in r:
        mount = r['model_mount']
        require(isinstance(mount, dict) and set(mount) == {'repo', 'revision', 'path'}, 'model mount fields')
        require(re.fullmatch(r'YUGOROU/[A-Za-z0-9_.-]+', mount['repo']) and
                re.fullmatch('[0-9a-f]{40}', mount['revision']) and mount['path'] == '/mounted-h3-assets',
                'private model mount pin/path required')
    if 'preview_only' in r:
        require(type(r['preview_only']) is bool, 'literal preview-only flag')
        if r['preview_only']:
            require('warm_start_remote' in r and 'inference' in r and 'cache_reuse' in r,
                    'preview-only requires pinned checkpoint/cache/cases')
    from research.modal_gpu_plan import authorized_rate_limit
    rate_limit = authorized_rate_limit(r)
    from research.h3_personal.execution_budget import cap
    cumulative_cap=cap(r.get('hf_price_precision_approval'))
    for key, low, high in (('overall_seconds', 1201, 23400),
                           ('rate_usd_hour', .001, rate_limit), ('spent_and_reserved_usd', 0, cumulative_cap)):
        trainer.finite_number(r.get(key), key, low, high)
    require(r['spent_and_reserved_usd'] + r['rate_usd_hour'] * r['overall_seconds'] / 3600 <= cumulative_cap,
            'cumulative budget exceeded')
    for key in ('output_repo', 'cache_repo'):
        repo = r.get(key)
        require(isinstance(repo, str) and len(repo.split('/')) == 2 and
                all(part and part not in ('.', '..') and
                    all(c.isascii() and (c.isalnum() or c in '-_.') for c in part)
                    for part in repo.split('/')), 'explicit private ' + key + ' required')
    require(r['cache_repo'] != r['output_repo'], 'independent cache dataset repo required')
    matrix=r.get('inference',{}).get('context_parallel_matrix')
    experiment=r.get('inference',{}).get('acceleration_experiment')
    if experiment is not None:
        from research.h3_personal.acceleration_experiment import validate as validate_experiment
        validate_experiment(experiment)
        require(r.get('inference',{}).get('native_continuation') is not None,
                'acceleration experiment requires explicit native branch')
    native=r.get('inference',{}).get('native_continuation')
    require(not (matrix and native),'matrix and native continuation are separate scopes')
    require(r.get('flavor') == ('h200x2' if matrix or native else 'h200'),
            'two H200 only for explicit matrix/native continuation; otherwise one H200')
    require(type(r.get('max_updates')) is int and r['max_updates'] > 0, 'max_updates required')
    require(type(r.get('stage_seconds')) is int and 1 <= r['stage_seconds'] <= 10500,
            'stage deadline required')
    require(r.get('source_revision') == trainer.helper().SOURCE_PINS['trainer'], 'trainer revision pin')
    require(r.get('runtime_source_revisions') == trainer.helper().SOURCE_PINS,
            'independent runtime source revisions required')
    for name in ('model_lock', 'runtime_lock', 'approval'):
        pinned(r[name]['path'], r[name]['sha256'])
    model = pinned(r['model_lock']['path'], r['model_lock']['sha256'])
    require(model.get('revision') == trainer.MODEL_PIN, 'model revision pin')
    require(model.get('schema') == 'h3-local-model-lock-v1', 'model lock schema')
    require(r['model_repo'] == 'MiniMaxAI/MiniMax-H3', 'model repo pin')
    require(model.get('files') and r.get('transformer_files'), 'explicit model subset required')
    require(all(n.startswith('transformer_ref/') for n in r['transformer_files']), 'Ref2VA subset only')
    runtime = pinned(r['runtime_lock']['path'], r['runtime_lock']['sha256'])
    require(runtime.get('schema') == 'h3-encode-runtime-v1', 'runtime schema')
    # These expectations must be independently supplied from reviewed source/environment,
    # not inferred from whatever happened to be installed on the worker.
    require(r.get('runtime_expected_versions') == runtime.get('versions') and
            bool(r.get('runtime_provenance')), 'independent runtime expectations required')
    require({'torch', 'diffusers', 'transformers', 'safetensors', 'Pillow', 'numpy',
             'accelerate', 'peft', 'huggingface-hub'} <= set(runtime['versions']), 'runtime versions incomplete')
    require(encoder.REQUIRED_MODULES <= set(runtime.get('modules', {})), 'runtime modules incomplete')
    require(set(r['input_documents']) == {'selection_path', 'manifest_path', 'summary_path',
            'source_map_path', 'reference_mapping_path', 'caption_mapping_path'}, 'exact input document map')
    for entry in r['input_documents'].values():
        pinned(entry['path'], entry['sha256'])
    require(r.get('sampling_mode', 'replacement') in ('replacement', 'coverage_first'), 'sampling mode')
    if 'cache_reuse' in r:
        reuse = r['cache_reuse']
        require(reuse['repo'] == r['cache_repo'] and re.fullmatch('[0-9a-f]{40}', reuse['revision'])
                and re.fullmatch(r'components/[0-9a-f]{64}', reuse['prefix'])
                and all(re.fullmatch('[0-9a-f]{64}', reuse[k]) for k in ('index_sha256', 'report_sha256')),
                'immutable completed cache pins required')
    if 'inference' in r:
        require('cache_reuse' in r, 'preview requires validated reusable cache')
        preview = r['inference']
        require(type(preview['seconds']) is int and 600 <= preview['seconds'] <= 3600
                and isinstance(preview['sample_ids'], list) and 1 <= len(preview['sample_ids']) <= 2
                and len(set(preview['sample_ids'])) == len(preview['sample_ids'])
                and preview['schedule_points'] == (9 if native else 50), 'bounded one/two preview cases required')
        if native:
            from research.h3_personal.continuation_contract import validate as check_native
            check_native(native)
            require(r.get('preview_only') is True and len(preview['sample_ids'])==1 and
                    all(k not in preview for k in ('tts','resident_batch','context_parallel_matrix')) and
                    isinstance(preview.get('prompt'),str) and bool(preview['prompt'].strip()),
                    'native continuation is single-case recorded-audio inference-only')
        if 'context_parallel_matrix' in preview:
            matrix=preview['context_parallel_matrix']
            from research.h3_personal.turbo_adapter import TURBO_SHA256
            require(r.get('preview_only') is True and len(preview['sample_ids'])==1 and
                    'tts' not in preview and 'resident_batch' not in preview and
                    matrix==dict(repo='lightx2v/Minimax-h3-Turbo',
                        revision='3ec17a324ced54151364f24f8b5fb6bf7e26414f',
                        path='/mounted-h3-turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',
                        file='minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',
                        bytes=1383677808,sha256=TURBO_SHA256),
                    'explicit fixed Ref2VA Turbo8 four-condition comparison contract required')
        if 'resident_batch' in preview:
            require(type(preview['resident_batch']) is bool and r.get('preview_only') is True and
                    len(preview['sample_ids'])==1 and 'tts' not in preview,
                    'same-condition resident benchmark is one recorded-audio inference case')
        if 'tts' in preview:
            tts=preview['tts'];voice=tts.get('voice',{});vox=tts.get('vox',{})
            require(r.get('preview_only') is True and len(preview['sample_ids'])==1,
                    'generated TTS is single-case inference-only')
            require(set(tts)=={'text','llm_model','llm_response_sha256','voice','vox'} and
                    isinstance(tts['text'],str) and 1<=len(tts['text'].strip())<=240 and
                    tts['llm_model']=='Qwen/Qwen3.8-27B:cerebras' and
                    re.fullmatch('[0-9a-f]{64}',tts['llm_response_sha256']), 'approved LLM response pin')
            require(set(voice)=={'repo','revision','path','sha256'} and
                    voice['repo']=='YUGOROU/h3-personal-20261011-data' and
                    re.fullmatch('[0-9a-f]{40}',voice['revision']) and
                    re.fullmatch(r'inference-inputs/review-[0-9]{8}T[0-9]{6}Z/voice-reference.wav',voice['path']) and
                    voice['sha256']=='37588c1ddc1cae2e804aa3f08af4038cac757205de0a1bc63a1bb7d77ed7a65d',
                    'existing private voice pin required')
            require(set(vox)=={'repo','revision','files'} and vox['repo']=='openbmb/VoxCPM2' and
                    vox['revision']=='32279effe8c19989596f05d353d1447f51d9e915' and
                    isinstance(vox['files'],dict) and len(vox['files'])==9,
                    'explicit pinned Vox preparation inventory')
        if 'prompt' in preview:
            require(isinstance(preview['prompt'], str) and 1 <= len(preview['prompt'].strip()) <= 2000,
                    'bounded nonempty new preview prompt')
        require(type(preview.get('seed', 42)) is int and 0 <= preview.get('seed', 42) < 2**32-2,
                'explicit bounded preview seed')
    return r


def approvals(r):
    evidence = pinned(r['approval']['path'], r['approval']['sha256'])
    encoder.license_gate(evidence)
    approval = r['territory_approval']
    trainer.territory_gate(approval)
    operator = evidence.get('evidence_mode') == 'operator_hf_acceptance'
    require((approval.get('evidence_kind') == 'operator_hf_acceptance') == operator,
            'approval evidence modes differ')
    if operator:
        # Explicit risk acceptance is not provider compute-country proof.
        for document, country_key in ((evidence, 'execution_country'), (approval, 'compute_country')):
            require(country_key in document and document[country_key] is None,
                    'operator acceptance must not claim a compute country')
            require(document.get('provider') == 'huggingface_jobs' and
                    document.get('risk_acknowledged') is True and document.get('approved') is True and
                    isinstance(document.get('approved_by'), str) and document['approved_by'].strip(),
                    'explicit HF operator risk acceptance required')
    require(approval['compute_country'] == evidence['execution_country'], 'country approvals differ')
    require(approval['evidence_sha256'] == evidence['country_evidence']['sha256'] and
            approval['evidence_file'] == evidence['country_evidence']['path'], 'country evidence differs')
    # Hosting or collection membership alone is never compute-location evidence.
    if not operator:
        description = evidence['country_evidence']['description'].lower()
        require('hosting' not in description and 'collection' not in description, 'hosting is not compute proof')


def private_repositories(r, api):
    """Read-only preflight: existing private repos and content-write authorization."""
    for key, kind in (('cache_repo', 'dataset'), ('output_repo', 'model')):
        require(api.repo_info(r[key], repo_type=kind).private is True,
                'already-private ' + kind + ' repository required')
        api.auth_check(r[key], repo_type=kind, write=True)


def model_root(r, out):
    return Path(r['model_mount']['path']) if 'model_mount' in r else out / 'model'


def provision(r, root, api=None, download=None):
    """Remote only. Compare actual bytes to revision-specific API Git/LFS metadata."""
    approvals(r)  # Before even fetching metadata or weights.
    if api is None:
        from huggingface_hub import HfApi, snapshot_download
        api, download = HfApi(), snapshot_download
    private_repositories(r, api)  # Fail closed before model metadata/download.
    lock = pinned(r['model_lock']['path'], r['model_lock']['sha256'])
    names = set(lock['files']) | set(r['transformer_files'])
    for name in names:
        require(not Path(name).is_absolute() and all(p not in ('', '.', '..') for p in name.split('/'))
                and '\\' not in name, 'unsafe asset path')
    info = api.model_info(r['model_repo'], revision=lock['revision'], files_metadata=True)
    require(info.sha == lock['revision'], 'API model revision differs')
    metadata = {s.rfilename: s for s in info.siblings}
    require(names <= set(metadata), 'API subset missing')
    if 'model_mount' in r:
        mount = r['model_mount']
        mounted = api.model_info(mount['repo'], revision=mount['revision'], files_metadata=True)
        require(mounted.private is True and mounted.sha == mount['revision'], 'private pinned model mount')
        served = {s.rfilename.removeprefix('assets/'): s for s in mounted.siblings
                  if s.rfilename.startswith('assets/')}
        require(set(served) == names, 'mounted repository working set differs')
        for name in names:
            original, replica = metadata[name], served[name]
            require(replica.size == original.size and
                    (replica.lfs.sha256 == original.lfs.sha256 if original.lfs and replica.lfs
                     else not original.lfs and not replica.lfs and replica.blob_id == original.blob_id),
                    'mounted replica metadata differs from original revision')
        require(root == Path(mount['path']) and root.is_dir(), 'model mount missing; no download fallback')
    else:
        encoder.private_root(root)
        download(repo_id=r['model_repo'], revision=lock['revision'], local_dir=str(root),
                 allow_patterns=sorted(names), max_workers=1)
        # Download bookkeeping is NOT part of the model inventory. Never mutate a mount.
        import shutil
        shutil.rmtree(root / '.cache', ignore_errors=True)
    require(not root.is_symlink(), 'symlink model root')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == names, 'unexpected asset inventory')
    inventory = {}
    verify_started = time.monotonic()
    for index, name in enumerate(sorted(names), 1):
        print(f'MODEL_VERIFY_BEGIN {index}/{len(names)} {name}', flush=True)
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'symlink asset')
        item = metadata[name]
        require(path.stat().st_size == item.size, 'API size mismatch')
        sha, blob = hash_asset(path, git=not bool(item.lfs))
        if item.lfs:
            require(sha == item.lfs.sha256, 'API LFS byte mismatch')
        else:
            require(blob == item.blob_id, 'API Git byte mismatch')
        if name in lock['files']:
            require(sha == lock['files'][name], 'operator model pin differs')
        inventory[name] = sha
        print(f'MODEL_VERIFY_DONE {index}/{len(names)} {name}', flush=True)
    encoder_inventory(inventory, lock)  # Layout/lock checks, without re-reading bytes.
    print('MODEL_VERIFY_SUMMARY', json.dumps(dict(files=len(inventory),
          bytes=sum(metadata[n].size for n in inventory), byte_passes=1,
          seconds=time.monotonic() - verify_started)), flush=True)
    return inventory


def prepare(r, out):
    from research import install_identity_checkpoint as installer
    approvals(r)
    encoder.verify_runtime(r['source_root'], pinned(r['runtime_lock']['path'], r['runtime_lock']['sha256']))
    # plan verifies exact upstream contexts/hashes; installer composes all new hooks.
    installer.install(r['source_root'], out / 'loader')
    files = {str(p.relative_to(out / 'loader')): digest(p)
             for p in (out / 'loader').rglob('*.py')}
    write_json(out / 'loader-lock.json', {'schema': 'h3-composed-loader-lock-v1', 'files': files})
    if 'cache_reuse' in r:
        restore_cache(r, out)
        inventory = provision(r, model_root(r, out))
        write_json(out / 'inventory.json', inventory)
        models = encoder_inventory(inventory, pinned(r['model_lock']['path'], r['model_lock']['sha256']))
        expected = encoder.identities(models, pinned(r['runtime_lock']['path'], r['runtime_lock']['sha256']))
        index = pinned(out / 'components/index.json', r['cache_reuse']['index_sha256'])
        require(all(e['attestation']['encoder_sha256'] == expected[e['kind']]
                    for e in index['components'].values()), 'reused encoder/model/runtime identities differ')
        return
    inputs = dict(r['production_inputs'])
    for key, entry in r['input_documents'].items():
        inputs[key] = entry['path']
    rows, receipt = production_inputs.stage(**inputs)
    require(len(rows) == 715, 'all715 required')
    write_json(out / 'inputs.json', inputs)
    write_json(out / 'input-receipt.json', receipt)
    inventory = provision(r, model_root(r, out))
    write_json(out / 'inventory.json', inventory)
    models = encoder_inventory(inventory, pinned(r['model_lock']['path'], r['model_lock']['sha256']))
    runtime = pinned(r['runtime_lock']['path'], r['runtime_lock']['sha256'])
    write_json(out / 'encoder-identities.json', encoder.identities(models, runtime))


def restore_cache(r, out):
    """Remote-only immutable cache fetch; no encoder calls or source-media copies."""
    from huggingface_hub import HfApi, snapshot_download
    import shutil
    from research.h3_personal import component_cache
    reuse = r['cache_reuse']
    api = HfApi()
    require(api.repo_info(reuse['repo'], repo_type='dataset', revision=reuse['revision']).private is True,
            'private pinned cache required')
    temp = out / 'cache-download'
    snapshot_download(reuse['repo'], repo_type='dataset', revision=reuse['revision'],
                      allow_patterns=[reuse['prefix'] + '/*'], local_dir=str(temp), max_workers=4)
    cache_root = temp / reuse['prefix']
    report = pinned(cache_root / 'report.json', reuse['report_sha256'])
    index = pinned(cache_root / 'index.json', reuse['index_sha256'])
    require(report.get('complete') is True and report.get('stage') == 'merge'
            and report.get('loader_verified') is True and report.get('sample_count') == 715
            and report.get('plan_sha256') == reuse['prefix'].split('/')[1], 'completed cache receipt required')
    frozen = pinned(r['input_documents']['selection_path']['path'], r['input_documents']['selection_path']['sha256'])
    require([(s['id'], s['quality_pool']) for s in index['samples']] ==
            [(s['id'], s['quality_pool']) for s in frozen['samples']], 'fixed715 cache order/pools differ')
    component_cache.validate_index(cache_root, index)
    shutil.move(str(cache_root), str(out / 'components'))
    shutil.rmtree(temp)
    write_json(out / 'cache-reuse-receipt.json', dict(reuse, validated=True, samples=715))
    print('CACHE_REUSED all715 validated', flush=True)


def encoder_command(r, out, stage, seconds, rank=0):
    command = [sys.executable, '-m', 'research.hf_identity_encode', stage,
               '--rank', str(rank), '--deadline-seconds', str(seconds),
               '--snapshot-repo', r['cache_repo']]
    paths = {'inputs': out / 'inputs.json', 'model-lock': Path(r['model_lock']['path']),
             'runtime-lock': Path(r['runtime_lock']['path']), 'approval': Path(r['approval']['path']),
             'loader-lock': out / 'loader-lock.json'}
    for key, path in paths.items():
        pin_key = key.replace('-', '_')
        expected = r[pin_key]['sha256'] if pin_key in ('model_lock', 'runtime_lock', 'approval') else digest(path)
        command += ['--' + key, str(path), '--' + key + '-sha256', expected]
    for key, path in {'model-root': model_root(r, out), 'source-root': r['source_root'],
                      'loader-root': out / 'loader', 'output': out / 'components'}.items():
        command += ['--' + key, str(path)]
    return command


def train_request(r, out, seconds):
    value = dict(schema='h3-identity-train-request-v1', run_id='h3-' + uuid.uuid4().hex,
                 max_updates=r['max_updates'], sampling_mode=r.get('sampling_mode', 'replacement'),
                 overall_seconds=seconds, overall_started_at=time.time(),
                 rate_usd_hour=r['rate_usd_hour'], spent_and_reserved_usd=r['spent_and_reserved_usd'],
                 territory_approval=r['territory_approval'],
                 pins={'sources': trainer.helper().SOURCE_PINS, 'models': {'transformer_ref':
                       {'repo_id': r['model_repo'], 'revision': trainer.MODEL_PIN}}},
                 frozen_selection=r['input_documents']['selection_path']['path'],
                 cache=str(out / 'components'), index_sha256=digest(out / 'components/index.json'),
                 model=str(model_root(r, out)), model_inventory=trainer.read_json(out / 'inventory.json'),
                 source=r['source_root'], work=str(out / 'training'), output_repo=r['output_repo'],
                 initial_checks=3, checkpoint_interval=r.get('checkpoint_interval', 25))
    if r.get('operator_managed_stages'):
        value.update(operator_managed_stages=True, preview_only=True)
    if 'compute_rate_approval' in r:
        value['compute_rate_approval'] = r['compute_rate_approval']
    if 'hf_price_precision_approval' in r:
        value['hf_price_precision_approval'] = r['hf_price_precision_approval']
    if r.get('inference',{}).get('native_continuation'):
        value['inference']={'native_continuation':r['inference']['native_continuation']}
        if r['inference'].get('acceleration_experiment') is not None:
            value['inference']['acceleration_experiment']=r['inference']['acceleration_experiment']
    if 'runtime_overlay' in r:
        value['runtime_overlay'] = r['runtime_overlay']
    if 'checkpoint_escrow' in r:
        require(type(r['checkpoint_escrow']) is bool, 'checkpoint escrow flag')
        value['checkpoint_escrow'] = r['checkpoint_escrow']
    if 'warm_start_remote' in r:
        remote = r['warm_start_remote']
        require(isinstance(remote, dict) and set(remote) == {'repo_id', 'revision', 'prefix', 'metadata_sha256'},
                'immutable remote weight handoff fields')
        from research.h3_personal.identity_handoff_fetch import fetch_handoff
        from huggingface_hub import constants
        previous = constants.HF_HUB_OFFLINE
        try:
            constants.HF_HUB_OFFLINE = False
            value['warm_start'] = fetch_handoff(**remote, destination=out / 'handoff-input',
                pins=value['pins'], expected_recipe_sha256=trainer.training_recipe_digest(value))
        finally:
            constants.HF_HUB_OFFLINE = previous
    trainer.validate_request(value)
    write_json(out / 'train-request.json', value)
    return value


def internal(stage, path, sha, seconds=None):
    r = validate(pinned(path, sha))
    out = Path(r['output'])
    approvals(r)
    if stage == 'prepare':
        prepare(r, out)
        return
    if stage == 'hardware':
        import torch
        count=2 if any(r.get('inference',{}).get(k) for k in ('context_parallel_matrix','native_continuation')) else 1
        require(torch.cuda.is_available() and torch.cuda.device_count() == count and
                all('H200' in torch.cuda.get_device_name(i) for i in range(count)),
                'exact configured H200 count required')
        return
    request = out / 'train-request.json'
    if stage == 'preview':
        from research.hf_identity_preview import run_previews
        run_previews(r, out, seconds)
        return
    if stage in ('train', 'handoff'):
        require(stage != 'handoff' or r.get('preview_only') is True, 'handoff stage is inference-only')
        value = train_request(r, out, seconds)
        work = Path(value['work'])
        work.mkdir(mode=0o700)
        trainer.materialize_runtime(value)
        write_json(out / 'train-request-pin.json', digest(request))
        os.environ.update(H3_IDENTITY_SUPERVISED='1', HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true')
        if r.get('preview_only', False):
            # No optimizer/training loop. Escrow is independently reverified by the next stage.
            # Fetched escrow is named simply checkpoint; upload requires the
            # canonical step parent used by normal training exports.
            import shutil
            original = Path(value['warm_start']['checkpoint'])
            metadata = trainer.read_json(value['warm_start']['metadata'])
            step = metadata['parent_step']
            require(type(step) is int and step >= 0, 'handoff parent step required')
            target = work / f'checkpoint-{step:07d}' / 'canonical'
            target.parent.mkdir(mode=0o700)
            shutil.copytree(original, target)
            checkpoint = str(target)
            write_json(work / 'training-snapshot.json', {'status': 'inference_only',
                       'checkpoint': checkpoint, 'quality_verified': False})
        else:
            # Existing worker loop, one production run; initial checks are inside it.
            trainer.main(['_train', '--request', str(request)])
    else:
        value = pinned(request, trainer.read_json(out / 'train-request-pin.json'))
        value.update(overall_seconds=seconds, overall_started_at=time.time())
        verification = out / 'verify-request.json'
        write_json(verification, value)
        checkpoint = trainer.read_json(out / 'training/training-snapshot.json')['checkpoint']
        if any(r.get('inference',{}).get(k) for k in ('context_parallel_matrix','native_continuation')):
            # Escrow verification remains a single-GPU fresh process; the later
            # matrix preview is a separate process with both GPUs visible.
            os.environ['CUDA_VISIBLE_DEVICES']='0'
        trainer.supervise(verification, 'verify-checkpoint', checkpoint)


def run(r, request, sha, runner=trainer.bounded_process):
    out = Path(r['output'])
    approvals(r)
    require(not out.exists(), 'exclusive new output required')
    os.umask(0o077)
    encoder.private_root(out)
    end = time.monotonic() + r['overall_seconds']
    status = {'schema': 'h3-identity-pipeline-report-v1', 'readiness': False,
              'quality_verified': False, 'c_gate_enabled': False}
    def command(stage, seconds):
        return [sys.executable, '-m', 'research.hf_identity_pipeline', '--request', str(request),
                '--request-sha256', sha, '--internal-stage', stage, '--seconds', str(seconds)]
    preview_seconds = r.get('inference', {}).get('seconds', 0)
    def stage(name, cmd, cap, reserve=None):
        if reserve is None:
            reserve = (600 if r.get('preview_only', False) else 1200) + preview_seconds
        status['stage'] = name
        print('H3_PHASE', name, flush=True)
        write_json(out / 'pipeline-report.json', status)
        if r.get('operator_managed_stages'):
            # No speculative per-phase cap/reserve. Provider timeout remains explicit.
            runner(cmd(r['overall_seconds']), deadline=None)
        else:
            seconds = min(cap, int(end - time.monotonic() - reserve))
            require(seconds > 0, 'verification reserve exhausted')
            runner(cmd(seconds), deadline=min(end - reserve, time.monotonic() + seconds))
    try:
        stage('hardware', lambda s: command('hardware', s), 60)
        stage('prepare', lambda s: command('prepare', s), r['stage_seconds'])
        if 'cache_reuse' not in r:
            for name, rank in (('media', 0), ('media', 1), ('reference', 0), ('text', 0), ('merge', 0)):
                stage(name + '-' + str(rank), lambda s, n=name, k=rank: encoder_command(r, out, n, s, k),
                      r['stage_seconds'])
        # Separate worker processes guarantee VAEs/optimizer unload before Qwen/reload.
        name = 'handoff' if r.get('preview_only', False) else 'train'
        stage(name, lambda s: command(name, s), 21600)
        stage('verify-upload', lambda s: command('verify-upload', s),
              600 if r.get('preview_only', False) else 1200, reserve=preview_seconds)
        if preview_seconds:
            stage('preview', lambda s: command('preview', s), preview_seconds, reserve=0)
        worker = trainer.read_json(out / 'training/worker-status.json')
        require(worker.get('uploaded') is True and
                worker['status'] == 'final_strict_reload_verified_quality_unverified', 'verification incomplete')
        status.update(readiness=True, canonical_reload_verified=True, private_uploaded=True)
    except BaseException as exc:
        status.update(readiness=False, failure_type=type(exc).__name__)
        raise
    finally:
        status['remaining_seconds'] = max(0, int(end - time.monotonic()))
        write_json(out / 'pipeline-report.json', status)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--request-sha256', required=True)
    p.add_argument('--execute-remote', action='store_true')
    p.add_argument('--internal-stage', choices=('hardware', 'prepare', 'handoff', 'train', 'verify-upload', 'preview'),
                   help=argparse.SUPPRESS)
    p.add_argument('--seconds', type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    os.umask(0o077)
    def terminate(*_):
        raise KeyboardInterrupt('pipeline termination')
    signal.signal(signal.SIGTERM, terminate)
    try:
        r = validate(pinned(args.request, args.request_sha256))
        approvals(r)
    except Exception as exc:
        print(json.dumps({'plan_only': True, 'readiness': False, 'failure_type': type(exc).__name__}))
        return 2 if args.execute_remote or args.internal_stage else 0
    if args.internal_stage:
        require(args.seconds is not None and 1 <= args.seconds <= 21600, 'stage seconds')
        internal(args.internal_stage, args.request, args.request_sha256, args.seconds)
    elif args.execute_remote:
        require(sys.platform == 'linux', 'execution is remote Linux only')
        run(r, args.request.resolve(), args.request_sha256)
    else:
        print(json.dumps({'plan_only': True, 'readiness': False, 'approval_valid': True,
                          'remaining_checks': ['private dataset/model content-write access',
                                               'remote runtime/source bytes', 'H200', 'provisioned model bytes',
                                               'all715 inputs', 'actual loader', 'train', 'fresh reload/upload']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
