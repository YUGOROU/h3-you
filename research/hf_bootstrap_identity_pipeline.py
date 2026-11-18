"""Stdlib-only remote bootstrap for the immutable identity deployment contract.
No package installation policy lives here: the independent runtime worker owns it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tarfile

INPUT_REPO = 'YUGOROU/h3-production-715-raw-20261012'
INPUT_REVISION = '60124b7a79c4711a14995594f8c95d54c6cc8924'
RAW_REVISION = '714cb0a67dd343c96276ae930ccd1029f9ebddb9'
ORIGINAL_REVISION = '851c993d1b460828160a04ddf977d6fe368cdf2b'
BUNDLE_SHA = '6204bf9c43f29990d18f528605f40c5e470c01525433109581c26ad6f552467f'
BUNDLE_BYTES = 129499
FREEZE_SHA = '17900bcdcbc5a6e2441b95b824d5d7778fdfd11a461bdbc3b4dc7f63ceacd634'
DATASET_SHA = '77e49d18d116be9717f8cd0558f2c1da765c3bb6db1fd27d9393d522fc22b0df'
TRAINER_PIN = '7a1563a267a17da898df3396acbd6a9d24fcb142'
MODEL_PIN = '42ed227ee7df40d41602854ae760620d6eb651fe'
DIFFUSERS_PIN = 'abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc'
DOCUMENTS = {'selection_path': 'frozen_selection.json',
             'manifest_path': 'raw/' + DATASET_SHA + '/manifest.json',
             'summary_path': 'raw/' + DATASET_SHA + '/summary.json',
             'source_map_path': 'production-inputs/source_map.json',
             'reference_mapping_path': 'production-inputs/reference_mapping.json',
             'caption_mapping_path': 'production-inputs/caption_mapping.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and
            not PurePosixPath(name).is_absolute() and
            all(p not in ('', '.', '..') for p in name.split('/')), 'unsafe relative path')
    return name


def inside(root, name):
    root = Path(root).resolve(strict=True)
    path = root / relative(name)
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent),
            'symlink input')
    require(path.resolve(strict=True).is_relative_to(root) and path.is_file(), 'nonregular input')
    return path


def pinned(root, entry):
    require(re.fullmatch('[0-9a-f]{64}', entry.get('sha256', '')), 'SHA256 required')
    path = inside(root, entry['path'])
    require(sha256(path) == entry['sha256'], 'document bytes differ')
    return path


def extract_bundle(archive, destination):
    """Validate the ENTIRE archive before writing anything; never tar.extractall."""
    require(Path(archive).stat().st_size == BUNDLE_BYTES and sha256(archive) == BUNDLE_SHA,
            'source bundle bytes differ')
    # Imports are from the trusted launcher locally or verified extracted bundle remotely.
    # The remote first-pass allowlist is intentionally static, not archive-supplied.
    workers = ('hf_identity_encode.py', 'hf_identity_train.py', 'hf_identity_pipeline.py',
               'hf_prepare_identity_media.py', 'install_identity_guidance.py',
               'install_identity_component_cache.py', 'install_identity_sampling.py',
               'install_identity_checkpoint.py')
    helpers = ('freeze_dataset.py', 'production_inputs.py', 'component_encode.py',
               'component_cache.py', 'identity_guidance.py', 'identity_sampling.py',
               'identity_checkpoint.py')
    source_names = ('__init__.py', 'checkpointing.py', 'config.py', 'constants.py',
                    'datasets.py', 'flow_matching.py', 'inference.py', 'logging_utils.py',
                    'lora.py', 'model_loader.py', 'packing.py', 'preprocessing/__init__.py',
                    'preprocessing/builder.py', 'preprocessing/encoders.py',
                    'preprocessing/media.py', 'quantization.py', 'timestep_samplers.py',
                    'trainer.py', 'training_strategies/__init__.py',
                    'training_strategies/base_strategy.py', 'training_strategies/flexible.py',
                    'validation_runner.py')
    allowed = {'research/' + n for n in workers} | {'research/h3_personal/' + n for n in helpers}
    allowed |= {'upstream/src/h3_trainer/' + n for n in source_names}
    allowed |= {'upstream/scripts/process_dataset.py', 'upstream/LICENSE'}
    require(len(allowed) == 39, 'bundle allowlist count')
    with tarfile.open(archive, 'r:gz') as tar:
        entries = tar.getmembers()
        names = [relative(m.name) for m in entries]
        require(len(names) == len(set(names)) and set(names) == allowed | {'CODE_MANIFEST.json'},
                'archive inventory differs')
        require(all(m.isfile() and 0 <= m.size <= 2 * 1024**2 for m in entries) and
                sum(m.size for m in entries) <= 8 * 1024**2, 'archive type/size rejected')
        contents = {m.name: tar.extractfile(m).read() for m in entries}
    manifest = json.loads(contents['CODE_MANIFEST.json'])
    require(manifest.get('schema') == 'h3-identity-source-bundle-v1' and
            manifest.get('trainer_revision') == TRAINER_PIN and
            set(manifest['files']) == allowed and manifest.get('contains_weights') is False and
            manifest.get('contains_credentials') is False, 'bundle manifest differs')
    for name in allowed:
        require(hashlib.sha256(contents[name]).hexdigest() == manifest['files'][name], 'archive hash differs')
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, data in contents.items():
        path = destination / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(data)
        path.chmod(0o600)
    return manifest


def build_request(deployment, deploy_root, input_root, source_root, rate, spent):
    """Resolve actual immutable mounts on the remote; no original/media download locally."""
    from research import hf_identity_pipeline as pipeline
    root, inputs = Path(deploy_root), Path(input_root)
    r = dict(deployment['pipeline'])
    for key in ('model_lock', 'runtime_lock', 'approval'):
        e = deployment['documents'][key]
        r[key] = {'path': str(pinned(root, e)), 'sha256': e['sha256']}
    approval = load(r['approval']['path'])
    for key in ('country_evidence', 'license_approval_evidence'):
        proof = approval[key]
        # Evidence documents contain stable absolute deployment-mount paths.
        require(Path(proof['path']).is_relative_to(root), 'evidence must be in deployment mount')
        pinned(root, {'path': str(Path(proof['path']).relative_to(root)), 'sha256': proof['sha256']})
    r['input_documents'] = {}
    for key, name in DOCUMENTS.items():
        path = inside(inputs, name)
        expected = deployment['input_document_sha256'][key]
        require(sha256(path) == expected, 'sensitive input document differs')
        r['input_documents'][key] = {'path': str(path), 'sha256': expected}
    require(r['input_documents']['selection_path']['sha256'] == FREEZE_SHA, 'frozen selection differs')
    summary = load(r['input_documents']['summary_path']['path'])
    require(summary['input_revision'] == ORIGINAL_REVISION and
            summary['complete_raw_preprocessing'] is True and summary['processed_clips'] == 715,
            'completed original raw receipt required')
    r.update(source_root=str(source_root), output='/work/identity', rate_usd_hour=rate,
             spent_and_reserved_usd=spent)
    r['production_inputs'] = dict(source_root=str(inputs / 'sources'),
        media_root=str(inputs / 'raw' / DATASET_SHA),
        reference_root=str(inputs / 'production-inputs'), code_root=str(inputs),
        expected_input_revision=ORIGINAL_REVISION)
    if 'prompt' in r.get('inference', {}):
        mapping = load(r['input_documents']['reference_mapping_path']['path'])['samples']
        r['production_inputs']['reference_images'] = {}
        for sid in r['inference']['sample_ids']:
            ref = mapping[sid]
            path = inside(inputs / 'production-inputs', ref['path'])
            require(sha256(path) == ref['sha256'], 'new preview pinned reference image differs')
            r['production_inputs']['reference_images'][sid] = {'path': str(path), 'sha256': ref['sha256']}
    return pipeline.validate(r)


def bounded_command(command, env, seconds):
    """Kill the entire stage process group on timeout/failure/termination."""
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        require(child.wait(timeout=seconds) == 0, 'remote stage failed')
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def execute(deploy_root, deployment_sha, rate, spent):
    require(sys.platform == 'linux', 'remote Linux only')
    root = Path(deploy_root)
    deployment = load(pinned(root, {'path': 'deployment.json', 'sha256': deployment_sha}))
    require(deployment['schema'] == 'h3-identity-deployment-v1', 'deployment schema')
    require(deployment['input_revision'] == INPUT_REVISION, 'complete inputs revision differs')
    work = Path('/work')
    work.mkdir(mode=0o700, exist_ok=False)
    extract_bundle(inside('/inputs', 'production-inputs/identity-workers-v2.tar.gz'), work / 'code')
    sys.path.insert(0, str(work / 'code'))
    from research.install_identity_guidance import PINS
    # Audit the native source root: no setup.py/pyproject/full checkout assumed.
    upstream = work / 'code/upstream'
    require(set(PINS) | {'LICENSE'} == {str(p.relative_to(upstream)) for p in upstream.rglob('*') if p.is_file()},
            'native source subset differs')
    for name, expected in PINS.items():
        require(sha256(inside(upstream, name)) == expected, 'upstream source pin differs')
    runtime = deployment['runtime_bootstrap']
    manifest_path = pinned('/runtime-inputs', runtime['manifest'])
    manifest = load(manifest_path)
    require(manifest['schema'] == 'h3-identity-gpu-runtime-bootstrap-v1' and
            manifest['gpu_bootstrap_supported'] is True and
            manifest['image'] == deployment['image'] and
            manifest['runtime_lock_sha256'] == deployment['documents']['runtime_lock']['sha256'] and
            manifest['native_source_inventory'] == 'PINS23+LICENSE' and
            manifest['entrypoint'] == runtime['entrypoint'] and manifest['module'] == runtime['module'],
            'reviewed GPU runtime manifest required')
    pinned('/runtime-inputs', runtime['entrypoint'])
    code_pins = manifest['bootstrap_files']
    require(isinstance(code_pins, dict) and code_pins.get(runtime['entrypoint']['path']) ==
            runtime['entrypoint']['sha256'], 'bootstrap code inventory missing')
    for name, expected in code_pins.items():
        pinned('/runtime-inputs', {'path': name, 'sha256': expected})
    # Importing -m may execute package initializers before the entrypoint itself.
    parts = runtime['module'].split('.')[:-1]
    for i in range(1, len(parts) + 1):
        name = '/'.join(parts[:i]) + '/__init__.py'
        require(not (Path('/runtime-inputs') / name).exists() or name in code_pins,
                'unpinned runtime package initializer')
    module = runtime['module']
    require(re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', module) and
            runtime['entrypoint']['path'] == module.replace('.', '/') + '.py', 'entrypoint module mismatch')
    receipt_path = work / 'runtime-ready.json'
    env = dict(os.environ, PYTHONPATH='/runtime-inputs', H3_RUNTIME_IMAGE=deployment['image'],
               H3_NATIVE_SOURCE_ROOT=str(upstream))
    inference=deployment['pipeline'].get('inference',{})
    if inference.get('context_parallel_matrix') or inference.get('native_continuation'):
        require(deployment['pipeline']['flavor']=='h200x2','explicit two-H200 inference required')
        env['H3_ACCELERATION_MATRIX' if inference.get('context_parallel_matrix') else 'H3_NATIVE_CONTINUATION']='1'
    bounded_command([sys.executable, '-m', module, '--manifest', str(manifest_path),
                     '--manifest-sha256', runtime['manifest']['sha256'],
                     '--output', '/work/runtime', '--receipt', str(receipt_path)],
                    env, None if deployment['pipeline'].get('operator_managed_stages') else runtime['timeout_seconds'])
    receipt = load(receipt_path)
    require(receipt.get('schema') == 'h3-identity-runtime-ready-v1' and
            receipt['manifest_sha256'] == runtime['manifest']['sha256'] and
            receipt['runtime_lock_sha256'] == deployment['documents']['runtime_lock']['sha256'],
            'independent runtime receipt differs')
    python = Path(receipt['python']).absolute()
    require(python.is_relative_to(work / 'runtime') and python.is_file(), 'runtime interpreter escapes output')
    # A venv may symlink its interpreter into the immutable image. Bind the actual binary.
    require(receipt['python_sha256'] == sha256(python) and
            receipt['python_sha256'] == manifest['python_sha256'], 'independent interpreter bytes differ')
    # The frozen input bundle remains immutable (including pristine upstream), but
    # workers must use the independently verified runtime closure, which contains
    # reviewed approval/diagnostic fixes added after that frozen bundle.
    sys.path.insert(0, '/runtime-inputs')
    r = build_request(deployment, root, '/inputs', upstream, rate, spent)
    from research import hf_identity_pipeline as pipeline
    pipeline.approvals(r)
    request = work / 'pipeline-request.json'
    dump(request, r)
    env = dict(os.environ, PYTHONPATH='/runtime-inputs', HF_HUB_DISABLE_TELEMETRY='1')
    experiment=r.get('inference',{}).get('acceleration_experiment')
    if experiment is not None:
        from research.h3_personal.acceleration_experiment import validate as validate_experiment
        experiment=validate_experiment(experiment)
        if experiment['quantization']!='none':
            # Separate experimental Job venv only; hash-pinned wheel, no dependency upgrades.
            bounded_command([str(python), '-m', 'research.prepare_h3_fp8_runtime', '--execute'],env,300)
    # The external runtime must supply ALL backend/dependency artifacts, not installed packages.
    bounded_command([str(python), '-m', 'research.hf_identity_pipeline', '--request', str(request),
                     '--request-sha256', sha256(request), '--execute-remote'],
                    env, None if r.get('operator_managed_stages') else r['overall_seconds'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-root', default='/deployment')
    p.add_argument('--deployment-sha256', required=True)
    p.add_argument('--rate', type=float, required=True)
    p.add_argument('--spent-and-reserved', type=float, required=True)
    a = p.parse_args(argv)
    os.umask(0o077)
    def terminate(*_):
        raise KeyboardInterrupt('remote bootstrap terminated')
    signal.signal(signal.SIGTERM, terminate)
    execute(a.deployment_root, a.deployment_sha256, a.rate, a.spent_and_reserved)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({'state': 'failed_closed', 'failure_type': type(exc).__name__}))
        raise SystemExit(2) from None
