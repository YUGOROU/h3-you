"""Remote component encoder. No job submission or model download API.

Run as python -m research.hf_identity_encode; see docs/h3-identity-encode-worker.md.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from research.h3_personal import component_cache as cache
from research.h3_personal import component_encode as encode
from research.h3_personal import production_inputs
from research.install_identity_guidance import PINS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return production_inputs._load(path)


def pinned_document(path, expected):
    require(cache.digest_ok(expected) and cache.sha256(path) == expected, 'document pin mismatch')
    return load(path)


def private_root(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not path.is_symlink() and path.stat().st_mode & 0o077 == 0,
            'output/snapshot root must be private (chmod 700)')
    return path


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def license_gate(evidence):
    require(evidence.get('schema') == 'h3-compute-approval-v1', 'compute approval schema')
    country = evidence.get('execution_country')
    operator = evidence.get('evidence_mode') == 'operator_hf_acceptance'
    if operator:
        # User contract interpretation/risk acceptance, not rights-holder proof.
        require('execution_country' in evidence and country is None,
                'operator acceptance must preserve unknown execution country')
        require(evidence.get('provider') == 'huggingface_jobs'
                and evidence.get('risk_acknowledged') is True
                and evidence.get('approved') is True
                and isinstance(evidence.get('approved_by'), str)
                and evidence['approved_by'].strip(), 'explicit HF operator acceptance required')
        require(evidence.get('license_allows_execution_country') is not True,
                'operator acceptance is not blanket territory permission')
    else:
        require(isinstance(country, str) and re.fullmatch('[A-Z]{2}', country), 'execution country required')
        from research.hf_identity_train import EXCLUDED, ISO_COUNTRIES
        require(country in ISO_COUNTRIES and country not in EXCLUDED,
                'MiniMax LICENSE excludes EU/UK/KR/US execution territories')
    for key in ('country_evidence', 'license_approval_evidence'):
        proof = evidence.get(key)
        require(isinstance(proof, dict) and proof.get('sha256') and cache.digest_ok(proof['sha256'])
                and isinstance(proof.get('description'), str) and proof['description'].strip(),
                'distinct country/license evidence required')
        require(cache.sha256(proof['path']) == proof['sha256'], 'approval evidence byte mismatch')
        require('collection' not in proof['description'].lower(), 'collection URL is not compute proof')
    require(evidence['country_evidence']['sha256'] != evidence['license_approval_evidence']['sha256'],
            'country proof must be distinct from license approval')
    require(evidence.get('approved') is True and bool(evidence.get('approved_by')),
            'explicit human license approval required')
    if operator:
        description = evidence['country_evidence']['description'].lower()
        require(all(word in description for word in ('unknown', 'official', 'deployment', 'observed'))
                and ('huggingface' in description or 'hugging face' in description
                     or re.search(r'\bhf\b', description)),
                'describe unknown country and observed official HF deployment')
        require(Path(evidence['country_evidence']['path']).resolve()
                != Path(evidence['license_approval_evidence']['path']).resolve(),
                'country/license evidence must be separate files')
    else:
        # This is operator evidence, not a geolocation service or legal opinion.
        require(evidence.get('license_allows_execution_country') is True, 'territory approval required')


MODEL_GROUPS = {'video': ('vae',), 'audio': ('audio_vae',), 'reference': ('vae',),
                'text': ('text_encoder', 'tokenizer', 'processor'),
                'identity_uncond': ('text_encoder', 'tokenizer', 'processor')}


def verify_models(root, lock):
    """Hash actual mounted files, including all configs/tokenizer assets and shards.

    lock is independently byte-pinned by the CLI. It is NOT old-cache provenance.
    No HF metadata, caller attestor flag, or directory name substitutes for bytes.
    """
    root = Path(root).resolve(strict=True)
    require(lock.get('schema') == 'h3-local-model-lock-v1' and
            re.fullmatch('[0-9a-f]{40}', lock.get('revision', '')), 'immutable model revision required')
    expected = lock.get('files', {})
    require(isinstance(expected, dict) and 'model_index.json' in expected, 'model_index.json pin required')
    actual = {'model_index.json'}
    for group in ('vae', 'audio_vae', 'text_encoder', 'tokenizer', 'processor'):
        directory = root / group
        require(directory.is_dir(), 'missing model component: ' + group)
        names = {str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file()}
        require(names and any(n.endswith('.json') for n in names), 'component config required')
        if group in ('vae', 'audio_vae', 'text_encoder'):
            require(any(n.endswith('.safetensors') for n in names), 'component weights required')
        actual.update(names)
    require(set(expected) == actual, 'lock must cover exact relevant model file inventory')
    hashes = {}
    for name in sorted(actual):
        path = production_inputs._inside(root, name)
        require(cache.digest_ok(expected[name]) and cache.sha256(path) == expected[name],
                'model file hash mismatch: ' + name)
        hashes[name] = expected[name]
        if name.endswith('.safetensors.index.json'):
            index = load(path)
            require(isinstance(index.get('weight_map'), dict) and index['weight_map'], 'invalid weight index')
            for shard in set(index['weight_map'].values()):
                require(str(Path(name).parent / shard) in actual, 'missing indexed weight shard')
    return hashes


REQUIRED_MODULES = {
    'diffusers.modular_pipelines.minimax_h3.packing',
    'diffusers.modular_pipelines.minimax_h3.packing_ref2va',
    'diffusers.models.autoencoders.autoencoder_kl_minimax_h3',
    'diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio',
    'transformers.models.qwen3_vl.modeling_qwen3_vl',
}


def verify_runtime(source_root, lock):
    root = Path(source_root).resolve(strict=True)
    # Verify before importing trainer code, then verify loaded locations too.
    for name in ('__init__.py', 'constants.py', 'model_loader.py', 'packing.py',
                 'preprocessing/__init__.py', 'preprocessing/encoders.py', 'preprocessing/media.py'):
        relative = 'src/h3_trainer/' + name
        require(cache.sha256(root / relative) == PINS[relative], 'unpinned trainer source')
    sys.path.insert(0, str(root / 'src'))
    require(lock.get('schema') == 'h3-encode-runtime-v1', 'runtime lock schema')
    require(REQUIRED_MODULES <= set(lock.get('modules', {})), 'runtime module pins missing')
    for package, version in lock['versions'].items():
        require(importlib.metadata.version(package) == version, 'runtime version mismatch')
    require({'torch', 'diffusers', 'transformers', 'safetensors', 'Pillow', 'numpy', 'accelerate'}
            <= set(lock['versions']), 'required runtime versions missing')
    for name, sha in lock['modules'].items():
        module = importlib.import_module(name)
        require(cache.digest_ok(sha) and cache.sha256(module.__file__) == sha, 'runtime module mismatch')
    for name in ('constants', 'model_loader', 'packing', 'preprocessing.encoders', 'preprocessing.media'):
        module = importlib.import_module('h3_trainer.' + name)
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(root) and cache.sha256(path) == PINS[str(path.relative_to(root))],
                'loaded trainer source mismatch')
    ffmpeg = Path(lock['ffmpeg']['path'])
    require(cache.sha256(ffmpeg) == lock['ffmpeg']['sha256'], 'ffmpeg pin mismatch')
    import shutil
    require(Path(shutil.which('ffmpeg') or '').resolve() == ffmpeg.resolve(), 'ffmpeg PATH mismatch')


def identities(model_files, runtime, text_device_map=None):
    worker_sources = {str(p.relative_to(Path(__file__).parent)): cache.sha256(p) for p in (
        Path(__file__), Path(encode.__file__), Path(cache.__file__))}
    return {kind: encode.digest({'weights_configs': {n: h for n, h in model_files.items()
                if n == 'model_index.json' or n.split('/')[0] in groups},
            'trainer': PINS, 'runtime': {k: v for k, v in runtime.items() if k != 'ffmpeg'},
            'ffmpeg_sha256': runtime['ffmpeg']['sha256'], 'worker': worker_sources,
            'policy': 'cuda/fp32-VAEs/bf16-text/seed42/batch1',
            'text_device_map': text_device_map if kind in ('text', 'identity_uncond') else None})
            for kind, groups in MODEL_GROUPS.items()}


def semantic_plan(plan):
    """Engine recipes already omit paths; receipt identity must omit them too."""
    value = json.loads(cache.canonical(plan))
    for unit in value['units'].values():
        for source in unit['sources'].values():
            source.pop('path', None)
    for sample in value['samples']:
        sample.pop('source', None)
    return encode.digest(value)


class RelocatedEngine(encode.Engine):
    def _receipt(self, name, bindings):
        payload = {'plan_sha256': semantic_plan(self.plan),
                   'bindings': {u: {'key': k, 'entry': e} for u, (k, e) in bindings.items()}}
        with (self.root / (name + '.json')).open('x') as f:
            json.dump(payload, f)
        return payload

    def _read(self, name):
        payload = load(self.root / (name + '.json'))
        require(payload['plan_sha256'] == semantic_plan(self.plan), 'receipt plan mismatch')
        return {u: (v['key'], v['entry']) for u, v in payload['bindings'].items()}


class PrivateSnapshots:
    """Incremental stage snapshots, objects first, receipt last; never model files."""
    def __init__(self, api, repo, root, prefix, max_bytes=256 * 1024**2):
        self.api, self.repo, self.root, self.prefix = api, repo, Path(root), prefix
        self.max_bytes = max_bytes
        require(re.fullmatch(r'components/[0-9a-f]{64}', prefix), 'snapshot prefix required')

    def upload(self, names):
        from huggingface_hub import CommitOperationAdd
        batch, size = [], 0
        def commit():
            require(self.api.repo_info(self.repo, repo_type='dataset').private is True,
                    'private snapshot dataset required')
            self.api.create_commit(repo_id=self.repo, repo_type='dataset', operations=batch,
                                   commit_message='Completed component snapshot')
        for name in dict.fromkeys(names):
            require(name.startswith('objects/') or name in {'media-0.json', 'media-1.json',
                    'references.json', 'text.json', 'index.json', 'report.json'}, 'snapshot allowlist')
            path = cache.contained_path(self.root, name)
            length = path.stat().st_size
            require(length <= self.max_bytes, 'single artifact exceeds snapshot byte limit')
            if batch and (size + length > self.max_bytes or len(batch) >= 16):
                commit()
                batch, size = [], 0
            batch.append(CommitOperationAdd(path_in_repo=self.prefix + '/' + name,
                                           path_or_fileobj=str(path)))
            size += length
        if batch:
            commit()


def construct(stage, model_root, source_root, text_device_map=None):
    from h3_trainer.preprocessing.encoders import H3Encoders
    import torch
    require(torch.cuda.is_available(), 'CUDA runtime unavailable')
    encoders = H3Encoders(model_path=str(model_root), device='cuda',
                          need_video=stage in ('media', 'reference'), need_audio=stage == 'media',
                          need_text=stage == 'text', text_device_map=text_device_map)
    return encode.PinnedBackend(encoders, source_root)


def validate_loader(root, source_root, lock):
    # Use the separately installed composed v2 loader, in a fresh process.
    require(lock.get('schema') == 'h3-composed-loader-lock-v1', 'composed loader lock required')
    source_root = Path(source_root).resolve(strict=True)
    files = {str(p.relative_to(source_root)) for p in source_root.rglob('*.py')}
    require({'src/h3_trainer/__init__.py', 'src/h3_trainer/datasets.py',
             'src/h3_trainer/component_cache.py'} <= files, 'composed v2 loader files required')
    require(files == set(lock['files']), 'loader source inventory mismatch')
    for name, sha in lock['files'].items():
        require(cache.sha256(production_inputs._inside(source_root, name)) == sha, 'loader source mismatch')
    script = ('import sys; sys.path.insert(0,sys.argv[1]); '
              'from h3_trainer.datasets import PrecomputedDataset; '
              'd=PrecomputedDataset(sys.argv[2], {k:"ignored" for k in '
              '("video","audio","reference","text","identity_uncond")}, val_split_every=0); '
              'assert len(d)==715; '
              '[d[i] for i in range(len(d))]')
    # Iterate without retaining the entire corpus.
    script = script.replace('[d[i] for i in range(len(d))]', 'exec("for i in range(len(d)):\\n d[i]")')
    process = subprocess.Popen([sys.executable, '-c', script, str(source_root / 'src'), str(root)],
                               start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        require(process.wait(timeout=600) == 0, 'actual v2 Dataset loader validation failed')
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def execute(stage, rank, root, plan, backend, reuse=None):
    engine = RelocatedEngine(root, plan, backend, reuse_catalog=reuse)
    if stage == 'media':
        return engine.run_media(rank), 'media-' + str(rank)
    if stage == 'reference':
        return engine.run_references(), 'references'
    if stage == 'text':
        return engine.run_text(), 'text'
    return engine.finish(), 'index'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('media', 'reference', 'text', 'merge'))
    p.add_argument('--rank', type=int, choices=(0, 1), default=0)
    for name in ('inputs', 'model-root', 'source-root', 'output', 'model-lock', 'runtime-lock',
                 'approval', 'loader-root', 'loader-lock'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('inputs', 'model-lock', 'runtime-lock', 'approval', 'loader-lock'):
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--deadline-seconds', type=int, default=9000)
    p.add_argument('--text-device-map', choices=('auto', 'balanced', 'balanced_low_0', 'sequential'))
    p.add_argument('--reuse-catalog', type=Path)
    p.add_argument('--snapshot-repo')
    args = p.parse_args(argv)
    require(1 <= args.deadline_seconds <= 10500, 'bounded deadline required')
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    root = private_root(args.output)
    report = {'schema': 'h3-identity-encode-report-v1', 'stage': args.stage,
              'complete': False, 'c_gate_enabled': False, 'legacy_cache_certified': False}
    def expired(*_):
        raise TimeoutError('worker deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.deadline_seconds)
    try:
        approval = pinned_document(args.approval, args.approval_sha256)
        license_gate(approval)
        inputs = pinned_document(args.inputs, args.inputs_sha256)
        rows, receipt = production_inputs.stage(**inputs)
        models = verify_models(args.model_root, pinned_document(args.model_lock, args.model_lock_sha256))
        runtime = pinned_document(args.runtime_lock, args.runtime_lock_sha256)
        verify_runtime(args.source_root, runtime)
        plan = encode.make_plan(rows, identities(models, runtime, args.text_device_map),
                                attested_by='verified-file-worker-v1',
                                media_shards=2)
        pid = semantic_plan(plan)
        # Output root is exclusive to one semantic plan; paths are allowed to relocate.
        binding = root / 'plan-id.json'
        if binding.exists():
            require(load(binding) == pid, 'output root belongs to a different plan')
        else:
            with binding.open('x') as f:
                json.dump(pid, f)
        reuse = None
        if args.reuse_catalog:
            document = load(args.reuse_catalog)
            require(document.get('schema') == cache.SCHEMA, 'only validated v2 reuse accepted')
            cache.validate_index(root, document)
            reuse = document['components']
        backend = None if args.stage == 'merge' else construct(
            args.stage, args.model_root, args.source_root, args.text_device_map)
        payload, name = execute(args.stage, args.rank, root, plan, backend, reuse)
        del backend
        gc.collect()
        if args.stage == 'merge':
            try:
                validate_loader(root, args.loader_root,
                                pinned_document(args.loader_lock, args.loader_lock_sha256))
            except BaseException:
                (root / 'index.json').unlink(missing_ok=True)
                raise
        report.update(complete=True, plan_sha256=pid, mounted_bytes_verified=True,
                      sample_count=len(rows), cuda_models_loaded=args.stage != 'merge',
                      loader_verified=args.stage == 'merge', input_revision=receipt['input_revision'])
        dump(root / 'report.json', report)
        if args.snapshot_repo:
            from huggingface_hub import HfApi
            snapshots = PrivateSnapshots(HfApi(), args.snapshot_repo, root, 'components/' + pid)
            names = [] if name == 'index' else [v['entry']['path'] for v in payload['bindings'].values()]
            # Model loading has finished. Permit only the explicit private snapshot API.
            from huggingface_hub import constants as hub_constants
            offline = hub_constants.HF_HUB_OFFLINE
            try:
                hub_constants.HF_HUB_OFFLINE = False
                snapshots.upload(names + [name + '.json', 'report.json'])
            finally:
                hub_constants.HF_HUB_OFFLINE = offline
        return 0
    except BaseException as exc:
        # No prompts, paths, tokens, model config, or raw-media content in failure reports.
        report.update(complete=False, failure_type=type(exc).__name__)
        dump(root / 'report.json', report)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
