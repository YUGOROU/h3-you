"""CPU-only private FastH3 preparation. Never submits, uploads or starts a runtime.

Network access is limited to pinned public source/text and Hub file metadata.
Large weights are represented by verified metadata, never downloaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request
import io
from concurrent.futures import ThreadPoolExecutor

from research.provision_modal_identity_secret import cached_token
from research.h3_personal.realtime_first import FASTVIDEO_SHA, RealtimeFirstConfig, budget_bound

BASE_SHA = '42ed227ee7df40d41602854ae760620d6eb651fe'
ADAPTER_SHA = 'f509e629374cac104e7f62daecce6d1488a3041d'
ADAPTER_DIGEST = '42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470'
IMAGE = 'ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:34ef8907bf0c4329d81be5fd4a6e3056b806e65fa0bba439733ab8a45bef31cf'
SCOPE = 'h3-realtime-first-fastvideo-v1'
ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    'research/hf_identity_encode.py': '6f8fd77bb924cf81e4c19e6ff64ca7f629c28282835403a22cfc9b6ec4743d7b',
    'research/h3_personal/component_encode.py': '2114f05051974013db7b538ff38dfaa654398109dbc94bbbbb4ed311511afd6c',
    'research/h3_personal/component_cache.py': '14513e8e11ad1788c999c8113c3aeaee14dac578183d3b513642955d87c8c5fa',
}
DIRECTORIES = ('transformer', 'text_encoder', 'tokenizer', 'processor', 'vae',
               'audio_vae', 'scheduler', 'audio_scheduler')
APPROVALS = ('country_evidence', 'approval', 'license_approval_evidence')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('unsafe asset/archive path')
    return p


def write_json(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    path.chmod(0o600)


def fetch(url, limit):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('download exceeds preparation size limit')
    return data


def source_archive():
    """Fetch code-only pinned tree, avoiding enormous unrelated demo media."""
    tree=json.loads(fetch(f'https://api.github.com/repos/hao-ai-lab/FastVideo/git/trees/{FASTVIDEO_SHA}?recursive=1',16*1024*1024))
    if tree.get('truncated'):raise ValueError('source tree truncated')
    entries=[e for e in tree['tree'] if e['type']=='blob' and (
        e['path'].startswith('fastvideo/') or e['path'] in ('pyproject.toml','LICENSE','README.md'))
        and Path(e['path']).suffix.lower() not in ('.mp4','.png','.jpg','.jpeg','.gif','.webp','.safetensors','.pt','.npy','.npz')]
    if not entries or sum(e['size'] for e in entries)>128*1024*1024:raise ValueError('unexpected code tree size')
    def download(entry):
        if entry['mode']!='100644' and entry['mode']!='100755':raise ValueError('nonregular code tree member')
        name=entry['path'];safe_path(name)
        data=fetch(f'https://raw.githubusercontent.com/hao-ai-lab/FastVideo/{FASTVIDEO_SHA}/{name}',16*1024*1024)
        if len(data)!=entry['size'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=entry['sha']:
            raise ValueError('source Git tree hash mismatch')
        return name,data
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as archive,ThreadPoolExecutor(max_workers=12) as pool:
        for name,data in pool.map(download,entries):
            member=tarfile.TarInfo(f'FastVideo-{FASTVIDEO_SHA}/{name}');member.size=len(data);member.mode=0o644
            archive.addfile(member,io.BytesIO(data))
    return out.getvalue()


def check_frozen():
    for name, sha in FROZEN.items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError(f'frozen producer changed: {name}')
    return dict(FROZEN)


def validate_archive(path):
    """Inspect only; links, devices, duplicate names and traversal are forbidden."""
    seen = set()
    prefix = f'FastVideo-{FASTVIDEO_SHA}'
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = member.name.rstrip('/')
            p = safe_path(name)
            if p.parts[0] != prefix or name in seen or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe source archive member')
            if member.mode & 0o6000:
                raise ValueError('privileged source archive member')
            seen.add(name)
    if not seen:
        raise ValueError('empty source archive')


def metadata_entry(file):
    safe_path(file.rfilename)
    size = file.size
    lfs = file.lfs
    sha = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
    blob = file.blob_id
    if type(size) is not int or size < 0:
        raise ValueError('missing file size')
    if sha is not None and re.fullmatch('[0-9a-f]{64}', sha) is None:
        raise ValueError('invalid LFS digest')
    if not sha and (not blob or re.fullmatch('[0-9a-f]{40}', blob) is None):
        raise ValueError('missing immutable file digest')
    return dict(path=file.rfilename, bytes=size, sha256=sha, git_blob=blob)


def verify_bytes(data, entry):
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if len(data) != entry['bytes'] or (entry['sha256'] and digest(data) != entry['sha256']):
        raise ValueError('asset size/SHA mismatch')
    if not entry['sha256'] and blob != entry['git_blob']:
        raise ValueError('asset Git blob mismatch')


def prepare(output, approval_spec, publication_repo, *, api=None):
    output = Path(output).expanduser().resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError('--output must be a new private directory outside the repository')
    if re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', publication_repo) is None:
        raise ValueError('invalid publication repository')
    frozen = check_frozen()
    original = json.loads(Path(approval_spec).expanduser().read_text())
    territory = original['pipeline']['territory_approval']
    if (territory.get('provider') != 'huggingface_jobs'
            or territory.get('evidence_mode') != 'operator_hf_acceptance'
            or territory.get('approved') is not True):
        raise ValueError('existing provider-bound HF operator acceptance required')
    documents = {}
    payloads = {}
    for key in APPROVALS:
        row = original['documents'][key]
        data = Path(row['path']).expanduser().read_bytes()
        if digest(data) != row['sha256']:
            raise ValueError(f'approval digest mismatch: {key}')
        json.loads(data)
        payloads[f'documents/{key}.json'] = data
        documents[key] = dict(path=f'documents/{key}.json', sha256=digest(data))
    config = RealtimeFirstConfig(FASTVIDEO_SHA, BASE_SHA, ADAPTER_SHA)
    config.validate()
    bound = budget_bound()
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=cached_token())
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output.chmod(0o700)
    def stage(name, data):
        safe_path(name)
        target = output / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o600)
    for name, data in payloads.items():
        stage(name, data)
    for name in ('research/run_h3_realtime_first.py', 'research/h3_personal/realtime_first.py',
                 'research/h3_personal/realtime_segments.py', 'research/bootstrap_h3_realtime_first.py'):
        stage(name, (ROOT / name).read_bytes())
    for name in ('research/__init__.py', 'research/h3_personal/__init__.py'):
        stage(name, b'')
    source_url = f'https://codeload.github.com/hao-ai-lab/FastVideo/tar.gz/{FASTVIDEO_SHA}'
    source = source_archive()
    archive_name = 'runtime/FastVideo.tar.gz'
    stage(archive_name, source)
    validate_archive(output / archive_name)
    manifests = []
    license_texts = {}
    repos = [('base', 'MiniMaxAI/MiniMax-H3', BASE_SHA),
             ('adapter', 'FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA', ADAPTER_SHA)]
    for role, repo, revision in repos:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError('Hub revision mismatch')
        entries = []
        for file in info.siblings:
            name = file.rfilename
            p = safe_path(name)
            root_text = len(p.parts) == 1 and (name.lower().startswith(('readme', 'license'))
                                               or name.endswith('.json'))
            selected = (role == 'base' and (p.parts[0] in DIRECTORIES or root_text)) or (
                role == 'adapter' and (root_text or name in (
                    'vsa-datafree/adapter_model.safetensors', 'vsa-datafree/adapter_config.json')))
            if not selected:
                continue
            entry = metadata_entry(file)
            entries.append(entry)
            if role == 'adapter' and name == 'vsa-datafree/adapter_model.safetensors':
                if entry['sha256'] != ADAPTER_DIGEST:
                    raise ValueError('FastH3 adapter digest mismatch')
            # Only public root legal/configuration texts are downloaded, never tensors.
            if root_text:
                from urllib.parse import quote
                data = fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{quote(name)}', 4 * 1024 * 1024)
                verify_bytes(data, entry)
                dest = f'public-texts/{role}/{name}'
                stage(dest, data)
                if name.lower().startswith(('readme', 'license')):
                    license_texts[dest] = data.decode('utf-8')
        paths = {e['path'] for e in entries}
        required = {'model_index.json'} if role == 'base' else {'vsa-datafree/adapter_model.safetensors'}
        if not required <= paths or 'README.md' not in paths:
            raise ValueError('required model files absent')
        if role == 'base' and any(not any(n.startswith(d + '/') for n in paths) for d in DIRECTORIES):
            raise ValueError('required base component absent')
        manifests.append(dict(role=role, repo=repo, revision=revision, files=entries))
    adapter_text = '\n'.join(v for k, v in license_texts.items() if '/adapter/' in k).lower()
    base_legal = [k for k in license_texts if '/base/' in k and 'license' in k.lower()]
    if 'license' not in adapter_text or not any(s in adapter_text for s in ('minimax', 'mini-max', 'h3')) or not base_legal:
        raise ValueError('FastH3 inherited H3 license evidence missing; inspect public texts')
    write_json(output / 'asset-manifest.json', dict(schema=SCOPE, assets=manifests))
    spec = dict(schema=SCOPE, config=asdict(config), publication_repo=publication_repo,
                image=IMAGE, documents=documents,
                source=dict(url=source_url, revision=FASTVIDEO_SHA, path=archive_name,
                            bytes=len(source), sha256=digest(source), extracted_on_host=False),
                frozen_producers=frozen,
                license_evidence=dict(mode='operator_hf_acceptance', provider='huggingface_jobs',
                    compute_country=territory.get('compute_country'), country_verified=False,
                    fast_h3_license='other; inherits MiniMax-H3; retained public texts require review',
                    public_text_sha256={k: digest(v.encode()) for k, v in license_texts.items()}),
                approval_scope=SCOPE,
                execution=dict(provider='huggingface_jobs', flavor='h200x2', max_jobs=1,
                    timeout_seconds=3000, job_bound_usd='8.33335', cumulative_cap_usd='170',
                    budget_bound={k: str(v) for k, v in bound.items()}, submitted=False),
                asset_manifest='asset-manifest.json')
    write_json(output / 'spec.json', spec)
    check_frozen()
    # The inventory cannot hash itself. All other stage files are included exactly.
    inventory = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            inventory.append(dict(path=path.relative_to(output).as_posix(),
                                  bytes=path.stat().st_size, sha256=digest(path.read_bytes())))
    write_json(output / 'upload-allowlist.json', dict(schema=SCOPE, private_only=True,
        self_excluded='upload-allowlist.json', files=inventory))
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approval-spec', type=Path, required=True,
                        help='existing corrected-native v6 spec; only three hashed approvals are copied')
    parser.add_argument('--publication-repo', default='YUGOROU/h3-realtime-first-20261016-v1')
    args = parser.parse_args(argv)
    prepare(args.output, args.approval_spec, args.publication_repo)
    print('Private CPU preparation complete; no upload or paid submission.')


if __name__ == '__main__':
    main()
