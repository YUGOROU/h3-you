"""Prepare independent H3 byte expectations; never download weights or submit Jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request

REPO = 'MiniMaxAI/MiniMax-H3'
REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
GROUPS = {'vae', 'audio_vae', 'text_encoder', 'tokenizer', 'processor'}
SMALL_CAP = 32 * 1024 * 1024
API_URL = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == 'https', 'non-HTTPS redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


class HubIO:
    """Bound every response and keep credentials out of receipts/errors."""
    def __init__(self, token=None):
        self.token = token
        self.opener = urllib.request.build_opener(SafeRedirect())

    def get(self, url, limit):
        headers = {'Accept-Encoding': 'identity'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        try:
            with self.opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
                data = response.read(limit + 1)
        except Exception:
            raise ValueError('Hub request failed (credentials and response omitted)') from None
        require(len(data) <= limit, 'Hub response exceeds byte limit')
        return data


def selected(name):
    parts = name.split('/')
    return (name == 'model_index.json' or parts[0] in GROUPS | {'transformer_ref'}) and not any(
        p.startswith('.') or p.lower() in {'docs', 'readme.md', 'license'} for p in parts)


def prepare(io):
    """io.get(url, limit) is injectable. Returns only JSON artifacts, not model bytes."""
    raw = io.get(API_URL, 8 * 1024 * 1024)
    info = json.loads(raw)
    require(info.get('sha') == REVISION, 'API revision mismatch')
    require(info.get('id') == REPO, 'API repository mismatch')
    require(isinstance(info.get('siblings'), list), 'missing complete file metadata')
    entries = {}
    seen = set()
    for item in info['siblings']:
        name = item.get('rfilename')
        require(isinstance(name, str) and not name.startswith('/') and '\\' not in name
                and all(p not in ('', '.', '..') for p in name.split('/')), 'unsafe API path')
        require(name not in seen, 'duplicate API path')
        seen.add(name)
        if not selected(name):
            continue
        require(type(item.get('size')) is int and item['size'] >= 0, 'invalid asset size')
        require(re.fullmatch('[0-9a-f]{40}', item.get('blobId', '')), 'missing Git blobId')
        lfs = item.get('lfs')
        if lfs is not None:
            require(isinstance(lfs, dict) and lfs.get('size') == item['size'] and
                    re.fullmatch('[0-9a-f]{64}', lfs.get('sha256', '')), 'invalid LFS metadata')
        else:
            require(name.endswith(('.json', '.txt')), 'refuse non-config/tokenizer Git asset')
        entries[name] = item
    require('model_index.json' in entries, 'missing model_index.json')
    for group in GROUPS | {'transformer_ref'}:
        names = [n for n in entries if n.startswith(group + '/')]
        require(names and any(n.endswith('.json') for n in names), 'missing component config: ' + group)
        if group in {'vae', 'audio_vae', 'text_encoder', 'transformer_ref'}:
            require(any(n.endswith('.safetensors') and entries[n].get('lfs') for n in names),
                    'missing LFS component weights: ' + group)
    # Deduplicate identical Git objects (tokenizer assets occur in three directories).
    unique = {}
    for name, item in sorted(entries.items()):
        if item.get('lfs') is None:
            key = item['blobId']
            if key in unique:
                require(unique[key][1] == item['size'], 'inconsistent Git object size')
            unique[key] = (name, item['size'])
    small_bytes = sum(size for _, size in unique.values())
    require(small_bytes <= SMALL_CAP, 'small-file budget exceeds 32 MiB; no partial/alternative lock')
    verified = {}
    for blob, (name, size) in sorted(unique.items()):
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name, safe="/")}'
        data = io.get(url, size)
        require(len(data) == size, 'Git size mismatch: ' + name)
        require(git_blob(data) == blob, 'Git blobId mismatch: ' + name)
        verified[blob] = data
    hashes, sources = {}, {}
    for name, item in sorted(entries.items()):
        lfs = item.get('lfs')
        digest = lfs['sha256'] if lfs else sha256(verified[item['blobId']])
        hashes[name] = digest
        sources[name] = {'size': item['size'], 'blobId': item['blobId'], 'sha256': digest,
                         'method': 'revision-api-lfs-sha256' if lfs else 'verified-git-blob-bytes'}
        if lfs:
            sources[name]['lfs'] = lfs
        if name.endswith('.safetensors.index.json'):
            require(not lfs, 'weight index must be small Git JSON')
            index = json.loads(verified[item['blobId']])
            mapping = index.get('weight_map')
            require(isinstance(mapping, dict) and mapping, 'invalid weight index')
            for shard in mapping.values():
                require(isinstance(shard, str) and '/' not in shard and '\\' not in shard
                        and shard.endswith('.safetensors'), 'unsafe indexed shard')
                target = name.rsplit('/', 1)[0] + '/' + shard
                require(target in entries and entries[target].get('lfs'), 'missing indexed LFS shard')
    component = {n: h for n, h in hashes.items() if not n.startswith('transformer_ref/')}
    transformer = {n: h for n, h in hashes.items() if n.startswith('transformer_ref/')}
    lock = {'schema': 'h3-local-model-lock-v1', 'revision': REVISION, 'files': component}
    transformer_doc = {'schema': 'h3-transformer-assets-v1', 'repo': REPO, 'revision': REVISION,
                       'transformer_files': sorted(transformer), 'files': transformer}
    artifacts = {'model-lock.json': canonical(lock),
                 'transformer-assets.json': canonical(transformer_doc), 'source-metadata.json': raw}
    totals = {'component_lfs_bytes': sum(entries[n]['size'] for n in component if entries[n].get('lfs')),
              'transformer_lfs_bytes': sum(entries[n]['size'] for n in transformer if entries[n].get('lfs')),
              'small_download_bytes': small_bytes,
              'small_inventory_bytes': sum(i['size'] for i in entries.values() if not i.get('lfs')),
              'component_files': len(component), 'transformer_files': len(transformer)}
    receipt = {'schema': 'h3-model-asset-source-receipt-v1', 'repo': REPO, 'revision': REVISION,
               'metadata_url': API_URL, 'small_cap_bytes': SMALL_CAP, 'totals': totals,
               'artifact_sha256': {n: sha256(b) for n, b in artifacts.items()}, 'expected_files': sources,
               'weights_downloaded': False, 'jobs_submitted': False}
    artifacts['source-receipt.json'] = canonical(receipt)
    return artifacts, receipt


def save(output, artifacts):
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not output.is_symlink() and output.stat().st_mode & 0o077 == 0, 'output must be private (0700)')
    require(all(not (output / n).exists() and not (output / n).is_symlink() for n in artifacts),
            'refuse to replace existing artifacts')
    for name, data in artifacts.items():
        fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--token-file', type=Path, help='explicit cached token override; never printed')
    args = parser.parse_args()
    token = args.token_file.expanduser().read_text().strip() if args.token_file else os.environ.get('HF_TOKEN')
    artifacts, receipt = prepare(HubIO(token))
    save(args.output, artifacts)
    print(json.dumps({'model_lock_sha256': receipt['artifact_sha256']['model-lock.json'],
                      **receipt['totals']}, sort_keys=True))


if __name__ == '__main__':
    main()
