"""Finite REMOTE CPU-only preparation of a private, pinned58-file mount.

Standalone worker: invoked under an explicitly authorized HF Job timeout.
One file at a time (no144GB local disk requirement); no retries or GPU allocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def hash_file(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    path = Path(args.request)
    assert hash_file(path) == args.sha256
    r = json.loads(path.read_text())
    assert r['source'] == 'MiniMaxAI/MiniMax-H3'
    assert r['revision'] == '42ed227ee7df40d41602854ae760620d6eb651fe'
    assert len(r['files']) == 58 and r['destination'].startswith('YUGOROU/')
    for name in r['files']:
        assert not name.startswith('/') and '\\' not in name
        assert all(part not in ('', '.', '..') for part in name.split('/'))
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    token = os.environ['HF_TOKEN']
    api = HfApi(token=token)
    assert api.repo_info(r['destination']).private is True
    api.auth_check(r['destination'], write=True)
    info = api.model_info(r['source'], revision=r['revision'], files_metadata=True)
    assert info.sha == r['revision']
    metadata = {s.rfilename: s for s in info.siblings}
    work = Path('/tmp/h3-curate-file')
    assert not work.exists()
    inventory = {}
    for i, name in enumerate(sorted(r['files']) + ['LICENSE']):
        work.mkdir(mode=0o700)
        try:
            src = Path(hf_hub_download(r['source'], name, revision=r['revision'],
                                      local_dir=work, token=token))
            item = metadata[name]
            assert not src.is_symlink() and src.stat().st_size == item.size
            sha = hash_file(src)
            if item.lfs:
                assert sha == item.lfs.sha256
            else:
                assert hash_file(src, git=True) == item.blob_id
            if name != 'LICENSE':
                assert sha == r['files'][name]['sha256']
                inventory[name] = sha
            api.create_commit(r['destination'], operations=[CommitOperationAdd(
                path_in_repo=('assets/' + name if name != 'LICENSE' else name),
                path_or_fileobj=str(src))], commit_message='Private pinned asset; no model or objective change')
            print(f'CURATED_ASSET {i+1} / 59', flush=True)
        finally:
            shutil.rmtree(work)
    receipt = dict(schema='h3-curated-model-assets-v1', source=r['source'],
                   source_revision=r['revision'], files=inventory, complete=True,
                   execution_country='unknown', invoice_verified=False)
    commit = api.create_commit(r['destination'], operations=[CommitOperationAdd(
        path_in_repo='CURATED.json', path_or_fileobj=json.dumps(receipt, sort_keys=True).encode())],
        commit_message='Complete private verified58-file model mount')
    final = api.model_info(r['destination'], revision=commit.oid, files_metadata=True)
    assert final.private is True
    assets = {s.rfilename.removeprefix('assets/'): s for s in final.siblings if s.rfilename.startswith('assets/')}
    assert set(assets) == set(r['files'])
    for name, item in assets.items():
        assert item.size == metadata[name].size
        if item.lfs:
            assert item.lfs.sha256 == inventory[name]
        else:
            assert item.blob_id == metadata[name].blob_id
    print(json.dumps({'state':'complete', 'repo':r['destination'], 'revision':commit.oid,
                      'mount_subfolder':'assets', 'files':58}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
