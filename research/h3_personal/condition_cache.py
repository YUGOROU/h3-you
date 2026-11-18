"""Private content-addressed conditioning artifacts; no torch/network at import.

Only a missing entry is a miss. Corrupt/partial/symlinked entries fail closed.
One writer per inference session; no prompt, image or credentials are stored here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from .component_cache import canonical, sha256

SCHEMA = 'h3-prompt-condition-cache-v1'


def address(identity):
    return hashlib.sha256(canonical(identity)).hexdigest()


def _root(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('condition cache root must not be a symlink')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir() or root.stat().st_mode & 0o077:
        raise ValueError('condition cache must be a private directory')
    return root


def load(root, identity, loader):
    root = _root(root); key = address(identity); entry = root / key
    if not entry.exists() and not entry.is_symlink():
        return None
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError('condition cache entry must be a regular directory')
    if {p.name for p in entry.iterdir()} != {'condition.safetensors', 'VERIFIED.json'}:
        raise ValueError('incomplete or unexpected condition cache inventory')
    payload = entry / 'condition.safetensors'; marker = entry / 'VERIFIED.json'
    if any(p.is_symlink() or not p.is_file() for p in (payload, marker)):
        raise ValueError('condition cache files must not be symlinks')
    if marker.stat().st_size > 4096:
        raise ValueError('oversized condition marker')
    meta = json.loads(marker.read_text())
    if (set(meta) != {'schema', 'key_sha256', 'payload_sha256', 'payload_bytes'} or
            meta['schema'] != SCHEMA or meta['key_sha256'] != key or
            meta['payload_bytes'] != payload.stat().st_size or
            meta['payload_sha256'] != sha256(payload)):
        raise ValueError('condition cache provenance/bytes differ')
    return loader(payload)


def store(root, identity, writer):
    root = _root(root); key = address(identity)
    if (root / key).exists() or (root / key).is_symlink():
        raise ValueError('condition cache entries are immutable; single writer required')
    staging = Path(tempfile.mkdtemp(prefix='.staging-', dir=root))
    try:
        payload = staging / 'condition.safetensors'; writer(payload)
        if payload.is_symlink() or not payload.is_file():
            raise ValueError('condition writer must produce a regular artifact')
        payload.chmod(0o600)
        marker = staging / 'VERIFIED.json'
        marker.write_bytes(canonical({'schema': SCHEMA, 'key_sha256': key,
            'payload_sha256': sha256(payload), 'payload_bytes': payload.stat().st_size}))
        marker.chmod(0o600)
        os.rename(staging, root / key)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return key
