"""Reuse byte-verified inventory within one preparation stage, not across jobs."""
import hashlib
import os
import re


GROUPS = ('vae', 'audio_vae', 'text_encoder', 'tokenizer', 'processor')


def hash_asset(path, *, git=False):
    """Compute both required digests in a single streaming read."""
    sha = hashlib.sha256()
    blob = hashlib.sha1() if git else None
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        if blob is not None:
            blob.update(('blob ' + str(size) + '\0').encode())
        count = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(chunk)
            sha.update(chunk)
            if blob is not None:
                blob.update(chunk)
    if count != size:
        raise ValueError('asset size changed while hashing')
    return sha.hexdigest(), blob.hexdigest() if blob is not None else None


def encoder_inventory(inventory, lock):
    """Select encoder identities from provision()'s SAME-stage verified hashes.

    Not a byte verifier or standalone receipt. Call only immediately after full
    API/revision/size/hash verification. No persisted verification bypass.
    """
    if (lock.get('schema') != 'h3-local-model-lock-v1' or
            re.fullmatch('[0-9a-f]{40}', lock.get('revision', '')) is None):
        raise ValueError('immutable model lock required')
    names = {'model_index.json'}
    for group in GROUPS:
        selected = {n for n in inventory if n.startswith(group + '/')}
        if not selected or not any(n.endswith('.json') for n in selected):
            raise ValueError('component config required: ' + group)
        if group in ('vae', 'audio_vae', 'text_encoder') and not any(n.endswith('.safetensors') for n in selected):
            raise ValueError('component weights required: ' + group)
        names.update(selected)
    if set(lock.get('files', {})) != names:
        raise ValueError('exact encoder inventory required')
    if any(n not in inventory or inventory[n] != lock['files'][n] or
           re.fullmatch('[0-9a-f]{64}', inventory[n]) is None for n in names):
        raise ValueError('verified inventory differs from model lock')
    return {n: inventory[n] for n in sorted(names)}
