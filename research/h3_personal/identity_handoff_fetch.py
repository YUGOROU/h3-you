"""Atomic private-revision checkpoint fetch for REMOTE workers only.

No GPU allocation/optimizer resume. Call under the provider supervisor timeout.
Credentials stay in the Hub SDK environment/managed secret, never in the spec.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile
from . import identity_checkpoint as cp, identity_handoff as handoff


def fetch_handoff(*, repo_id, revision, prefix, metadata_sha256, destination,
                  pins, expected_recipe_sha256, repo_info=None, download=None):
    if (not isinstance(repo_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo_id)
            or not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision)
            or not isinstance(metadata_sha256, str) or not re.fullmatch('[0-9a-f]{64}', metadata_sha256)):
        raise ValueError('immutable private repo/revision/metadata digest required')
    if not isinstance(prefix, str):
        raise ValueError('canonical escrow prefix required')
    match = re.fullmatch(r'identity/([A-Za-z0-9][A-Za-z0-9_-]{0,127})/escrow/checkpoint-([0-9]+)', prefix)
    if match is None:
        raise ValueError('canonical escrow prefix required')
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if repo_info is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download
        repo_info = repo_info or HfApi().repo_info
        download = download or hf_hub_download
    info = repo_info(repo_id=repo_id, repo_type='model', revision=revision)
    if info.private is not True or info.sha != revision:
        raise ValueError('exact revision of private model repository required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.h3-handoff-fetch-', dir=destination.parent))
    try:
        checkpoint = staging / 'checkpoint'
        checkpoint.mkdir(mode=0o700)
        # Fetch/pin small metadata first; reject recipe before adapter download.
        source = Path(download(repo_id=repo_id, repo_type='model', revision=revision,
                               filename=f'{prefix}/HANDOFF.json'))
        if source.stat().st_size > 1024**2 or cp.sha256(source) != metadata_sha256:
            raise ValueError('handoff metadata size/hash mismatch')
        metadata = json.loads(source.read_text())
        if (not isinstance(metadata, dict) or metadata.get('recipe_sha256') != expected_recipe_sha256
                or metadata.get('parent_run_id') != match[1]
                or type(metadata.get('parent_step')) is not int
                or metadata['parent_step'] != int(match[2])):
            raise ValueError('handoff source recipe/run/update mismatch')
        shutil.copyfile(source, staging / 'HANDOFF.json')
        if cp.sha256(staging / 'HANDOFF.json') != metadata_sha256:
            raise ValueError('handoff metadata changed during copy')
        for name in handoff.ARTIFACTS:
            source = Path(download(repo_id=repo_id, repo_type='model', revision=revision,
                                   filename=f'{prefix}/{name}'))
            shutil.copyfile(source, checkpoint / name)
        handoff.validate_handoff(checkpoint, metadata, pins=pins,
                                 expected_recipe_sha256=expected_recipe_sha256)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(checkpoint=str(destination / 'checkpoint'), metadata=str(destination / 'HANDOFF.json'),
                metadata_sha256=metadata_sha256)
