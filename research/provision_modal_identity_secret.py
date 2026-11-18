"""Provision one managed HF secret; no token files, output, overwrites or GPU calls."""
import re
from pathlib import Path


def cached_token(path=None):
    """Explicit user cache only; ignore harness HF_TOKEN/HF_HOME and never print."""
    path = Path(path) if path is not None else Path.home() / '.cache/huggingface/token'
    token = path.read_text().strip()
    if not token:
        raise ValueError('user cached credential unavailable')
    return token


def ensure_secret(*, manager, name, token):
    if not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', name):
        raise ValueError('explicit dedicated managed secret name required')
    if not isinstance(token, str) or not token or token.strip() != token:
        raise ValueError('in-memory HF credential required')
    # Never inspect values or overwrite an existing operator-managed secret.
    existing = manager.list(max_objects=1000)
    if any(secret.name == name for secret in existing):
        return {'managed_secret_exists': True, 'created': False, 'values_verified': False}
    manager.create(name, {'HF_TOKEN': token}, allow_existing=False)
    return {'managed_secret_exists': True, 'created': True, 'values_verified': False}


def check_hub_permissions(api, *, input_repo, cache_repo, output_repo):
    for repo, kind, write in ((input_repo, 'dataset', False), (cache_repo, 'dataset', True),
                              (output_repo, 'model', True)):
        if api.repo_info(repo, repo_type=kind).private is not True:
            raise ValueError('existing private input/cache/output repositories required')
        api.auth_check(repo, repo_type=kind, write=write)
    return {'private_repository_permissions_checked': True}
