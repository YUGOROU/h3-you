"""Explicit H3 Job compute credentials; assets/Endpoint/Pi credentials stay main.

Loading performs authentication GET only, never Job creation or account mutation.
No environment token fallback, global auth switch, or credential persistence.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
import stat

MAIN_NAMESPACE = 'YUGOROU'
BACKUP_NAMESPACE = '2nd2nd'
CANONICAL_ENDPOINT = 'https://huggingface.co'
TERMINAL = frozenset(('CANCELED', 'COMPLETED', 'ERROR', 'DELETED'))


def validate_namespace(namespace):
    if type(namespace) is not str or namespace not in (MAIN_NAMESPACE, BACKUP_NAMESPACE):
        raise ValueError('explicit supported H3 Job namespace required')
    return namespace


def _token(value):
    if (type(value) is not str or not 1 <= len(value) <= 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in value)):
        raise ValueError('bounded private compute credential required')
    return value


def _symlinked(path):
    """Reject a symlinked credential or a symlinked directory below the home root.

    Platform roots above the home directory may legitimately be symlinks
    (/var -> /private/var on macOS); they are not part of the credential's
    operator-controlled path and are never traversed by this reader.
    """
    home = Path.home()
    for parent in path.parents:
        if parent == home:
            return False
        if parent.is_symlink():
            return True
    return True


def _read_backup_token():
    path = Path.home()/'.cache/huggingface-backup/token-backup'
    if path.is_symlink() or _symlinked(path):
        raise ValueError('nonsymlink backup credential required')
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= 8192:
                raise ValueError('bounded regular backup credential required')
            raw = stream.read(8193)
            after = os.fstat(stream.fileno())
            identity = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            if identity(before) != identity(after) or len(raw) != before.st_size:
                raise ValueError('backup credential changed during reading')
        current = path.stat(follow_symlinks=False)
        if (path.is_symlink() or _symlinked(path)
                or identity(current) != identity(before)):
            raise ValueError('backup credential path changed during reading')
        return _token(raw.decode('ascii').strip())
    except (OSError, UnicodeError):
        raise ValueError('backup credential unavailable or invalid') from None


def _jobs_scope(identity):
    """Reject known-denied token metadata, without inventing a permission API.

    Missing scope metadata is not proof of Job-create eligibility: actual Jobs
    GET/POST authorization remains authoritative and never triggers fallback.
    """
    auth = identity.get('auth', {})
    if not isinstance(auth, dict):
        return
    access = auth.get('accessToken', {})
    if not isinstance(access, dict):
        return
    role = access.get('role')
    if role == 'read':
        raise ValueError('read-only token cannot own H3 Jobs')
    fine = access.get('fineGrained')
    if isinstance(fine, dict) and 'global' in fine:
        permissions = fine['global']
        if (type(permissions) is not list or any(type(p) is not str for p in permissions)
                or not {'job.read', 'job.write'} <= set(permissions)):
            raise ValueError('explicit Job read/write token permissions required')


@dataclass(frozen=True)
class ComputeProfile:
    api: object = field(repr=False)
    token: str = field(repr=False)
    namespace: str

    def __post_init__(self):
        validate_namespace(self.namespace)
        _token(self.token)
        if getattr(self.api, 'endpoint', None) != CANONICAL_ENDPOINT:
            raise ValueError('canonical compute API required')

    def metadata(self):
        """Allowlisted receipt fields only; never serialize api/token objects."""
        return {'namespace': self.namespace}

    def assert_idle(self):
        """GET only. The caller must separately check the main account as well."""
        jobs = self.api.list_jobs(namespace=self.namespace)
        if any(getattr(getattr(job, 'status', None), 'stage', None) not in TERMINAL
               for job in jobs):
            raise ValueError('H3 compute namespace is not idle')


def load_compute_profile(main_api, main_token, *, backup=False):
    if type(backup) is not bool:
        raise ValueError('explicit backup Job selection required')
    main_token = _token(main_token)
    if getattr(main_api, 'endpoint', None) != CANONICAL_ENDPOINT:
        raise ValueError('canonical main API required')
    identity = main_api.whoami()
    if type(identity) is not dict or identity.get('name') != MAIN_NAMESPACE:
        raise ValueError('expected main asset account required')
    if not backup:
        return ComputeProfile(main_api, main_token, MAIN_NAMESPACE)
    token = _read_backup_token()
    if token == main_token:
        raise ValueError('distinct backup compute credential required')
    from huggingface_hub import HfApi
    api = HfApi(endpoint=CANONICAL_ENDPOINT, token=token)
    identity = api.whoami()
    if type(identity) is not dict or identity.get('name') != BACKUP_NAMESPACE:
        raise ValueError('expected backup compute account required')
    _jobs_scope(identity)
    return ComputeProfile(api, token, BACKUP_NAMESPACE)
