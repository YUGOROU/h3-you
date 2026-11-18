"""Location of the private working area, outside this repository.

The pipeline keeps raw media, generated artifacts, owner locks and the compute
ledger outside version control. Nothing here is uploaded by the code; callers
only resolve paths under one root so that every lock and fence lives in the same
place.

Resolution order:

1. ``H3_PRIVATE_ROOT`` (absolute path) when set.
2. ``$H3_PRIVATE_BASE/<H3_PRIVATE_SESSION>`` where ``H3_PRIVATE_BASE`` defaults
   to ``~/Downloads/H3Personal`` and ``H3_PRIVATE_SESSION`` defaults to
   ``2026-10-12/production-715``.

The default reproduces the layout used while developing this pipeline. Set
``H3_PRIVATE_ROOT`` to point at any other private directory.
"""
import os
from pathlib import Path

ENV_VAR = 'H3_PRIVATE_ROOT'
BASE_ENV_VAR = 'H3_PRIVATE_BASE'
SESSION_ENV_VAR = 'H3_PRIVATE_SESSION'
DEFAULT_BASE = Path('~/Downloads/H3Personal')
DEFAULT_SESSION = Path('2026-10-12/production-715')


def base(environ=None):
    """Parent of the session directory. Raw media may live here too."""
    environ = os.environ if environ is None else environ
    raw = environ.get(BASE_ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_BASE.expanduser()


def session(environ=None):
    """Session directory name (a literal relative path, never absolute)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(SESSION_ENV_VAR)
    return Path(raw) if raw else DEFAULT_SESSION


def root(environ=None):
    """Absolute private root. ``H3_PRIVATE_ROOT`` wins when it is set."""
    environ = os.environ if environ is None else environ
    override = environ.get(ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError(ENV_VAR + ' must be an absolute path')
        return path
    return base(environ) / session(environ)


def ledger(environ=None):
    """File the compute ledger and reservation guards live in.

    It sits beside the private root, not inside it, so the ledger is shared by
    every session directory under the same parent.
    """
    return root(environ).parent / 'compute-budget.json'


def describe(environ=None):
    """Non-secret summary for startup preflight checks."""
    return dict(env_var=ENV_VAR, base=str(base(environ)),
                session=str(session(environ)), root=str(root(environ)),
                ledger=str(ledger(environ)))
