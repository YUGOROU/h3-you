"""Read-only namespace catalog observation; never runtime or launch approval."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from research.provision_modal_identity_secret import cached_token

COMMAND = ('hf', 'endpoints', 'hardware', '--namespace', 'YUGOROU',
           '--instance-type', 'nvidia-h200', '--all', '--format', 'json')
MAX_BYTES = 1024 * 1024
TIMEOUT = 30
REGIONS = {
    'aws': {'ap-northeast-1', 'ap-northeast-3', 'ap-southeast-2', 'ap-southeast-4'},
    'gcp': {'asia-northeast1', 'asia-northeast2', 'australia-southeast1', 'australia-southeast2'},
    'azure': {'japaneast', 'japanwest', 'australiaeast', 'australiasoutheast', 'australiacentral', 'australiacentral2'},
}


class PreflightError(ValueError):
    """Only constant, non-sensitive messages may escape the query boundary."""


def bounded_runner(command, *, env, timeout):
    """Bound both pipes in memory and wall time; discard raw diagnostics."""
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, True)
                selector.register(process.stderr, selectors.EVENT_READ, False)
                output = bytearray()
                total = 0
                deadline = time.monotonic() + timeout
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PreflightError('catalog timeout')
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise PreflightError('catalog output limit')
                        if key.data:
                            output.extend(chunk)
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                return subprocess.CompletedProcess(command, code, bytes(output), b'')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def sanitize(payload):
    if not isinstance(payload, list) or len(payload) > 4096:
        raise PreflightError('invalid catalog')
    observations = []
    for row in payload:
        if not isinstance(row, dict):
            raise PreflightError('invalid catalog')
        if row.get('instance_type') != 'nvidia-h200' or row.get('instance_size') != 'x2':
            continue
        vendor, region = row.get('vendor'), row.get('region')
        # Unknown regions are not echoed, including arbitrary strings or credentials.
        if not isinstance(vendor, str) or vendor not in REGIONS:
            continue
        if not isinstance(region, str) or region not in REGIONS[vendor]:
            continue
        used, maximum, count = (row.get(k) for k in
                                ('used_accelerators', 'max_accelerators', 'num_accelerators'))
        if (row.get('status') != 'available' or row.get('accelerator') != 'gpu'
                or any(type(n) is not int or n < 0 for n in (used, maximum, count))
                or count != 2 or used + 2 > maximum):
            continue
        observations.append({k: row[k] for k in (
            'vendor', 'region', 'instance_type', 'instance_size', 'num_accelerators',
            'status', 'used_accelerators', 'max_accelerators')})
    return observations


def preflight(*, query=False, directory=None, runner=bounded_runner, token_getter=cached_token):
    result = {'scope': 'single namespace catalog observation; not runtime, gateway or launch approval',
              'namespace': 'YUGOROU', 'queried': False, 'observations': []}
    if not query:
        return result
    if directory is None:
        raise PreflightError('fresh external directory required')
    root = Path(os.path.abspath(directory))
    repo = Path(__file__).resolve().parents[1]
    if any(p.is_symlink() for p in (root, *root.parents)) or root.exists():
        raise PreflightError('fresh non-symlink directory required')
    if root == repo or repo in root.parents:
        raise PreflightError('external directory required')
    try:
        root.mkdir(mode=0o700)  # parent must already exist; no implicit hierarchy
        os.chmod(root, 0o700)
        token = token_getter()
        if not isinstance(token, str) or not token or token.strip() != token:
            raise PreflightError('credential unavailable')
        env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ}
        env['HF_TOKEN'] = token
        response = runner(COMMAND, env=env, timeout=TIMEOUT)
        if response.returncode != 0:
            raise PreflightError('catalog query failed')
        raw = response.stdout
        if not isinstance(raw, (str, bytes)) or len(raw) > MAX_BYTES:
            raise PreflightError('invalid catalog output')
        if token in (raw.decode('utf-8') if isinstance(raw, bytes) else raw):
            raise PreflightError('credential in catalog output')
        result['observations'] = sanitize(json.loads(raw))
        result['queried'] = True
        result['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
        result['catalog_entry_count'] = len(json.loads(raw))
        path = root / 'endpoint-catalog.json'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        return result
    except Exception:
        raise PreflightError('catalog observation failed') from None


def main(argv=None, *, runner=bounded_runner, token_getter=cached_token):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', action='store_true')
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args(argv)
    try:
        result = preflight(query=args.query, directory=args.directory,
                           runner=runner, token_getter=token_getter)
    except PreflightError:
        print('endpoint catalog preflight failed')
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
