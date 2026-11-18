"""Bounded Modal carrier for a reviewed, byte-bound H3 deployment closure.

Defining an App does not submit it. Caller must separately approve the actual
multi-hour spec and provide the completed provider-neutral bootstrap closure.
No volumes, warm pool, fallback GPU, retries or unbounded log streaming.
"""
import hashlib
from pathlib import Path
import re
from research.modal_gpu_plan import hourly, multiplier, validate_placement

IMAGE = 'pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385'
RATE = 6.774192  # H200 + hard4 physical CPU +256GiB, base region pricing


def validate(spec, root):
    if set(spec) - {'placement'} != {'schema', 'files', 'entrypoint', 'seconds', 'startup_seconds', 'cap_usd', 'region_multiplier'}:
        raise ValueError('literal bounded carrier spec required; no credentials in spec')
    if spec['schema'] != 'h3-modal-carrier-v1':
        raise ValueError('carrier schema')
    for key in ('seconds', 'startup_seconds'):
        if type(spec[key]) is not int or not 1 <= spec[key] <= 21600:
            raise ValueError('finite carrier deadlines required')
    if spec['region_multiplier'] != multiplier(spec.get('placement', {}).get('region')):
        raise ValueError('regional premium must match published Modal region pricing')
    region = spec.get('placement', {}).get('region')
    exposure = hourly('H200', region) * (spec['seconds'] + spec['startup_seconds'] + 2) / 3600
    if type(spec['cap_usd']) not in (int, float) or not exposure <= spec['cap_usd'] <= 49:
        raise ValueError('bounded function time exposure exceeds forward training cap')
    if 'placement' in spec:
        from research import hf_bootstrap_identity_pipeline as wire
        placement = spec['placement']
        if (set(placement) != {'cloud', 'region', 'country', 'pricing_evidence'}
                or placement['cloud'] not in {'aws', 'gcp', 'provider_selected'}):
            raise ValueError('reviewed placement fields required')
        evidence = wire.load(wire.pinned(root, placement['pricing_evidence']))
        validate_placement(placement, evidence)
    files = spec['files']
    if not isinstance(files, dict) or not files or len(files) > 512:
        raise ValueError('explicit source/document closure required')
    root = Path(root).resolve()
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(files):
        raise ValueError('closure inventory mismatch')
    for name, digest in files.items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or name != path.as_posix()
                or not (name.endswith(('.py', '.json', '.txt', '.toml', '.gz')) or path.name == 'LICENSE')
                or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('unsafe closure file')
        source = root / path
        if source.is_symlink() or not source.resolve().is_relative_to(root) or source.stat().st_size > 8 * 1024**2:
            raise ValueError('source/document payload bounds')
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('closure hash mismatch')
    if spec['entrypoint'] not in files or not spec['entrypoint'].endswith('.py'):
        raise ValueError('byte-bound Python bootstrap entrypoint required')
    return exposure


def safe_diagnostic(path):
    """Return only bounded predefined phase/type fields, never raw logs/paths."""
    import json
    path = Path(path)
    try:
        if path.is_symlink() or path.stat().st_size > 1024:
            return None
        value = json.loads(path.read_text())
        if (value.get('schema') != 'h3-modal-bootstrap-status-v1' or
                value.get('stage') not in {'validate', 'inputs', 'runtime', 'pipeline', 'complete'}):
            return None
        failure = value.get('failure_type')
        if failure is not None and (not isinstance(failure, str) or
                                    not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,127}', failure)):
            return None
        return {'stage': value['stage'], 'failure_type': failure}
    except (OSError, ValueError, AttributeError):
        return None


def safe_progress(line):
    """Only controlled stage names and numeric preview counters; never raw child logs."""
    if len(line) > 4096:
        return None
    import re
    value = line.strip()
    if re.fullmatch(r'H3_PHASE (validate|runtime|inputs|pipeline|hardware|prepare|handoff|train|verify-upload|preview|complete)', value):
        return value
    if re.fullmatch(r'PREVIEW_STAGE [12] (generate|decode)', value):
        return value
    if re.fullmatch(r'PREVIEW_DENOISE [0-9]{1,3} / [0-9]{1,3}', value):
        return value
    if re.fullmatch(r'PREVIEWS_UPLOADED [12]', value):
        return value
    return None


def build_app(spec, source_root, *, hf_secret_name=None, hf_token=None, modal_module=None):
    """Lazy SDK definition only; source upload/build/remote call are NOT performed."""
    validate(spec, source_root)
    if (hf_secret_name is None) == (hf_token is None):
        raise ValueError('choose exactly one dynamic token or managed secret reference')
    if hf_secret_name is not None and (not isinstance(hf_secret_name, str) or
                                      not re.fullmatch('[A-Za-z0-9_-]{1,128}', hf_secret_name)):
        raise ValueError('explicit managed HF secret name required; never token value')
    if hf_token is not None and (not isinstance(hf_token, str) or not hf_token or hf_token.strip() != hf_token):
        raise ValueError('explicit in-memory cached credential required')
    carrier = 'research/modal_identity_worker.py'
    if (spec['files'].get(carrier) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('closure must contain the actual reviewed carrier module')
    if modal_module is None:
        import modal as modal_module
    modal = modal_module
    image = modal.Image.from_registry(IMAGE).add_local_dir(str(Path(source_root).resolve()), '/deployment', copy=True).env({'PYTHONPATH': '/deployment', 'PYTHONDONTWRITEBYTECODE': '1'})
    app = modal.App('h3-identity-bounded')
    secret = (modal.Secret.from_dict({'HF_TOKEN': hf_token}) if hf_token is not None else
              modal.Secret.from_name(hf_secret_name, required_keys=['HF_TOKEN']))

    placement = ({k: spec['placement'][k] for k in ('cloud', 'region')} if 'placement' in spec else {})
    if placement.get('cloud') == 'provider_selected':
        placement.pop('cloud')  # Account API rejects cloud pinning; still enforce Japan region.
    @app.function(**placement, image=image, secrets=[secret], gpu='H200', cpu=(4, 4), memory=(262144, 262144),
                  ephemeral_disk=524288, timeout=spec['seconds'], startup_timeout=spec['startup_seconds'],
                  max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
                  retries=0, single_use_containers=True, serialized=True, include_source=False)
    def execute():
        # The closure must implement provisioning, approvals and same-Job previews;
        # the carrier itself cannot be mistaken for a production-ready Trainer.
        import os
        import subprocess
        import time
        os.umask(0o077)
        validate(spec, '/deployment')
        env = dict(os.environ, H3_RUNTIME_IMAGE=IMAGE, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', WANDB_DISABLED='true', H3_CARRIER_SECONDS=str(spec['seconds']),
                   H3_CARRIER_RATE=str(hourly('H200', spec.get('placement', {}).get('region'))),
                   H3_REQUESTED_PLACEMENT=__import__('json').dumps(spec.get('placement', {}), sort_keys=True))
        started = time.monotonic()
        child = subprocess.Popen(['/opt/conda/bin/python', '/deployment/' + spec['entrypoint']],
                                 env=env, cwd='/deployment', stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True)
        import threading
        def forward():
            for line in iter(lambda: child.stdout.readline(4097), ''):
                value = safe_progress(line)
                if value is not None:
                    print(value, flush=True)
        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        try:
            code = child.wait(timeout=max(1, spec['seconds'] - 10))
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
            code = 124
        reader.join(timeout=2)
        return {'schema': 'h3-modal-carrier-result-v1', 'exit_code': code,
                'elapsed_seconds': time.monotonic() - started, 'quality_verified': False,
                'diagnostic': safe_diagnostic('/tmp/h3-bootstrap-status.json')}
    return app, execute
