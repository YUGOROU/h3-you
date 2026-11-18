"""Materialize isolated Trainer/Diffusers source overlay; never edit site-packages.

Run only after pristine runtime validation, before model construction/imports.
Inputs must be bound by the new deployment closure, not self-certified on worker.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from research import install_identity_routing as overlay


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    root = Path(root).resolve()
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('source symlinks forbidden')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            result[str(path.relative_to(root))] = sha256(path)
    if not result:
        raise ValueError('source inventory empty')
    return result


def inventory_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def install(stage, *, stage_sha256, base_diffusers, base_inventory_sha256, destination):
    stage, base, destination = Path(stage), Path(base_diffusers), Path(destination).absolute()
    if sha256(stage / 'OVERLAY_MANIFEST.json') != stage_sha256:
        raise ValueError('source overlay manifest pin mismatch')
    source = overlay.inspect_stage(stage)
    baseline = inventory(base)
    if inventory_digest(baseline) != base_inventory_sha256:
        raise ValueError('independently pinned pristine Diffusers inventory differs')
    module = 'models/transformers/transformer_minimax_h3.py'
    if baseline.get(module) != overlay.DIFFUSERS_SHA256:
        raise ValueError('pristine H3 implementation required')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.h3-runtime-overlay-', dir=destination.parent))
    try:
        for name in source['files']:
            if name.startswith('trainer/'):
                target = temporary / name.removeprefix('trainer/')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / name, target)
        shadow = temporary / 'diffusers-shadow/diffusers'
        for name in baseline:
            target = shadow / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / name, target)
        shutil.copyfile(stage / ('diffusers/src/diffusers/' + module), shadow / module)
        files = inventory(temporary)
        receipt = dict(schema='h3-isolated-runtime-overlay-v1', source_overlay_sha256=stage_sha256,
                       base_inventory_sha256=base_inventory_sha256, files=files,
                       site_packages_modified=False, actual_h3_gpu_verified=False)
        (temporary / 'RUNTIME_OVERLAY.json').write_text(json.dumps(receipt, indent=2, sort_keys=True))
        inspect(temporary, stage_sha256=stage_sha256)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def inspect(root, *, stage_sha256):
    root = Path(root)
    receipt = json.loads((root / 'RUNTIME_OVERLAY.json').read_text())
    if (receipt.get('schema') != 'h3-isolated-runtime-overlay-v1'
            or receipt.get('source_overlay_sha256') != stage_sha256
            or receipt.get('site_packages_modified') is not False
            or receipt.get('actual_h3_gpu_verified') is not False):
        raise ValueError('isolated overlay receipt policy mismatch')
    actual = inventory(root)
    actual.pop('RUNTIME_OVERLAY.json', None)
    if actual != receipt.get('files'):
        raise ValueError('isolated overlay source integrity differs')
    return receipt
