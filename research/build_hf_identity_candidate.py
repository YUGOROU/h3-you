"""Reuse the previously reviewed HF training-only spec with a new runtime closure.

Pure builder, no uploads/submissions/automatic provider fallback. Existing HF
approval documents stay byte-identical; requested provider switch is explicit.
"""
import copy
from pathlib import Path
import re
from research import hf_bootstrap_identity_pipeline as wire
from research import hf_restore_identity_runtime as runtime


def build(previous_spec, *, runtime_mount, runtime_repo, runtime_revision,
          rate_usd_hour, total_seconds=18000):
    old = wire.load(previous_spec)
    if old['schema'] != 'h3-identity-launch-spec-v1':
        raise ValueError('previous reviewed HF launch spec required')
    if not re.fullmatch('[0-9a-f]{40}', runtime_revision):
        raise ValueError('new uploaded immutable runtime revision required')
    root = Path(runtime_mount).absolute()
    manifest_path = root / 'runtime-manifest.json'
    manifest = wire.load(manifest_path)
    if manifest['schema'] != runtime.SCHEMA or manifest['image'] != old['image']:
        raise ValueError('same pristine runtime/image contract required')
    for name, digest in {**manifest['bootstrap_files'], **manifest['artifact_files']}.items():
        wire.pinned(root, {'path': name, 'sha256': digest})
    if not 0 < rate_usd_hour <= 10 or type(total_seconds) is not int or not 6000 <= total_seconds <= 18000:
        raise ValueError('finite production candidate within HF training reserve')
    if rate_usd_hour * total_seconds / 3600 > 25:
        raise ValueError('HF candidate exceeds protected25 dollar reserve')
    spec = copy.deepcopy(old)
    r = spec['pipeline']
    if r['territory_approval'].get('provider') != 'huggingface_jobs':
        raise ValueError('existing provider-bound HF approval required')
    # Preserve objective/base/optimizer/sampling; new recoverable production run.
    r.update(overall_seconds=total_seconds - 1800, rate_usd_hour=rate_usd_hour,
             checkpoint_escrow=True, checkpoint_interval=10)
    if len(r.get('inference', {}).get('sample_ids', [])) not in (1, 2):
        raise ValueError('retain same-Job short preview cases')
    r['runtime_provenance']['runtime_mount_revision'] = runtime_revision
    for key in ('runtime_manifest', 'runtime_lock'):
        path = manifest_path if key == 'runtime_manifest' else root / manifest['documents']['runtime_lock']['path']
        spec['documents'][key] = {'path': str(path), 'sha256': wire.sha256(path)}
    spec['runtime_bootstrap'].update(repo=runtime_repo, revision=runtime_revision,
        module=manifest['module'], entrypoint=manifest['entrypoint'],
        manifest={'path': 'runtime-manifest.json', 'sha256': wire.sha256(manifest_path)}, timeout_seconds=1800)
    return spec
