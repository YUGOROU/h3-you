"""Finite remote CPU runtime preparation and atomic private receipt upload. No Jobs API."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from research import prepare_identity_runtime as prep


def bounded(command, deadline):
    remaining = deadline - time.monotonic()
    prep.require(remaining > 0, 'runtime preparation deadline exhausted')
    child = subprocess.Popen(command, start_new_session=True,
                             stdout=None, stderr=subprocess.DEVNULL)
    try:
        prep.require(child.wait(timeout=remaining) == 0, 'runtime preparation child failed')
    finally:
        # Kill descendants too, including a stuck pip/APT/import or upload helper.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def repository(repo):
    prep.require(isinstance(repo, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo)
                 and all(p not in ('.', '..') for p in repo.split('/')), 'private dataset repo required')
    return repo


def upload(api, repo, root, prefix, operation=None):
    """Report, lock and base interpreter in ONE private commit; no repo creation."""
    repository(repo)
    prep.require(re.fullmatch(r'runtimes/[0-9a-f]{64}', prefix), 'runtime snapshot prefix required')
    report_path = Path(root) / 'runtime-report.json'
    lock_path = Path(root) / 'runtime-lock.json'
    report = json.loads(report_path.read_text())
    prep.require(report.get('complete') is True and report.get('cpu_imports_verified') is True,
                 'only completed CPU runtime may be published')
    prep.require(report['runtime_lock_sha256'] == prep.sha256(lock_path), 'report/lock binding mismatch')
    interpreter_path = Path(root) / 'base-interpreter-receipt.json'
    interpreter = json.loads(interpreter_path.read_text())
    prep.require(interpreter.get('schema') == 'h3-runtime-base-interpreter-v1' and
                 interpreter.get('complete') is True and interpreter.get('image') == report.get('image') and
                 interpreter.get('cpu_report_sha256') == prep.sha256(report_path) and
                 isinstance(interpreter.get('python'), str) and interpreter['python'].startswith('/') and
                 re.fullmatch('[0-9a-f]{64}', interpreter.get('python_sha256', '')),
                 'bound actual base interpreter receipt required')
    prep.require(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 2 * 1024**2 and
                     p.stat().st_mode & 0o077 == 0 for p in (report_path, lock_path, interpreter_path)),
                 'private bounded artifacts required')
    prep.require(api.repo_info(repo, repo_type='dataset').private is True, 'existing private dataset required')
    api.auth_check(repo, repo_type='dataset', write=True)
    if operation is None:
        from huggingface_hub import CommitOperationAdd
        operation = CommitOperationAdd
    operations = [operation(path_in_repo=prefix + '/' + p.name, path_or_fileobj=str(p))
                  for p in (lock_path, report_path, interpreter_path)]
    # Re-check privacy immediately before publication. No raw media/model/source archive upload.
    info = api.repo_info(repo, repo_type='dataset')
    prep.require(info.private is True, 'dataset became public')
    return api.create_commit(repo_id=repo, repo_type='dataset', operations=operations,
                             parent_commit=info.sha, commit_message='Verified CPU-only identity runtime')


def execute(args, runner=bounded):
    prep.require(sys.platform == 'linux', 'execute only on remote Linux')
    prep.require(type(args.deadline_seconds) is int and 60 <= args.deadline_seconds <= 7200,
                 'finite 60..7200 second deadline required')
    c, _ = prep.expectations(args.expectations, args.expectations_sha256)
    prep.check_trainer(args.source_root, c)
    if args.snapshot_repo:
        repository(args.snapshot_repo)
    prep.require(os.environ.get('H3_RUNTIME_IMAGE') == c['image'], 'immutable image launch binding required')
    os.umask(0o077)
    root = prep.private_new(args.output)
    end = time.monotonic() + args.deadline_seconds
    report = {'schema': 'h3-identity-runtime-prep-report-v1', 'complete': False,
              'cpu_imports_verified': False, 'weights_loaded': False, 'inference_executed': False,
              'training_executed': False, 'gpu_verified': False, 'c_gate_enabled': False,
              'expectations_sha256': args.expectations_sha256,
              'image': c['image'], 'image_identity_basis': 'operator launch binding, not in-container attestation',
              'source_revisions': c['source_revisions'],
              'independent_expectations': {'versions': c['versions'], 'modules': c['modules'],
                  'transformers_source_revision': c['transformers_source_revision'],
                  'diffusers_inventory_sha256': c['diffusers_inventory_sha256']},
              'private_upload_requested': bool(args.snapshot_repo)}
    os.environ.update(H3_RUNTIME_PREP_SUPERVISED='1', CUDA_VISIBLE_DEVICES='',
                      WANDB_MODE='disabled', WANDB_DISABLED='true', PYTHONNOUSERSITE='1')
    command = [sys.executable, '-m', 'research.prepare_identity_runtime', '--internal', 'bootstrap',
               '--expectations', str(args.expectations.resolve()),
               '--expectations-sha256', args.expectations_sha256,
               '--source-root', str(args.source_root.resolve()), '--output', str(root)]
    try:
        runner(command, end - (120 if args.snapshot_repo else 0))
        report.update(complete=True, cpu_imports_verified=True,
                      runtime_lock_sha256=prep.sha256(root / 'runtime-lock.json'),
                      ffmpeg_package_evidence=json.loads((root / 'apt-evidence.json').read_text()),
                      elapsed_seconds=round(args.deadline_seconds - (end - time.monotonic()), 3))
        prep.atomic_json(root / 'runtime-report.json', report)
        # Capture the executing immutable-image base, never the overlay or local builder.
        prep.atomic_json(root / 'base-interpreter-receipt.json', {
            'schema': 'h3-runtime-base-interpreter-v1', 'complete': True,
            'image': c['image'], 'python': str(Path(sys.executable).absolute()),
            'python_sha256': prep.sha256(sys.executable),
            'cpu_report_sha256': prep.sha256(root / 'runtime-report.json')})
        if args.snapshot_repo:
            runner([str(root / 'venv/bin/python'), '-m', 'research.hf_prepare_identity_runtime',
                    '--internal-upload', '--output', str(root), '--snapshot-repo', args.snapshot_repo,
                    '--expectations-sha256', args.expectations_sha256], end)
        return 0
    except BaseException as exc:
        # Failed preparation never leaves a publishable lock/report pair.
        (root / 'runtime-lock.json').unlink(missing_ok=True)
        (root / 'candidate-lock.json').unlink(missing_ok=True)
        (root / 'base-interpreter-receipt.json').unlink(missing_ok=True)
        report.update(complete=False, cpu_imports_verified=False, failure_type=type(exc).__name__)
        prep.atomic_json(root / 'runtime-report.json', report)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expectations', type=Path, default=prep.CONFIG / 'expectations.json')
    p.add_argument('--expectations-sha256', required=True)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshot-repo')
    p.add_argument('--deadline-seconds', type=int, default=3600)
    p.add_argument('--execute-remote', action='store_true')
    p.add_argument('--internal-upload', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    os.umask(0o077)
    def stop(*_):
        raise KeyboardInterrupt('runtime preparation terminated')
    signal.signal(signal.SIGTERM, stop)
    if a.internal_upload:
        prep.require(os.environ.get('H3_RUNTIME_PREP_SUPERVISED') == '1', 'supervised upload only')
        # Import offline for preparation; enable only the explicit private receipt API here.
        os.environ['HF_HUB_OFFLINE'] = '0'
        from huggingface_hub import HfApi
        upload(HfApi(), a.snapshot_repo, a.output, 'runtimes/' + a.expectations_sha256)
        return 0
    prep.require(a.source_root is not None, 'original pinned trainer source root required')
    if not a.execute_remote:
        return prep.main(['--expectations', str(a.expectations), '--expectations-sha256', a.expectations_sha256,
                          '--source-root', str(a.source_root)])
    return execute(a)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        # Deliberately no traceback, subprocess output, environment or credentials.
        print(json.dumps(prep.failure_summary(exc)))
        raise SystemExit(2)
