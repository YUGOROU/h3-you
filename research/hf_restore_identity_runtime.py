"""Byte-bound GPU runtime restoration; no Jobs API, weights or model construction."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sys

from research import prepare_identity_runtime as prep
from research import hf_bootstrap_identity_pipeline as wire

MODULE = 'research.hf_restore_identity_runtime'
SCHEMA = 'h3-identity-gpu-runtime-bootstrap-v1'


def entry(root, path):
    path = Path(path).absolute()
    root = Path(root).absolute()
    name = str(path.relative_to(root))
    wire.relative(name)
    wire.inside(root, name)
    return {'path': name, 'sha256': prep.sha256(path)}


def strict_lock(c, lock):
    prep.require(set(lock) == {'schema', 'versions', 'modules', 'ffmpeg'} and
                 lock['schema'] == 'h3-encode-runtime-v1' and
                 lock['versions'] == c['versions'] and
                 lock['modules'] == {k: v['sha256'] for k, v in c['modules'].items()},
                 'reviewed lock differs from independent expectations')
    prep.require(set(lock['ffmpeg']) == {'path', 'sha256'} and
                 lock['ffmpeg']['path'] == '/usr/bin/ffmpeg' and
                 re.fullmatch('[0-9a-f]{64}', lock['ffmpeg']['sha256']), 'FFmpeg lock malformed')


def package_fields(evidence):
    fields = dict(line.split(': ', 1) for line in evidence['package_stanza'].splitlines()
                  if ': ' in line and not line.startswith(' '))
    prep.require(fields.get('Package') == 'ffmpeg' and fields.get('Architecture') == 'amd64' and
                 re.fullmatch('[a-zA-Z0-9.+:~_-]+', fields.get('Version', '')) and
                 fields.get('SHA256') == evidence['deb_sha256'] and
                 re.fullmatch('[0-9a-f]{64}', evidence['deb_sha256']), 'CPU APT evidence malformed')
    return fields


def reviewed_inputs(root, documents):
    report = wire.load(wire.pinned(root, documents['cpu_report']))
    lock_path = wire.pinned(root, documents['runtime_lock'])
    lock = wire.load(lock_path)
    expectation_path = wire.pinned(root, documents['expectations'])
    c, tree = prep.expectations(expectation_path, documents['expectations']['sha256'])
    strict_lock(c, lock)
    prep.require(report['schema'] == 'h3-identity-runtime-prep-report-v1' and
                 report.get('complete') is True and report.get('cpu_imports_verified') is True and
                 report.get('weights_loaded') is False and report.get('gpu_verified') is False and
                 report['image'] == c['image'] and report['source_revisions'] == c['source_revisions'] and
                 report['expectations_sha256'] == documents['expectations']['sha256'] and
                 report['runtime_lock_sha256'] == documents['runtime_lock']['sha256'] and
                 report['independent_expectations'] == {
                     'versions': c['versions'], 'modules': c['modules'],
                     'transformers_source_revision': c['transformers_source_revision'],
                     'diffusers_inventory_sha256': c['diffusers_inventory_sha256']},
                 'reviewed CPU report/lock/expectations binding mismatch')
    # This receipt is externally reviewed evidence from the SAME immutable CPU image.
    # Never hash the builder's Mac interpreter or manufacture a GPU success flag.
    interpreter = wire.load(wire.pinned(root, documents['interpreter_receipt']))
    prep.require(interpreter.get('schema') == 'h3-runtime-base-interpreter-v1' and
                 interpreter.get('image') == c['image'] and interpreter.get('complete') is True and
                 interpreter.get('cpu_report_sha256') == documents['cpu_report']['sha256'] and
                 isinstance(interpreter.get('python'), str) and
                 interpreter['python'].startswith('/') and
                 re.fullmatch('[0-9a-f]{64}', interpreter.get('python_sha256', '')),
                 'trusted recorded immutable-image interpreter receipt required')
    package_fields(report['ffmpeg_package_evidence'])
    return c, tree, lock, report, interpreter


def build_manifest(*, mount_root, cpu_report, runtime_lock, expectations,
                   interpreter_receipt, source_root):
    """Pure local builder. Inputs already staged at relative paths in runtime mount.

    Returns the literal launcher contract; does not upload, execute, or guess pins.
    source_root is a pristine PINS trainer subset plus independently pinned pyproject.
    """
    root = Path(mount_root).absolute()
    documents = {k: entry(root, v) for k, v in {
        'cpu_report': cpu_report, 'runtime_lock': runtime_lock, 'expectations': expectations,
        'interpreter_receipt': interpreter_receipt}.items()}
    c, tree, lock, report, interpreter = reviewed_inputs(root, documents)
    source = prep.check_trainer(source_root, c)
    from research.install_identity_guidance import PINS
    source_hashes = {name: prep.sha256(wire.inside(source, name)) for name in PINS}
    source_hashes['pyproject.toml'] = c['trainer_pyproject_sha256']
    source_hashes['LICENSE'] = prep.sha256(wire.inside(source, 'LICENSE'))
    # Pin the entire stdlib helper import closure and package initializers, not only entrypoint.
    files = {str(p.relative_to(root)): prep.sha256(wire.inside(root, str(p.relative_to(root))))
             for p in (root / 'research').rglob('*.py')}
    for name in ('research/hf_restore_identity_runtime.py', 'research/prepare_identity_runtime.py',
                 'research/hf_bootstrap_identity_pipeline.py', 'research/hf_identity_encode.py',
                 'research/install_identity_guidance.py', 'research/install_identity_checkpoint.py',
                 'research/h3_personal/identity_checkpoint.py'):
        prep.require(name in files, 'runtime helper prerequisite missing: ' + name)
    # Ensure staged workers are the actual reviewed local implementation, not renamed counterfeits.
    local = Path(__file__).resolve().parents[1]
    required_files = {str(p.relative_to(local)) for p in (local / 'research').rglob('*.py')}
    prep.require(set(files) == required_files, 'complete reviewed research helper closure required')
    for name, digest in files.items():
        prep.require(prep.sha256(wire.inside(local, name)) == digest, 'staged bootstrap source differs')
    artifacts = {documents[k]['path']: documents[k]['sha256'] for k in documents}
    for name in ('image-manifest.json', 'image-config.json', 'diffusers-source.json'):
        e = entry(root, Path(expectations).parent / name)
        artifacts[e['path']] = e['sha256']
    source_relative = str(source.relative_to(root.resolve()))
    wire.relative(source_relative)
    artifacts.update({source_relative + '/' + n: h for n, h in source_hashes.items()})
    return {'schema': SCHEMA, 'gpu_bootstrap_supported': True,
            'image': c['image'], 'runtime_lock_sha256': documents['runtime_lock']['sha256'],
            'native_source_inventory': 'PINS23+LICENSE', 'module': MODULE,
            'entrypoint': {'path': MODULE.replace('.', '/') + '.py',
                           'sha256': files[MODULE.replace('.', '/') + '.py']},
            'bootstrap_files': files, 'artifact_files': artifacts, 'documents': documents,
            'source_root': source_relative, 'source_hashes': source_hashes,
            'source_revisions': c['source_revisions'], 'python_sha256': interpreter['python_sha256'],
            'versions': c['versions'], 'modules': c['modules'], 'wheels': c['wheels'],
            'diffusers_inventory_sha256': c['diffusers_inventory_sha256'],
            'ffmpeg': lock['ffmpeg'], 'ffmpeg_package': package_fields(report['ffmpeg_package_evidence']),
            'gpu_policy': {'count': 1, 'name_contains': 'H200', 'cuda': c['cuda']}}


def inspect_gpu(c):
    """Import CUDA build on GPU without constructing any model or tensor."""
    torch = importlib.import_module('torch')
    prep.require(torch.version.cuda == c['cuda'] and torch.cuda.is_available(), 'CUDA unavailable/build mismatch')
    comparison=any(os.environ.get(k)=='1' for k in ('H3_ACCELERATION_MATRIX','H3_NATIVE_CONTINUATION'))
    count=2 if comparison else 1
    prep.require(torch.cuda.device_count() == count and
                 all('H200' in torch.cuda.get_device_name(i) for i in range(count)),
                 'exact configured H200 comparison count required')


def probe(c, source, lock):
    # Reuse independent byte/version/location checks individually. CPU prep.probe is unchanged.
    strict_lock(c, lock)
    prep.check_trainer(source, c)
    from research.hf_identity_encode import verify_runtime
    verify_runtime(source, lock)
    inspect_gpu(c)
    for name in ('torchvision', 'torchaudio', 'peft', 'h3_trainer.trainer'):
        importlib.import_module(name)
    for name, version in c['versions'].items():
        prep.require(importlib.metadata.version(name) == version, 'independent version expectation mismatch')


def provision_ffmpeg(root, expected, cpu_evidence):
    """Same signed Ubuntu trust chain as CPU prep, with exact reviewed version selection."""
    fields = package_fields(cpu_evidence)
    prep.require(os.geteuid() == 0, 'APT image root required')
    apt = root / 'apt'
    apt.mkdir()
    for name in ('lists/partial', 'archives/partial', 'debs', 'extracted'):
        (apt / name).mkdir(parents=True)
    keyring = Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg')
    prep.require(prep.sha256(keyring) == cpu_evidence['keyring_sha256'], 'Ubuntu trust anchor drift')
    sources = apt / 'sources.list'
    official = ''.join(f'deb [arch=amd64 signed-by={keyring}] https://{host}/ubuntu {suite} main universe\n'
                      for host, suite in [('archive.ubuntu.com', 'jammy'),
                                          ('archive.ubuntu.com', 'jammy-updates'),
                                          ('security.ubuntu.com', 'jammy-security')])
    prep.require(cpu_evidence['sources'] == official, 'CPU APT source policy mismatch')
    sources.write_text(official)
    opts = ['-o', 'Dir::Etc::sourcelist=' + str(sources), '-o', 'Dir::Etc::sourceparts=-',
            '-o', 'Dir::State::lists=' + str(apt / 'lists'),
            '-o', 'Dir::Cache::archives=' + str(apt / 'archives'), '-o', 'APT::Sandbox::User=root',
            '-o', 'APT::Get::AllowUnauthenticated=false', '-o', 'Acquire::AllowInsecureRepositories=false',
            '-o', 'Acquire::AllowDowngradeToInsecureRepositories=false', '-o', 'Acquire::Check-Valid-Until=true']
    prep.run(['apt-get', *opts, 'update'])
    version = fields['Version']
    stanza = prep.run(['apt-cache', *opts, 'show', 'ffmpeg=' + version]).split('\n\n')[0]
    current = dict(line.split(': ', 1) for line in stanza.splitlines()
                   if ': ' in line and not line.startswith(' '))
    prep.require(all(current.get(k) == fields[k] for k in ('Package', 'Version', 'Architecture', 'SHA256')),
                 'signed index differs from reviewed exact FFmpeg package')
    prep.run(['apt-get', *opts, 'download', 'ffmpeg=' + version], cwd=apt / 'debs')
    debs = list((apt / 'debs').glob('*.deb'))
    prep.require(len(debs) == 1 and prep.sha256(debs[0]) == fields['SHA256'], 'FFmpeg package drift')
    prep.run(['dpkg-deb', '--extract', str(debs[0]), str(apt / 'extracted')])
    os.environ['DEBIAN_FRONTEND'] = 'noninteractive'
    prep.run(['apt-get', *opts, 'install', '-y', '--no-install-recommends', 'ffmpeg=' + version])
    digest = prep.verify_native_package(debs[0], fields['SHA256'],
                                       apt / 'extracted/usr/bin/ffmpeg', expected['path'])
    prep.require(digest == expected['sha256'] and
                 prep.run([expected['path'], '-version']) == cpu_evidence['version_output'] and
                 Path(shutil.which('ffmpeg') or '').resolve() == Path(expected['path']),
                 'exact captured FFmpeg binary/version differs')


def restore(c, tree, root, lock, report):
    """GPU equivalent of shared CPU bootstrap; never hides CUDA or relaxes CPU policy."""
    prep.require(sys.platform == 'linux' and platform.machine() == 'x86_64' and
                 sys.version_info[:2] == (3, 11), 'Linux amd64 Python 3.11 required')
    prep.require(os.environ.get('H3_RUNTIME_IMAGE') == c['image'] and
                 'VERSION_ID="22.04"' in Path('/etc/os-release').read_text(), 'immutable jammy image required')
    prep.require(importlib.metadata.version('torch') == c['versions']['torch'], 'base torch mismatch')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      WANDB_MODE='disabled', WANDB_DISABLED='true', PYTHONNOUSERSITE='1',
                      PIP_CONFIG_FILE='/dev/null', HF_HUB_DISABLE_TELEMETRY='1')
    python = root / 'venv/bin/python'
    prep.run([sys.executable, '-m', 'venv', '--system-site-packages', str(root / 'venv')])
    wheels = root / 'wheels'
    wheels.mkdir()
    lines = []
    for pin in c['wheels'].values():
        target = wheels / pin['filename']
        prep.fetch(pin['url'], target, 150 * 1024**2, pin['sha256'])
        lines.append(str(target) + ' --hash=sha256:' + pin['sha256'])
    req = root / 'requirements.txt'
    req.write_text('\n'.join(lines) + '\n')
    prep.run([str(python), '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
              '--require-hashes', '--upgrade', '-r', str(req)])
    archive = root / 'diffusers.tar.gz'
    prep.fetch('https://codeload.github.com/huggingface/diffusers/tar.gz/' + tree['revision'],
               archive, 100 * 1024**2)
    source = root / 'diffusers-source'
    source.mkdir()
    prep.extract_source(archive, source, tree['git_blob_sha1'])
    prep.run([str(python), '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
              '--no-build-isolation', str(source)])
    prep.run([str(python), '-m', 'pip', '--isolated', 'check'])
    provision_ffmpeg(root, lock['ffmpeg'], report['ffmpeg_package_evidence'])
    return python


def execute(manifest_path, manifest_sha256, output, receipt, internal_probe=False):
    manifest_path = Path(manifest_path).absolute()
    prep.require(prep.sha256(manifest_path) == manifest_sha256, 'manifest byte mismatch')
    m = wire.load(manifest_path)
    mount = Path(__file__).absolute().parents[1]
    prep.require(m['schema'] == SCHEMA and m['module'] == MODULE and
                 m['gpu_bootstrap_supported'] is True and m['native_source_inventory'] == 'PINS23+LICENSE',
                 'launcher manifest policy mismatch')
    for inventory in ('bootstrap_files', 'artifact_files'):
        for name, digest in m[inventory].items():
            wire.pinned(mount, {'path': name, 'sha256': digest})
    c, tree, lock, report, interpreter = reviewed_inputs(mount, m['documents'])
    rebuilt = build_manifest(mount_root=mount,
                             **{k: wire.pinned(mount, v) for k, v in m['documents'].items()},
                             source_root=mount / m['source_root'])
    prep.require(m == rebuilt, 'literal reviewed runtime contract differs')
    prep.require(prep.sha256(sys.executable) == interpreter['python_sha256'], 'base interpreter bytes differ')
    output, receipt = Path(output).absolute(), Path(receipt).absolute()
    prep.require(output == Path('/work/runtime') and receipt == Path('/work/runtime-ready.json'),
                 'launcher private work paths required')
    # Native source remains unchanged; assemble pristine supplementary pyproject separately.
    native = Path(os.environ['H3_NATIVE_SOURCE_ROOT'])
    for name, digest in m['source_hashes'].items():
        if name != 'pyproject.toml':
            wire.pinned(native, {'path': name, 'sha256': digest})
    source = output / 'trainer-source'
    if internal_probe:
        prep.require(output.is_dir() and output.stat().st_mode & 0o077 == 0, 'private runtime required')
        probe(c, source, lock)
        return
    prep.require(not receipt.exists() and not receipt.is_symlink(), 'exclusive receipt required')
    os.umask(0o077)
    prep.private_new(output)
    source.mkdir()
    for name, digest in m['source_hashes'].items():
        src = wire.pinned(mount / m['source_root'], {'path': name, 'sha256': digest})
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    prep.check_trainer(source, c)
    python = restore(c, tree, output, lock, report)
    prep.require(prep.sha256(python) == m['python_sha256'], 'restored interpreter bytes differ')
    prep.run([str(python), '-m', MODULE, '--manifest', str(manifest_path),
              '--manifest-sha256', manifest_sha256, '--output', str(output),
              '--receipt', str(receipt), '--internal-probe'])
    # Receipt published only after fresh venv imports pass, bound to original lock BYTES.
    prep.atomic_json(receipt, {'schema': 'h3-identity-runtime-ready-v1',
                              'manifest_sha256': manifest_sha256,
                              'runtime_lock_sha256': m['runtime_lock_sha256'],
                              'python': str(python), 'python_sha256': prep.sha256(python),
                              'gpu_imports_verified': True,
                              'gpu_count_verified': 2 if any(os.environ.get(k)=='1' for k in ('H3_ACCELERATION_MATRIX','H3_NATIVE_CONTINUATION')) else 1,
                              'weights_loaded': False,
                              'models_constructed': False, 'c_gate_enabled': False})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--internal-probe', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    execute(a.manifest, a.manifest_sha256, a.output, a.receipt, a.internal_probe)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({'complete': False, 'failure_type': type(exc).__name__}))
        raise SystemExit(2) from None
