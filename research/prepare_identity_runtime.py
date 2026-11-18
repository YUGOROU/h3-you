"""Pinned source/wheel bootstrap for the official Linux PyTorch image, no models.

Default CLI validates inputs only. Execution is supervised by hf_prepare_identity_runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request

CONFIG = Path(__file__).parent / 'configs/identity-runtime'
IMAGE = 'pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385'


class RuntimeCheckError(ValueError):
    """Only static, credential-free check labels may use this exception."""


def failure_summary(exc):
    result = {'complete': False, 'failure_type': type(exc).__name__}
    encoder_checks = {'unpinned trainer source', 'runtime lock schema', 'runtime module pins missing',
                      'runtime version mismatch', 'required runtime versions missing',
                      'runtime module mismatch', 'loaded trainer source mismatch',
                      'ffmpeg pin mismatch', 'ffmpeg PATH mismatch'}
    if isinstance(exc, RuntimeCheckError) or (type(exc) is ValueError and str(exc) in encoder_checks):
        result['check'] = str(exc)
    # Location/type identify import failures without exposing exception text or paths.
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        frames.append({'file': Path(frame.f_code.co_filename).name,
                       'function': frame.f_code.co_name, 'line': tb.tb_lineno})
        tb = tb.tb_next
    if frames:
        result['locations'] = frames[-8:]
    if isinstance(exc, ImportError) and isinstance(exc.name, str) and re.fullmatch(r'[A-Za-z0-9_.]+', exc.name):
        result['import_name'] = exc.name
    return result


def require(ok, message):
    if not ok:
        raise RuntimeCheckError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        os.chmod(temporary, 0o600)
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def private_new(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), 'exclusive new output required')
    path.mkdir(parents=True, mode=0o700)
    return path


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and
            not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')),
            'unsafe source path')
    return name


def expectations(path, expected_sha256):
    require(re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') and
            sha256(path) == expected_sha256, 'expectations byte pin mismatch')
    c = json.loads(Path(path).read_text())
    from research.h3_personal.identity_checkpoint import SOURCE_PINS
    from research.hf_identity_encode import REQUIRED_MODULES
    require(c['schema'] == 'h3-runtime-expectations-v1' and c['image'] == IMAGE and
            c['source_revisions'] == SOURCE_PINS, 'runtime/source policy mismatch')
    require(c['python'] == '3.11' and c['cuda'] == '12.8' and c['platform'] == 'linux/amd64',
            'platform policy mismatch')
    require(REQUIRED_MODULES <= set(c['modules']), 'required independent module pins missing')
    for name, pin in c['modules'].items():
        require(re.fullmatch('[a-zA-Z0-9_.]+', name) and
                re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'module pin malformed')
        revision = (c['source_revisions']['diffusers'] if name.startswith('diffusers.')
                    else c['transformers_source_revision'])
        package = name.split('.')[0]
        require(pin['url'] == f'https://raw.githubusercontent.com/huggingface/{package}/{revision}/src/'
                + name.replace('.', '/') + '.py', 'independent source URL mismatch')
    for name, pin in c['wheels'].items():
        require(c['versions'][name] == pin['version'] and
                re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'wheel/version pin mismatch')
        u = urllib.parse.urlsplit(pin['url'])
        require(u.scheme == 'https' and u.hostname == 'files.pythonhosted.org' and
                not u.query and not u.fragment and not u.username and
                Path(u.path).name == safe_name(pin['filename']) and pin['filename'].endswith('.whl'),
                'wheel URL not allowlisted')
    image_manifest = Path(path).parent / 'image-manifest.json'
    image_config = Path(path).parent / 'image-config.json'
    require('sha256:' + sha256(image_manifest) == c['image'].split('@')[1] and
            'sha256:' + sha256(image_config) == c['image_config_digest'], 'registry evidence byte mismatch')
    manifest = json.loads(image_manifest.read_text())
    config = json.loads(image_config.read_text())
    require(manifest['config']['digest'] == c['image_config_digest'] and
            config['os'] == 'linux' and config['architecture'] == 'amd64', 'registry platform mismatch')
    inventory = Path(path).parent / 'diffusers-source.json'
    require(sha256(inventory) == c['diffusers_inventory_sha256'], 'source inventory byte mismatch')
    tree = json.loads(inventory.read_text())
    require(tree['revision'] == c['source_revisions']['diffusers'], 'source inventory revision mismatch')
    for name, blob in tree['git_blob_sha1'].items():
        safe_name(name)
        require((name.startswith('src/diffusers/') and (name.endswith('.py') or name.endswith('py.typed'))
                 or name in {'setup.py', 'setup.cfg', 'pyproject.toml', 'LICENSE', 'README.md'}) and
                re.fullmatch('[0-9a-f]{40}', blob), 'source inventory not allowlisted')
    return c, tree


def check_trainer(root, c):
    from research.install_identity_guidance import PINS
    from research.install_identity_checkpoint import plan
    root = Path(root).resolve(strict=True)
    for name, digest in PINS.items():
        file = root / name
        require(not file.is_symlink() and sha256(file) == digest, 'trainer byte mismatch')
    require(sha256(root / 'pyproject.toml') == c['trainer_pyproject_sha256'], 'trainer pyproject mismatch')
    plan(root)  # Exact composed checkpoint/guidance/cache/sampling contexts, without installing.
    return root


def fetch(url, path, max_bytes, expected=None):
    """No auth; bounded HTTPS source/wheel transfer. Never an HF/model endpoint."""
    u = urllib.parse.urlsplit(url)
    require(u.scheme == 'https' and u.hostname in {'files.pythonhosted.org', 'codeload.github.com'},
            'bootstrap download host not allowlisted')
    with urllib.request.urlopen(url, timeout=60) as response, Path(path).open('xb') as out:
        require(urllib.parse.urlsplit(response.url).hostname == u.hostname, 'download redirect denied')
        total = 0
        while block := response.read(1024 * 1024):
            total += len(block)
            require(total <= max_bytes, 'download size limit exceeded')
            out.write(block)
    if expected:
        require(sha256(path) == expected, 'download byte mismatch')


def git_blob(payload):
    return hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()


def extract_source(archive, destination, inventory):
    """Extract ONLY reviewed source files; verify Git blobs before touching disk."""
    seen = set()
    with tarfile.open(archive, 'r:gz') as t:
        for member in t:
            # GitHub archives have a single diffusers-<commit> root.
            parts = member.name.split('/', 1)
            if len(parts) != 2 or parts[1] not in inventory:
                continue
            name = safe_name(parts[1])
            require(member.isfile() and not member.issym() and member.size <= 8 * 1024**2 and
                    name not in seen, 'unsafe/duplicate archive member')
            payload = t.extractfile(member).read()
            require(git_blob(payload) == inventory[name], 'independent Git source blob mismatch')
            file = Path(destination) / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(payload)
            seen.add(name)
    require(seen == set(inventory), 'source archive inventory incomplete')


def command_label(command):
    """Static labels only: never expose paths, URLs, arguments or credentials."""
    if 'research.prepare_identity_runtime' in command:
        return 'runtime import probe'
    if 'venv' in command:
        return 'venv creation'
    if 'pip' in command:
        if 'check' in command:
            return 'pip dependency check'
        if '--require-hashes' in command:
            return 'pinned wheel installation'
        return 'Diffusers source installation'
    return Path(command[0]).name


def safe_dependency_failures(output):
    # pip check messages contain package/version/specifier tokens, not arbitrary errors.
    token = r'[A-Za-z0-9_.+!<>=,*() -]+'
    patterns = (rf'{token} requires {token}, which is not installed\.',
                rf'{token} has requirement {token}, but you have {token}\.')
    return [line for line in output.splitlines()
            if len(line) <= 500 and any(re.fullmatch(p, line) for p in patterns)][:20]


def run(command, *, cwd=None):
    # Parent worker owns this entire process group and enforces the wall deadline.
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=1200, check=False)
    label = command_label(command)
    if result.returncode != 0 and label == 'pip dependency check':
        failures = safe_dependency_failures(result.stdout.decode('utf-8', errors='replace'))
        print(json.dumps({'stage': label, 'dependency_failures': failures}), flush=True)
    if result.returncode != 0 and 'research.prepare_identity_runtime' in command:
        # This module emits only controlled summaries; never forward raw pip/APT output.
        for line in result.stdout.decode('utf-8', errors='replace').splitlines():
            try:
                summary = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(summary, dict) and summary.get('complete') is False and isinstance(summary.get('failure_type'), str):
                # Select the fields emitted by failure_summary, not arbitrary child JSON.
                allowed = {k: summary[k] for k in ('complete', 'failure_type', 'check', 'locations', 'import_name', 'version_mismatches') if k in summary}
                print(json.dumps(allowed), flush=True)
    require(result.returncode == 0, 'runtime preparation subprocess failed: ' + label)
    return result.stdout.decode('utf-8', errors='strict')


def verify_native_package(deb, expected_deb_sha256, extracted_binary, installed_binary):
    require(re.fullmatch('[0-9a-f]{64}', expected_deb_sha256) and
            sha256(deb) == expected_deb_sha256, 'trusted package SHA256 mismatch')
    binary = Path(installed_binary)
    require(not binary.is_symlink() and sha256(binary) == sha256(extracted_binary),
            'installed FFmpeg differs from authenticated package')
    return sha256(binary)


def provision_ffmpeg(root):
    """Ubuntu signed APT -> independent package SHA256 -> extracted/installed byte match.

    Package version is captured remotely, not presented as a pre-reviewed version pin.
    All APT invocations use only the image's Ubuntu keyring and explicit official sources.
    """
    require(os.geteuid() == 0, 'APT bootstrap requires image root')
    apt = root / 'apt'
    apt.mkdir()
    for name in ('lists/partial', 'archives/partial', 'debs', 'extracted'):
        (apt / name).mkdir(parents=True)
    sources = apt / 'sources.list'
    keyring = Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg')
    require(keyring.is_file(), 'image Ubuntu archive trust anchor missing')
    sources.write_text(''.join(
        f'deb [arch=amd64 signed-by={keyring}] https://{host}/ubuntu {suite} main universe\n'
        for host, suite in [('archive.ubuntu.com', 'jammy'), ('archive.ubuntu.com', 'jammy-updates'),
                            ('security.ubuntu.com', 'jammy-security')]))
    opts = ['-o', 'Dir::Etc::sourcelist=' + str(sources), '-o', 'Dir::Etc::sourceparts=-',
            '-o', 'Dir::State::lists=' + str(apt / 'lists'),
            '-o', 'Dir::Cache::archives=' + str(apt / 'archives'),
            '-o', 'APT::Sandbox::User=root', '-o', 'APT::Get::AllowUnauthenticated=false',
            '-o', 'Acquire::AllowInsecureRepositories=false',
            '-o', 'Acquire::AllowDowngradeToInsecureRepositories=false',
            '-o', 'Acquire::Check-Valid-Until=true']
    run(['apt-get', *opts, 'update'])
    version = run(['apt-cache', *opts, 'policy', 'ffmpeg']).split('Candidate: ', 1)[1].splitlines()[0].strip()
    require(re.fullmatch('[a-zA-Z0-9.+:~_-]+', version), 'APT candidate unavailable')
    stanza = run(['apt-cache', *opts, 'show', 'ffmpeg=' + version]).split('\n\n')[0]
    fields = dict(line.split(': ', 1) for line in stanza.splitlines() if ': ' in line and not line.startswith(' '))
    require(fields.get('Architecture') == 'amd64' and fields.get('Package') == 'ffmpeg' and
            fields.get('Version') == version and re.fullmatch('[0-9a-f]{64}', fields.get('SHA256', '')),
            'APT signed index package evidence missing')
    run(['apt-get', *opts, 'download', 'ffmpeg=' + version], cwd=apt / 'debs')
    debs = list((apt / 'debs').glob('*.deb'))
    require(len(debs) == 1 and sha256(debs[0]) == fields['SHA256'], 'trusted package SHA256 mismatch')
    run(['dpkg-deb', '--extract', str(debs[0]), str(apt / 'extracted')])
    os.environ['DEBIAN_FRONTEND'] = 'noninteractive'
    run(['apt-get', *opts, 'install', '-y', '--no-install-recommends', 'ffmpeg=' + version])
    binary = Path('/usr/bin/ffmpeg')
    verify_native_package(debs[0], fields['SHA256'], apt / 'extracted/usr/bin/ffmpeg', binary)
    require(Path(shutil.which('ffmpeg') or '').resolve() == binary, 'unexpected FFmpeg PATH')
    require(len(list((apt / 'lists').glob('*InRelease'))) == 3 and
            len(list((apt / 'lists').glob('*Packages*'))) >= 3, 'signed APT evidence incomplete')
    return {'path': str(binary), 'sha256': sha256(binary)}, {
        'basis': 'Ubuntu signed InRelease/Packages, APT SHA256, extracted .deb comparison',
        'version_status': 'remote-captured authenticated package, not independent exact version expectation',
        'package_stanza': stanza, 'deb_sha256': sha256(debs[0]),
        'keyring_sha256': sha256(keyring), 'sources': sources.read_text(),
        'inrelease_sha256': {p.name: sha256(p) for p in (apt / 'lists').glob('*InRelease')},
        'index_sha256': {p.name: sha256(p) for p in (apt / 'lists').glob('*Packages*')},
        'installed_packages': run(['dpkg-query', '-W', '-f=${Package}\t${Version}\t${Architecture}\n']),
        'version_output': run([str(binary), '-version'])}


def bootstrap(c, tree, root):
    require(sys.platform == 'linux' and platform.machine() == 'x86_64' and
            sys.version_info[:2] == (3, 11), 'Linux amd64 Python 3.11 required')
    require(os.environ.get('H3_RUNTIME_IMAGE') == c['image'], 'explicit immutable launch image required')
    require(not list(Path('/dev').glob('nvidia[0-9]*')), 'CPU-only preparation: GPU devices forbidden')
    require('VERSION_ID="22.04"' in Path('/etc/os-release').read_text(), 'Ubuntu jammy required')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', WANDB_DISABLED='true',
                      PYTHONNOUSERSITE='1', PIP_CONFIG_FILE='/dev/null')
    # Preserve official torch/CUDA binaries; never allow pip to resolve/reinstall torch.
    require(importlib.metadata.version('torch') == c['versions']['torch'], 'base torch version mismatch')
    env = root / 'venv'
    run([sys.executable, '-m', 'venv', '--system-site-packages', str(env)])
    python = env / 'bin/python'
    wheels = root / 'wheels'
    wheels.mkdir()
    req = root / 'requirements.txt'
    lines = []
    for name, pin in c['wheels'].items():
        target = wheels / pin['filename']
        fetch(pin['url'], target, 150 * 1024**2, pin['sha256'])
        lines.append(str(target) + ' --hash=sha256:' + pin['sha256'])
    req.write_text('\n'.join(lines) + '\n')
    # venv seeds its own setuptools. --ignore-installed leaves both old/new dist-info,
    # so metadata.version can select the stale seed. Normal upgrade removes venv-owned
    # seeds; pip does not uninstall inherited packages outside this venv. No resolver.
    run([str(python), '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
         '--require-hashes', '--upgrade', '-r', str(req)])
    archive = root / 'diffusers.tar.gz'
    fetch('https://codeload.github.com/huggingface/diffusers/tar.gz/' + tree['revision'],
          archive, 100 * 1024**2)
    source = root / 'diffusers-source'
    source.mkdir()
    extract_source(archive, source, tree['git_blob_sha1'])
    run([str(python), '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
         '--no-build-isolation', str(source)])
    run([str(python), '-m', 'pip', '--isolated', 'check'])
    ffmpeg, evidence = provision_ffmpeg(root)
    return python, ffmpeg, evidence


def version_mismatches(versions, lookup=importlib.metadata.version):
    mismatches = []
    for package, expected in versions.items():
        try:
            actual = lookup(package)
        except importlib.metadata.PackageNotFoundError:
            actual = '<missing>'
        if actual != expected:
            # Package keys/expected values come from pinned policy; actual is untrusted metadata.
            safe_actual = actual if isinstance(actual, str) and re.fullmatch(r'[A-Za-z0-9_.+!<>-]{1,100}', actual) else '<invalid>'
            mismatches.append({'package': package, 'expected': expected, 'actual': safe_actual})
    return mismatches


def probe(c, source_root, lock):
    """CPU imports only; compare independent expectations, never invent module pins."""
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', WANDB_DISABLED='true')
    from research.hf_identity_encode import verify_runtime
    mismatches = version_mismatches(lock['versions'])
    if mismatches:
        print(json.dumps({'complete': False, 'failure_type': 'RuntimeCheckError',
                          'check': 'runtime version mismatch', 'version_mismatches': mismatches}), flush=True)
        require(False, 'runtime version mismatch')
    verify_runtime(source_root, lock)
    import torch
    require(torch.version.cuda == c['cuda'] and not torch.cuda.is_available(), 'CPU CUDA build probe failed')
    importlib.import_module('torchvision')
    importlib.import_module('torchaudio')
    importlib.import_module('peft')
    importlib.import_module('h3_trainer.trainer')
    for package, version in c['versions'].items():
        require(importlib.metadata.version(package) == version, 'independent version expectation mismatch')
    # Import checks above instantiate no models, pipelines, adapters or training loops.


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expectations', type=Path, default=CONFIG / 'expectations.json')
    p.add_argument('--expectations-sha256', required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--internal', choices=('bootstrap', 'probe'), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    c, tree = expectations(a.expectations, a.expectations_sha256)
    source = check_trainer(a.source_root, c)
    if not a.internal:
        print(json.dumps({'plan_only': True, 'image': c['image'], 'cpu_only': True,
                          'weights_loaded': False, 'runtime_verified': False}))
        return 0
    require(os.environ.get('H3_RUNTIME_PREP_SUPERVISED') == '1' and a.output is not None,
            'use bounded hf_prepare_identity_runtime worker')
    root = a.output.resolve(strict=True)
    require(not root.is_symlink() and root.stat().st_mode & 0o077 == 0, 'private worker output required')
    if a.internal == 'bootstrap':
        python, ffmpeg, apt = bootstrap(c, tree, root)
        lock = {'schema': 'h3-encode-runtime-v1', 'versions': c['versions'],
                'modules': {k: v['sha256'] for k, v in c['modules'].items()}, 'ffmpeg': ffmpeg}
        atomic_json(root / 'candidate-lock.json', lock)
        atomic_json(root / 'apt-evidence.json', apt)
        run([str(python), '-m', 'research.prepare_identity_runtime', '--internal', 'probe',
             '--expectations', str(a.expectations.resolve()), '--expectations-sha256', a.expectations_sha256,
             '--source-root', str(source), '--output', str(root)])
        (root / 'candidate-lock.json').replace(root / 'runtime-lock.json')
    else:
        lock = json.loads((root / 'candidate-lock.json').read_text())
        require(lock['versions'] == c['versions'] and lock['modules'] ==
                {k: v['sha256'] for k, v in c['modules'].items()}, 'candidate expectations mismatch')
        probe(c, source, lock)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(failure_summary(exc)))
        raise SystemExit(2)
