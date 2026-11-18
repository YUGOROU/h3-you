"""Finite isolated Qwen server owner; no import-time install/download/GPU access.

Public urllib OCI audit: official vllm/vllm-omni:v0.30.0 resolves to
index fc77cf...; linux/amd64 b09c9e... (constants below). Release source
is a8576c..., NOT adapter OMNI_REVISION 7d2240.... Both speech stream and
YAML hashes differ. Explicit revision-change acceptance is mandatory; this
is NOT proof that the old protocol works with the release. Registry labels
are inherited vLLM labels, not Omni source attestations. Installed package
versions AND shipped source hashes must be checked inside the container.
Official Dockerfile.cuda uses vllm-openai:v0.30.0; CUDA image config reports
13.0.2, SM8.9 included, CUDA compatibility disabled: check host driver first.

L4 candidate: ORIGINAL 0.6B weights, explicitly accepted BF16 (not Irodori's
FP32 invariant), no quantization. Official profile verified H100 only:
MRV2 experimental, 64 sequences, graph capture, 0.3 memory fraction per
stage, stage1 BF16, no explicit attention backend. L4 must not inherit an
SM90-only attention choice. No forced backend is asserted qualified here.
Use a separately reviewed deploy config for L4; preserving async_chunk,
codec streaming and context sizes is required. No automatic OOM fallback.

Supported minimal CLI is `vllm serve /absolute/snapshot --omni ...`, NOT
`vllm-omni serve`. Parent supplies reviewed config and offline snapshot,
owns common Job worker/location/finite cloud lease. This module never creates
cloud resources. Context expiry also reaps the local server process group.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

from .qwen3_streaming_candidate import MODEL_ID, MODEL_REVISION

RELEASE_REVISION = 'a8576ccb725c4e21cd13c3eb5f9a546b21149d2b'
IMAGE_INDEX = 'sha256:fc77cfaac7b1b43c30cbf32eaea214864434c6e0f303e6e9eb9f64104241a994'
IMAGE_AMD64 = 'sha256:b09c9eab7d2e30817208de92ccb482756d13d4339377d7cc4c4ce9f47bf04174'
IMAGE = 'vllm/vllm-omni@' + IMAGE_AMD64
SOURCE_HASHES = {
    'vllm_omni/deploy/qwen3_tts.yaml': '28426bce3af2cb7c3f750f592a7ca0f339e947c617b80c730585b882ae4d6e0f',
    'vllm_omni/entrypoints/openai/serving_speech_stream.py': 'e7547ac1618b943b0d2e2cead1135768942f44d50b07a5384aaed2c01a0a41a3',
}


def verify_registry():
    """Public metadata only; hash-check index and amd64 manifest, no layers."""
    root = 'https://registry-1.docker.io/v2/vllm/vllm-omni/'
    with urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:vllm/vllm-omni:pull', timeout=20) as r:
        token = json.load(r)['token']
    def manifest(ref, expected):
        req = urllib.request.Request(root + 'manifests/' + ref, headers={
            'Authorization': 'Bearer ' + token,
            'Accept': 'application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json'})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024 or 'sha256:' + hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('OCI digest mismatch')
        return json.loads(raw)
    index = manifest('v0.30.0', IMAGE_INDEX)
    if not any(x['digest'] == IMAGE_AMD64 and x['platform'] == {'architecture': 'amd64', 'os': 'linux'} for x in index['manifests']):
        raise RuntimeError('OCI platform mismatch')
    return manifest(IMAGE_AMD64, IMAGE_AMD64)


def verify_install(source_root, *, version):
    """Call inside selected image, before GPU initialization. No version guess."""
    for package in ('vllm', 'vllm-omni'):
        if version(package).split('+')[0] != '0.30.0':
            raise RuntimeError('Installed dependency version mismatch: ' + package)
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((Path(source_root) / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Omni source-hash mismatch: ' + name)


def stage_model(*, snapshot_download=None):
    """Resolve full pinned snapshot from existing cache ONLY; never network.

    Parent may pre-stage full snapshot in a separately approved preparation
    step. No allow_patterns: embedded speech_tokenizer weights are required.
    """
    if snapshot_download is None:
        from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                  local_files_only=True)).resolve()
    if path.name != MODEL_REVISION or not (path / 'config.json').is_file():
        raise RuntimeError('Wrong/incomplete pinned model snapshot')
    codec = path / 'speech_tokenizer'
    if not (codec / 'config.json').is_file() or not list(codec.glob('*.safetensors')):
        raise RuntimeError('Embedded tokenizer weights absent')
    if not list(path.glob('*.safetensors')):
        raise RuntimeError('Model weights absent')
    return path


def command(snapshot, deploy_config, *, accepted_revision, accepted_bf16, port=8091, reference_directory=None):
    if accepted_revision != RELEASE_REVISION or accepted_bf16 is not True:
        raise RuntimeError('Explicit release source change and BF16 candidate acceptance required')
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('Invalid loopback port')
    paths = [Path(snapshot), Path(deploy_config)]
    if not all(p.is_absolute() and p.exists() for p in paths):
        raise ValueError('Existing absolute offline model/config paths required')
    argv=['vllm', 'serve', str(paths[0]), '--omni', '--served-model-name', MODEL_ID,
          '--deploy-config', str(paths[1]), '--host', '127.0.0.1', '--port', str(port)]
    if reference_directory is not None:
        reference_directory=Path(reference_directory)
        if not reference_directory.is_absolute() or not reference_directory.is_dir() or str(reference_directory) in ('/','/tmp'):
            raise ValueError('dedicated local reference directory required')
        argv+=['--allowed-local-media-path',str(reference_directory)]
    return argv


def _reap(proc):
    # Signal group even if leader exited: workers may still own CUDA memory.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=3)


@contextmanager
def server(snapshot, deploy_config, *, accepted_revision, accepted_bf16,
           lifetime=300, readiness=180, port=8091, reference_directory=None):
    """ONE server per context; serial owner/no retry; no proxy health requests.

    Parent MUST call verify_install first and enforce singleton across workers.
    Timer kills at lifetime even if caller stalls. Readiness is not synthesis
    acceptance. Public startup diagnostics are forwarded only before any private
    request; after readiness raw logs are drained/discarded (OOM type only).
    """
    if not all(math.isfinite(x) and 0 < x <= 900 for x in (lifetime, readiness)) or readiness > lifetime:
        raise ValueError('Invalid finite deadlines')
    argv = command(snapshot, deploy_config, accepted_revision=accepted_revision,
                   accepted_bf16=accepted_bf16, port=port,reference_directory=reference_directory)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1')
    env.pop('HF_TOKEN',None);env.pop('H3_PROBE_KEY',None)
    # Refuse a pre-existing listener; never mistake another owner's /health for ours.
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    proc = subprocess.Popen(argv, env=env, start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    ready_logs_stop=threading.Event()
    def drain():
        if getattr(proc,'stdout',None) is None:return
        while True:
            line=proc.stdout.readline(65536)
            if not isinstance(line,bytes) or not line:return
            # No private reference/text is submitted before health/readiness.
            # After readiness all raw server logs are discarded.
            if not ready_logs_stop.is_set():
                print('[qwen-startup] '+line.decode('utf-8','replace').rstrip(),flush=True)
            elif b'OutOfMemoryError' in line:
                print('[qwen-runtime] OutOfMemoryError',flush=True)
    drainer=threading.Thread(target=drain,daemon=True);drainer.start()
    timer = threading.Timer(lifetime, _reap, args=(proc,))
    timer.daemon = True
    timer.start()
    try:
        deadline = time.monotonic() + readiness
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if proc.poll() is not None:
                raise RuntimeError('Qwen server exited before readiness')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Qwen readiness deadline')
            try:
                with opener.open(f'http://127.0.0.1:{port}/health', timeout=min(1, remaining)) as r:
                    if r.status == 200 and proc.poll() is None:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        ready_logs_stop.set()
        yield proc
    finally:
        timer.cancel()
        timer.join()
        _reap(proc)
        drainer.join(timeout=2)
