"""Pinned CPU-host/T4-device runtime; no model substitution or CPU fallback."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from gate import run

RUNTIME_SHA = 'aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515'
MODEL_SHA = 'a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024):
            result.update(block)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    deadline = int(os.environ['H3_ASR_EXPIRES_AT'])
    # Only an immutable, explicitly selected long-session marker raises the
    # native process bound; all historical deployments retain the1200s cap.
    lifetime = 10800 if os.environ.get('H3_ASR_SESSION_SECONDS') == '10800' else 1200
    remaining = min(lifetime, deadline-int(time.time()))
    if remaining <= 0:
        raise SystemExit('Expired; no runtime start')
    if len(os.environ.get('H3_ASR_KEY', '')) < 32:
        raise SystemExit('Owner stream key required')
    for name, expected in [('runtime.tar.gz', RUNTIME_SHA), ('model.gguf', MODEL_SHA)]:
        if sha(root/name) != expected:
            raise SystemExit('Artifact hash mismatch')
    target = Path('/tmp/nemo')
    target.mkdir(exist_ok=False)
    with tarfile.open(root/'runtime.tar.gz') as archive:
        archive.extractall(target, filter='data')
    binaries = list(target.glob('*/bin/nemo-speech'))
    if len(binaries) != 1:
        raise SystemExit('Expected one pinned runtime binary')
    binary = binaries[0]
    # Explicit CUDA rejects unsupported devices; authenticated /ready must ALSO
    # report GPU before the client is permitted to upload public audio.
    command = [str(binary), '--quiet', 'serve', '--host', '127.0.0.1', '--port', '7861',
               '--threads', '2', '--no-ui', '--device', 'cuda', '--asr.backend.gpu', '0',
               '--asr-model', str(root/'model.gguf'), '--asr.batching.enabled=false',
               '--asr.batching.state_arena_slots', '1', '--asr.endpointing.enable=true',
               '--asr.endpointing.stop_history_eou_ms', '800', '--max-upload-mb', '2']
    child = subprocess.Popen(command)
    threading.Thread(target=run, name='owner-key-ws-gate', daemon=True).start()
    try:
        code = child.wait(timeout=remaining)
        print('Native runtime exited:', code, flush=True)
        return code or 1
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
        return 124
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=10)


if __name__ == '__main__':
    sys.exit(main())
