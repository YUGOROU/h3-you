"""RTC-only ASR Endpoint supervisor; original native runtime/model/argv pins."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

RUNTIME_SHA='aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515'
MODEL_SHA='a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        while body:=source.read(1024*1024):digest.update(body)
    return digest.hexdigest()


def native_command(binary,model):
    return [str(binary),'--quiet','serve','--host','127.0.0.1','--port','7861','--threads','2',
            '--no-ui','--device','cuda','--asr.backend.gpu','0','--asr-model',str(model),
            '--asr.batching.enabled=false','--asr.batching.state_arena_slots','1',
            '--asr.endpointing.enable=true','--asr.endpointing.stop_history_eou_ms','800','--max-upload-mb','2']


def main():
    os.umask(0o077)
    root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
    if os.environ.get('H3_ASR_MEDIA_TRANSPORT')!='webrtc-exact-v1':raise ValueError('sealed RTC selection required')
    expires=int(os.environ['H3_ASR_EXPIRES_AT']);seconds=min(1200,expires-int(time.time()))
    if seconds<=0 or len(os.environ.get('H3_ASR_KEY',''))<32:raise ValueError('finite owner required')
    seal=json.loads((root/'asr-rtc.json').read_text())
    expected={'schema':'h3-asr-rtc-runtime-v1','runtime_sha256':RUNTIME_SHA,'model_sha256':MODEL_SHA}
    if any(seal.get(k)!=v for k,v in expected.items()) or set(seal)!=set(expected)|{'project_sha256','lock_sha256'}:
        raise ValueError('RTC runtime seal')
    for name,digest in (('runtime.tar.gz',RUNTIME_SHA),('model.gguf',MODEL_SHA),
                        ('research/webrtc_experiment/pyproject.toml',seal['project_sha256']),
                        ('research/webrtc_experiment/uv.lock',seal['lock_sha256'])):
        if sha(root/name)!=digest:raise ValueError('pinned ASR artifact drift')
    def interrupted(*_):raise InterruptedError('finite ASR supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    deadline=time.monotonic()+seconds;children=[]
    from research.h3_personal.webrtc_sidecar import bootstrap,clean_env
    with tempfile.TemporaryDirectory(prefix='h3-asr-rtc-') as temporary:
        target=Path(temporary)/'native';target.mkdir()
        with tarfile.open(root/'runtime.tar.gz') as archive:archive.extractall(target,filter='data')
        binaries=list(target.glob('*/bin/nemo-speech'))
        if len(binaries)!=1:raise ValueError('one pinned native binary required')
        try:
            children.append(subprocess.Popen(native_command(binaries[0],root/'model.gguf')))
            workspace=Path(temporary)/'rtc';workspace.mkdir()
            python=bootstrap(root,workspace,deadline)
            env=dict(clean_env(),PYTHONPATH=str(root),H3_ASR_KEY=os.environ['H3_ASR_KEY'],
                     H3_ASR_EXPIRES_AT=str(expires),H3_ASR_MEDIA_TRANSPORT='webrtc-exact-v1')
            children.append(subprocess.Popen([str(python),'-m','research.run_h3_asr_rtc_service'],cwd=root,env=env))
            while time.monotonic()<deadline:
                if any(child.poll() is not None for child in children):raise RuntimeError('ASR model or RTC service exited')
                time.sleep(.2)
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)

if __name__=='__main__':
    try:main()
    except BaseException:raise SystemExit(1)
