"""Finite supervisor: unchanged native Vox runtime plus isolated RTC gateway."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def stop_children(children):
    """Own process groups, including uv descendants after leader exit."""
    errors=[]
    for child in reversed(children):
        try:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        except BaseException as error:errors.append(error)
        finally:
            try:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=3)
            except BaseException as error:errors.append(error)
    if errors:raise errors[0]


def main():
    os.umask(0o077)
    root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
    if os.environ.get('H3_VOX_MEDIA_TRANSPORT')!='webrtc-exact-v1':raise ValueError('explicit RTC deployment required')
    expiry=float(os.environ['H3_TTS_EXPIRES_AT'])
    if not 0<expiry-time.time()<=1200:raise ValueError('finite owner lease required')
    seal=json.loads((root/'vox-rtc.json').read_text())
    if set(seal)!={'schema','project_sha256','lock_sha256'} or seal['schema']!='h3-vox-rtc-runtime-v1':raise ValueError('RTC seal')
    for name,key in (('pyproject.toml','project_sha256'),('uv.lock','lock_sha256')):
        if hashlib.sha256((root/'research/webrtc_experiment'/name).read_bytes()).hexdigest()!=seal[key]:raise ValueError('RTC lock drift')
    deadline=time.monotonic()+expiry-time.time();children=[]
    def interrupted(*_):raise InterruptedError('finite Vox supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    from research.h3_personal.webrtc_sidecar import bootstrap,clean_env
    with tempfile.TemporaryDirectory(prefix='h3-vox-rtc-') as temporary:
        try:
            children.append(subprocess.Popen([sys.executable,str(root/'entrypoint.py'),str(root)],cwd=root,start_new_session=True))
            workspace=Path(temporary)/'rtc';workspace.mkdir()
            python=bootstrap(root,workspace,deadline)
            names=('H3_TTS_KEY','H3_TTS_SESSION','H3_TTS_BOOT','H3_TTS_EXPIRES_AT','H3_VOX_MEDIA_TRANSPORT')
            env=dict(clean_env(),PYTHONPATH=str(root),**{k:os.environ[k] for k in names})
            children.append(subprocess.Popen([str(python),'-m','research.run_h3_voxcpm_rtc_service'],cwd=root,env=env,start_new_session=True))
            while time.monotonic()<deadline:
                if any(child.poll() is not None for child in children):raise RuntimeError('native or RTC child exited')
                time.sleep(.2)
        finally:
            stop_children(children)

if __name__=='__main__':
    try:main()
    except BaseException:raise SystemExit(1)
