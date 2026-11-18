"""Finite CPU RTC signaling worker, reusable peer path; no H3/private media/devices."""
import asyncio
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile


def bootstrap(root,spec):
    uvdir=Path('/tmp/h3-rtc-uv');uvdir.mkdir(mode=0o700)
    wheel=uvdir/'uv.whl'
    with urllib.request.urlopen(spec['uv']['url'],timeout=30) as response:
        body=response.read(32*1024*1024+1)
    if len(body)>32*1024*1024 or hashlib.sha256(body).hexdigest()!=spec['uv']['sha256']:raise ValueError('uv pin')
    wheel.write_bytes(body)
    with zipfile.ZipFile(wheel) as z:
        names=[n for n in z.namelist() if n.endswith('/uv')]
        if len(names)!=1:raise ValueError('uv executable inventory')
        uv=uvdir/'uv';uv.write_bytes(z.read(names[0]));uv.chmod(0o700)
    project=root/'research/webrtc_experiment';venv=Path('/tmp/h3-rtc-env');req=Path('/tmp/h3-rtc-requirements.txt')
    env=dict(os.environ,UV_NO_CACHE='1',PYTHONPATH=str(root))
    def run(args):
        left=spec['expires_at']-time.time()-15
        if left<=0:raise TimeoutError('bootstrap expired')
        subprocess.run([str(uv),*map(str,args)],env=env,check=True,timeout=min(180,left),stdout=subprocess.DEVNULL)
    run(['venv','--python','3.12',venv])
    python=venv/'bin/python'
    run(['export','--project',project,'--frozen','--no-dev','--no-emit-project','--python',python,'--output-file',req])
    run(['pip','install','--python',python,'--require-hashes','-r',req])
    os.execve(str(python),[str(python),str(root/'research/hf_h3_webrtc_worker.py'),str(root),'--rtc'],env)


def synthetic_tracks(gate,start,seconds=4):
    import av
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    class Track(MediaStreamTrack):
        def __init__(self,kind):super().__init__();self.kind=kind;self.index=0
        async def recv(self):
            await gate.wait();i=self.index;audio=self.kind=='audio';rate=50 if audio else 24
            if i>=seconds*rate:self.stop();raise MediaStreamError
            await asyncio.sleep(max(0,start[0]+i/rate-time.monotonic()))
            if audio:
                f=av.AudioFrame(format='s16',layout='mono',samples=960)
                values=[round(8192*math.sin(2*math.pi*440*(i*960+j)/48000)) for j in range(960)]
                f.planes[0].update(struct.pack('<960h',*values));f.sample_rate=48000
                f.pts=i*960;f.time_base=Fraction(1,48000)
            else:
                f=av.VideoFrame(252,448,'yuv420p')
                for n,p in enumerate(f.planes):p.update(bytes([16+(i%200) if n==0 else 128])*p.buffer_size)
                f.pts=i*3750;f.time_base=Fraction(1,90000)
            self.index+=1;return f
    return [Track('audio'),Track('video')]


async def serve(spec):
    from research.h3_personal.webrtc_live_peer import Session
    from research.h3_personal.webrtc_signal_server import SignalServer
    from research.h3_personal.webrtc_ice import HOST_ONLY, HF_CLOUDFLARE_TURN_TLS, ice_urls
    from research.h3_personal.webrtc_turn import validate_turn_config
    profile=spec.get('ice_profile',HOST_ONLY);urls=ice_urls(profile)
    turn=(validate_turn_config(json.loads(os.environ.pop('H3_RTC_ICE_CREDENTIALS')))
          if profile==HF_CLOUDFLARE_TURN_TLS else None)
    expires=spec['expires_at'];gate=asyncio.Event();start=[None]
    deadline=time.monotonic()+expires-time.time()
    async with Session(deadline=deadline,tracks=synthetic_tracks(gate,start),ice_profile=profile,turn_config=turn) as peer:
        server=SignalServer(session=peer,loop=asyncio.get_running_loop(),session_id=spec['session'],
            key=os.environ['H3_RTC_KEY'],expires_at=expires-1,host='0.0.0.0',port=7860)
        with server:
            print(json.dumps(dict(type='h3_rtc_signaling_ready',session=spec['session'],private_media=False,
                gpu_used=False,ice_profile=profile,external_ice_servers=urls,expires_at=expires-1)),flush=True)
            await peer.wait_control()
            if await peer.receive()!='start':raise ValueError('expected start')
            start[0]=time.monotonic();gate.set();await peer.send('started')
            if await peer.receive()!='stop':raise ValueError('expected stop')
            print(json.dumps(dict(type='h3_rtc_peer_stopped',generated={t.kind:t.index for t in peer._tracks},
                actual_playback_verified=False)),flush=True)


def main():
    os.umask(0o077);root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root));os.chdir(root)
    os.environ.pop('HF_TOKEN',None)
    spec=json.loads((root/'request.json').read_text());remaining=spec['expires_at']-time.time()
    if not 0<remaining<=600:raise ValueError('finite CPU peer lease')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('peer lease expired')))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    if len(sys.argv)==2:bootstrap(root,spec)
    elif sys.argv[2:]==['--rtc']:asyncio.run(serve(spec))
    else:raise ValueError('entry mode')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps(dict(type='h3_rtc_worker_failed',error_type=type(e).__name__)),flush=True)
        raise SystemExit(1)
