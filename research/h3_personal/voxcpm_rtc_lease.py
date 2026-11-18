"""One finite lease on the existing Vox L4 Endpoint; no media/device fallback."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import secrets
import time
import uuid

from research.run_h3_native_coupled import PRIVATE,GUARD,preflight,upload_verified
from research.run_h3_irodori_endpoint import control,pause_owned,persist
from research.run_h3_asr_endpoint import assert_deployment
from research.h3_personal.voxcpm_endpoint_protocol import PINS
from research.h3_personal.native_tts_backend import validate_native_config

NAME='h3-voxcpm-l4-a474fd64f740'


@contextmanager
def lease(directory,*,min_remaining=420):
    """Explicit live API: update/resume once; pause/confirm on every exit."""
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    from research.prepare_h3_voxcpm_rtc import stage,deployment
    if type(min_remaining) is not int or not 420<=min_remaining<=900:raise ValueError('bounded remaining lease required')
    root=Path(os.path.abspath(directory));locks=[];owned=False;token=None
    path='/v2/endpoint/YUGOROU/'+NAME
    if root.exists() or not root.is_relative_to(PRIVATE) or any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('fresh private lease directory required')
    try:
        for name in ('.voxcpm-endpoint-owner.lock','.irodori-endpoint-owner.lock','.native-streaming-tts-owner.lock','.network-tts-owner.lock'):
            fd=os.open(PRIVATE/name,os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW,0o600);locks.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in (GUARD.name,'.irodori-endpoint-unresolved.json','.native-streaming-tts-unresolved.json','.network-tts-unresolved.json'):
            if (PRIVATE/name).exists() or (PRIVATE/name).is_symlink():raise RuntimeError('unresolved prior resource')
        if any(k in os.environ for k in ('H3_NATIVE_TTS_KEY','H3_NATIVE_TTS_HF_TOKEN')):raise ValueError('preexisting native credentials')
        root.mkdir(mode=0o700)
        token=cached_token();api=HfApi(token=token)
        spec,hardware=preflight(api,token)
        current=control(token,path)
        if (current.get('name')!=NAME or current.get('status',{}).get('state')!='paused'
                or current.get('provider')!={'vendor':'aws','region':'us-east-1'}
                or current.get('compute',{}).get('instanceType')!='nvidia-l4'
                or current['compute'].get('instanceSize')!='x1'
                or current.get('type')!='authenticated'):
            raise ValueError('existing approved paused L4 Endpoint required')
        expires=int(time.time())+1200;spec=dict(spec,expires_at=expires)
        artifact=stage(root/'deployment',spec)
        repo='YUGOROU/h3-voxcpm-rtc-runtime-'+uuid.uuid4().hex[:12]
        revision=upload_verified(api,artifact,repo)
        session=uuid.uuid4().hex;boot=uuid.uuid4().hex;key=secrets.token_urlsafe(40)
        intent=deployment(repo,revision,NAME,session,boot,expires)
        persist(root/'intent.json',intent);persist(root/'hardware.json',hardware)
        persist(GUARD,dict(name=NAME,namespace='YUGOROU',directory=str(root),expires_at=expires))
        owned=True
        payload=dict(model={**intent['model'],'secrets':{'HF_TOKEN':token,'H3_TTS_KEY':key}},compute=intent['compute'])
        control(token,path,payload,'PUT')
        current=control(token,path);assert_deployment(current,intent)
        control(token,path+'/resume',{},'POST')
        while time.time()<expires-min_remaining:
            current=control(token,path);assert_deployment(current,intent)
            state=current.get('status',{}).get('state')
            if state=='running':break
            if state in ('failed','paused'):raise RuntimeError('RTC Endpoint failed to start')
            time.sleep(5)
        else:raise TimeoutError('RTC Endpoint readiness deadline')
        config=validate_native_config(dict(schema='h3-native-voxcpm-rtc-v3',url=current['status']['url'].rstrip('/'),
            session=session,boot=boot,expires_at=expires,pins=dict(PINS),
            preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0)))
        persist(root/'native-config.json',config)
        yield dict(config=config,token=token,key=key,root=root)
    finally:
        try:
            if owned:
                current=control(token,path);assert_deployment(current,intent)
                state=pause_owned(token,path)
                current=control(token,path);assert_deployment(current,intent)
                if current.get('status',{}).get('state')!='paused':raise RuntimeError('independent Endpoint pause not confirmed')
                # Preserve original guard on any unconfirmed cleanup.
                persist(root/'terminal.json',dict(name=NAME,state=state,independent_confirmation=True))
                import json
                if GUARD.exists() and json.loads(GUARD.read_text()).get('directory')==str(root):GUARD.unlink()
        finally:
            for fd in reversed(locks):os.close(fd)
