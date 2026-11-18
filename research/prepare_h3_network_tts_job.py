"""Offline immutable L4 Job artifact/payload builder. Never submits a Job.

Caller must perform original model/license/private-reference/placement gates,
use fresh per-run network names, fence uncertain creation, and own cancellation.
TLS private material belongs in Job secrets only, not source ZIP/config/logs.
"""
import hashlib
import json
from pathlib import Path
import re
import zipfile
from research.run_h3_native_coupled import FILES,ROOT
from research.h3_personal.director_tts_job import IMAGE
from research.h3_personal.job_network import network
from research.h3_personal.voxcpm_endpoint_protocol import PINS

EXTRA=('research/hf_h3_voxcpm_network_worker.py',
       'research/h3_personal/network_voxcpm.py',
       'research/h3_personal/network_voxcpm_service.py',
       'research/h3_personal/tts_preroll.py',
       'research/endpoints/voxcpm_l4/entrypoint.py')


def bundle(directory,spec):
    directory=Path(directory)
    if directory.exists() or any(p.is_symlink() for p in (directory,*directory.parents)):
        raise ValueError('fresh immutable artifact directory required')
    service=spec.get('network_service',{})
    if (type(service) is not dict or set(service)!={'group','alias','port','certificate_sha256'}
            or spec.get('native_pins')!=PINS or spec.get('candidate')!='voxcpm2'
            or spec.get('voxcpm_optimize',False) is not False):raise ValueError('native network source selection required')
    network(service['group'],[service['alias']])
    if (type(service['port']) is not int or not 1024<=service['port']<=65535
            or type(service['certificate_sha256']) is not str
            or not re.fullmatch('[a-f0-9]{64}',service['certificate_sha256'])):raise ValueError('TLS endpoint manifest')
    if any(k in spec for k in ('secrets','key','private_key','token','certificate_pem')):raise ValueError('credentials forbidden in artifact')
    directory.mkdir(mode=0o700,parents=True);path=directory/'runtime.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in dict.fromkeys((*FILES,*EXTRA)):
            source=ROOT/name
            if source.is_symlink() or not source.is_file():raise ValueError('first-party source required')
            z.writestr(name,source.read_bytes())
        for name in ('research/__init__.py','research/h3_personal/__init__.py'):z.writestr(name,'')
        z.writestr('request.json',json.dumps(spec,allow_nan=False))
        if len(z.infolist())>=32:raise ValueError('bootstrap source file bound')
    if path.stat().st_size>2*1024*1024:raise ValueError('source archive byte bound')
    return path


def payload(repo,revision,digest,*,group,session,boot,token,key,certificate,private_key,alias='voxcpm'):
    if (not re.fullmatch('YUGOROU/[a-zA-Z0-9_-]+',repo) or not re.fullmatch('[a-f0-9]{40}',revision)
            or not re.fullmatch('[a-f0-9]{64}',digest)
            or any(not re.fullmatch('[a-f0-9]{32}',v) for v in (session,boot))):raise ValueError('immutable source identity')
    if not all(type(v) is str and v for v in (token,key,certificate,private_key)):raise ValueError('in-memory source secrets required')
    from research.run_h3_native_tts_probe import bootstrap
    script=bootstrap(repo,revision,digest)
    old="root/'research/hf_h3_native_tts_probe.py'"
    if script.count(old)!=1:raise ValueError('bootstrap template drift')
    script=script.replace(old,"root/'research/hf_h3_voxcpm_network_worker.py'")
    # The pinned pytorch image exposes `python` on PATH and has no `/opt/venv`
    # (that path belongs to the custom H3 runtime image). Using it here made the
    # container fail before start: exec /opt/venv/bin/python: no such file.
    return dict(dockerImage=IMAGE,command=['python','-c',script],arguments=[],
        environment={'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1'},
        secrets=dict(HF_TOKEN=token,H3_TTS_KEY=key,H3_TTS_SESSION=session,H3_TTS_BOOT=boot,
            H3_NATIVE_TLS_CERT=certificate,H3_NATIVE_TLS_KEY=private_key),
        flavor='l4x1',timeoutSeconds=1200,attempts=1,network=network(group,[alias]),
        labels=dict(h3session=session,h3boot=boot,deployrev=revision))
