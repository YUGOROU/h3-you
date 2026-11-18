"""Dedicated finite L4 native Vox Job service; no Endpoint or public media port.

Staged request.json + Job secrets are owned by a separate paired-job launcher.
No launcher/default switch, model fallback or provider placement inference here.

Scheduling wait is separate from the runtime lease: however long the Job waits
for hardware, the actual source lease is clamped to LEASE_SECONDS from container
start (and the Job's own timeoutSeconds=1200 kills it regardless).
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import ssl
import sys
import tempfile
import time

LEASE_SECONDS = 1200
QUEUE_SECONDS_MAX = 4 * 60 * 60


def lease(expires_at, now):
    """Split the submitted deadline into queue allowance + fixed runtime lease.

    `expires_at` is fixed at submission, so a long scheduling wait must not
    silently shorten (or extend) the source's own runtime lease: the worker
    clamps its effective lease to `now+LEASE_SECONDS` and that clamped value is
    what the parent accepts from READY.
    """
    if type(expires_at) not in (int, float) or not math.isfinite(expires_at):
        raise ValueError('finite source lease')
    if not 0 < expires_at - now <= QUEUE_SECONDS_MAX + LEASE_SECONDS:
        raise ValueError('finite source lease')
    return min(expires_at, now + LEASE_SECONDS)


def main():
    os.umask(0o077)
    root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root));os.chdir(root)
    spec=json.loads((root/'request.json').read_text());expires=spec['expires_at']
    expires=lease(expires,time.time())
    def expired(*_):raise TimeoutError('finite L4 source expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,expires-time.time())
    from research.h3_personal.network_voxcpm import validate_config,address,SCHEMA
    service=spec['network_service']
    if type(service) is not dict or set(service)!={'group','alias','port','certificate_sha256'}:raise ValueError('sealed network service required')
    session,boot=(os.environ[k] for k in ('H3_TTS_SESSION','H3_TTS_BOOT'))
    config=validate_config(dict(schema=SCHEMA,connection=dict(service,server_job_id=os.environ['JOB_ID']),
        session=session,boot=boot,expires_at=expires,pins=spec['native_pins'],
        preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0)))
    address(config)  # Provider prefix must exist before model bootstrap.
    key=os.environ['H3_TTS_KEY']
    if len(key)<32 or not key.isascii() or any(ord(c)<33 or ord(c)>126 for c in key):raise ValueError('source owner key')
    cert=os.environ['H3_NATIVE_TLS_CERT'];private=os.environ['H3_NATIVE_TLS_KEY']
    if hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert)).hexdigest()!=service['certificate_sha256']:
        raise ValueError('source certificate manifest mismatch')
    from research.hf_h3_native_tts_probe import bootstrap,models,validate_profile
    if (spec['candidate']!='voxcpm2' or validate_profile(spec) or any(spec.get(k,False) for k in
            ('voxcpm_trace_abba','voxcpm_supply_diagnostic','voxcpm_stage_diagnostic','voxcpm_lm_compile'))):
        raise ValueError('native eager profile required')
    if len(sys.argv)==2:
        bootstrap(root,spec,reexec_target=Path(__file__).resolve());return
    if sys.argv[2:]!=['--native']:raise ValueError('entry mode')
    with tempfile.TemporaryDirectory(prefix='h3-native-tls-') as directory:
        directory=Path(directory);cert_file=directory/'cert.pem';key_file=directory/'key.pem'
        cert_file.write_text(cert);key_file.write_text(private)
        cert_file.chmod(0o600);key_file.chmod(0o600)
        os.environ.pop('H3_NATIVE_TLS_KEY',None);os.environ.pop('H3_NATIVE_TLS_CERT',None)
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(cert_file,key_file)
        import torch
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA L4':raise ValueError('exact L4 required')
        from research.h3_personal.voxcpm2_streaming_candidate import load,MODEL_ID,MODEL_REVISION
        from research.h3_personal.voxcpm_endpoint_protocol import PINS
        if (spec['model']['repo']!=MODEL_ID or spec['model']['revision']!=MODEL_REVISION
                or spec['reference']['sha256']!=PINS['reference_sha256']):raise ValueError('native source pins')
        model,reference=models(spec);candidate=load(model,reference,optimize=False)
        from research.endpoints.voxcpm_l4.entrypoint import LoggedCandidate,emit
        from research.h3_personal.network_voxcpm_service import Server
        wrapped=LoggedCandidate(candidate,lambda:server.gate.epoch)
        server=Server(('0.0.0.0',service['port']),wrapped,certificate=str(cert_file),key_file=str(key_file),
            key=key,session=session,boot=boot,expires_at=expires,max_requests=2)
        wrapped.runtime=dict(type='h3_voxcpm_network_ready',device_name=torch.cuda.get_device_name(0),
            device_count=1,total_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch=str(torch.__version__),cuda=str(torch.version.cuda),optimize=False,sample_rate=48000)
        try:
            emit(dict(type='h3_voxcpm_network_ready',job_id=os.environ['JOB_ID'],
                config=config,runtime=wrapped.runtime))
            server.serve_forever(poll_interval=.1)
        finally:server.gate.close();server.server_close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(type='h3_voxcpm_network_failed',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1)
