"""Synthetic CPU-only Network Group TLS probe; not Vox/model acceptance."""
import hashlib
import json
import os
from pathlib import Path
import signal
import ssl
import struct
import sys
import tempfile
import time


def main():
    os.umask(0o077);root=Path(sys.argv[1]);sys.path.insert(0,str(root))
    spec=json.loads((root/'request.json').read_text());expires=spec['expires_at']
    if not 0<expires-time.time()<=600:raise ValueError('finite CPU probe')
    def expired(*_):raise TimeoutError('CPU probe expired')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    signal.setitimer(signal.ITIMER_REAL,expires-time.time())
    from research.h3_personal.network_voxcpm import RemoteNetworkVox,SCHEMA
    from research.h3_personal.voxcpm_endpoint_protocol import PINS
    config=dict(schema=SCHEMA,connection=dict(spec['connection'],server_job_id=os.environ.get('H3_PROBE_SERVER_JOB_ID',os.environ['JOB_ID'])),
        session=spec['session'],boot=spec['boot'],expires_at=expires,pins=dict(PINS),
        preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0))
    if os.environ['H3_PROBE_ROLE']=='server':
        from research.h3_personal.network_voxcpm_service import Server
        class Candidate:
            sample_rate=48000
            def stream(self,text):yield [.125,-.5,1.0,.0,.25]
        with tempfile.TemporaryDirectory(prefix='h3-cpu-tls-') as tmp:
            cert=Path(tmp)/'cert.pem';key=Path(tmp)/'key.pem'
            cert.write_text(os.environ.pop('H3_NATIVE_TLS_CERT'));key.write_text(os.environ.pop('H3_NATIVE_TLS_KEY'))
            cert.chmod(0o600);key.chmod(0o600)
            server=Server(('0.0.0.0',config['connection']['port']),Candidate(),certificate=str(cert),key_file=str(key),
                key=os.environ['H3_PROBE_KEY'],session=config['session'],boot=config['boot'],expires_at=expires)
            try:
                print(json.dumps(dict(type='h3_network_cpu_ready',synthetic=True,gpu=False)),flush=True)
                server.serve_forever(poll_interval=.1)
            finally:server.server_close()
    elif os.environ['H3_PROBE_ROLE']=='client':
        client=RemoteNetworkVox(config,key=os.environ['H3_PROBE_KEY'])
        try:
            client.read('tts_ready',time.monotonic()+15)
            expected=struct.pack('<5f',.125,-.5,1.0,.0,.25)
            for epoch in (1,2):
                body=b''.join(client.stream('synthetic fixture',epoch,time.monotonic()+15))
                if body!=expected:raise ValueError('synthetic bytes mismatch')
            print(json.dumps(dict(type='h3_network_cpu_result',synthetic=True,gpu=False,turns=2,
                bytes_per_turn=len(expected),sha256=hashlib.sha256(expected).hexdigest(),exact_match=True,tls_pinned=True)),flush=True)
        finally:client.close()
    else:raise ValueError('CPU probe role')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps(dict(type='h3_network_cpu_failed',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1)
