"""Offline by default. Explicit finite HF T4 ASR validation, public audio only.
One Endpoint, min1/max3 REPLICAS, not three separate Endpoints. Pause in finally.
"""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid
import wave

from research.h3_personal.ref2va_budget import HF_CAP
CAP = HF_CAP
RESERVATION = Decimal('0.75')  # three T4 replicas * .50/hr * maximum30min
BASE = 'python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258'
RUNTIME_SHA = 'aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515'
MODEL_SHA = 'a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def control(token, path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request('https://api.endpoints.huggingface.cloud'+path, data=data,
        headers={'Authorization':'Bearer '+token, 'Content-Type':'application/json'}, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def pause_owned(token, path):
    """Always try pause FIRST: a failed preliminary GET must not block cleanup."""
    for attempt in range(3):
        try:
            response = control(token,path+'/pause',{},'POST')
            if response.get('status',{}).get('state')=='paused':
                return 'paused'
            response = control(token,path)
            if response.get('status',{}).get('state')=='paused':
                return 'paused'
        except urllib.error.HTTPError as exc:
            if exc.code==404:
                return 'not-created'
        except Exception:
            pass
        time.sleep(2*(attempt+1))
    raise RuntimeError('PAUSE NOT CONFIRMED; operator must stop owned Endpoint')


def assert_deployment(current, expected):
    for field in ('name','type','provider'):
        if current.get(field)!=expected[field]:
            raise ValueError('owned Endpoint identity/access/provider changed')
    for field in ('accelerator','instanceType','instanceSize'):
        if current.get('compute',{}).get(field)!=expected['compute'][field]:
            raise ValueError('owned Endpoint hardware changed')
    if current.get('compute',{}).get('scaling',{}).get('schedule'):
        raise ValueError('unapproved replica schedule')
    for field in ('minReplica','maxReplica'):
        if current.get('compute',{}).get('scaling',{}).get(field)!=expected['compute']['scaling'][field]:
            raise ValueError('owned Endpoint replica bounds changed')
    for field in ('repository','revision','framework','image','command','env'):
        if current.get('model',{}).get(field)!=expected['model'][field]:
            raise ValueError('immutable runtime/deadline changed')
    if (current.get('model',{}).get('args') or [])!=expected['model']['args']:
        raise ValueError('immutable runtime arguments changed')


def reserve(ledger, key, scope='hf-endpoint-ASR-T4-public-validation'):
    if scope not in ('hf-endpoint-ASR-T4-public-validation','hf-endpoint-ASR-T4-attended-comet'):
        raise ValueError('unknown bounded ASR scope')
    with ledger.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text())
        if key in data['reservations']:
            raise ValueError('reservation replay')
        exposure = Decimal(str(data['spent_usd'])) + sum(Decimal(str(r['upper_bound_usd'])) for r in data['reservations'].values())
        if exposure+RESERVATION > CAP:
            raise ValueError('existing cumulative HF cap exceeded')
        data['reservations'][key] = dict(upper_bound_usd=float(RESERVATION), scope=scope,
                                       created_at=time.time(), max_replicas=3, max_seconds=1800, rate_per_replica='0.50')
        ledger.write_text(json.dumps(data, indent=2))


def deployment(repository, revision, name, deadline, *, rtc=False, session_seconds=1800):
    if type(rtc) is not bool:raise ValueError('explicit ASR transport selection')
    if type(session_seconds) is not int or session_seconds not in (1800, 10800):
        raise ValueError('ASR session_seconds must be1800 or explicit10800')
    value = dict(name=name, type='authenticated', provider=dict(vendor='aws', region='us-east-1'),
        compute=dict(accelerator='gpu',instanceType='nvidia-t4',instanceSize='x1',
                     scaling=dict(minReplica=1,maxReplica=3)),
        model=dict(repository=repository,revision=revision,framework='custom',task='custom',
                   image=dict(custom=dict(url=BASE,port=7860,healthRoute='/ready')),
                   command=['python','/repository/entrypoint.py'],args=[],
                   env=dict(H3_ASR_EXPIRES_AT=str(deadline))),cacheHttpResponses=False,
        tags=['h3-asr-public-only','finite-validation'])
    if session_seconds == 10800:
        value['model']['env']['H3_ASR_SESSION_SECONDS']='10800'
    if rtc:
        value['compute']['scaling']['maxReplica']=1
        value['model']['env']['H3_ASR_MEDIA_TRANSPORT']='webrtc-exact-v1'
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if not args.run:
        print(json.dumps(dict(model='nvidia/nemotron-3.5-asr-streaming-0.6b',precision='Q8_0 ASR only',
                              instance='nvidia-t4 x1',min_replica=1,max_replica=3,max_seconds=1800,
                              reservation_usd=str(RESERVATION),public_samples_only=True,upload_private_repo=True)))
        return
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    from research.probe_h3_asr_cpu_space import http_ready, stream_public
    root = args.directory.resolve()
    if (root/'submission.json').exists() or (root/'intent.json').exists():
        raise ValueError('do not replay existing attempt; use fresh immutable attempt')
    token = cached_token()
    api = HfApi(token=token)
    owner = api.whoami()['name']
    if owner != 'YUGOROU':
        raise ValueError('expected owned namespace')
    # Namespace-scoped provider endpoint; public /v2/provider reports zero quotas!
    providers = control(token,'/v2/provider/'+owner)
    candidates = [c for v in providers['vendors'] if v['name']=='aws'
                  for rg in v['regions'] if rg['name']=='us-east-1'
                  for c in rg['computes'] if c['instanceType']=='nvidia-t4' and c['instanceSize']=='x1']
    if len(candidates)!=1 or candidates[0]['status']!='available' or candidates[0]['quota']['maxAccelerators']-candidates[0]['quota']['usedAccelerators']<3 or Decimal(str(candidates[0]['pricePerHour']))!=Decimal('.5'):
        raise ValueError('live T4 rate/three-replica quota gate failed')
    (root/'preflight.json').write_text(json.dumps(dict(provider=candidates[0], endpoints_before=control(token,'/v2/endpoint/'+owner)['totalItems']),indent=2))
    runtime = root/'runtime.tar.gz'
    model = root.parent/'asr-provider-check-20261019/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf'
    if sha(runtime)!=RUNTIME_SHA or sha(model)!=MODEL_SHA:
        raise ValueError('pinned artifact SHA mismatch')
    artifact = root/'deployment'
    artifact.mkdir()
    for path, name in ((runtime,'runtime.tar.gz'),(model,'model.gguf')):
        os.link(path, artifact/name)
    for name in ('gate.py','entrypoint.py'):
        shutil.copyfile(Path(__file__).parent/'endpoints/nemotron_t4'/name, artifact/name)
    (artifact/'README.md').write_text('# Private finite Nemotron ASR runtime\nASR-only Q8_0 from NVIDIA Nemotron3.5, OpenMDW1.1.\nOfficial NeMo-Speech.cpp v0.2.0 CUDA runtime; bundled notices apply.\nPublic benchmark audio is NOT included. No personal media or credentials.\n')
    identity = uuid.uuid4().hex[:10]
    repo = owner+'/h3-asr-t4-runtime-'+identity
    name = 'h3-asr-t4-'+identity
    api.create_repo(repo,repo_type='model',private=True,exist_ok=False)
    commit = api.upload_folder(repo_id=repo,repo_type='model',folder_path=artifact,commit_message='Pinned finite Nemotron CUDA public-sample validation runtime')
    deadline = int(time.time())+1800
    spec = deployment(repo,commit.oid,name,deadline)
    # Never persist/print the secret-bearing request or unfiltered API response.
    (root/'intent.json').write_text(json.dumps(spec,indent=2))
    key = 'hf-asr-endpoint-'+identity
    reserve(root.parent/'compute-budget.json',key)
    app_key = secrets.token_urlsafe(40)
    spec['model']['secrets'] = {'H3_ASR_KEY':app_key}
    path = '/v2/endpoint/'+owner+'/'+name
    owned = False
    result = dict(name=name,namespace=owner,repository=repo,revision=commit.oid,
                  deadline=deadline,reservation=key,reservation_usd=str(RESERVATION),max_replicas=3,
                  public_samples_only=True,invoice_verified=False,tests=[])
    try:
        # Request could be accepted even if its response times out; finally
        # looks up/pause by this unique owned name even when owned staysFalse.
        response = control(token,'/v2/endpoint/'+owner,spec,'POST')
        owned = True
        (root/'submission.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(name=name,state=response.get('status',{}).get('state'),reservation_usd=str(RESERVATION))),flush=True)
        wait_end = time.monotonic()+1000
        while True:
            response = control(token,path)
            state = response.get('status',{}).get('state')
            print(json.dumps(dict(name=name,state=state)),flush=True)
            if state=='running':
                break
            if state in ('failed','paused') or time.monotonic()>wait_end or time.time()>deadline-180:
                raise RuntimeError('endpoint not ready within finite startup bound')
            time.sleep(15)
        url = response['status']['url']
        parsed = urlsplit(url)
        if parsed.scheme!='https' or not parsed.hostname.endswith('.endpoints.huggingface.cloud') or parsed.path not in ('','/'):
            raise ValueError('unexpected HF endpoint URL')
        host = parsed.hostname
        anon, _ = http_ready(host)
        if anon not in (401,403,404):
            raise ValueError('anonymous endpoint access not denied')
        status, body = http_ready(host,token)
        ready = json.loads(body)
        result.update(url=url,anonymous_status=anon,ready=ready)
        if status!=200 or ready.get('device') not in ('cuda','gpu'):
            raise ValueError('actual CUDA readiness required; no CPU fallback')
        sample_root = root.parent/'asr-provider-check-20261019'
        receipt = json.loads((sample_root/'model-sample-receipt.json').read_text())
        if receipt['sample_url']!='https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/v0.2.0/test_files/asr/wav/test/jfk.wav' or sha(sample_root/'jfk.wav')!=receipt['sample_sha256']:
            raise ValueError('pinned public sample required')
        with wave.open(str(sample_root/'jfk.wav'),'rb') as wav:
            if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,16000):
                raise ValueError('expected public monoPCM16k')
            pcm = wav.readframes(wav.getnframes())
        # Two sequential fresh WS sessions; models resident. First11s, then33s
        # repeated PUBLIC sample to detect accumulating backlog, not accuracy.
        for repeat in (1,3):
            test = stream_public(host,token,pcm*repeat,16000,{'X-H3-ASR-Key':app_key})
            result['tests'].append(test)
            (root/'result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({k:v for k,v in test.items() if k not in ('events','final_segments')}),flush=True)
            if not test['final_received'] or test['errors']:
                raise ValueError('public streaming completion failed')
        result['japanese_verified'] = False
        result['causal_av_verified'] = False
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        # No HTTP body/request/secret dump, including authorization failures.
        print(json.dumps(dict(name=name,error_type=type(exc).__name__)),flush=True)
        raise
    finally:
        result['terminal_state'] = 'unconfirmed'
        try:
            result['terminal_state'] = pause_owned(token,path)
            print(json.dumps(dict(name=name,terminal_state=result['terminal_state'])),flush=True)
        finally:
            # Retain finite reservation; persist even if cleanup requests fail.
            (root/'result.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
