"""Explicit finite native FP32 L4 Endpoint validation. Offline by default.

One owned Endpoint/replica, no H3 or devices, no ledger writes or replay. Always
attempt provider pause even after an uncertain create response.
"""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time
import uuid

from research.h3_personal.director_tts_job import IMAGE, REFERENCE_REPO, REFERENCE_REV
from research.h3_personal.director_tts_job import SOURCE, MODEL, CODEC, LOCK_SHA
from research.h3_personal.irodori_endpoint_client import RemoteIrodoriTTS
from research.run_h3_asr_endpoint import assert_deployment
from research.h3_personal.live_relay_client import _DeadlineSocket
from research.h3_personal.private_root import root as _private_root

ROOT = Path(__file__).resolve().parents[1]
OWNER_LOCK = _private_root()/'.irodori-endpoint-owner.lock'
OWNER_GUARD = OWNER_LOCK.with_name('.irodori-endpoint-unresolved.json')
FILES = ('research/__init__.py', 'research/h3_personal/__init__.py',
         'research/hf_h3_live_voice.py', 'research/hf_irodori_tts_diagnose.py',
         'research/h3_personal/director_irodori.py', 'research/h3_personal/director_tts_job.py',
         'research/h3_personal/irodori_endpoint_service.py')


class ControlError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__('bounded Endpoint control failed')


def control(token, path, body=None, method=None):
    end = time.monotonic()+30
    def remaining():
        left = end-time.monotonic()
        if left <= 0: raise TimeoutError('finite control deadline')
        return left
    conn = http.client.HTTPSConnection('api.endpoints.huggingface.cloud', timeout=30)
    try:
        conn.connect(); conn.sock = _DeadlineSocket(conn.sock, remaining)
        data = None if body is None else json.dumps(body).encode()
        conn.request(method or ('POST' if body is not None else 'GET'), path, data,
                     headers={'Authorization':'Bearer '+token, 'Content-Type':'application/json'})
        response = conn.getresponse()
        if not 200 <= response.status < 300: raise ControlError(response.status)
        payload = response.read(1024*1024+1)
        if len(payload) > 1024*1024: raise ValueError('control response limit')
        return json.loads(payload)
    finally:
        conn.close()


def pause_owned(token, path):
    for attempt in range(3):
        try:
            response = control(token, path+'/pause', {}, 'POST')
            if response.get('status',{}).get('state') == 'paused': return 'paused'
        except Exception:
            pass
        # Even a 404 after an uncertain create is not proof of non-creation.
        # Reconcile the owned name; never turn repeated absence into success.
        try:
            response = control(token, path)
            if response.get('status',{}).get('state') == 'paused': return 'paused'
        except Exception:
            pass
        time.sleep(attempt+1)
    raise RuntimeError('Endpoint pause not confirmed; operator action required')


def persist(path, row):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def deployment(repo, revision, name, session, expires):
    return dict(name=name, type='authenticated', provider=dict(vendor='aws', region='us-east-1'),
        compute=dict(accelerator='gpu', instanceType='nvidia-l4', instanceSize='x1',
                     scaling=dict(minReplica=1, maxReplica=1)),
        model=dict(repository=repo, revision=revision, framework='custom', task='custom',
                   image=dict(custom=dict(url=IMAGE, port=7860, healthRoute='/health')),
                   command=['python', '/repository/entrypoint.py'], args=[],
                   env=dict(H3_TTS_EXPIRES_AT=str(expires), H3_TTS_SESSION=session)),
        cacheHttpResponses=False, tags=['h3-irodori-fp32-l4', 'finite-validation'])


def copy_runtime(artifact, source_root=ROOT):
    for name in FILES:
        target = artifact/name; target.parent.mkdir(parents=True, exist_ok=True)
        source = source_root/name
        if name.endswith('/__init__.py') and not source.exists():
            target.write_text('')  # ROOT uses namespace packages; deployment is explicit.
        else:
            shutil.copyfile(source, target)
    shutil.copyfile(source_root/'research/endpoints/irodori_l4/entrypoint.py', artifact/'entrypoint.py')


def stage(root):
    # Resolve from HOME explicitly; never depend on the current shell directory.
    source = _private_root()/'gpu-hf-irodori-diagnostic-v1'
    body = (source/'request.json').read_bytes()
    if hashlib.sha256(body).hexdigest() != 'b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':
        raise ValueError('original TTS manifest drift')
    request = json.loads(body)
    if (request['source_revision'] != SOURCE or request['uv_lock_sha256'] != LOCK_SHA
            or not request.get('license')):
        raise ValueError('original source/license binding required')
    artifact = root/'deployment'; artifact.mkdir(mode=0o700)
    copy_runtime(artifact, ROOT)
    for name in ('NOTICE', 'GEMMA_TERMS_OF_USE.md', 'GEMMA_PROHIBITED_USE_POLICY.md'):
        shutil.copyfile(source/name, artifact/name)
    persist(artifact/'tts-manifest.json', {k:request[k] for k in
        ('source_revision', 'uv_lock_sha256', 'uv_version', 'models', 'license')})
    (artifact/'README.md').write_text('# Private finite native Irodori L4 runtime\n'
        'Native FP32, original pinned Gemma-derived model/codec/SilentCipher. Voice reference is fetched privately by hash, not included here. '
        'No H3, user text/audio, auth tokens or extra dependencies are published in this repository. Owner validates and pauses the Endpoint.\n')
    return artifact, request


def run(root):
    fd = os.open(OWNER_LOCK, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if OWNER_GUARD.is_symlink() or OWNER_GUARD.exists():
            raise RuntimeError('unresolved owned Endpoint; reconcile before new launch')
        return _run_owned(root)
    finally:
        os.close(fd)


def _run_owned(root):
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    os.umask(0o077)
    root = Path(os.path.abspath(root))
    if root.exists() or root.is_relative_to(ROOT) or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('fresh external private directory required')
    root.mkdir(mode=0o700)
    token = cached_token(); api = HfApi(token=token)
    if api.whoami().get('name') != 'YUGOROU': raise ValueError('owned namespace required')
    providers = control(token, '/v2/provider/YUGOROU')
    candidates = [c for v in providers['vendors'] if v['name']=='aws'
        for region in v['regions'] if region['name']=='us-east-1'
        for c in region['computes'] if c['instanceType']=='nvidia-l4' and c['instanceSize']=='x1']
    if (len(candidates) != 1 or candidates[0]['status'] != 'available'
            or candidates[0]['quota']['usedAccelerators'] + 1 > candidates[0]['quota']['maxAccelerators']
            or candidates[0].get('numAccelerators') != 1
            or candidates[0].get('gpuMemoryGb', 0) < 24
            or candidates[0].get('memoryGb', 0) < 24
            or Decimal(str(candidates[0]['pricePerHour'])) > Decimal('10')):
        raise ValueError('native L4 quota/rate scope unavailable')
    persist(root/'hardware-preflight.json', {k:candidates[0][k] for k in
        ('instanceType','instanceSize','numAccelerators','gpuMemoryGb','memoryGb','numCpus','status')})
    existing = control(token, '/v2/endpoint/YUGOROU')
    if type(existing.get('items')) is not list: raise ValueError('Endpoint inventory schema unavailable')
    for e in existing['items']:
        if e.get('name', '').startswith('h3-irodori-l4-') and e.get('status', {}).get('state') not in ('paused', 'scaledToZero'):
            raise ValueError('previous TTS Endpoint not paused')
    api.auth_check(REFERENCE_REPO, repo_type='dataset', write=False)
    if api.repo_info(REFERENCE_REPO, repo_type='dataset', revision=REFERENCE_REV).private is not True:
        raise ValueError('private consenting reference required')
    artifact, request = stage(root)
    for m in request['models']:
        api.auth_check(m['repo'], write=False)
        if api.model_info(m['repo'], revision=m['revision']).sha != m['revision']:
            raise ValueError('native model pin unavailable')
        if m['repo']=='sony/silentcipher' and api.model_info(m['repo'], revision='main').sha != m['revision']:
            raise ValueError('native watermark pin drift')
    identity = uuid.uuid4().hex[:12]
    name, repo = 'h3-irodori-l4-'+identity, 'YUGOROU/h3-irodori-l4-runtime-'+identity
    api.create_repo(repo, repo_type='model', private=True, exist_ok=False)
    commit = api.upload_folder(repo_id=repo, repo_type='model', folder_path=artifact,
                               commit_message='Immutable finite native FP32 Irodori L4 runtime')
    info = api.repo_info(repo, repo_type='model', revision=commit.oid, files_metadata=True)
    local = {str(p.relative_to(artifact)):p for p in artifact.rglob('*') if p.is_file()}
    remote = {e.rfilename:e for e in info.siblings if e.rfilename != '.gitattributes'}
    if not info.private or info.sha != commit.oid or set(local) != set(remote):
        raise ValueError('immutable deployment inventory differs')
    for path, f in local.items():
        body=f.read_bytes(); entry=remote[path]
        digest=hashlib.sha256(body).hexdigest() if entry.lfs else hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if entry.size!=len(body) or digest!=(entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError('immutable deployment digest differs')
    expires = int(time.time())+1200
    session = 'tts-'+identity
    spec = deployment(repo, commit.oid, name, session, expires)
    persist(root/'intent.json', spec)
    key = secrets.token_urlsafe(40)
    spec['model']['secrets'] = dict(HF_TOKEN=token, H3_TTS_KEY=key)
    path = '/v2/endpoint/YUGOROU/'+name
    result = dict(schema='h3-irodori-l4-validation-v1', name=name, namespace='YUGOROU',
                  repository=repo, revision=commit.oid, expires_at=expires,
                  vendor='aws', region='us-east-1', instance_type='nvidia-l4', replicas=1,
                  gpu_fit_verified=False, h3_separation_integrated=False, playback_verified=False, tests=[])
    client = None
    rejected = False
    persist(OWNER_GUARD, dict(name=name, namespace='YUGOROU', directory=str(root), expires_at=expires))
    try:
        try:
            response = control(token, '/v2/endpoint/YUGOROU', spec, 'POST')  # one POST, no replay
        except ControlError as error:
            if error.status in (400, 401, 403, 422):
                rejected = True
                result['create_rejected_status'] = error.status
            raise
        persist(root/'submission.json', {k:result[k] for k in ('name','namespace','repository','revision','expires_at')})
        print(json.dumps(dict(type='h3_tts_endpoint_created', name=name, state=response.get('status',{}).get('state'))), flush=True)
        startup_end = time.monotonic()+900
        previous = None
        while True:
            current = control(token, path)
            assert_deployment(current, spec)
            state = current.get('status',{}).get('state')
            if state != previous:
                print(json.dumps(dict(type='h3_tts_endpoint_state', name=name, state=state)), flush=True)
                previous = state
            if state == 'running': break
            if state in ('failed','paused') or time.monotonic() >= startup_end or time.time() > expires-270:
                raise RuntimeError('finite endpoint readiness failed')
            time.sleep(10)
        url = current['status']['url'].rstrip('/')
        client = RemoteIrodoriTTS(url, token=token, key=key, session=session, expires_at=expires)
        anon = http.client.HTTPSConnection(client._host, timeout=10)
        try:
            anon.request('GET', '/ready'); status = anon.getresponse().status
        finally: anon.close()
        if status not in (401,403,404): raise ValueError('anonymous access not denied')
        ready = client.read('tts_ready', time.monotonic()+30)
        result['ready'] = ready
        result['anonymous_status'] = status
        texts = ('おはようございます。今日は空を眺めながら少し散歩して、温かいお茶を飲んでから、落ち着いて一日を始めようと思います。',
                 '短い休憩でも、好きな音楽を思い浮かべたり窓の外を見たりすると、気分が少し変わります。無理せず自分のペースで進めましょう。')
        for i, text in enumerate(texts):
            started = time.monotonic()
            audio = client.synthesize(text, f'probe-{i+1}', root/f'audio-{i+1}',
                                      time.monotonic()+min(120, expires-time.time()-30))
            receipt = audio['receipt']
            result['tests'].append(dict(sequence=i+1, wall_seconds=time.monotonic()-started,
                samples=receipt['samples'], sample_rate=receipt['sample_rate'], channels=receipt['channels'],
                watermark_applied=receipt['watermark_applied'], metrics=audio['metrics'],
                generated_audio_retained_privately=True))
            print(json.dumps(dict(type='h3_tts_endpoint_synthesis_complete', **result['tests'][-1])), flush=True)
        result['gpu_fit_verified'] = True
    except BaseException as error:
        result['error_type'] = type(error).__name__
        raise
    finally:
        primary = sys.exc_info()[1]
        if client is not None: client.close()
        cleanup_error = None
        try:
            result['terminal_state'] = ('not-created-explicit-rejection' if rejected else pause_owned(token, path))
            if json.loads(OWNER_GUARD.read_text()).get('name') != name:
                raise RuntimeError('owned Endpoint guard mismatch')
            OWNER_GUARD.unlink()
        except BaseException as error:
            cleanup_error = error
            result['terminal_state'] = 'pause-unconfirmed'
            result['cleanup_error_type'] = type(error).__name__
        try:
            persist(root/'result.json', result)
        except BaseException as error:
            cleanup_error = cleanup_error or error
        print(json.dumps(dict(type='h3_tts_endpoint_terminal', name=name,
                              terminal_state=result.get('terminal_state'), gpu_fit_verified=result['gpu_fit_verified'])), flush=True)
        if cleanup_error is not None:
            if primary is not None:
                primary.add_note('Endpoint cleanup/persistence incomplete; inspect owned Endpoint immediately')
            else:
                raise cleanup_error
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path)
    p.add_argument('--run', action='store_true')
    a=p.parse_args(argv)
    if not a.run:
        print(json.dumps(dict(scope='offline only', native_precision='fp32', hardware='L4 x1',
                              provider='aws', region='us-east-1', max_replicas=1, max_seconds=1200)))
        return 0
    if a.directory is None: p.error('--directory required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        run(a.directory)
    except BaseException as error:
        print(json.dumps(dict(type='h3_tts_endpoint_owner_failed', error_type=type(error).__name__)), flush=True)
        return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
