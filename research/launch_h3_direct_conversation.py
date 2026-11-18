"""Finite direct authenticated Jobs conversation; offline inspect by default.

--run reads 1–2 fresh RAM-only typed turns by default; optional attended ASR.
No Space, mailbox,
SDK upgrade, device setup or automatic OBS playback. TTS defaults to complete-WAV.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import sys
import time
from types import FunctionType
import uuid

from research.launch_h3_conversation_scenario import ROOT, PRIVATE, _bound as arithmetic_bound, closure, rewrite_allowlist
from research.launch_h3_relay_conversation import canonical, sdk_redirect_guard, TERMINAL


def inspect(stage):
    from research.launch_h3_live_voice import inspect as shared_inspect
    from research.hf_h3_direct_conversation import validate_spec
    spec, inventory = shared_inspect(stage)
    validate_spec(spec)
    return spec, inventory


def first_party_sources(root=ROOT):
    """Never stage nested uv environments, caches or installed dependencies."""
    for path in sorted((root/'research').rglob('*.py')):
        parts = path.relative_to(root/'research').parts
        if any(part.startswith('.') or part in ('__pycache__', 'site-packages', 'venv') for part in parts):
            continue
        if path.is_symlink():
            raise ValueError('first-party Python symlink forbidden')
        yield path


def ordered_selection(value, *, confirm_private_reference_upload=False, require_confirmation=False):
    """Fresh local validation only; confirmation is explicit operator scope, never go."""
    if type(confirm_private_reference_upload) is not bool:
        raise ValueError('explicit private reference upload confirmation required')
    if value is None:
        if confirm_private_reference_upload:
            raise ValueError('upload confirmation requires ordered references')
        return None, {}
    if require_confirmation and not confirm_private_reference_upload:
        raise ValueError('new private references require explicit cloud scope confirmation before resources')
    from research.h3_personal.ordered_references import verify_local_files,content_identity,content_sha256
    selected=verify_local_files(value)
    return selected, dict(ordered_reference_identity=content_identity(selected,short_edge=1024),
                          ordered_reference_sha256=content_sha256(selected,short_edge=1024))


def stage_ordered_inputs(stage, selected):
    """Copy exact bytes only after caller's confirmation, retaining original inputs."""
    from research.h3_personal.ordered_references import verify_local_files,MAX_FILE_BYTES,reference_roles
    selected=verify_local_files(selected)
    # Staged names follow the explicit schema roles; a single-reference schema
    # stages only person.png and never a truncated/zipped surroundings name.
    names=tuple(role+'.png' for role in reference_roles(selected['schema']))
    if len(names)!=len(selected['references']):
        raise ValueError('staged names must match explicit reference cardinality')
    destination=Path(stage)/'ordered-inputs'
    destination.mkdir(mode=0o700,exist_ok=False)
    for row,name in zip(selected['references'],names):
        path=destination/name
        digest=hashlib.sha256();count=0
        with open(row['path'],'rb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW)) as source, open(path,'xb',opener=lambda p,f:os.open(p,f,0o600)) as target:
            while True:
                data=source.read(1024*1024)
                if not data:break
                count+=len(data)
                if count>MAX_FILE_BYTES:raise ValueError('ordered input byte bound exceeded')
                digest.update(data);target.write(data)
            target.flush();os.fsync(target.fileno())
        if digest.hexdigest()!=row['sha256']:
            raise ValueError('ordered input changed during staging')
    verify_local_files(selected)


def prepare(directory, row, api, *, source_snapshot=None, ordered_references=None,
            confirm_private_reference_upload=False):
    """Shared preparation; private image bytes only under explicit scoped opt-in."""
    selected, selection=ordered_selection(ordered_references,
        confirm_private_reference_upload=confirm_private_reference_upload,require_confirmation=True)
    if any(row.get(k)!=v for k,v in selection.items()) or (not selection and
            any(k in row for k in ('ordered_reference_identity','ordered_reference_sha256'))):
        raise ValueError('ordered reference immutable row mismatch')
    if selected is not None and row.get('visual_intents') is not True:
        raise ValueError('ordered references require visual intents')
    from research import prepare_h3_fast_ref2va_gpu as base
    from research.launch_h3_live_voice import EXTRA
    from research.hf_h3_direct_conversation import endurance_limits
    _, _, timeout = endurance_limits(row.get('five_minute_test', False), row.get('ten_minute_test', False), row.get('long_session', False))
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('private staging outside repository required')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stage = directory/'deployment'
    if stage.exists():
        raise ValueError('fresh deployment required')
    dependencies = dict(base.prepare.__globals__, budget_bound=arithmetic_bound)
    preparer = FunctionType(base.prepare.__code__, dependencies, base.prepare.__name__, base.prepare.__defaults__)
    preparer(stage, PRIVATE/'gpu-hf-realtime-first-v2/deployment', PRIVATE/'compute-budget.json', api,
             publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-direct-'+uuid.uuid4().hex[:12],
             timeout_seconds=1200 if row.get('long_session') else timeout, attention_mode='fa4-packed', layout='portrait',
             runtime_policy='resident-visual-cached', locked_speech=None)
    if row.get('tts_mode','complete') != 'native_voxcpm':
        source = (PRIVATE/'gpu-hf-irodori-diagnostic-v1/request.json').read_bytes()
        if hashlib.sha256(source).hexdigest() != 'b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':
            raise ValueError('TTS manifest drift')
        manifest = json.loads(source)
        (stage/'tts-manifest.json').write_text(json.dumps({k:manifest[k] for k in ('source_revision','uv_lock_sha256','uv_version','models','license')}))
    source_root=ROOT if source_snapshot is None else Path(source_snapshot)
    for path in first_party_sources(source_root):
        target = stage/path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    if (row.get('media_transport')=='webrtc-rtp-v1'
            or row.get('native_tts',{}).get('schema')=='h3-native-voxcpm-rtc-v3'):
        for name in ('pyproject.toml','uv.lock'):
            source=ROOT/'research/webrtc_experiment'/name
            if source.is_symlink():raise ValueError('RTC lock symlink forbidden')
            target=stage/'research/webrtc_experiment'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    if selected is not None:
        stage_ordered_inputs(stage,selected)
    closure(stage, [p[:-3].replace('/', '.') for p in EXTRA] + ['research.hf_h3_direct_conversation'])
    spec = json.loads((stage/'spec.json').read_text())
    if row.get('long_session'):
        spec['config']['timeout_seconds'] = timeout
        spec['execution']['timeout_seconds'] = timeout
    spec['live_voice_session'] = dict(id=row['session'], max_turns=1000 if row.get('long_session') else 2, fresh_input_required=True)
    spec['live_direct_session'] = row
    (stage/'spec.json').write_text(json.dumps(spec))
    rewrite_allowlist(stage)
    inspect(stage)
    return stage


def durable_json(path, row):
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f'unsupported durable JSON type: {type(value).__name__}')
    # Validate before creating the exclusive receipt: no partial JSON on error.
    payload = json.dumps(row, default=encode, allow_nan=False)
    with open(path, 'x', opener=lambda p, flags: os.open(p, flags, 0o600)) as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_review_previews(root, *, max_seconds=120):
    """Offline local viewing renditions AFTER GPU/Endpoint cleanup.

    NUTs remain authoritative encoder bytes, not device playback. MOV previews
    never invent time, pad audio, concatenate discontinuities or launch a player.
    """
    import subprocess
    root=Path(root).resolve(strict=True)
    if root.is_relative_to(ROOT) or root.is_symlink():raise ValueError('private review root required')
    if type(max_seconds) not in (int,float) or not 0<max_seconds<=120:raise ValueError('bounded preview time')
    index=json.loads((root/'review-index.json').read_text())
    cap=min(index['max_bytes'],2*1024**3);deadline=time.monotonic()+max_seconds;records=[]
    def media_info(path):
        value=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=nb_read_frames','-of','json',str(path)],capture_output=True,
            timeout=max(.001,deadline-time.monotonic()),check=True)
        return int(json.loads(value.stdout)['streams'][0]['nb_read_frames'])
    def audio_hash(path):
        value=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:a:0',
            '-c:a','pcm_s16le','-f','hash','-hash','sha256','pipe:1'],capture_output=True,
            timeout=max(.001,deadline-time.monotonic()),check=True)
        return value.stdout.strip().decode('ascii')
    def digest_file(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def safe(name):
        if type(name) is not str or Path(name).name!=name or (root/name).is_symlink():
            raise ValueError('owned review basename required')
        return root/name
    for entry in index['entries']:
        files=entry['files'];kind=entry['kind']
        if kind=='encoder-span':
            src=safe(next(x['name'] for x in files if x['name'].endswith('.nut')))
            inputs=['-i',str(src)];codec=['-c:v','libx264','-preset','fast','-crf','12','-pix_fmt','yuv420p']
            label='viewing_transcode_not_native_pixel_or_real_time_supply_proof'
        elif kind=='full-native-cut-reference':
            src=safe(next(x['name'] for x in files if x['name'].endswith('.mp4')))
            pcm=safe(next(x['name'] for x in files if x['name'].endswith('.s16le')))
            inputs=['-i',str(src),'-f','s16le','-ar','48000','-ac','1','-i',str(pcm)]
            codec=['-c:v','copy'];label='full_native_reference_includes_omitted_prefix_not_presented'
        else:continue
        out=src.with_suffix('.mov')
        if out.exists():raise ValueError('preview replay forbidden')
        used=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        limit=min(256*1024**2,cap-used-256*1024)
        if limit<=0:raise ValueError('shared review byte cap')
        fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
        command=['ffmpeg','-v','error','-nostdin','-y',*inputs,'-map','0:v:0',
                 '-map',('1:a:0' if kind=='full-native-cut-reference' else '0:a:0'),
                 *codec,'-threads','1','-c:a','pcm_s16le','-fs',str(limit),'-movflags','+faststart',str(out)]
        try:
            result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                timeout=max(.001,deadline-time.monotonic()),check=False)
            if result.returncode or not out.stat().st_size or out.stat().st_size>=limit:
                raise ValueError('preview decode/byte limit failed; originals retained')
            if media_info(src)!=media_info(out):raise ValueError('preview frame truncation')
            expected_audio=audio_hash(src) if kind=='encoder-span' else 'SHA256='+digest_file(pcm)
            if expected_audio!=audio_hash(out):raise ValueError('preview PCM changed')
            records.append(dict(file=out.name,source=src.name,bytes=out.stat().st_size,
                sha256=digest_file(out),source_sha256=digest_file(src),
                kind=kind,semantics=label,global_mapping=entry['metadata'],
                real_time_supply_proof=False,device_output_verified=False))
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as error:
            out.unlink(missing_ok=True)
            records.append(dict(source=src.name,status='preview_unavailable_original_retained',error_type=type(error).__name__))
            if time.monotonic()>=deadline:break
    durable_json(root/'review-previews.json',dict(schema='h3-review-previews-v1',previews=records,
        no_automatic_playback=True,no_external_upload=True,
        note='Each file keeps its own span/packet origin. No montage, missing-time padding, or real playback claim.'))
    if sum(p.stat().st_size for p in root.iterdir() if p.is_file())>cap:raise ValueError('total review cap exceeded')
    return dict(directory=str(root),manifest=str(root/'review-previews.json'),files=records)


from research.h3_personal.direct_job_api import JobCreationRejected


def _backup_compute_binding(profile, main_api, main_token):
    """Verify the explicit alternate Job owner without changing asset credentials."""
    if (profile is None or getattr(profile, 'namespace', None) != '2nd2nd'
            or getattr(profile, 'token', None) == main_token
            or type(getattr(profile, 'token', None)) is not str or not profile.token
            or getattr(getattr(profile, 'api', None), 'endpoint', None) != 'https://huggingface.co'
            or getattr(profile.api, 'token', None) != profile.token
            or not callable(getattr(profile, 'assert_idle', None))):
        raise ValueError('explicit distinct canonical backup Job credentials required')
    if profile.api.whoami().get('name') != '2nd2nd':
        raise ValueError('backup Job owner differs')
    return profile


def submit(api, stage, token, producer_key, consumer_key, *, adoption=None, compute_profile=None):
    canonical()
    from huggingface_hub import CommitOperationAdd
    from research.launch_h3_live_voice import pre_submit
    from research.h3_personal.direct_job_api import submit_job
    import httpx
    spec, inventory = inspect(stage)
    job_api, job_token, job_namespace = api, token, 'YUGOROU'
    if compute_profile is not None:
        compute_profile = _backup_compute_binding(compute_profile, api, token)
        row = spec.get('live_direct_session', {})
        if (adoption is not None or row.get('stable_selfie') is not True
                or row.get('attended_visual', False) or row.get('turn_interruptions', False)
                or row.get('paired_cuts', False) or row.get('long_session', False)
                or row.get('tts_mode') != 'native_voxcpm'
                or row.get('media_transport', 'http-paired-v1') != 'http-paired-v1'
                or row.get('native_tts', {}).get('schema') != 'h3-native-voxcpm-backend-v1'
                or 'expected_asr' in row):
            raise ValueError('backup billing supports device-free stable native HTTPS only')
        job_api, job_token, job_namespace = compute_profile.api, compute_profile.token, compute_profile.namespace
        compute_profile.assert_idle()
    from research.h3_personal.native_tts_backend import native_job_secrets
    native_secrets=native_job_secrets(spec)
    guard = stage.parent/'submission_guard.json'
    if guard.exists():
        raise ValueError('unknown/submitted outcome: NEVER replay')
    owner = api.whoami()
    if owner.get('name') != 'YUGOROU' or owner.get('isPro') is not True:
        raise ValueError('Jobs-enabled owner required')
    from research.h3_personal.job_network import assert_idle, native_config, network
    if adoption is not None:
        from research.h3_personal.h3_first_owner import DeferredH3Owner
        if not isinstance(adoption,DeferredH3Owner):raise ValueError('explicit campaign-owned H3 adoption required')
    admission=assert_idle if adoption is None else adoption.admit
    admission(api,spec,token)
    pre_submit(api, spec, stage)
    repo = spec['publication_repo']
    with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as control:
        try:
            response = control.post('https://huggingface.co/api/repos/create',
                headers={'Authorization':'Bearer '+token}, json=dict(name=repo.split('/')[1], organization='YUGOROU', type='dataset', private=True))
        except httpx.TransportError:
            # The POST may have succeeded. Never replay it; only adopt the exact
            # fresh random deployment repo when it is still empty and private.
            try:
                remote = api.repo_info(repo, repo_type='dataset')
                verified = (remote.id == repo and remote.id.split('/')[0] == owner['name']
                            and remote.private is True
                            and all(s.rfilename == '.gitattributes' for s in remote.siblings))
            except Exception:
                verified = False
            if not verified:
                raise
        else:
            if response.status_code not in (200, 201):
                raise RuntimeError('dataset creation failed; no retry')
    if api.repo_info(repo, repo_type='dataset').private is not True:
        raise ValueError('private deployment required')
    names = [r['path'] for r in inventory['files']] + ['upload-allowlist.json']
    commit = api.create_commit(repo, repo_type='dataset', operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(stage/n)) for n in names], commit_message='Immutable finite direct conversation runtime')
    remote = api.repo_info(repo, repo_type='dataset', revision=commit.oid, files_metadata=True)
    entries = {s.rfilename:s for s in remote.siblings}
    if remote.sha != commit.oid or remote.private is not True or set(entries)-{'.gitattributes'} != set(names):
        raise ValueError('remote inventory differs')
    for n in names:
        body = (stage/n).read_bytes(); entry = entries[n]
        digest = hashlib.sha256(body).hexdigest() if entry.lfs else hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if entry.size != len(body) or digest != (entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError('remote digest differs')
    bootstrap = 'research/bootstrap_h3_fast_ref2va.py'
    script = ("import hashlib,os,runpy;from huggingface_hub import hf_hub_download;"
              f"p=hf_hub_download({repo!r},{bootstrap!r},repo_type='dataset',revision={commit.oid!r},token=os.environ['HF_TOKEN']);"
              f"assert hashlib.sha256(open(p,'rb').read()).hexdigest()=={hashlib.sha256((stage/bootstrap).read_bytes()).hexdigest()!r};runpy.run_path(p,run_name='__main__')")
    command = ['/opt/venv/bin/python','-c',script,repo,commit.oid,hashlib.sha256((stage/'upload-allowlist.json').read_bytes()).hexdigest()]
    payload = dict(dockerImage=spec['image'], command=command, arguments=[],
        environment={'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1'},
        secrets={'HF_TOKEN':token,'H3_DIRECT_PRODUCER_KEY':producer_key,'H3_DIRECT_CONSUMER_KEY':consumer_key,**native_secrets},
        flavor='h200x2', timeoutSeconds=spec.get('config', {}).get('timeout_seconds', 1200), attempts=1,
        expose={'ports':[7860],'portsPublic':[]}, labels={'deployrev':commit.oid})
    source=native_config(spec)
    if source is not None:payload['network']=network(source['connection']['group'])
    native=spec.get('live_direct_session',{}).get('native_tts',{})
    if native.get('schema')=='h3-native-voxcpm-rtc-v3':
        payload['environment']['H3_SESSION_NOT_AFTER']=str(native['expires_at'])
    # Recheck idleness/pins after upload, immediately before the one POST.
    pre_submit(api, spec, stage)
    admission(api,spec,token)
    if compute_profile is not None:compute_profile.assert_idle()
    durable_json(guard, dict(state='unknown_no_replay', revision=commit.oid,
                            job_namespace=job_namespace, main_asset_namespace='YUGOROU'))
    job = None
    def on_created(created):
        nonlocal job
        # Own the ID before receipt I/O, which may itself be interrupted.
        job = created
        durable_json(stage.parent/'submission.json', dict(job_id=job.id, repo=repo, revision=commit.oid,
                                                        job_namespace=job_namespace, main_asset_namespace='YUGOROU'))
    try:
        if adoption is not None:
            job,image,actual_command=adoption.launch(api,stage,token,payload,commit.oid,
                hashlib.sha256((stage/'upload-allowlist.json').read_bytes()).hexdigest())
            on_created(job)
            return job,image,actual_command
        with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as control:
            job = submit_job(control, job_token, payload, on_created=on_created,
                **({'namespace':job_namespace} if compute_profile is not None else {}),
                **({'long_session':True} if spec.get('live_direct_session',{}).get('long_session') else {}))
        print(json.dumps(dict(type='h3_job_submitted', job_id=job.id)), flush=True)
        return job, spec['image'], command
    except BaseException as original:
        if isinstance(original,JobCreationRejected):
            durable_json(stage.parent/'submission-rejection.json',dict(
                status=original.status,detail=original.detail,revision=commit.oid))
        if job is not None:
            try:
                job_api.cancel_job(job_id=job.id, **({'namespace':job_namespace} if compute_profile is not None else {}))
            except BaseException as error:
                original.add_note('Job cancellation failed: '+type(error).__name__)
            # submit() may fail before the caller receives the owned Job ID.
            # Its own exception path therefore owns independent terminal proof.
            try:
                confirm_submit_failure_terminal(job_api, job, stage.parent,
                                                job_namespace=job_namespace)
            except BaseException as error:
                original.add_note('Job terminal confirmation failed: '+type(error).__name__)
        raise


def confirm_submit_failure_terminal(api, job, directory, *, job_namespace):
    from research.h3_personal.job_compute_profile import validate_namespace, TERMINAL
    validate_namespace(job_namespace)
    ns = {'namespace': job_namespace} if job_namespace != 'YUGOROU' else {}
    last_error = None
    for attempt in range(45):
        try:
            state = api.inspect_job(job_id=job.id, **ns).status.stage
        except Exception as error:
            last_error = error
        else:
            if state in TERMINAL:
                durable_json(Path(directory)/'job-terminal.json', dict(
                    job_id=job.id, status=state, job_namespace=job_namespace,
                    independent_confirmation=True, cleanup_owner='submit-exception'))
                return
        if attempt < 44:
            time.sleep(1)
    raise RuntimeError('owned submitted Job terminal status unconfirmed') from last_error


def native_source2_healthy(worker, endpoint):
    """Independent EOS/native-byte proof, not voice quality or cancelled-cache proof."""
    from research.h3_personal.voxcpm_endpoint_protocol import PINS
    try:
        receipt=next(r for r in worker['receipts'] if r['epoch']==2)
        model=next(r for r in endpoint['evidence']['completions'] if r['epoch']==2)
        proof=model['completion']
        return bool(receipt['completion']=='protocol_END' and receipt['pins']==PINS
            and type(model['samples']) is int and model['samples']>0
            and model['samples']==receipt['native_samples']
            and model['native_float_sha256']==receipt['native_sha256']
            and proof['reason']=='natural_stop' and proof['natural_stop'] is True
            and proof['stream_exhausted'] is True and type(proof['stop_index']) is int
            and type(proof['effective_max_len']) is int
            and 0<=proof['stop_index']<proof['effective_max_len']-1)
    except (KeyError,TypeError,StopIteration):return False


def validate_native_cancel_evidence(worker, control, endpoint):
    """Pure cross-proof; no fake EOS/cache health from iterator tests.

    Times subtracted below are ONLY L4 native/control monotonic nanoseconds.
    No historical full-drain counterfactual or speedup percentage is invented.
    """
    from research.h3_personal.voxcpm_endpoint_protocol import PINS
    if any(type(v) is not dict for v in (worker,control,endpoint)) or type(endpoint.get('evidence')) is not dict:
        raise ValueError('native cancel evidence objects required')
    receipts=worker.get('receipts'); rows=endpoint.get('evidence',{}).get('completions')
    runtime=endpoint.get('evidence',{}).get('runtime',{})
    if (type(receipts) is not list or type(rows) is not list or type(runtime) is not dict
            or any(type(r) is not dict for r in receipts+rows)
            or [r.get('epoch') for r in receipts]!=[1,2]
            or [r.get('epoch') for r in rows]!=[1,2]
            or type(runtime.get('device_count')) is not int or runtime['device_count']!=1
            or runtime.get('device_name')!='NVIDIA L4' or runtime.get('optimize') is not False
            or receipts[0].get('completion')!='interrupted_protocol_END'
            or receipts[1].get('completion')!='protocol_END'):
        raise ValueError('native cancel/reuse objective not exercised')
    for receipt,row in zip(receipts,rows):
        if (receipt.get('pins')!=PINS or row.get('samples')!=receipt.get('native_samples')
                or row.get('native_float_sha256')!=receipt.get('native_sha256')
                or type(row.get('samples')) is not int or row['samples']<=0):
            raise ValueError('native model/transport prefix mismatch')
    old,new=rows[0]['completion'],rows[1]['completion']
    if (type(old) is not dict or type(new) is not dict or old.get('reason')!='cancelled' or old.get('cancel_prior_reason')!='unknown'
            or old.get('natural_stop') is not False or old.get('stream_exhausted') is not False
            or new.get('reason')!='natural_stop' or new.get('natural_stop') is not True
            or new.get('stream_exhausted') is not True
            or type(new.get('stop_index')) is not int or type(new.get('effective_max_len')) is not int
            or not 0<=new['stop_index']<new['effective_max_len']-1):
        raise ValueError('old native generation or next natural EOS not independently proven')
    controls=control.get('receipts',[])
    if type(controls) is not list or any(type(r) is not dict for r in controls):
        raise ValueError('native cancel control objects required')
    matches=[r for r in controls if r.get('epoch')==1 and r.get('status')=='requested']
    if len(matches)!=1:raise ValueError('bound cancellation control receipt required')
    accepted=matches[0].get('ack',{}).get('cancel_monotonic_ns')
    closed=old.get('ended_monotonic_ns'); started=new.get('started_monotonic_ns')
    if (any(type(v) is not int for v in (accepted,closed,started))
            or not 0<accepted<=closed<=started):
        raise ValueError('L4 cancel/stop/next-request clock mismatch')
    return dict(native_cancel_exercised=True,next_native_natural_eos_verified=True,
        same_gate_lease_reused=True,old_full_utterance_completed=False,
        clock='l4_native_monotonic_ns',cancel_to_native_stop_seconds=(closed-accepted)/1e9,
        native_stop_to_next_request_seconds=(started-closed)/1e9,
        historical_same_condition_speedup_verified=False)


def collect_native_worker_receipts(api,job,directory,deadline,*,visual_intents=False,paired_cuts=False,native_tts_cancel=False,ordered_reference_sha256=None,long_session=False,stable_selfie=False,job_namespace='YUGOROU'):
    """Wait for flushed bounded post-join metadata before cancelling the Job."""
    if job_namespace not in ('YUGOROU', '2nd2nd'):raise ValueError('explicit known Job namespace required')
    if type(stable_selfie) is not bool or (stable_selfie and visual_intents is not True):
        raise ValueError('stable ancestry collection requires explicit visual selection')
    if ordered_reference_sha256 is not None:
        if not visual_intents or type(ordered_reference_sha256) is not str or not re.fullmatch('[a-f0-9]{64}',ordered_reference_sha256):
            raise ValueError('ordered metadata requires expected visual READY identity')
    while time.monotonic()<deadline:
        count=0;visual_windows={};visual_received={};visual_session=None;cut_windows={};measurements=[];cancel_evidence=None
        candidate_binding=None;ancestry_anchor=None
        for line in api.fetch_job_logs(job_id=job.id,namespace=job_namespace):
            count+=1
            if count>(100000 if long_session else 10000) or time.monotonic()>deadline:raise TimeoutError('bounded native receipt collection')
            if len(line)>65536:raise ValueError('native log line bound')
            try:row=json.loads(line)
            except (ValueError,TypeError):continue
            if paired_cuts and type(row) is dict and row.get('type') in (
                    'h3_stage_timing','h3_window_substages','h3_audio_coverage','h3_producer_timing'):
                if len(measurements)>=512:raise ValueError('bounded review metadata')
                measurements.append(row)
            if paired_cuts and type(row) is dict and row.get('type')=='h3_paired_cut':
                index=row.get('sequence')
                if (type(index) is not int or not 0<=index<(2542 if long_session else 144) or index in cut_windows
                        or row.get('visual_achievement')!='unverified'):
                    raise ValueError('strict once-only cut evidence')
                from research.h3_personal.paired_av import PresentationCut
                cut=PresentationCut(**row['cut'])
                if (cut.frame!=row['previous_native_committed']
                        or row['frame_start']!=row['previous_native_emitted']
                        or not row['frame_start']<cut.frame<row['frame_stop']):
                    raise ValueError('cut frontier evidence mismatch')
                cut_windows[index]=row
            if (not stable_selfie and type(row) is dict and row.get('type')=='h3_visual_conditioning'
                    and 'visual_ancestry_selection' in row):
                raise ValueError('unexpected stable ancestry selection')
            if visual_intents and type(row) is dict and row.get('type') in ('h3_visual_conditioning','h3_visual_intent_received'): 
                session=row.get('session')
                if type(session) is not str or not re.fullmatch('[a-f0-9]{32}',session):raise ValueError('visual session binding')
                if visual_session is not None and session!=visual_session:raise ValueError('mixed visual sessions')
                visual_session=session
                target=row.get('target_epoch')
                if type(target) is not int or not 0<=target<=(1000 if long_session else 16) or row.get('visual_achievement')!='unverified':
                    raise ValueError('visual receipt cannot claim achieved action')
                if row['type']=='h3_visual_conditioning':
                    index=row.get('sequence')
                    if (type(index) is not int or not 0<=index<(2542 if long_session else 144) or index in visual_windows
                            or row.get('stage')!='native_conditioning_applied'
                            or type(row.get('prompt_sha256')) is not str
                            or not re.fullmatch('[a-f0-9]{64}',row['prompt_sha256'])):
                        raise ValueError('bounded once-only visual native receipt required')
                    if ordered_reference_sha256 is not None:
                        from research.h3_personal.ordered_metadata import validate_metadata
                        metadata=validate_metadata(row.get('ordered_metadata'),
                            expected_identity=ordered_reference_sha256,visual_binding=row,sequence=index)
                        binding=metadata['ordered_references']['candidate_binding_sha256']
                        if candidate_binding is not None and candidate_binding!=binding:
                            raise ValueError('mixed ordered candidate/base bindings')
                        candidate_binding=binding
                        row['ordered_metadata']=metadata
                    elif 'ordered_metadata' in row:
                        raise ValueError('unexpected ordered metadata without READY identity')
                    if stable_selfie:
                        from research.h3_personal.visual_ancestry_policy import validate_selection
                        selection=validate_selection(row.get('visual_ancestry_selection'),index,ancestry_anchor)
                        if index==1:ancestry_anchor=selection['anchor_sha256']
                        row['visual_ancestry_selection']=selection
                    visual_windows[index]=row
                else:
                    if not 1<=target<=(1000 if long_session else 2) or target in visual_received:raise ValueError('bounded ordered visual turns required')
                    visual_received[target]=row
            if native_tts_cancel and type(row) is dict and row.get('type')=='h3_native_tts_cancel_receipts':
                rows=row.get('receipts')
                if (cancel_evidence is not None or type(rows) is not list or not 1<=len(rows)<=2
                        or any(type(r) is not dict or type(r.get('epoch')) is not int
                               or r['epoch'] not in (1,2) or r.get('status') not in
                               ('requested','bounded_drain','already_terminal','terminal_timeout') for r in rows)
                        or len({r['epoch'] for r in rows})!=len(rows)):
                    raise ValueError('bounded once-only native cancel evidence required')
                cancel_evidence=row
            if type(row) is dict and row.get('type')=='h3_native_tts_receipts':
                receipts=row.get('receipts')
                if type(receipts) is not list or not 1<=len(receipts)<=(1000 if long_session else 2):raise ValueError('bounded native completions required')
                if any(type(r) is not dict for r in receipts):raise ValueError('native receipt objects')
                if [r.get('epoch') for r in receipts]!=list(range(1,len(receipts)+1)):raise ValueError('native receipt epochs')
                durable_json(Path(directory)/'native-worker-evidence.json',row)
                if native_tts_cancel:
                    if cancel_evidence is None:raise ValueError('native cancel receipt missing before completion')
                    durable_json(Path(directory)/'native-cancel-worker-evidence.json',cancel_evidence)
                if paired_cuts:
                    durable_json(Path(directory)/'paired-cut-worker-evidence.json',dict(cuts=list(cut_windows.values()),
                        native_cut_exercised=bool(cut_windows),visual_achievement='unverified'))
                    durable_json(Path(directory)/'review-worker-timing.json',dict(events=measurements,
                        producer_timing_may_follow_tts_receipts=True))
                if visual_intents:
                    if ((set(visual_received)!=set(range(1,len(receipts)+1)) if long_session else set(visual_received)!={1,2}) or set(visual_windows)!=set(range(len(visual_windows)))
                            or (not stable_selfie and not {1,2}<={r['target_epoch'] for r in visual_windows.values()})
                            or (stable_selfie and (
                                not visual_windows or max(r['target_epoch'] for r in visual_windows.values()) != len(receipts)
                                or any(type(r.get('intent_epoch')) is not int or r['intent_epoch'] != 0
                                       or r.get('instruction_sha256', 'missing') is not None
                                       for r in visual_received.values())
                                or any(type(r.get('intent_epoch')) is not int or r['intent_epoch'] != 0
                                       for r in visual_windows.values())))):
                        raise ValueError('both visual turn applications must be independently recorded')
                    if stable_selfie and (3 not in visual_windows
                            or visual_windows[3]['visual_ancestry_selection']['kind']!='anchor'):
                        raise ValueError('stable selfie must exercise actual sequence3 conversation anchor')
                    durable_json(Path(directory)/'visual-worker-evidence.json',dict(
                        received=list(visual_received.values()),windows=list(visual_windows.values()),
                        native_application_verified=True,visual_achievement='unverified'))
                return row
        time.sleep(.2)
    raise TimeoutError('native worker receipt unavailable before cleanup')


def validate_ready(row, session, *, five_minute_test=False, ten_minute_test=False, tts_mode='complete', overlap_send=False, http_keepalive=False, continuous_av=False, native_tts=None, media_transport='http-paired-v1', turn_interruptions=False, visual_intents=False, paired_cuts=False, native_tts_cancel=False, native_cancel_trigger=None, ordered_reference_sha256=None, attended_visual=False, long_session=False, stable_selfie=False):
    from research.hf_h3_direct_conversation import endurance_limits, CONVERSATION_ANCHOR_POLICY
    windows, frames, timeout = endurance_limits(five_minute_test, ten_minute_test, long_session)
    from research.hf_h3_direct_conversation import validate_tts_mode
    validate_tts_mode(tts_mode)
    if type(overlap_send) is not bool:
        raise ValueError('explicit overlap_send bool required')
    if type(http_keepalive) is not bool:
        raise ValueError('explicit http_keepalive bool required')
    if type(continuous_av) is not bool:
        raise ValueError('explicit continuous_av bool required')
    keys = {'type','status','session','profile_sha256','epoch','max_segments','max_total_frames','port','expires_at'}
    if long_session:
        keys.add('long_session')
        if not attended_visual or row.get('long_session') is not True:
            raise ValueError('three-hour READY requires immutable attended selection')
    if ordered_reference_sha256 is not None:
        keys.add('ordered_reference_sha256')
        if (type(ordered_reference_sha256) is not str or not re.fullmatch('[a-f0-9]{64}',ordered_reference_sha256)
                or not visual_intents or type(row) is not dict
                or row.get('ordered_reference_sha256')!=ordered_reference_sha256):
            raise ValueError('ordered reference READY mismatch')
    if media_transport=='webrtc-rtp-v1':
        keys.update(('media_transport','rtc_boot','ice_profile'))
        if (type(row) is not dict or row.get('media_transport')!=media_transport
                or row.get('ice_profile')!='hf-cloudflare-turn-tls-v1'
                or type(row.get('rtc_boot')) is not str
                or not re.fullmatch('[a-f0-9]{32}',row['rtc_boot'])
                or continuous_av):raise ValueError('RTC READY binding mismatch')
    elif media_transport!='http-paired-v1':raise ValueError('invalid media transport')
    if type(turn_interruptions) is not bool:raise ValueError('explicit interruption READY required')
    if turn_interruptions:
        keys.add('turn_interruptions')
        if row.get('turn_interruptions') is not True or media_transport != 'http-paired-v1':
            raise ValueError('interruptible HTTPS READY mismatch')
    if native_cancel_trigger not in (None,'native-preroll-v1'):
        raise ValueError('explicit native cancel trigger required')
    if native_cancel_trigger is not None:
        keys.add('native_cancel_trigger')
        if not native_tts_cancel or type(row) is not dict or row.get('native_cancel_trigger')!=native_cancel_trigger:
            raise ValueError('native trigger READY mismatch')
    if type(native_tts_cancel) is not bool:raise ValueError('explicit native cancel READY required')
    if native_tts_cancel:
        keys.add('native_tts_cancel')
        if not turn_interruptions or type(row) is not dict or row.get('native_tts_cancel') is not True:
            raise ValueError('native cancel READY binding mismatch')
    if (type(native_tts) is dict and native_tts.get('schema')=='h3-native-voxcpm-cancel-v4') != native_tts_cancel:
        raise ValueError('native cancel READY/config mismatch')
    if type(paired_cuts) is not bool:raise ValueError('explicit paired cut READY required')
    if paired_cuts:
        keys.add('paired_cuts')
        if not visual_intents or row.get('paired_cuts') is not True:
            raise ValueError('paired cut READY binding mismatch')
    if type(visual_intents) is not bool:raise ValueError('explicit visual READY selection required')
    if type(attended_visual) is not bool:raise ValueError('explicit attended visual READY selection required')
    if type(stable_selfie) is not bool:
        raise ValueError('explicit stable selfie READY selection required')
    if stable_selfie:
        keys.update(('stable_selfie', 'anchor_policy'))
        if (not visual_intents or turn_interruptions or paired_cuts
                or type(row) is not dict or row.get('stable_selfie') is not True
                or row.get('anchor_policy') != CONVERSATION_ANCHOR_POLICY):
            raise ValueError('stable selfie READY binding mismatch')
    if attended_visual and (not visual_intents or turn_interruptions):
        raise ValueError('attended visual READY forbids headless interruption binding')
    if visual_intents:
        keys.add('visual_intents')
        if not (turn_interruptions or attended_visual or stable_selfie) or row.get('visual_intents') is not True:
            raise ValueError('visual READY binding mismatch')
        if attended_visual:
            keys.add('attended_visual')
            if type(row) is not dict or row.get('attended_visual') is not True:
                raise ValueError('attended visual READY binding mismatch')
    if continuous_av:
        keys.add('continuous_av')
        if type(row) is not dict or row.get('continuous_av') is not True:
            raise ValueError('direct READY continuous_av mismatch')
    if http_keepalive:
        keys.add('http_keepalive')
        if type(row) is not dict or row.get('http_keepalive') is not True:
            raise ValueError('direct READY http_keepalive mismatch')
    if overlap_send:
        keys.add('overlap_send')
        if type(row) is not dict or row.get('overlap_send') is not True:
            raise ValueError('direct READY overlap_send mismatch')
    if tts_mode in ('sentence_incremental','native_voxcpm'):
        keys.add('tts_mode')
    if tts_mode=='native_voxcpm':
        from research.h3_personal.native_tts_backend import config_sha256
        keys.add('native_tts_config_sha256')
        if native_tts is None or type(row) is not dict or row.get('native_tts_config_sha256')!=config_sha256(native_tts):
            raise ValueError('native TTS READY selection mismatch')
        if type(row.get('expires_at')) not in (int,float) or row['expires_at']>native_tts['expires_at']:
            raise ValueError('direct READY outlives native TTS lease')
    elif native_tts is not None:
        raise ValueError('native config forbidden for legacy READY')
    if type(row) is dict and row.get('tts_mode', 'complete') != tts_mode:
        raise ValueError('direct READY TTS mode mismatch')
    if (type(row) is not dict or set(row) != keys or row['type'] != 'h3_direct_ready'
            or row['session'] != session or row['status'] != 'ready'
            or type(row['profile_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', row['profile_sha256'])):
        raise ValueError('invalid direct READY')
    for key, expected in (('epoch',1),('max_segments',windows),('max_total_frames',frames),('port',7860)):
        if type(row[key]) is not int or row[key] != expected:
            raise ValueError('invalid direct READY limits')
    expiry = row['expires_at']
    if type(expiry) not in (int,float) or not math.isfinite(expiry) or not time.time() < expiry <= time.time()+timeout:
        raise ValueError('invalid boot-relative READY expiry')
    return row


def await_ready(api, job, session, deadline, *, five_minute_test=False, ten_minute_test=False, tts_mode='complete', overlap_send=False, http_keepalive=False, continuous_av=False, native_tts=None, media_transport='http-paired-v1', turn_interruptions=False, visual_intents=False, paired_cuts=False, native_tts_cancel=False, native_cancel_trigger=None, ordered_reference_sha256=None, attended_visual=False, long_session=False, stable_selfie=False, job_namespace='YUGOROU'):
    if job_namespace not in ('YUGOROU', '2nd2nd'):raise ValueError('explicit known Job namespace required')
    from research.hf_h3_direct_conversation import validate_tts_mode
    validate_tts_mode(tts_mode)
    import httpx
    failures = 0
    while time.monotonic() < deadline:
        fetch_error = None
        try:
            # The iterator can fail after yielding logs. Retry only this read,
            # never submission, validation, HTTP status errors or inspection.
            for log in api.fetch_job_logs(job_id=job.id, follow=False,
                                          **({'namespace':job_namespace} if job_namespace != 'YUGOROU' else {})):
                text = log if isinstance(log,str) else getattr(log,'data','')
                if not isinstance(text,str) or len(text) > 4096:
                    continue
                try:
                    row = json.loads(text)
                except ValueError:
                    continue
                if type(row) is dict and row.get('type') == 'h3_direct_ready':
                    return validate_ready(row, session, five_minute_test=five_minute_test, ten_minute_test=ten_minute_test, tts_mode=tts_mode, overlap_send=overlap_send, http_keepalive=http_keepalive, continuous_av=continuous_av, native_tts=native_tts, media_transport=media_transport, turn_interruptions=turn_interruptions, visual_intents=visual_intents, paired_cuts=paired_cuts, native_tts_cancel=native_tts_cancel, native_cancel_trigger=native_cancel_trigger, ordered_reference_sha256=ordered_reference_sha256, attended_visual=attended_visual, long_session=long_session, stable_selfie=stable_selfie)
        except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as error:
            failures += 1
            fetch_error = error
        else:
            failures = 0
        if api.inspect_job(job_id=job.id, **({'namespace':job_namespace} if job_namespace != 'YUGOROU' else {})).status.stage in TERMINAL:
            raise RuntimeError('Job ended before READY')
        if failures >= 3:
            raise fetch_error
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2, remaining))
    raise TimeoutError('finite direct READY wait')


def attended_visual_capability(*, visual_intents, attended_visual, headless_interrupt=False,
        capture_comet=False, asr_parent=None, zoom_input=None, media_transport='http-paired-v1',
        asr_rtc_factory=None, allow_obs=False, obs_port=None, direct_audio_binary=None,
        start_gate=None, idle_anchor=False, local_sink_factory=None):
    """Validate the explicit attended capability selection for the verified device route.

    Permits exactly the already-verified E2E I/O: real Comet ASR input plus the split
    output (video to OBS, original PCM to the audio device). Validation and projection
    only: it starts no process, opens no socket and requests no provider.
    """
    if type(attended_visual) is not bool or type(visual_intents) is not bool:
        raise ValueError('explicit attended visual capability bools required')
    if not attended_visual:
        return {}
    if not visual_intents:
        raise ValueError('attended visual capability requires explicit visual intents')
    if headless_interrupt:
        raise ValueError('attended visual capability is separate from headless interruption')
    if media_transport != 'http-paired-v1' or asr_rtc_factory is not None:
        raise ValueError('attended visual capability permits only the approved HTTPS ASR input')
    if zoom_input is not None:
        raise ValueError('attended visual capability permits only the existing Comet ASR input')
    if capture_comet is not True or asr_parent is None:
        raise ValueError('attended visual capability requires the existing attended ASR input')
    if allow_obs is not True or obs_port is None or direct_audio_binary is None:
        raise ValueError('attended visual capability requires the verified OBS video plus PCM audio output')
    if start_gate is not None or local_sink_factory is not None:
        raise ValueError('attended visual capability uses the split OBS route only')
    if idle_anchor:
        raise ValueError('attended visual capability forbids periodic anchor reset')
    return dict(attended_visual=True, visual_intents=True)


def attended_session_fields(attended_fields):
    """Project the validated attended capability into immutable session fields.

    Pure data only: no process, socket, provider or device access. The same field
    must appear in the immutable direct session, the READY request and the
    relayed worker options; a missing field on any boundary fails closed there.
    """
    if type(attended_fields) is not dict or attended_fields != dict(
            attended_visual=True, visual_intents=True):
        raise ValueError('exact attended visual capability required')
    return dict(attended_visual=True)


def attended_execution_available():
    """Whether the existing relay runner accepts the attended visual route.

    Read purely from the existing callable signature; nothing is started. Until
    the relay boundary exists this stays false so a half-wired attended run can
    never reach a GPU, provider or device.
    """
    import inspect as _inspect
    from research.run_h3_relay_conversation import run_conversation
    return 'attended_visual' in _inspect.signature(run_conversation).parameters


def run(directory, turns=None, *, capture_comet=False, asr_parent=None, allow_obs=False, obs_port=None,
        idle_anchor=False, conversation_anchor=False, direct_audio_binary=None, audio_delay_seconds=0.2,
        today_pause_on_underflow=False, long_session=False,
        five_minute_test=False, ten_minute_test=False, tts_mode='complete',
        local_sink_factory=None, prefetch_receive=False, start_gate=None, overlap_send=False,
        http_keepalive=False, continuous_av=False, zoom_input=None, asr_capture_seconds=45, native_tts=None,
        attended_visual=False, stable_selfie=False,
        media_transport='http-paired-v1', on_browser=None, asr_rtc_factory=None, h3_adoption=None,
        headless_interrupt=False, visual_intents=False, review_av=False, native_tts_cancel=False,
        native_cancel_trigger=None, ordered_references=None, confirm_private_reference_upload=False,
        h3_backup_jobs=False, _compute_profile=None):
    if type(stable_selfie) is not bool or (stable_selfie and (
            visual_intents is not True or conversation_anchor is not True
            or idle_anchor or headless_interrupt or native_tts_cancel
            or native_cancel_trigger is not None or media_transport != 'http-paired-v1'
            or (review_av and attended_visual) or tts_mode != 'native_voxcpm')):
        raise ValueError('stable selfie requires native visual transport and conversation anchor without movement/interruption')
    if stable_selfie and not attended_visual and (
            capture_comet or asr_parent is not None or zoom_input is not None
            or allow_obs or obs_port is not None or direct_audio_binary is not None
            or local_sink_factory is not None or on_browser is not None or asr_rtc_factory is not None
            or start_gate is not None or long_session or today_pause_on_underflow
            or five_minute_test or ten_minute_test
            or type(turns) not in (list, tuple) or len(turns) != 2
            or any(type(t) is not str or not t.strip() or len(t.encode()) > 2000 for t in turns)):
        raise ValueError('device-free stable selfie requires exactly two typed turns and no device/capture adapters')
    if visual_intents and conversation_anchor and not stable_selfie:
        raise ValueError('visual conversation anchor requires explicit stable selfie mode')
    if type(h3_backup_jobs) is not bool:
        raise ValueError('explicit backup Job billing boolean required')
    if h3_backup_jobs:
        # Initial split scope only: the existing device-free stable native HTTPS
        # route. Asset/Endpoint/Pi credentials stay on the main account.
        if not stable_selfie or attended_visual or headless_interrupt or media_transport != 'http-paired-v1':
            raise ValueError('backup Job billing supports device-free stable native HTTPS only')
        if h3_adoption is not None or long_session or asr_parent is not None:
            raise ValueError('backup Job billing rejects adoption/long-session/ASR')
    if _compute_profile is not None and not h3_backup_jobs:
        raise ValueError('compute profile requires explicit backup Job billing')
    ordered_metadata={}
    if ordered_references is not None or confirm_private_reference_upload:
        ordered_references, ordered_metadata=ordered_selection(ordered_references,
            confirm_private_reference_upload=confirm_private_reference_upload,require_confirmation=True)
    if ordered_references is not None and visual_intents is not True:
        raise ValueError('ordered references require visual intents')
    if media_transport not in ('http-paired-v1','webrtc-rtp-v1'):raise ValueError('explicit media transport required')
    rtc=media_transport=='webrtc-rtp-v1'
    if type(review_av) is not bool or (review_av and not visual_intents):
        raise ValueError('review retention requires explicit device-free visual route')
    if review_av and attended_visual:
        # Same fail-closed rule as the CLI: paired review cuts need the headless
        # interruption gate, so the attended route must not accept and drop them.
        raise ValueError('review retention is unavailable on the attended route; '
                         'paired review cuts require the headless interruption gate')
    attended_fields = attended_visual_capability(visual_intents=visual_intents,
        attended_visual=attended_visual, headless_interrupt=headless_interrupt,
        capture_comet=capture_comet, asr_parent=asr_parent, zoom_input=zoom_input,
        media_transport=media_transport, asr_rtc_factory=asr_rtc_factory, allow_obs=allow_obs,
        obs_port=obs_port, direct_audio_binary=direct_audio_binary, start_gate=start_gate,
        idle_anchor=idle_anchor, local_sink_factory=local_sink_factory)
    if type(visual_intents) is not bool or (visual_intents and
            (not (headless_interrupt or attended_visual or stable_selfie) or idle_anchor
             or (conversation_anchor and not (attended_visual or stable_selfie)))):
        raise ValueError('visual intent requires explicit headless or attended route with latest-tail continuation')
    if native_cancel_trigger not in (None,'native-preroll-v1') or (native_cancel_trigger is not None and not native_tts_cancel):
        raise ValueError('native preroll trigger requires explicit native cancel')
    if type(native_tts_cancel) is not bool or (native_tts_cancel and not headless_interrupt):
        raise ValueError('native cancel requires explicit headless interruption mode')
    if type(headless_interrupt) is not bool:raise ValueError('explicit headless interruption mode required')
    if headless_interrupt and (rtc or capture_comet or zoom_input is not None or allow_obs
            or obs_port is not None or direct_audio_binary is not None or local_sink_factory is not None
            or tts_mode != 'native_voxcpm' or not continuous_av
            or five_minute_test or ten_minute_test or start_gate is not None):
        raise ValueError('headless interruption requires device-free native HTTPS continuous route')
    if rtc and (not callable(on_browser) or allow_obs or obs_port is not None or direct_audio_binary is not None
                or local_sink_factory is not None or continuous_av or prefetch_receive or ten_minute_test):
        raise ValueError('RTC requires private browser callback and no legacy media sink')
    if not rtc and on_browser is not None:raise ValueError('browser callback is RTC only')
    if type(asr_capture_seconds) is not int or not 8 <= asr_capture_seconds <= 45:
        raise ValueError('finite ASR capture8..45 seconds required')
    if zoom_input is not None:
        from research.h3_personal.zoom_turn_input import ZoomTurnInput
        if capture_comet or not isinstance(zoom_input, ZoomTurnInput):
            raise ValueError('explicit mutually exclusive Zoom pipeline required')
    capture_asr = capture_comet or zoom_input is not None
    if asr_rtc_factory is not None and (not capture_asr or getattr(asr_rtc_factory,'media_transport',None)!='webrtc-exact-v1'
            or not callable(getattr(asr_rtc_factory,'open_turn',None))):raise ValueError('explicit capture and exact RTC ASR factory required')
    if type(overlap_send) is not bool:
        raise ValueError('explicit overlap_send bool required')
    if type(http_keepalive) is not bool:
        raise ValueError('explicit http_keepalive bool required')
    if type(continuous_av) is not bool:
        raise ValueError('explicit continuous_av bool required')
    if type(prefetch_receive) is not bool:
        raise TypeError('explicit prefetch_receive bool required')
    if start_gate is not None and (not isinstance(start_gate, (str, Path))
                                   or not str(start_gate).strip()):
        raise TypeError('explicit start gate path required')
    if local_sink_factory is not None:
        if not callable(local_sink_factory):
            raise TypeError('explicit local sink factory required')
        if (allow_obs or obs_port is not None or direct_audio_binary is not None
                or (capture_asr and not attended_visual)):
            raise ValueError('local typed playback cannot also use OBS, split audio or capture')
    if attended_fields and not attended_execution_available():
        # Validation/propagation seam only; never a silent partial GPU launch.
        raise ValueError('attended visual execution is not yet wired: the relay runner has no '
                         'attended route; no process, network or GPU started')
    from research.hf_h3_direct_conversation import validate_tts_mode
    validate_tts_mode(tts_mode)
    if type(long_session) is not bool or (long_session and
            (not attended_visual or not today_pause_on_underflow or five_minute_test or ten_minute_test)):
        raise ValueError('three-hour mode requires explicit attended pause route')
    if type(today_pause_on_underflow) is not bool or (today_pause_on_underflow and
            (not attended_visual or direct_audio_binary is None)):
        raise ValueError('today pause workaround requires attended split output')
    if tts_mode=='native_voxcpm':
        from research.h3_personal.native_tts_backend import validate_native_config, native_job_secrets
        native_tts=validate_native_config(native_tts)
        if (native_tts['schema']=='h3-native-voxcpm-cancel-v4') != native_tts_cancel:
            raise ValueError('native cancel selection/config mismatch')
        if headless_interrupt and native_tts['schema'] != (
                'h3-native-voxcpm-cancel-v4' if native_tts_cancel else 'h3-native-voxcpm-backend-v1'):
            raise ValueError('headless interruption requires explicit HTTPS Vox backend')
        if native_tts['schema']=='h3-native-voxcpm-network-v2' and capture_asr and asr_rtc_factory is None:
            raise ValueError('hybrid conversation forbids legacy WAN ASR audio')
        native_job_secrets({'live_direct_session':{'tts_mode':tts_mode,'native_tts':native_tts}})
    elif native_tts is not None:
        raise ValueError('native config requires explicit native mode')
    from research.hf_h3_direct_conversation import (ANCHOR_ENV, ANCHOR_POLICY,
        CONVERSATION_ANCHOR_ENV, CONVERSATION_ANCHOR_POLICY)
    from research.hf_h3_direct_conversation import endurance_limits
    windows, frames, timeout = endurance_limits(five_minute_test, ten_minute_test, long_session)
    if type(conversation_anchor) is not bool or (conversation_anchor and idle_anchor):
        raise ValueError('explicit mutually exclusive conversation anchor required')
    if type(idle_anchor) is not bool:
        raise ValueError('idle_anchor must be an explicit boolean')
    if ANCHOR_ENV in os.environ or CONVERSATION_ANCHOR_ENV in os.environ:
        raise ValueError('preexisting idle anchor environment forbidden')
    from research.run_h3_relay_conversation import direct_audio_pin
    audio_pin = None
    if direct_audio_binary is not None:
        if allow_obs is not True or obs_port is None:
            raise PermissionError('split route requires explicit OBS approval and port')
        audio_pin = direct_audio_pin(direct_audio_binary, audio_delay_seconds)
    elif audio_delay_seconds != 0.2:
        raise ValueError('audio delay requires direct audio binary')
    canonical()
    if obs_port is not None and not allow_obs:
        raise PermissionError('OBS approval required')
    if bool(capture_asr) != (asr_parent is not None):
        raise ValueError('explicit capture source and asr_parent are both required for ASR')
    if capture_asr:
        if turns is not None:
            raise ValueError('typed turns are forbidden in ASR mode')
        from research.h3_personal.resident_endpoint import ResidentEndpoint, PRIVATE as ASR_PRIVATE
        from research.h3_personal.direct_asr_input import DirectASRInput
        if not Path(directory).resolve().is_relative_to(ASR_PRIVATE.resolve()):
            raise ValueError('ASR directory must reside under endpoint PRIVATE')
    elif (type(turns) not in (list,tuple) or not 1 <= len(turns) <= 2
            or any(type(t) is not str or not t.strip() or len(t.encode()) > 2000 for t in turns)):
        raise ValueError('one or two fresh bounded typed turns required')
    if headless_interrupt and len(turns) != 2:
        raise ValueError('device-free interruption needs exactly two typed turns')
    # Resolve pending direct API/endpoint/consumer contracts BEFORE any GPU.
    from research.h3_personal.direct_job_api import submit_job
    from research.h3_personal.job_av_endpoint import attest_job
    from research.h3_personal.live_relay_client import SocketAdapter
    from research.h3_personal.paired_av import LiveAVBinding
    from research.run_h3_relay_conversation import PiDirector, run_conversation
    from research.provision_modal_identity_secret import cached_token
    from huggingface_hub import HfApi, set_client_factory
    import httpx
    if not callable(submit_job) or not callable(attest_job):
        raise RuntimeError('direct Jobs API contracts unavailable')
    if not all(shutil.which(p) for p in ('node','ffmpeg','ffprobe')):
        raise RuntimeError('existing node/ffmpeg/ffprobe required; no installs')
    if allow_obs:
        from research.h3_personal.obs_av_sink import output_arguments
        output_arguments(obs_port)
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT) or directory.exists():
        raise ValueError('fresh private directory outside repository required')
    directory.mkdir(mode=0o700, parents=True)
    if audio_pin is not None:
        durable_json(directory/'av-output-route.json', dict(
            schema='h3-av-output-route-v1', route='split',
            binary_sha256=audio_pin['binary_sha256'],
            audio_delay_seconds=audio_pin['audio_delay_seconds'],
            obs_port=obs_port, obs_audio_disabled=True, remote_verified=False))
    # Prove receiver output is writable before reserving compute.
    (directory/'receiver').mkdir(mode=0o700)
    set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(120 if headless_interrupt or attended_visual else 20,connect=5), follow_redirects=True,
        event_hooks={'response':[sdk_redirect_guard]}, trust_env=False))
    token = cached_token(); api = HfApi(endpoint='https://huggingface.co', token=token)
    # Compute profile owns ONLY Job create/inspect/log/cancel/attest/gateway and
    # the consumer token. Asset staging, payload secrets, Pi and Endpoints stay
    # on the main account above; there is no fallback between the two.
    compute_profile = _compute_profile
    if h3_backup_jobs:
        if compute_profile is None:
            from research.h3_personal.job_compute_profile import load_compute_profile
            compute_profile = load_compute_profile(api, token, backup=True)
        compute_profile = _backup_compute_binding(compute_profile, api, token)
        # Both namespaces must be idle before any Vox lease or Job POST.
        compute_profile.assert_idle()
        from research.h3_personal.job_network import assert_idle as _assert_main_idle
        _assert_main_idle(api, {'live_direct_session': {}}, token)
    job_api = compute_profile.api if compute_profile is not None else api
    job_token = compute_profile.token if compute_profile is not None else token
    job_namespace = compute_profile.namespace if compute_profile is not None else 'YUGOROU'
    session = uuid.uuid4().hex; pk = secrets.token_hex(32); ck = secrets.token_hex(32)
    job = consumer = director = endpoint = source = native_observer = None
    terminated = False
    def terminate(signum, frame):
        nonlocal terminated
        if not terminated:
            terminated = True
            raise InterruptedError('direct conversation received SIGTERM')
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        if capture_asr:
            endpoint = ResidentEndpoint.open(directory/'asr', Path(asr_parent), accounting='operator-managed',
                **({'coowned_vox_directory': directory.parent/'vox'} if attended_visual and tts_mode=='native_voxcpm' else {}),
                **({'session_seconds':10800} if long_session else {}),
                **({'rtc_factory':asr_rtc_factory} if asr_rtc_factory is not None else {}))
            source = (DirectASRInput(endpoint, capture_seconds=asr_capture_seconds,
                **({'max_attempts':1000,'max_turns':1000} if long_session else {})) if zoom_input is None else
                      DirectASRInput(endpoint, capture_seconds=asr_capture_seconds, input_pipeline=zoom_input))
        director = PiDirector(token, deadline=time.monotonic()+(10800 if long_session else 1750), max_turns=1000 if long_session else 2,
            **({'long_session':True} if long_session else {}),
            **({'stable_selfie':True} if stable_selfie else {}),
            **({'visual_intents':True} if visual_intents else {}))
        row = dict(schema='h3-live-direct-session-v1', session=session, port=7860,
                   max_windows=windows, max_frames=frames, lease_seconds=timeout)
        row.update(ordered_metadata)
        if long_session:
            row['long_session'] = True
            row['session_seconds'] = 10800
        if rtc:row.update(media_transport=media_transport,ice_profile='hf-cloudflare-turn-tls-v1')
        if continuous_av:
            row['continuous_av'] = True
        if headless_interrupt:
            row['turn_interruptions'] = True
        if visual_intents:row['visual_intents'] = True
        if stable_selfie:
            row['stable_selfie'] = True
            row['turn_interruptions'] = False
        if attended_fields:
            row.update(attended_session_fields(attended_fields))
            row['turn_interruptions'] = False
        if review_av and not stable_selfie:row['paired_cuts'] = True
        if native_tts_cancel:row['native_tts_cancel'] = True
        if native_cancel_trigger is not None:row['native_cancel_trigger'] = native_cancel_trigger
        if http_keepalive:
            row['http_keepalive'] = True
        if overlap_send:
            row['overlap_send'] = True
        if tts_mode in ('sentence_incremental','native_voxcpm'):
            row['tts_mode'] = tts_mode
        if tts_mode=='native_voxcpm':
            row['native_tts']=native_tts
        if five_minute_test:
            row['five_minute_test'] = True
        if ten_minute_test:
            row['ten_minute_test'] = True
        if idle_anchor:
            row['anchor_policy'] = ANCHOR_POLICY
        elif conversation_anchor:
            row['anchor_policy'] = CONVERSATION_ANCHOR_POLICY
        if source is not None:
            row['expected_asr'] = source.expected_asr
            if asr_rtc_factory is not None:row['asr_media_transport']='webrtc-exact-v1'
        stage = prepare(directory, row, api,
            **({'ordered_references':ordered_references,'confirm_private_reference_upload':True} if ordered_references is not None else {}),
            **({'source_snapshot':h3_adoption.snapshot} if h3_adoption is not None else {}))
        job, image, command = submit(api, stage, token, pk, ck,
            **({'adoption':h3_adoption} if h3_adoption is not None else {}),
            **({'compute_profile':compute_profile} if compute_profile is not None else {}))
        ready = await_ready(job_api, job, session, time.monotonic()+min(1050, timeout),
                            **({'job_namespace':job_namespace} if job_namespace != 'YUGOROU' else {}),
                            **({'ordered_reference_sha256':ordered_metadata['ordered_reference_sha256']} if ordered_metadata else {}),
                            **({'media_transport':media_transport} if rtc else {}),
                            **({'continuous_av': True} if continuous_av else {}),
                            **({'turn_interruptions': True} if headless_interrupt else {}),
                            **({'visual_intents': True} if visual_intents else {}),
                            **({'attended_visual': True, 'turn_interruptions': False} if attended_fields else {}),
                            **({'long_session':True} if long_session else {}),
                            **({'stable_selfie':True} if stable_selfie else {}),
                            **({'paired_cuts': True} if review_av and not stable_selfie else {}),
                            **({'native_tts_cancel':True} if native_tts_cancel else {}),
                            **({'native_cancel_trigger':native_cancel_trigger} if native_cancel_trigger is not None else {}),
                            **({'http_keepalive': True} if http_keepalive else {}),
                            **({'overlap_send': True} if overlap_send else {}),
                            **({'tts_mode': tts_mode} if tts_mode in ('sentence_incremental','native_voxcpm') else {}),
                            **({'native_tts':native_tts} if tts_mode=='native_voxcpm' else {}),
                            **({'five_minute_test': True} if five_minute_test else
                               {'ten_minute_test': True} if ten_minute_test else {}))
        if start_gate is not None:
            # Local operator gate only: the caller reopens its capture source
            # immediately before AV begins, then releases the gate. No device,
            # provider or OBS API access happens here; the lease clock is not
            # extended and the wait stays inside the existing READY lease.
            gate = Path(start_gate)
            limit = time.monotonic() + min(300, ready['expires_at'] - time.time() - 60)
            print(json.dumps({'type': 'h3_start_gate_waiting',
                              'max_wait_seconds': max(0, min(300, ready['expires_at'] - time.time() - 60))}), flush=True)
            while not gate.exists():
                if time.monotonic() >= limit:
                    raise TimeoutError('start gate not released before lease margin')
                time.sleep(0.5)
        permit = attest_job(job_api, job_id=job.id, port=7860,
                            expected_image=image, expected_command=command,
                            **({'namespace':job_namespace} if job_namespace != 'YUGOROU' else {}))
        adapter_type=SocketAdapter
        if continuous_av:
            from research.h3_personal.continuous_av_client import ContinuousConsumerAdapter
            adapter_type=ContinuousConsumerAdapter
        # Ordered candidate refresh can exceed the legacy 10s AV-progress gap.
        # Opt in only on this device-free route; the absolute READY lease remains.
        candidate_progress = (dict(http_timeout=120, stream_progress_timeout=120)
                              if continuous_av and (headless_interrupt or (stable_selfie and not attended_visual)) and ordered_references is not None else {})
        consumer = adapter_type(permit.base_url, session=session, not_after=ready['expires_at'],
            role='consumer', role_key=ck, hf_token=job_token, endpoint_permit=permit,
            **candidate_progress)
        consumer.open_claim()
        binding = LiveAVBinding(session, ready['profile_sha256'], ready['epoch'], max_segments=windows, max_total_frames=frames,
            **({'long_session': True} if long_session else {}))
        if audio_pin is not None and direct_audio_pin(direct_audio_binary, audio_delay_seconds) != audio_pin:
            raise ValueError('direct audio binary changed')
        if not rtc and (five_minute_test or ten_minute_test) and ready['expires_at'] - time.time() < (620 if ten_minute_test else 320):
            raise TimeoutError('insufficient lease for first AV and endurance conversation')
        if rtc:
            from research.h3_personal.webrtc_conversation import run_conversation as rtc_conversation,collect_delivery
            result=rtc_conversation(consumer,binding,directory/'receiver',turns,director,
                ready=ready,token=token,key=ck,job_id=job.id,on_browser=on_browser,input_source=source,
                completion=lambda deadline:collect_delivery(api,job,session,deadline))
            if tts_mode=='native_voxcpm':
                collect_native_worker_receipts(job_api,job,directory,time.monotonic()+15,
                    **({'job_namespace':job_namespace} if job_namespace != 'YUGOROU' else {}))
            return result
        if native_cancel_trigger is not None:
            from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
            from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL
            native_observer=RemoteVoxCPM(native_tts['url'],boot=native_tts['boot'],
                session=native_tts['session'],expires_at=native_tts['expires_at'],
                token=os.environ['H3_NATIVE_TTS_HF_TOKEN'],key=os.environ['H3_NATIVE_TTS_KEY'],
                cancel_protocol=CANCEL_PROTOCOL)
        result = run_conversation(consumer, binding, directory/'receiver', turns, director,
            ready=ready, max_seconds=min(10800 if long_session else 300 if five_minute_test else 600,ready['expires_at']-time.time()), max_turns=1000 if long_session else 2,
            **({'long_session':True} if long_session else {}),
            **({'hold_listening': True, 'start_on_first_av': True} if five_minute_test or ten_minute_test else {}),
            **({'ten_minute_test': True} if ten_minute_test else {}),
            allow_obs=allow_obs, obs_port=obs_port, prefetch_receive=prefetch_receive,
            **({'today_pause_on_underflow': True} if today_pause_on_underflow else {}),
            **({'headless_interrupt': True} if headless_interrupt else {}),
            **({'visual_intents': True} if visual_intents else {}),
            **({'attended_visual': True} if attended_fields else {}),
            **({'stable_selfie': True} if stable_selfie else {}),
            **({'review_av': True} if review_av else {}),
            **({'native_cancel_trigger':native_cancel_trigger,'native_progress':native_observer.progress,
                'native_preroll_samples':native_tts['preroll']['reserve_samples']}
               if native_cancel_trigger is not None else {}),
            **({'sink_factory': local_sink_factory} if local_sink_factory is not None else {}),
            **({'input_source': source} if source is not None else {}),
            **({'direct_audio_binary': audio_pin['audio_binary'],
                'audio_delay_seconds': audio_pin['audio_delay_seconds']} if audio_pin is not None else {}))
        if tts_mode=='native_voxcpm':
            collect_native_worker_receipts(job_api,job,directory,
                time.monotonic()+min(15,max(0,ready['expires_at']-time.time())),
                **({'job_namespace':job_namespace} if job_namespace != 'YUGOROU' else {}),
                **({'long_session':True} if long_session else {}),
                **({'visual_intents':True} if visual_intents else {}),
                **({'stable_selfie':True} if stable_selfie else {}),
                **({'ordered_reference_sha256':ready['ordered_reference_sha256']} if ordered_metadata else {}),
                **({'paired_cuts':True} if review_av and not stable_selfie else {}),
                **({'native_tts_cancel':True} if native_tts_cancel else {}))
            if visual_intents:
                result['visual_conditioning']=json.loads((directory/'visual-worker-evidence.json').read_text())
        if headless_interrupt:
            evidence = json.loads((directory/'native-worker-evidence.json').read_text())['receipts']
            source = next(r for r in evidence if r['epoch'] == 2)
            actual = result['sink_timing']['decoded_source_output']['2']
            if (actual['samples'] != source['converted_samples']
                    or actual['sha256'] != source['converted_sha256']):
                raise ValueError('actual headless new-source PCM differs from native TTS')
            result['new_source_sink_pcm_exact'] = True
            result.pop('output', None)  # child temporary file is not durable output
            durable_json(directory/'conversation-result.json', result)
        return result
    finally:
        original = sys.exc_info()[1]
        errors = []
        def cleanup(label, operation):
            try:
                return operation()
            except BaseException as error:
                errors.append((label, error))
        try:
            if native_observer is not None:
                cleanup('native observer close',native_observer.close)
            if consumer is not None:
                cleanup('consumer close', consumer.close)
            if director is not None:
                cleanup('director close', director.close)
            if endpoint is not None:
                cleanup('ASR endpoint close', endpoint.close)
            if job is not None:
                # Every compute-side operation is bound to the explicit Job
                # namespace; a backup-billed Job is never inspected or cancelled
                # through the main account.
                ns = {'namespace':job_namespace} if job_namespace != 'YUGOROU' else {}
                status = cleanup('Job inspection', lambda: job_api.inspect_job(job_id=job.id, **ns))
                # Unknown status must never prevent a cancellation attempt.
                if status is None or status.status.stage not in TERMINAL:
                    cleanup('Job cancellation', lambda: job_api.cancel_job(job_id=job.id, **ns))
                if rtc or headless_interrupt or compute_profile is not None:
                    def confirm_terminal():
                        attempts = 45 if (headless_interrupt or compute_profile is not None) else 8
                        for _ in range(attempts):
                            state=job_api.inspect_job(job_id=job.id, **ns).status.stage
                            if state in TERMINAL:
                                durable_json(directory/('rtc-job-terminal.json' if rtc else 'job-terminal.json'),
                                             dict(job_id=job.id,status=state,job_namespace=job_namespace));return
                            time.sleep(1)
                        raise RuntimeError('owned Job terminal status unconfirmed')
                    cleanup('Job terminal confirmation',confirm_terminal)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        if errors:
            primary = original if original is not None else errors[0][1]
            for label, error in errors:
                primary.add_note(label+' failed: '+type(error).__name__)
            if original is None:
                raise primary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--sentence-incremental-tts', action='store_true',
        help='Experimental sentence-level watermarked TTS; requires --run')
    endurance = p.add_mutually_exclusive_group()
    endurance.add_argument('--ten-minute-test', action='store_true',
        help='Explicit 600s after first AV endurance; finite 1200s provider/lease; requires --run')
    endurance.add_argument('--five-minute-test', action='store_true',
        help='Explicit 300s after first AV endurance; finite 1200s provider/lease; requires --run')
    anchors = p.add_mutually_exclusive_group()
    anchors.add_argument('--conversation-anchor', action='store_true',
        help='Experimental reviewed conversation-tail E2E candidate; requires --run')
    anchors.add_argument('--idle-anchor', action='store_true',
        help='Explicit visual candidate experiment; live speech handoff not proven (requires --run)')
    p.add_argument('--continuous-av', action='store_true',
                   help='experimental one-shot incremental AV response; requires --run')
    p.add_argument('--http-keepalive', action='store_true',
                   help='explicit bounded authenticated HTTP connection reuse; requires --run')
    p.add_argument('--overlap-send', action='store_true',
                   help='explicit bounded GPU generation/local-enqueue overlap; requires --run')
    p.add_argument('--start-gate', type=Path,
                   help='local file whose appearance starts AV after READY (operator-managed source reopen)')
    p.add_argument('--prefetch-receive', action='store_true',
                   help='opt-in bounded receive-ahead of one next AV pair (no quality change)')
    p.add_argument('--headless-interrupt', action='store_true',
                   help='existing Vox lease + interruptible HTTPS H3 + observed headless sink, no devices')
    p.add_argument('--visual-intents', action='store_true',
                   help='versioned Pi camera/action intent through native conditioning; requires --headless-interrupt, latest-tail only')
    p.add_argument('--attended-visual', action='store_true',
                   help='explicit attended capability: existing Comet ASR input + existing local HTTPS browser sink, '
                        'no headless interruption/OBS/devices; validated seam only, no execution from this CLI')
    p.add_argument('--stable-selfie', action='store_true',
                   help='attended static composition with conversation-tail-every3-v1; camera/action movement disabled')
    p.add_argument('--h3-backup-jobs', action='store_true',
                   help='bill ONLY the H3 Job to the explicit backup compute account; assets/Pi/Endpoints stay main')
    p.add_argument('--review-av', action='store_true',
                   help='bounded private review AV retention and explicit paired suffix cuts; requires --visual-intents; no playback/upload')
    p.add_argument('--native-cancel-trigger',choices=['native-preroll-v1'],
                   help='explicit native progress + local AV preparatory cancellation, not heard-speech barge-in')
    p.add_argument('--native-tts-cancel', action='store_true',
                   help='explicit native-yield Vox cancel v2; requires --headless-interrupt and reviewed Vox parent')
    p.add_argument('--ordered-references-config',type=Path,
                   help='local bounded two-reference config; requires --visual-intents, offline inspection by default')
    p.add_argument('--confirm-private-reference-upload',action='store_true',
                   help='explicit scoped approval for these two images into HF private staging/original H3 territory; storage territory not guaranteed')
    p.add_argument('--vox-parent', type=Path, help='reviewed paused existing Vox lease parent')
    p.add_argument('--capture-comet', action='store_true')
    p.add_argument('--asr-parent', type=Path)
    p.add_argument('--allow-obs', action='store_true')
    p.add_argument('--obs-port', type=int)
    p.add_argument('--direct-blackhole-audio', type=Path)
    p.add_argument('--three-hour-session', action='store_true', help='Explicit attended finite three-hour owner')
    p.add_argument('--today-pause-on-underflow', action='store_true',
                   help='Temporary attended-only paired pause; not continuous playback acceptance')
    p.add_argument('--audio-delay-seconds', type=float, default=None)
    p.add_argument('--native-tts-config',type=Path,
                   help='explicit native Vox Endpoint/preroll manifest; offline validates only, no secrets in file')
    a = p.parse_args(argv)
    if a.stable_selfie and (not a.visual_intents or a.headless_interrupt
            or a.idle_anchor or a.native_tts_cancel or a.native_cancel_trigger is not None):
        p.error('--stable-selfie requires --visual-intents without movement/interruption modes')
    if a.stable_selfie and not a.attended_visual and (
            a.vox_parent is None or a.capture_comet or a.asr_parent is not None
            or a.allow_obs or a.obs_port is not None or a.direct_blackhole_audio is not None
            or a.native_tts_config is not None or a.sentence_incremental_tts
            or a.five_minute_test or a.ten_minute_test or a.start_gate is not None
            or a.three_hour_session or a.today_pause_on_underflow):
        p.error('device-free --stable-selfie requires --vox-parent and no device/capture/alternate modes')
    if a.visual_intents and a.conversation_anchor and not a.stable_selfie:
        p.error('--visual-intents with --conversation-anchor requires --stable-selfie')
    if a.h3_backup_jobs and (not a.stable_selfie or a.attended_visual or a.headless_interrupt
            or a.three_hour_session or a.asr_parent is not None):
        p.error('--h3-backup-jobs supports only the device-free --stable-selfie native HTTPS route')
    selected_references=None;selected_metadata={}
    selected_path=getattr(a,'ordered_references_config',None)
    confirmed_references=getattr(a,'confirm_private_reference_upload',False)
    if selected_path is not None:
        if not a.visual_intents:p.error('--ordered-references-config requires --visual-intents')
        try:
            path=selected_path
            if path.is_symlink() or not path.is_file() or path.stat().st_size>32768:
                raise ValueError('bounded regular reference configuration required')
            from research.h3_personal.voxcpm_endpoint_protocol import loads
            selected_references,selected_metadata=ordered_selection(loads(path.read_bytes()),
                confirm_private_reference_upload=confirmed_references,require_confirmation=a.run)
        except (ValueError,OSError):
            p.error('valid fresh ordered reference config and explicit cloud confirmation required for --run')
    elif confirmed_references:
        p.error('--confirm-private-reference-upload requires --ordered-references-config')
    if a.native_cancel_trigger is not None and not a.native_tts_cancel:
        p.error('--native-cancel-trigger requires --native-tts-cancel')
    if a.native_tts_cancel and not a.headless_interrupt:
        p.error('--native-tts-cancel requires --headless-interrupt')
    if a.review_av and not a.visual_intents:p.error('--review-av requires --visual-intents')
    if a.review_av and getattr(a,'attended_visual',False):
        # Review retention and paired suffix cuts are bound to the headless
        # interruption presentation gate in the immutable spec, READY and worker
        # (paired_cuts requires turn_interruptions). The attended split route has
        # no interruption controls, so it cannot carry them. Reject instead of
        # accepting the flag and silently dropping it: a 3h attended session
        # without review AV leaves an interrupted visual defect undiagnosable.
        p.error('--review-av is not available on the attended route: paired review cuts '
                'require the headless interruption gate, which attended sessions forbid')
    if a.visual_intents and (not (a.headless_interrupt or getattr(a,'attended_visual',False) or a.stable_selfie) or a.idle_anchor):
        p.error('--visual-intents requires --headless-interrupt or explicit --attended-visual and no periodic anchor reset')
    if getattr(a,'attended_visual',False):
        try:
            attended_visual_capability(visual_intents=a.visual_intents, attended_visual=True,
                headless_interrupt=a.headless_interrupt, capture_comet=a.capture_comet,
                asr_parent=a.asr_parent, media_transport='http-paired-v1',
                allow_obs=a.allow_obs, obs_port=a.obs_port,
                direct_audio_binary=a.direct_blackhole_audio, start_gate=a.start_gate,
                idle_anchor=a.idle_anchor, local_sink_factory=None)
        except ValueError as error:
            p.error(str(error))
    if a.headless_interrupt and (a.vox_parent is None or a.capture_comet or a.asr_parent is not None
            or a.allow_obs or a.obs_port is not None or a.direct_blackhole_audio is not None
            or a.native_tts_config is not None or a.sentence_incremental_tts
            or a.five_minute_test or a.ten_minute_test or a.idle_anchor or a.start_gate is not None):
        p.error('--headless-interrupt requires --vox-parent and no device/legacy/alternate modes')
    if a.vox_parent is not None and not (a.headless_interrupt or getattr(a,'attended_visual',False) or a.stable_selfie):
        p.error('--vox-parent is only for --headless-interrupt, --attended-visual or --stable-selfie')
    native_config=None
    if a.native_tts_config is not None:
        if a.sentence_incremental_tts:
            p.error('native fulltext and sentence TTS are mutually exclusive')
        from research.h3_personal.native_tts_backend import validate_native_config
        try:
            if a.native_tts_config.stat().st_size>8192:
                raise ValueError()
            from research.h3_personal.voxcpm_endpoint_protocol import loads
            native_config=validate_native_config(loads(a.native_tts_config.read_bytes()))
        except (ValueError,OSError):
            p.error('bounded valid native TTS configuration required (no secrets)')
    if a.sentence_incremental_tts and not a.run:
        p.error('--sentence-incremental-tts requires --run')
    if a.continuous_av and not a.run:
        p.error('--continuous-av requires --run')
    if a.http_keepalive and not a.run:
        p.error('--http-keepalive requires --run')
    if a.overlap_send and not a.run:
        p.error('--overlap-send requires --run')
    if a.prefetch_receive and not a.run:
        p.error('--prefetch-receive requires --run')
    if a.five_minute_test and not a.run:
        p.error('--five-minute-test requires --run')
    if a.ten_minute_test and not a.run:
        p.error('--ten-minute-test requires --run')
    if a.three_hour_session and (not a.run or not a.attended_visual or not a.today_pause_on_underflow
            or a.five_minute_test or a.ten_minute_test):
        p.error('--three-hour-session requires --run --attended-visual --today-pause-on-underflow')
    if a.today_pause_on_underflow and (not a.attended_visual or not a.run
            or a.direct_blackhole_audio is None):
        p.error('--today-pause-on-underflow requires attended split output and --run')
    direct_kwargs = ({'five_minute_test': True} if a.five_minute_test else
                     {'ten_minute_test': True} if a.ten_minute_test else {})
    if a.three_hour_session:
        direct_kwargs['long_session'] = True
    if a.today_pause_on_underflow:
        direct_kwargs['today_pause_on_underflow'] = True
    if a.stable_selfie:
        direct_kwargs['stable_selfie'] = True
    if selected_references is not None:
        direct_kwargs.update(ordered_references=selected_references,
                             confirm_private_reference_upload=confirmed_references)
    if a.sentence_incremental_tts:
        direct_kwargs['tts_mode'] = 'sentence_incremental'
    if native_config is not None:
        direct_kwargs.update(tts_mode='native_voxcpm',native_tts=native_config)
    if a.continuous_av:
        direct_kwargs['continuous_av'] = True
    if a.http_keepalive:
        direct_kwargs['http_keepalive'] = True
    if a.overlap_send:
        direct_kwargs['overlap_send'] = True
    if a.prefetch_receive:
        direct_kwargs['prefetch_receive'] = True
    if a.start_gate is not None:
        if not a.run:
            p.error('--start-gate requires --run')
        if not str(a.start_gate).strip():
            p.error('--start-gate requires a non-empty path')
        direct_kwargs['start_gate'] = a.start_gate
    if a.audio_delay_seconds is not None and a.direct_blackhole_audio is None:
        p.error('--audio-delay-seconds requires --direct-blackhole-audio')
    if a.direct_blackhole_audio is not None:
        if not a.run or not a.allow_obs or a.obs_port is None:
            p.error('--direct-blackhole-audio requires --run, --allow-obs and --obs-port')
        from research.run_h3_relay_conversation import direct_audio_pin
        delay = 0.2 if a.audio_delay_seconds is None else a.audio_delay_seconds
        try:
            pin = direct_audio_pin(a.direct_blackhole_audio, delay)
        except (ValueError, OSError):
            p.error('valid executable binary and finite audio delay within 0..1 required')
        direct_kwargs.update(direct_audio_binary=pin['audio_binary'], audio_delay_seconds=delay)
    if a.conversation_anchor and not a.run:
        p.error('--conversation-anchor requires --run')
    if a.idle_anchor and not a.run:
        p.error('--idle-anchor requires --run')
    if a.capture_comet != (a.asr_parent is not None):
        p.error('--capture-comet and --asr-parent are both required for ASR')
    if a.obs_port is not None and not a.allow_obs:
        p.error('--obs-port requires --allow-obs')
    if not a.run:
        if selected_metadata:
            print(json.dumps(dict(type='h3_ordered_references_offline',**selected_metadata,
                                  visual_acceptance='unreviewed',cloud_operation=False),sort_keys=True))
        if (a.directory/'deployment').exists():
            inspect(a.directory/'deployment')
        if a.native_cancel_trigger is not None:
            print('Native preroll trigger: authenticated source1 progress + local first AV; preparatory cancellation, not heard-speech barge-in. Offline only.')
        if a.native_tts_cancel:
            print('Native cancel v2: independent authenticated control, verified terminal, retained interrupted prefix; legacy drain remains default. Offline only.')
        if a.review_av:
            print('Review AV: private NUT spans, max2GiB; ' + ('continuous static archive' if a.stable_selfie else 'paired suffix cut plus reference video/PCM/sidecars') + '; no capture/playback/upload. Offline only.')
        if a.stable_selfie:
            print('Stable selfie: conversation-tail-every3-v1 with camera/action movement disabled; quality unverified. No provider/device operation.')
        elif a.visual_intents:
            print('Visual intent: versioned proposal/command/native prompt-cache binding; latest-tail continuation, achievement unverified. No provider/device operation.')
        print('Offline. --run prepares a fresh private deployment and one finite direct H200x2 Job; uses typed input or explicitly selected attended ASR. No Discord join.')
        return
    if a.capture_comet:
        if getattr(a,'attended_visual',False):
            if a.vox_parent is None:
                p.error('--attended-visual requires a reviewed paused --vox-parent lease')
            if not a.allow_obs or a.obs_port is None or a.direct_blackhole_audio is None:
                p.error('--attended-visual requires --allow-obs, --obs-port and --direct-blackhole-audio')
            from research.h3_personal.voxcpm_https_lease import lease as vox_lease, PRIVATE
            attended_root = Path(os.path.abspath(a.directory))
            if (attended_root.exists() or not attended_root.is_relative_to(PRIVATE)
                    or any(p.is_symlink() for p in (attended_root,*attended_root.parents))):
                p.error('fresh private campaign directory required')
            os.umask(0o077); attended_root.mkdir(mode=0o700)
            previous_alarm = signal.signal(signal.SIGALRM,
                lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('finite conversation owner deadline')))
            previous_term = signal.signal(signal.SIGTERM,
                lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('conversation owner terminated')))
            signal.setitimer(signal.ITIMER_REAL,10800 if a.three_hour_session else 1500)
            try:
                with vox_lease(attended_root/'vox', min_remaining=600,
                               parent_campaign=a.vox_parent,
                               **({'session_seconds':10800} if a.three_hour_session else {})) as source:
                    os.environ.update(H3_NATIVE_TTS_HF_TOKEN=source['token'],
                                      H3_NATIVE_TTS_KEY=source['key'])
                    try:
                        result = run(attended_root/'h3', None, tts_mode='native_voxcpm',
                            native_tts=source['config'], capture_comet=True, asr_parent=a.asr_parent,
                            allow_obs=True, obs_port=a.obs_port, attended_visual=True,
                            visual_intents=a.visual_intents,
                            conversation_anchor=a.stable_selfie or not a.visual_intents,
                            **direct_kwargs)
                    finally:
                        os.environ.pop('H3_NATIVE_TTS_HF_TOKEN',None)
                        os.environ.pop('H3_NATIVE_TTS_KEY',None)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,previous_alarm)
                signal.signal(signal.SIGTERM,previous_term)
            print(json.dumps(dict(result=result), ensure_ascii=False, default=str))
            print('Session ended; Discord reception is not independently verified.')
            return
        run(a.directory, capture_comet=True, asr_parent=a.asr_parent,
            allow_obs=a.allow_obs, obs_port=a.obs_port, **direct_kwargs, **({'idle_anchor': True} if a.idle_anchor else
                {'conversation_anchor': True} if a.conversation_anchor else {}))
        print('Session ended; Discord reception is not independently verified.')
        return
    turns = []
    for _ in range(2):
        try:
            line = input('Fresh typed turn (blank ends): ')
        except EOFError:
            break
        if not line.strip():
            break
        turns.append(line)
    if a.headless_interrupt or (a.stable_selfie and not a.attended_visual):
        if len(turns) != 2:p.error('exactly two typed turns required before leasing GPU')
        if selected_references is not None:
            selected_references,_=ordered_selection(selected_references,
                confirm_private_reference_upload=confirmed_references,require_confirmation=True)
        from research.h3_personal.voxcpm_https_lease import lease, PRIVATE
        root = Path(os.path.abspath(a.directory))
        if (root.exists() or not root.is_relative_to(PRIVATE)
                or any(p.is_symlink() for p in (root,*root.parents))):
            p.error('fresh private campaign directory required')
        os.umask(0o077); root.mkdir(mode=0o700)
        previous_alarm = signal.signal(signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(TimeoutError('finite conversation owner deadline')))
        previous_term = signal.signal(signal.SIGTERM,
            lambda *_: (_ for _ in ()).throw(InterruptedError('conversation owner terminated')))
        signal.setitimer(signal.ITIMER_REAL,1500)
        try:
            with lease(root/'vox', min_remaining=600, parent_campaign=a.vox_parent,
                       **({'native_tts_cancel':True} if a.native_tts_cancel else {})) as source:
                os.environ.update(H3_NATIVE_TTS_HF_TOKEN=source['token'],H3_NATIVE_TTS_KEY=source['key'])
                try:
                    result = run(root/'h3', turns, tts_mode='native_voxcpm', native_tts=source['config'],
                        conversation_anchor=a.stable_selfie or not a.visual_intents, prefetch_receive=True, overlap_send=True,
                        http_keepalive=True, continuous_av=True, headless_interrupt=a.headless_interrupt,
                        **({'stable_selfie':True} if a.stable_selfie else {}),
                        **({'h3_backup_jobs':True} if a.h3_backup_jobs else {}),
                        **({'ordered_references':selected_references,'confirm_private_reference_upload':confirmed_references} if selected_references is not None else {}),
                        **({'visual_intents':True} if a.visual_intents else {}),
                        **({'review_av':True} if a.review_av else {}),
                        **({'native_tts_cancel':True} if a.native_tts_cancel else {}),
                        **({'native_cancel_trigger':a.native_cancel_trigger} if a.native_cancel_trigger is not None else {}))
                finally:
                    os.environ.pop('H3_NATIVE_TTS_HF_TOKEN',None); os.environ.pop('H3_NATIVE_TTS_KEY',None)
            if a.native_tts_cancel:
                try:
                    result['native_cancel_validation']=validate_native_cancel_evidence(
                        json.loads((root/'h3/native-worker-evidence.json').read_text()),
                        json.loads((root/'h3/native-cancel-worker-evidence.json').read_text()),
                        json.loads((root/'vox/native-completion-evidence.json').read_text()))
                except (ValueError,KeyError,TypeError,OSError):
                    result['native_cancel_validation']=dict(native_cancel_exercised=False,
                        next_native_natural_eos_verified=False,
                        reason='missing_or_mismatched_native_cancel_reuse_evidence')
            if a.native_tts_cancel:
                result['native_cancel_exercised']=result['native_cancel_validation']['native_cancel_exercised']
                try:
                    healthy=native_source2_healthy(
                        json.loads((root/'h3/native-worker-evidence.json').read_text()),
                        json.loads((root/'vox/native-completion-evidence.json').read_text()))
                except (ValueError,OSError):healthy=False
                result['new_source_healthy']=bool(healthy and result.get('new_source_sink_pcm_exact'))
                result['presentation_interrupted']=bool(result.get('sink_timing',{}).get('discontinuities'))
            if a.review_av:
                result['review_files']=create_review_previews(root/'h3/receiver/review')
            durable_json(root/'result.json',dict(result,complete=True,devices_used=False,cleanup_confirmed=True))
            print('Device-free conversation ended; see result.json for separate sink-stop and response timings.')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous_alarm)
            signal.signal(signal.SIGTERM,previous_term)
        return
    run(a.directory, turns, allow_obs=a.allow_obs, obs_port=a.obs_port, **direct_kwargs,
        **({'idle_anchor': True} if a.idle_anchor else
           {'conversation_anchor': True} if a.conversation_anchor else {}))
    print('Session ended; Discord reception is not independently verified.')


if __name__ == '__main__':
    main()
