"""Finite Comet-process audio → fresh T4 Endpoint → exact Pi director.

Offline default. Explicit --run/--capture-comet uploads in-memory Comet audio;
all Comet tabs are included, NOT Discord-isolated. No microphone, saved raw audio,
H3 GPU launch, routing change, browser control or automatic Discord join.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import math
import json
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import time
import uuid

from research.run_h3_asr_endpoint import (
    control, assert_deployment, deployment, pause_owned, reserve, sha,
)
from research.h3_personal.private_root import root as _private_root


def stop_capture(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
        raise RuntimeError('capture required SIGKILL; CoreAudio cleanup unconfirmed')
    if process.returncode not in (0,143):
        raise RuntimeError('Comet capture failed; do not submit transcript to Pi')


def validate_capture_time(value, now=None):
    if value is not None:
        now = time.time() if now is None else now
        if not math.isfinite(value) or not now <= value <= now + 600:
            raise ValueError('capture-not-before must be a future UNIX timestamp within 600 seconds')
    return value


def capture_guard(planned, deadline):
    now = time.time()
    if now + 60 >= deadline:
        raise RuntimeError('insufficient finite deadline for capture')
    if planned is not None and now > planned + 3:
        raise RuntimeError('missed attended capture window')
    return now


def wait_for_capture(planned, deadline):
    started = capture_guard(planned, deadline)
    # Monotonic bound also prevents a backwards wall-clock jump extending wait.
    end = time.monotonic() + max(0, (planned or started) - started) + 3
    while planned is not None and time.time() < planned:
        capture_guard(planned, deadline)
        if time.monotonic() >= end:
            raise RuntimeError('finite capture wait expired')
        time.sleep(min(.25, max(0, planned - time.time())))
    actual = capture_guard(planned, deadline)
    return dict(planned_start=planned, planned_start_utc=None if planned is None else datetime.fromtimestamp(planned, timezone.utc).isoformat(),
                actual_time=actual, actual_utc=datetime.fromtimestamp(actual, timezone.utc).isoformat(),
                wait_status='waited' if planned is not None and started < planned else 'not_needed')


def parent_intent(original, campaign, info):
    expected = original
    if campaign is not None:
        expected = json.loads((campaign / 'intent.json').read_text())
        result = json.loads((campaign / 'result.json').read_text())
        if result.get('terminal_state') != 'paused':
            raise ValueError('parent terminal state must be paused')
        # Only revision and finite expiration may differ from original pins.
        reviewed = deployment(original['model']['repository'], expected['model']['revision'],
                              original['name'], int(expected['model']['env']['H3_ASR_EXPIRES_AT']))
        assert_deployment(expected, reviewed)
    if not info.private or info.sha != expected['model']['revision'] or expected['model']['repository'] != original['model']['repository']:
        raise ValueError('private runtime branch does not match reviewed parent')
    return expected


def run(args):
    validate_capture_time(getattr(args, 'capture_not_before', None))
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    from research.probe_h3_asr_cpu_space import http_ready
    from research.h3_personal.endpoint_asr import transcribe
    private=_private_root()
    root=args.directory.resolve()
    if root.exists() or not root.is_relative_to(private):
        raise ValueError('new private session directory required')
    if not args.capture_comet:
        raise ValueError('explicit Comet audio upload authorization required')
    binary=Path('/tmp/h3-comet-pcm')
    capture_source=Path(__file__).parent/'stream_h3_comet_pcm.swift'
    subprocess.run(['swiftc','-swift-version','6','-strict-concurrency=complete','-warnings-as-errors',str(capture_source),'-o',str(binary)],check=True,timeout=60,capture_output=True)
    binary_sha=sha(binary)
    parent=private/'hf-asr-endpoint-20261019-v1'
    original=json.loads((parent/'intent.json').read_text())
    token=cached_token()
    api=HfApi(token=token)
    if api.whoami()['name']!='YUGOROU':raise ValueError('owned namespace required')
    path='/v2/endpoint/YUGOROU/'+original['name']
    current=control(token,path)
    if current['status']['state']!='paused':raise ValueError('no concurrent allocation permitted')
    active=[e for e in control(token,'/v2/endpoint/YUGOROU')['items'] if e['status']['state'] not in ('paused','scaledToZero')]
    if active:raise ValueError('another allocated Endpoint exists')
    providers=control(token,'/v2/provider/YUGOROU')
    rows=[c for v in providers['vendors'] if v['name']=='aws' for rg in v['regions'] if rg['name']=='us-east-1' for c in rg['computes'] if c['instanceType']=='nvidia-t4' and c['instanceSize']=='x1']
    if len(rows)!=1 or rows[0]['status']!='available' or Decimal(str(rows[0]['pricePerHour']))!=Decimal('.5') or rows[0]['quota']['maxAccelerators']-rows[0]['quota']['usedAccelerators']<3:
        raise ValueError('fresh T4 rate/quota gate failed')
    info=api.repo_info(original['model']['repository'],repo_type='model')
    previous = parent_intent(original, getattr(args, 'parent_campaign', None), info)
    assert_deployment(current, previous)
    # Pin reviewed executable bytes in a NEW immutable revision. Inherit original
    # model and native runtime; entrypoint independently verifies both SHA pins.
    root.mkdir(mode=0o700)
    stage=root/'deployment';stage.mkdir(mode=0o700)
    source=Path(__file__).parent/'endpoints/nemotron_t4'
    for name in ('gate.py','entrypoint.py'):shutil.copyfile(source/name,stage/name)
    pins={n:sha(stage/n) for n in ('gate.py','entrypoint.py')}
    pins.update(capture_source=sha(capture_source),capture_binary=binary_sha)
    (root/'source-pins.json').write_text(json.dumps(pins,indent=2))
    commit=api.upload_folder(repo_id=original['model']['repository'],repo_type='model',folder_path=stage,commit_message='Fresh finite attended Comet ASR session; no media or secrets')
    deadline=int(time.time())+1800
    expected=deployment(original['model']['repository'],commit.oid,original['name'],deadline)
    (root/'intent.json').write_text(json.dumps(expected,indent=2))
    key='hf-asr-attended-'+uuid.uuid4().hex[:12]
    reserve(private/'compute-budget.json',key,scope='hf-endpoint-ASR-T4-attended-comet')
    owner_key=secrets.token_urlsafe(40)
    result=dict(name=original['name'],reservation=key,deadline=deadline,revision=commit.oid,
                origin='comet_process_audio_endpoint_asr',raw_audio_saved=False,
                discord_connection_verified=False,discord_reception_verified=False,
                h3_generated=False,terminal_state='unconfirmed')
    process=None
    try:
        patch=dict(model=expected['model'],tags=['h3-asr-attended','finite-owner'])
        patch['model']['secrets']={'H3_ASR_KEY':owner_key}
        control(token,path,patch,'PUT')
        current=control(token,path)
        if current['status']['state']=='paused':control(token,path+'/resume',{},'POST')
        wait=time.monotonic()+300
        while True:
            current=control(token,path)
            print(json.dumps(dict(stage='endpoint',state=current['status']['state'])),flush=True)
            if current['status']['state']=='running':break
            if current['status']['state']=='failed' or time.monotonic()>wait:raise RuntimeError('finite readiness failed')
            time.sleep(10)
        assert_deployment(current,expected)
        url=current['status']['url']
        from research.h3_personal.endpoint_asr import _host
        host=_host(url)
        anon,_=http_ready(host)
        code,body=http_ready(host,token)
        if anon not in (401,403,404) or code!=200 or json.loads(body).get('device')!='cuda':
            raise ValueError('authenticated CUDA/anonymous denial required')
        timing = wait_for_capture(getattr(args, 'capture_not_before', None), deadline)
        print(json.dumps(dict(stage='capture_ready',seconds=45,**timing,warning='Speak via joined iOS Discord; all Comet audio included')),flush=True)
        # Metadata file contains diagnostics only; binary PCM remains pipe-only.
        if sha(binary)!=binary_sha:raise ValueError('capture binary drifted during readiness')
        with (root/'capture-metadata.jsonl').open('xb') as metadata:
            def start_capture():
                nonlocal process
                capture_guard(getattr(args, 'capture_not_before', None), deadline)
                process=subprocess.Popen([str(binary),'--stream'],stdout=subprocess.PIPE,stderr=metadata)
                return process.stdout
            try:
                transcript=transcribe(url,token=token,owner_key=owner_key,pcm=start_capture,timeout=60,max_audio_seconds=45,
                                      completion_policy='input-finalized')
            finally:
                try:stop_capture(process)
                finally:
                    if process is not None:process.stdout.close()
        result['asr']=transcript
        diagnostics=[json.loads(line) for line in (root/'capture-metadata.jsonl').read_text().splitlines()]
        if not diagnostics or diagnostics[-1].get('type')!='stopped' or diagnostics[-1].get('cleanup') is not True:
            raise ValueError('CoreAudio cleanup receipt required')
        if diagnostics[-1].get('frames',0)<=0 or diagnostics[-1].get('nonzero_samples',0)<=0 or transcript['audio_sent_seconds']<=0:
            raise ValueError('nonzero captured and transmitted audio required')
        # Capture is fully stopped before Pi or any future outbound playback.
        transcript['capture_stopped']=True
        result['asr']=transcript
        if not transcript['final'] or transcript['errors'] or not transcript['cleanup_complete']:
            raise ValueError('native final and complete capture/transport cleanup required')
        text=transcript['text'].strip()
        if not 1<=len(text)<=160:raise ValueError('bounded short utterance required')
        result['terminal_state']=pause_owned(token,path)
        if result['terminal_state']!='paused':raise RuntimeError('owned endpoint disappeared')
        # Exact model, two-tool/no-fallback Pi contracts live in director owner.
        # Never externally kill the director: it owns its Node child/accounting.
        prompt='以下はASRから返った音声認識テキストです。原則として聞き返さず、文脈から意図を推測して誤認識を補正し、質問に答えてください。補正で意味を変えたり、事実を捏造したりしないでください。\n'+text
        packet=json.dumps(dict(op='turn',turn_id='turn-'+uuid.uuid4().hex,text=prompt))+'\n'
        pi=subprocess.run(['python3','-m','research.run_h3_director','--run','--serve','--max-turns','1'],input=packet,capture_output=True,text=True)
        if pi.returncode:raise RuntimeError('Pi failed; no automatic fallback')
        proposal=next(r for r in map(json.loads,pi.stdout.splitlines()) if r.get('type')=='turn')
        if proposal['model']!='Qwen/Qwen3.8-27B:cerebras':raise ValueError('exact Pi model required')
        result['proposal']=proposal
        print(json.dumps(dict(stage='pi_complete',asr=text,reply=proposal['reply'],h3_generated=False),ensure_ascii=False),flush=True)
    except Exception as exc:
        result['error_type']=type(exc).__name__
        raise
    finally:
        try:
            if process is not None and process.poll() is None:stop_capture(process)
        finally:
            try:result['terminal_state']=pause_owned(token,path)
            finally:(root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--capture-comet',action='store_true')
    p.add_argument('--capture-not-before',type=float,default=None)
    p.add_argument('--parent-campaign',type=Path,default=None)
    args=p.parse_args()
    try:validate_capture_time(args.capture_not_before)
    except ValueError as exc:p.error(str(exc))
    if not args.run:
        print(json.dumps(dict(mode='offline',path='Comet process audio -> T4 ASR -> Pi',max_capture_seconds=45,
                              endpoint_max_seconds=1800,endpoint_reservation_usd='.75',h3_launch=False,
                              all_comet_tabs=True,capture_requires_explicit_flag=True)))
        return
    def interrupted(_number,_frame):
        raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:run(args)
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()
