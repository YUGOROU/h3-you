"""Finite on-demand conversation owner; no per-turn Pause/cancel, no renewals.

One local command service owns ASR, H3, Pi and bounded worker queues. Its private
Unix socket accepts explicit turns only: it does not watch for or invent speech.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import selectors
import socket
import subprocess
import sys
import threading
import time
import uuid

from research.h3_personal.resident_admission import (
    MAX_PI_REQUESTS, MAX_CAPTURE_ATTEMPTS, admit_capture, reserve_pi_request,
)
from research.h3_personal.private_root import root as _private_root

PRIVATE = _private_root()
MAX_TURNS = MAX_PI_REQUESTS


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    os.replace(temporary, path)


def command(directory, packet):
    if packet.get('op') in ('turn', 'recover') and (directory/'operator-hold.json').exists():
        raise RuntimeError('operator hold: no capture or automatic H3 launch')
    address = json.loads((directory/'owner.json').read_text())['socket']
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(10)
        client.connect(address)
        client.sendall(json.dumps(packet).encode()+b'\n')
        return json.loads(client.makefile('r').readline(10000))


def full_budget_gate(ledger):
    from decimal import Decimal
    from research.h3_personal.ref2va_budget import HF_CAP
    data = json.loads(ledger.read_text())
    bound = Decimal(str(data['spent_usd'])) + sum(Decimal(str(x['upper_bound_usd'])) for x in data['reservations'].values())
    required = Decimal('3.333340')+Decimal('.75')+Decimal('.20')*MAX_TURNS
    if bound+required > HF_CAP:
        raise ValueError('full1200s H3 +1800s ASR +4 Pi turns exceeds authority')
    return str(required)


def serve(root, parent, obs):
    from huggingface_hub import HfApi, hf_hub_download
    from research.provision_modal_identity_secret import cached_token
    from research.h3_personal.resident_endpoint import ResidentEndpoint, NoSpeech, TurnRejected
    from research.h3_personal.resident_delivery import ResidentDelivery
    from research.h3_personal.resident_recovery import ResidentRecovery, plan_turn_recovery
    from research.wait_h3_live_ready import wait
    from research.launch_h3_live_voice import main as launch
    full_budget_gate(PRIVATE/'compute-budget.json')
    api = HfApi(token=cached_token())
    if any(j.status.stage not in ('CANCELED','ERROR','COMPLETED','DELETED') for j in api.list_jobs(namespace='YUGOROU')):
        raise ValueError('another active HF Job exists')
    lock = threading.RLock()
    state = dict(status='preparing_pi', turns={}, maximum_turns=MAX_TURNS,
                 maximum_capture_attempts=MAX_CAPTURE_ATTEMPTS, pi_requests=0,
                 realtime_verified=False)
    def update(**values):
        with lock:
            state.update(values);write(root/'status.json', state)
    recovery = ResidentRecovery(state, lock, lambda: write(root/'status.json', state))
    endpoint = delivery = pi = None
    job = None
    end_reason = 'owner_error'
    stop = threading.Event()
    pools = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=n) for n in ('h3-input','h3-generate','h3-play')]
    input_pool, generate_pool, play_pool = pools
    input_future = None
    sequence = 0
    video_sequence = 0
    server = socket.socket(socket.AF_UNIX)
    address = json.loads((root/'owner.json').read_text())['socket']
    server.bind(address);os.chmod(address,0o600);server.listen(2);server.settimeout(.5)
    def interrupted(_number, _frame):
        nonlocal end_reason
        end_reason='explicit_stop'
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    def pi_row():
        line = pi.stdout.readline(16000)
        if not line:raise RuntimeError('Pi owner terminated')
        return json.loads(line)
    try:
        # Persistent Pi owns its .80 reservation, Node child, history and accounting.
        pi = subprocess.Popen([sys.executable,'-m','research.run_h3_director','--run','--serve','--max-turns',str(MAX_TURNS)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        ready = pi_row()
        if ready.get('type')!='ready' or ready.get('model')!='Qwen/Qwen3.8-27B:cerebras':raise RuntimeError('exact Pi readiness required')
        update(status='preparing_asr')
        endpoint = ResidentEndpoint.open(root/'asr', parent)
        deadline = endpoint.deadline-10
        update(status='asr_ready_preflight', asr_deadline=endpoint.deadline, deadline=deadline)
        def prepare_h3():
            nonlocal job, delivery, deadline
            update(status='preparing_h3')
            h3 = root/'h3'
            old_args = sys.argv
            try:
                sys.argv = ['launch_h3_live_voice','--directory',str(h3),'--asr-mode','endpoint-native-v1','--session-seconds','1200','--submit']
                launch()
            finally:sys.argv = old_args
            submission_path = h3/'submission.json'
            submission = json.loads(submission_path.read_text());job=submission['job_id']
            if stop.is_set():
                api.cancel_job(job_id=job,namespace='YUGOROU')
                raise RuntimeError('session stopped during preparation')
            def fetch(repo,name,revision):
                return json.loads(Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=cached_token(),etag_timeout=15)).read_text())
            ready = wait(h3,api,fetch,running_seconds=1200,cancel_on_failure=False)
            deadline = min(endpoint.deadline-10, ready['expires_at']-10)
            submission.update(resident_lease_id=endpoint.lease_id, deadline=ready['expires_at'], resident_asr_deadline=endpoint.deadline)
            write(submission_path,submission)
            delivery = ResidentDelivery(h3,obs)
            update(status='ready', deadline=deadline, job_id=job, h3_ready_at=ready['observed_at'])
        def playback(asset, seq, video_seq):
            try:
                if stop.is_set():return
                delivery.play(asset,video_seq)
                recovery.complete(seq, 'played')
            except BaseException as exc:recovery.fail(seq, 'playback', exc)
        def render(result, turn, seq, video_seq, pending=None):
            phase = 'generation_submit' if pending is None else 'generation_finish'
            try:
                if stop.is_set():raise RuntimeError('session stopping')
                if pending is None:
                    pending = delivery.submit(result,turn,video_seq)
                phase = 'generation_finish'
                asset = delivery.finish(pending)
                recovery.submit(play_pool, playback, asset, seq, video_seq)
            except BaseException as exc:recovery.fail(seq, phase, exc)
        def recovery_task(seq, fault, action):
            return plan_turn_recovery(seq, fault, action, state=state,
                endpoint=endpoint, delivery=delivery, complete=recovery.complete,
                render=render, playback=playback)
        def input_turn(seq, planned):
            nonlocal video_sequence
            phase = 'capture'
            try:
                turn=root/'asr'/f'turn-{seq:03d}'
                started=time.time()
                a=endpoint.capture(turn,planned)
                if state['status']=='degraded':raise TurnRejected('operator_review_required')
                packet=dict(op='turn',turn_id='turn-'+uuid.uuid4().hex,text='以下はASRから返った音声認識テキストです。原則聞き返さず文脈で誤認識を補正し、意味を変えず質問に答えてください。事実は捏造しないでください。\n'+a['text'])
                if len(packet['text'].encode())>2000:raise TurnRejected('director_input_byte_limit')
                phase = 'pi_request'
                with lock:
                    state['turns'][str(seq)]['pi_request_index'] = reserve_pi_request(state)
                    write(root/'status.json',state)
                pi.stdin.write(json.dumps(packet)+'\n');pi.stdin.flush()
                proposal=pi_row()
                if proposal.get('type')!='turn' or proposal.get('turn_id')!=packet['turn_id']:raise RuntimeError('Pi turn failed; no fallback')
                result=dict(origin='comet_process_audio_endpoint_asr', terminal_state='running', resident_lease_id=endpoint.lease_id, deadline=endpoint.deadline, asr=a, proposal=proposal)
                write(turn/'director-result.json',result)
                with lock:
                    state['turns'][str(seq)].update(stage='generation_queued',asr=a['text'],reply=proposal['reply'],asr_pi_seconds=time.time()-started)
                    write(root/'status.json',state)
                if delivery is None:
                    with lock:
                        state['turns'][str(seq)]['stage']='preflight_only_not_AV_input'
                        write(root/'status.json',state)
                    phase = 'prepare_h3'
                    prepare_h3()
                    return
                video_sequence+=1
                with lock:
                    state['turns'][str(seq)]['video_sequence'] = video_sequence
                    write(root/'status.json',state)
                phase = 'generation_submit'
                recovery.submit(generate_pool,render,result,turn,seq,video_sequence)
            except NoSpeech:
                with lock:
                    state['turns'][str(seq)]['stage']='no_speech'
                    write(root/'status.json',state)
            except TurnRejected as exc:
                with lock:
                    state['turns'][str(seq)].update(stage='rejected_input',reason=str(exc))
                    write(root/'status.json',state)
            except BaseException as exc:recovery.fail(seq, phase, exc)
        while not stop.is_set() and time.time()<deadline:
            try:client,_=server.accept()
            except socket.timeout:continue
            with client:
                client.settimeout(3)
                try:
                    packet=json.loads(client.makefile('r').readline(4096));op=packet.get('op')
                    if op=='stop':stop.set();reply=dict(stopping=True)
                    elif op=='status':reply=state
                    elif op=='recover':
                        if (root/'operator-hold.json').exists():raise ValueError('operator hold')
                        seq=packet.get('sequence');action=packet.get('action','retry')
                        recovery.recover(seq,lambda fault: recovery_task(seq,fault,action),
                            generate_pool,now=time.time(),deadline=deadline,stopping=stop.is_set())
                        reply=dict(recovering=True,sequence=seq,action=action)
                    elif op=='turn':
                        if (root/'operator-hold.json').exists():raise ValueError('operator hold')
                        if state['status'] not in ('ready','asr_ready_preflight'):raise ValueError('not ready for input')
                        planned=packet['start']
                        if type(planned) not in (int,float) or not time.time()+3<=planned<=min(time.time()+120,deadline-200):raise ValueError('future finite window with200s remaining required')
                        if input_future is not None and not input_future.done():raise ValueError('capture/Pi busy')
                        with lock:admit_capture(sequence, state['pi_requests'])
                        # Bounded backpressure: at most one generated/playing predecessor.
                        if sequence>1 and state['turns'][str(sequence-1)]['stage'] not in ('played','no_speech','rejected_input','preflight_only_not_AV_input','skipped_failure'):raise ValueError('playout queue full')
                        sequence+=1;state['turns'][str(sequence)]=dict(stage='capturing',planned_start=planned)
                        write(root/'status.json',state)
                        input_future=recovery.submit(input_pool,input_turn,sequence,planned)
                        reply=dict(accepted=True,sequence=sequence,start=planned,capture_seconds=45,input_boundary='minimum10s_then3s_quiet_after_speech')
                    else:raise ValueError('unknown command')
                except Exception as exc:reply=dict(error=type(exc).__name__)
                client.sendall(json.dumps(reply,ensure_ascii=False).encode()+b'\n')
        end_reason = 'explicit_stop' if stop.is_set() else 'session_deadline'
    except BaseException as exc:
        update(status='error',error_type=type(exc).__name__)
        raise
    finally:
        stop.set();server.close()
        # Never discard warm H200s because their local owner encountered an error.
        # The original provider timeout still bounds cost; leave an adoption receipt.
        if job and end_reason == 'owner_error':
            update(h3_terminal='kept_until_original_provider_timeout',end_reason=end_reason)
        if job and end_reason != 'owner_error':
            terminal='unconfirmed'
            for _ in range(3):
                try:
                    terminal=api.inspect_job(job_id=job,namespace='YUGOROU').status.stage
                    if terminal in ('CANCELED','ERROR','COMPLETED','DELETED'):break
                    api.cancel_job(job_id=job,namespace='YUGOROU')
                except Exception as exc:update(h3_cleanup_error=type(exc).__name__)
                time.sleep(1)
            update(h3_terminal=terminal)
        if input_future is not None:
            try:input_future.result(timeout=140)
            except BaseException:pass
        if endpoint:
            for _ in range(3):
                try:
                    endpoint.close();update(asr_terminal=endpoint.receipt['terminal_state'])
                    if endpoint.closed:break
                except BaseException as exc:update(asr_cleanup_error=type(exc).__name__)
        for pool in pools:pool.shutdown(wait=False,cancel_futures=True)
        if delivery:delivery.close()
        if pi:
            pi.stdin.close()
            # Director owns timeout/cancellation/accounting; never externally kill it.
            remaining=[];end=time.monotonic()+90
            with selectors.DefaultSelector() as selector:
                selector.register(pi.stdout,selectors.EVENT_READ)
                while time.monotonic()<end:
                    if not selector.select(min(.5,max(0,end-time.monotonic()))):continue
                    line=pi.stdout.readline(16000)
                    if not line:break
                    remaining.append(json.loads(line))
            update(pi_closing_rows=remaining,pi_closed=pi.poll() is not None)
        Path(address).unlink(missing_ok=True)
        confirmed=(not job or state.get('h3_terminal') in ('CANCELED','ERROR','COMPLETED','DELETED')) and (not endpoint or state.get('asr_terminal')=='paused')
        update(status='stopped' if confirmed else 'cleanup_unconfirmed',stopped_at=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('op',choices=('start','serve','status','turn','recover','stop'))
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--parent-campaign',type=Path)
    p.add_argument('--obs-receipt',type=Path)
    p.add_argument('--start-time',type=float)
    p.add_argument('--sequence',type=int)
    p.add_argument('--action',choices=('retry','skip'),default='retry')
    p.add_argument('--run',action='store_true');p.add_argument('--capture-comet',action='store_true')
    a=p.parse_args();root=a.directory.resolve()
    if a.op=='status':print((root/'status.json').read_text());return
    if not a.run:print(json.dumps(dict(mode='offline',network=False,max_turns=MAX_TURNS,h3_seconds=1200,asr_seconds=1800)));return
    if not root.is_relative_to(PRIVATE):raise ValueError('private root required')
    if a.op=='start':
        if root.exists() or not a.capture_comet or not a.parent_campaign or not a.obs_receipt:raise ValueError('fresh directory and explicit capture required')
        full_budget_gate(PRIVATE/'compute-budget.json');root.mkdir(mode=0o700)
        address='/tmp/h3-resident-'+uuid.uuid4().hex+'.sock'
        write(root/'owner.json',dict(socket=address))
        with (root/'owner.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-m','research.run_h3_resident_conversation','serve','--directory',str(root),'--parent-campaign',str(a.parent_campaign),'--obs-receipt',str(a.obs_receipt),'--run','--capture-comet'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        write(root/'process.json',dict(pid=child.pid,started=time.time(),max_lease_seconds=1800))
        print(json.dumps(dict(started=True,pid=child.pid)));return
    if a.op=='serve':
        if not a.capture_comet:raise ValueError('explicit capture required')
        serve(root,a.parent_campaign,a.obs_receipt);return
    print(json.dumps(command(root,dict(op=a.op,start=a.start_time,sequence=a.sequence,action=a.action))))


if __name__=='__main__':main()
