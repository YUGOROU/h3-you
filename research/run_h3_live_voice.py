"""Foreground live-call owner: READY → fresh Comet ASR → Pi → resident TTS/H3 → OBS.
No GPU launch, background watcher, old speech replay or automatic camera control.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from io import BytesIO
from contextlib import ExitStack


def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    path.chmod(0o600)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--state',action='store_true');p.add_argument('--turn',action='store_true');p.add_argument('--wait-seconds',type=int,default=0);p.add_argument('--countdown',type=int,default=5);p.add_argument('--obs-receipt',type=Path);a=p.parse_args()
    if a.state==a.turn or not 0<=a.wait_seconds<=700 or not 0<=a.countdown<=10:p.error('one action, bounded wait/countdown required')
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    from research.provision_modal_identity_secret import cached_token
    d=a.directory.resolve();spec=json.loads((d/'deployment/spec.json').read_text());submission=json.loads((d/'submission.json').read_text());session=spec['live_voice_session']['id'];repo=spec['publication_repo'];api=HfApi(token=cached_token())
    def job_stage():
        if submission.get('provider')=='modal':
            control=subprocess.run([str(Path.home()/'.local/share/uv/tools/modal/bin/python3'),'-m','research.modal_h3_live_function','--status',str(d/'carrier')],capture_output=True,text=True,timeout=135)
            if control.returncode:raise RuntimeError('Modal control-plane status unavailable; no fresh capture')
            carrier=json.loads(control.stdout.splitlines()[-1])
            if carrier.get('full_reservation_retained') is False:return 'ERROR' if carrier.get('status')=='ERROR' else 'COMPLETED'
            return 'RUNNING'
        return api.inspect_job(job_id=submission['job_id'],namespace='YUGOROU').status.stage
    def fetch(name):
        info=api.repo_info(repo,repo_type='dataset')
        if not info.private:raise ValueError('private live session required')
        if name not in api.list_repo_files(repo,repo_type='dataset',revision=info.sha):return None
        path=hf_hub_download(repo,name,repo_type='dataset',revision=info.sha,token=cached_token())
        return json.loads(Path(path).read_text())
    def state():
        value=fetch('live/state.json')
        if value and value.get('session_id')!=session:raise ValueError('wrong session')
        return value
    deadline=time.monotonic()+a.wait_seconds
    while True:
        value=state()
        if value or time.monotonic()>=deadline:break
        stage=job_stage()
        if stage in ('ERROR','COMPLETED','CANCELED','DELETED'):
            print(json.dumps(dict(status=stage,job_id=submission.get('job_id'))));return
        time.sleep(3)
    if a.state:
        print(json.dumps(value or dict(status='preparing',job_id=submission['job_id'])));return
    minimum_window=45 if spec['config']['timeout_seconds']<=720 else 90
    if not value or value.get('status') not in ('ready','output') or value.get('expires_at',0)-time.time()<minimum_window:raise ValueError('fresh live-input window unavailable; no capture started')
    if a.obs_receipt is None:p.error('--turn needs existing --obs-receipt')
    obs=json.loads(a.obs_receipt.read_text())
    if obs['host']!='127.0.0.1' or not 1024<=obs['port']<=65535 or obs.get('audio_monitor_device')!='BlackHole2ch_UID':raise ValueError('generated AV route differs')
    if str(obs['pid']) not in subprocess.check_output(['pgrep','-x','OBS'],text=True).split():raise ValueError('prepared OBS process not running')
    with ExitStack() as owners, (d/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=sorted(d.glob('turn-*/delivered.json'));sequence=len(completed)+1
        if sequence>spec['live_voice_session']['max_turns']:raise ValueError('finite turn limit')
        turn=d/f'turn-{sequence:03d}';turn.mkdir(mode=0o700)
        time.sleep(a.countdown)
        previous=[]
        if completed:
            before=completed[-1].parent
            previous=[dict(user=json.loads((before/'asr.json').read_text())['text'],assistant=json.loads((before/'proposal.json').read_text())['reply'])]
        def propose(utterance):
            text=('会話履歴:'+json.dumps(previous,ensure_ascii=False)+'\n' if previous else '')+'現在の通話相手の発話:'+utterance+'\n日本語15〜25文字程度の短い自然な一文で返答してください。'
            if len(text)>500:raise ValueError('bounded context exceeded; no silent truncation')
            llm=subprocess.run(['python3','-m','research.run_h3_director','--run','--text',text],capture_output=True,text=True,timeout=150)
            if llm.returncode:raise RuntimeError('Pi failed; no model fallback')
            return next(json.loads(line) for line in llm.stdout.splitlines() if json.loads(line).get('type')=='turn')
        policy=spec.get('asr_policy','apple-final')
        if policy=='application-stream-v1':
            from research.h3_personal.streaming_asr import StreamingASR
            stream=owners.enter_context(StreamingASR(propose))
            try:
                asr,proposal=stream.capture(['/tmp/h3-comet-speech-stream','--stream-turn'])
            except Exception as exc:
                write(turn/'capture-or-proposal-failure.json',dict(asr_policy=policy,error_type=type(exc).__name__))
                raise
        elif policy=='apple-final':
            capture=subprocess.run(['/tmp/h3-comet-speech','--listen-turn'],capture_output=True,text=True,timeout=55)
            if capture.returncode:
                write(turn/'asr-failure.json',dict(exit_code=capture.returncode,output=capture.stdout))
                raise RuntimeError('ASR failed; '+capture.stdout.strip())
            rows=[json.loads(line) for line in capture.stdout.splitlines()]
            asr=next(r for r in rows if r.get('type')=='transcript')
            if not asr.get('final') or not asr.get('on_device') or not asr.get('capture_stopped') or not 1<=len(asr['text'])<=160:raise ValueError('fresh final short call utterance required')
            proposal=propose(asr['text'])
        else:raise ValueError('unknown immutable ASR policy')
        write(turn/'asr.json',asr)
        write(turn/'proposal.json',proposal)
        row=dict(schema='h3-live-voice-input-v1',session_id=session,sequence=sequence,origin='discord_received_audio_ondevice_asr',asr_final=asr['final'],asr_stopped=True,director_model=proposal['model'],reply=proposal['reply'],asr_sha256=hashlib.sha256(asr['text'].encode()).hexdigest(),expires_at=time.time()+45)
        if policy=='application-stream-v1':
            row.update(asr_endpoint_source=asr['endpoint_source'],asr_revision=asr['revision'])
        # The immutable backend validates policy; never forge Apple-final to reuse an old job.
        from research.hf_h3_live_voice import validate_request
        validate_request(row,session,sequence,policy)
        if value.get('expires_at',0)-time.time()<65 or job_stage()!='RUNNING':raise ValueError('GPU window closed after ASR/Pi; no upload')
        api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=f'live/input-{sequence:03d}.json',path_or_fileobj=BytesIO(json.dumps(row).encode()))],commit_message='Fresh policy-labelled call ASR-directed turn; no raw capture')
        wait=time.monotonic()+65;result=None
        while time.monotonic()<wait:
            result=fetch(f'live/turn-{sequence:03d}/result.json')
            if result:break
            if job_stage() in ('ERROR','CANCELED','DELETED'):raise RuntimeError('live GPU terminated')
            time.sleep(1)
        if not result or result['session_id']!=session or result['sequence']!=sequence or result['reply_sha256']!=hashlib.sha256(proposal['reply'].encode()).hexdigest() or result['previous_audio_reused']:raise RuntimeError('no matching fresh generated output')
        info=api.repo_info(repo,repo_type='dataset');revision=info.sha
        name=f'live/turn-{sequence:03d}/review.mp4'
        remote=Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=cached_token()));data=remote.read_bytes()
        if hashlib.sha256(data).hexdigest()!=result['files'][name]:raise ValueError('generated AV SHA differs')
        media=turn/'reply.mp4'
        with media.open('xb') as f:f.write(data)
        media.chmod(0o600)
        destination=f"udp://127.0.0.1:{obs['port']}?pkt_size=1316"
        subprocess.run(['ffmpeg','-nostdin','-v','error','-re','-i',str(media),'-map','0:v:0','-map','0:a:0','-c:v','copy','-bsf:v','h264_mp4toannexb','-c:a','aac','-b:a','128k','-f','mpegts','-mpegts_flags','+resend_headers','-muxdelay','0',destination],check=True,timeout=35)
        receipt=dict(sequence=sequence,asr=asr['text'],asr_policy=policy,apple_final=asr['final'],reply=proposal['reply'],fresh_audio_and_video=True,output_revision=revision,generation_seconds=result['generated_seconds'],sent_to_OBS=True,discord_reception_verified=False,realtime_verified=False)
        write(turn/'delivered.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
