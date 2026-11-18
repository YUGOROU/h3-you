"""Deliver exactly one fresh native Endpoint/Pi result; offline unless --run.

No capture, GPU launch, routing change, OBS launch or Discord control.
"""
import argparse
import fcntl
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import time

from research.hf_h3_live_voice import validate_request
from research.h3_personal.endpoint_asr import native_completed_text, finalized_covers


def request(result, session, now=None):
    now=time.time() if now is None else now
    asr=result['asr'];proposal=result['proposal']
    finals=[e for e in asr.get('events',[]) if e.get('final') is True]
    whole = asr.get('completion_policy') == 'input-finalized'
    if whole and (asr.get('input_committed') is not True or
                  not isinstance(asr.get('audio_sent_seconds'), (int, float)) or
                  asr['audio_sent_seconds'] <= 0 or
                  not isinstance(asr.get('finalized_audio_seconds'), (int, float)) or
                  not finalized_covers(asr['finalized_audio_seconds'], asr['audio_sent_seconds'])):
        raise ValueError('whole input finalized coverage required')
    expected_text = native_completed_text(finals) if whole else (finals[-1].get('text') if finals else None)
    if (result.get('origin')!='comet_process_audio_endpoint_asr' or result.get('error_type') or
        result.get('terminal_state')!='paused' or asr.get('final') is not True or
        asr.get('capture_stopped') is not True or asr.get('on_device') is not False or
        asr.get('cleanup_complete') is not True or asr.get('errors') or not finals or
        finals[-1].get('endpoint_source')!='native_completed' or
        expected_text!=asr.get('text') or not isinstance(asr.get('text'),str) or
        not 1<=len(asr['text'])<=160):raise ValueError('clean genuine native final required')
    row=dict(schema='h3-live-voice-input-v1',session_id=session,sequence=1,
             origin=result['origin'],asr_final=True,asr_stopped=True,asr_on_device=False,
             endpoint_source='native_completed',director_model=proposal['model'],reply=proposal['reply'],
             asr_sha256=hashlib.sha256(asr['text'].encode()).hexdigest(),expires_at=now+45)
    return validate_request(row,session,1,'endpoint-native-v1')


def verify_output(result, session, row):
    if (result.get('schema')!='h3-live-voice-state-v1' or result.get('status')!='output' or
        result.get('session_id')!=session or result.get('sequence')!=1 or
        result.get('previous_audio_reused') is not False or
        result.get('reply_sha256')!=hashlib.sha256(row['reply'].encode()).hexdigest()):
        raise ValueError('fresh bound output required')


def verify_codecs(probe):
    streams=probe.get('streams',[])
    if ([s.get('codec_name') for s in streams if s.get('codec_type')=='video']!=['h264'] or
        [s.get('codec_name') for s in streams if s.get('codec_type')=='audio']!=['alac']):
        raise ValueError('H264/ALAC required')


def run(args):
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from research.provision_modal_identity_secret import cached_token
    d=args.directory.resolve();submission_path=d/'submission.json'
    submission_bytes=submission_path.read_bytes();submission=json.loads(submission_bytes)
    spec=json.loads((d/'deployment/spec.json').read_text())
    if submission.get('provider','huggingface_jobs')!='huggingface_jobs' or not submission.get('job_id'):
        raise ValueError('HF own latest submission required')
    if spec.get('asr_policy')!='endpoint-native-v1' or spec['config']['timeout_seconds']!=660:
        raise ValueError('native finite660s session required')
    repo=spec['publication_repo'];session=spec['live_voice_session']['id']
    if not repo.startswith('YUGOROU/'):raise ValueError('owned private repo required')
    token=cached_token();api=HfApi(token=token);job_id=submission['job_id']
    with (d/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (d/'turn-001').exists():raise ValueError('one turn only; no replay')
        try:
            def snapshot():
                info=api.repo_info(repo,repo_type='dataset')
                if not info.private:raise ValueError('private repo required')
                return info,api.list_repo_files(repo,repo_type='dataset',revision=info.sha)
            def fetch(name,revision):
                return Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=token))
            info,names=snapshot()
            if any(n.startswith('live/input-') or n.startswith('live/turn-') for n in names):raise ValueError('new sequence1 required')
            ready=json.loads(fetch('live/state.json',info.sha).read_text())
            if ready.get('session_id')!=session or ready.get('status')!='ready' or ready.get('previous_audio_reused') is not False or not 65<ready.get('expires_at',0)-time.time()<=660:
                raise ValueError('ready/deadline required')
            # READY is the latest commit in a fresh repo. Both capture diagnostics
            # and completed ASR/Pi result must be newer, not copied historical text.
            ready_at=info.last_modified.timestamp()
            source=args.asr_result.resolve();capture=source.parent/'capture-metadata.jsonl'
            if min(source.stat().st_mtime,capture.stat().st_mtime)<=ready_at:raise ValueError('capture/result must be NEW after READY')
            diagnostics=[json.loads(line) for line in capture.read_text().splitlines()]
            if (not diagnostics or diagnostics[-1].get('type')!='stopped' or diagnostics[-1].get('cleanup') is not True or diagnostics[-1].get('frames',0)<=0 or diagnostics[-1].get('nonzero_samples',0)<=0):raise ValueError('stopped nonzero capture receipt required')
            row=request(json.loads(source.read_text()),session)
            obs=json.loads(args.obs_receipt.read_text())
            if obs.get('host')!='127.0.0.1' or type(obs.get('port')) is not int or not 1024<=obs['port']<=65535 or obs.get('audio_monitor_device')!='BlackHole2ch_UID':raise ValueError('existing OBS route required')
            if str(obs['pid']) not in subprocess.check_output(['pgrep','-x','OBS'],text=True,timeout=5).split():raise ValueError('existing OBS not running')
            if api.inspect_job(job_id=job_id,namespace='YUGOROU').status.stage!='RUNNING':raise ValueError('own job not running')
            current,current_names=snapshot()
            if current.sha!=info.sha or submission_path.read_bytes()!=submission_bytes:raise ValueError('session changed before upload')
            if ready['expires_at']-time.time()<65:raise ValueError('deadline closed before upload')
            validate_request(row,session,1,'endpoint-native-v1')
            turn=d/'turn-001';turn.mkdir(mode=0o700)
            api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,operations=[CommitOperationAdd(path_in_repo='live/input-001.json',path_or_fileobj=BytesIO(json.dumps(row).encode()))],commit_message='NEW native Endpoint/Pi sequence1 after READY')
            end=time.monotonic()+min(65,ready['expires_at']-time.time()-5)
            while time.monotonic()<end:
                output_info,output_names=snapshot()
                name='live/turn-001/result.json'
                if name in output_names:
                    result=json.loads(fetch(name,output_info.sha).read_text());break
                if api.inspect_job(job_id=job_id,namespace='YUGOROU').status.stage!='RUNNING':raise RuntimeError('own job terminated')
                time.sleep(1)
            else:raise TimeoutError('bounded output wait')
            if output_info.sha==info.sha:raise ValueError('old output revision')
            verify_output(result,session,row)
            name='live/turn-001/review.mp4';data=fetch(name,output_info.sha).read_bytes()
            if hashlib.sha256(data).hexdigest()!=result['files'][name]:raise ValueError('AV SHA differs')
            media=turn/'reply.mp4'
            with media.open('xb') as f:f.write(data)
            media.chmod(0o600)
            probe=subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(media)],check=True,capture_output=True,text=True,timeout=10)
            verify_codecs(json.loads(probe.stdout))
            subprocess.run(['ffmpeg','-nostdin','-v','error','-re','-i',str(media),'-map','0:v:0','-map','0:a:0','-c:v','copy','-bsf:v','h264_mp4toannexb','-c:a','aac','-b:a','128k','-f','mpegts','-mpegts_flags','+resend_headers','-muxdelay','0',f"udp://127.0.0.1:{obs['port']}?pkt_size=1316"],check=True,timeout=35)
            receipt=dict(sequence=1,session_id=session,output_revision=output_info.sha,sent_to_OBS=True,discord_reception_verified=False,realtime_verified=False)
            with (turn/'delivered.json').open('x') as f:json.dump(receipt,f)
            print(json.dumps(receipt))
        finally:
            # Never enumerate/cancel other Jobs, nor target a replaced submission.
            if submission_path.read_bytes()==submission_bytes:
                api.cancel_job(job_id=job_id,namespace='YUGOROU')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--asr-result',type=Path,required=True)
    p.add_argument('--obs-receipt',type=Path,required=True)
    p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:
        print(json.dumps(dict(mode='offline',policy='endpoint-native-v1',sequence=1,max_turns=1,network=False)))
        return
    run(args)


if __name__=='__main__':main()
