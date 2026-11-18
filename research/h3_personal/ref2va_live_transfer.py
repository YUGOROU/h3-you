"""Generated-only incremental HF transfer. No playback, capture, or Discord API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

NAME=re.compile(r'segment-00[0-3]\.mp4\Z')


def publish(repo,path):
    from huggingface_hub import HfApi,CommitOperationAdd
    path=Path(path)
    if not NAME.fullmatch(path.name) or path.is_symlink() or not path.is_file():raise ValueError('generated segment file required')
    api=HfApi()
    if not api.repo_info(repo,repo_type='dataset').private:raise ValueError('private publication required')
    result=api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo='outputs/'+path.name,path_or_fileobj=str(path))],commit_message='Generated segment available during GPU execution; no capture or inputs')
    print(json.dumps(dict(name=path.name,revision=result.oid,sha256=hashlib.sha256(path.read_bytes()).hexdigest())),flush=True)


def publisher(repo):
    def send(encoded,cancel):
        if cancel.is_set():raise RuntimeError('incremental publication canceled')
        env=dict(os.environ)
        # Online SDK only in a model-free child; model parent's offline flags stay set.
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):env.pop(key,None)
        result=subprocess.run([sys.executable,'-m','research.h3_personal.ref2va_live_transfer','publish','--repo',repo,'--path',encoded['path']],env=env,capture_output=True,text=True,timeout=90)
        if result.returncode:raise RuntimeError('incremental publication failed (no automatic retry)')
        return dict(encoded,remote_publication=json.loads(result.stdout))
    return send


def receive(repo,output,deadline_seconds):
    import signal
    from huggingface_hub import HfApi,hf_hub_download
    from types import SimpleNamespace as NS
    from research.run_h3_realtime_first import validate_packets
    root=Path(output).resolve();root.mkdir(mode=0o700,exist_ok=False)
    def expired(*args):raise TimeoutError('finite receive deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(deadline_seconds)
    from research.provision_modal_identity_secret import cached_token
    token=cached_token();api=HfApi(token=token);seen={};started=time.monotonic();end=started+deadline_seconds
    while time.monotonic()<end:
        info=api.repo_info(repo,repo_type='dataset',files_metadata=True)
        if not info.private:raise ValueError('private source required')
        for row in info.siblings:
            name=Path(row.rfilename).name
            if row.rfilename!='outputs/'+name or not NAME.fullmatch(name) or name in seen:continue
            src=Path(hf_hub_download(repo,row.rfilename,repo_type='dataset',revision=info.sha,token=token));body=src.read_bytes()
            digest=hashlib.sha256(body).hexdigest() if row.lfs else hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
            if digest!=(row.lfs.sha256 if row.lfs else row.blob_id):raise ValueError('remote hash mismatch')
            path=root/name
            with path.open('xb') as f:f.write(body)
            path.chmod(0o600);index=int(name[8:11]);validate_packets(path,NS(pts_start=index*124,newly_emitted_frames=124))
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,timeout=30)
            seen[name]=dict(revision=info.sha,sha256=hashlib.sha256(body).hexdigest(),received_elapsed_seconds=time.monotonic()-started,pts_full_decode_verified=True)
            (root/'receive-receipt.json').write_text(json.dumps(dict(segments=seen,playback_started=False,capture_accessed=False,discord_connected=False),indent=2)+'\n')
            print('RECEIVED',name,info.sha,flush=True)
        if len(seen)==4:signal.alarm(0);return
        time.sleep(2)
    raise TimeoutError('finite receive deadline reached')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('publish','receive'));p.add_argument('--repo',required=True);p.add_argument('--path',required=True);p.add_argument('--deadline-seconds',type=int,default=1200);a=p.parse_args()
    if not 1<=a.deadline_seconds<=1500:raise ValueError('finite receive timeout required')
    if a.action=='publish':publish(a.repo,a.path)
    else:receive(a.repo,a.path,a.deadline_seconds)

if __name__=='__main__':main()
