"""Finite existing-L4-Endpoint native RTC acceptance. Offline unless --run.

No H3, capture, browser, playback or calls. Native floats stay private; HTTP is
control/signaling only outside the container. Original model/reference unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from research.h3_personal.voxcpm_rtc_lease import lease,PRIVATE
from research.h3_personal.voxcpm_rtc_factory import VoxRTCFactory
from research.h3_personal.webrtc_upstream_adapter import RemoteVoxCPMRTC
from research.run_h3_irodori_endpoint import persist

TEXTS=('接続を確認しています。落ち着いて、ゆっくりお話しします。','準備ができました。今日もよろしくお願いします。')


def validate_source(config,evidence,turns):
    if evidence.get('binding')!=dict(config['pins'],session=config['session'],boot=config['boot'],expires_at=config['expires_at'],epoch=0):
        raise ValueError('native evidence binding')
    runtime=evidence.get('evidence',{}).get('runtime',{})
    if (runtime.get('device_name')!='NVIDIA L4' or type(runtime.get('device_count')) is not int
            or runtime['device_count']!=1 or runtime.get('optimize') is not False
            or runtime.get('sample_rate')!=48000 or not str(runtime.get('torch','')).startswith('2.8.0')):
        raise ValueError('original native runtime evidence required')
    completions=evidence['evidence'].get('completions',[])
    if len(completions)!=2 or len(turns)!=2:raise ValueError('two native completions required')
    for turn in turns:
        matches=[r for r in completions if r.get('epoch')==turn['epoch']]
        if (len(matches)!=1 or matches[0].get('samples')!=turn['samples']
                or matches[0].get('native_float_sha256')!=turn['sha256']
                or not turn['carrier'] or turn['carrier'].get('transport_complete') is not True):
            raise ValueError('native source/carrier evidence mismatch')
        c=matches[0].get('completion',{})
        if (c.get('reason')!='natural_stop' or c.get('natural_stop') is not True
                or c.get('stream_exhausted') is not True or type(c.get('stop_index')) is not int
                or type(c.get('effective_max_len')) is not int or not 0<=c['stop_index']<c['effective_max_len']-1
                or type(c.get('decision_monotonic_ns')) is not int):
            raise ValueError('independent natural stop required')


def run(directory):
    root=Path(os.path.abspath(directory))
    if root.exists() or not root.is_relative_to(PRIVATE):raise ValueError('fresh private directory required')
    root.mkdir(mode=0o700);result=dict(schema='h3-voxcpm-rtc-acceptance-v1',complete=False,device=False,h3=False,playback=False,turns=[])
    client=None
    try:
        with lease(root/'lease') as owner:
            c=owner['config'];factory=VoxRTCFactory()
            def stream(text,epoch,deadline):
                return factory.open_stream(endpoint=c['url'],token=owner['token'],key=owner['key'],
                    session=c['session'],boot=c['boot'],expires_at=c['expires_at'],text=text,epoch=epoch,deadline=deadline)
            client=RemoteVoxCPMRTC(c['url'],open_stream=stream,token=owner['token'],key=owner['key'],
                session=c['session'],boot=c['boot'],expires_at=c['expires_at'])
            client.read('tts_ready',time.monotonic()+30)
            for epoch,text in enumerate(TEXTS,1):
                digest=hashlib.sha256();samples=0;started=time.monotonic();first=None
                path=root/f'native-{epoch}.float32le'
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'wb') as output:
                    for pcm in client.stream(text,epoch,time.monotonic()+120):
                        if first is None:first=time.monotonic()-started
                        digest.update(pcm);samples+=len(pcm)//4;output.write(pcm)
                result['turns'].append(dict(epoch=epoch,samples=samples,sha256=digest.hexdigest(),
                    first_pcm_seconds=first,elapsed_seconds=time.monotonic()-started,carrier=factory.last_receipt))
            evidence=client._request('GET','/evidence',None,time.monotonic()+15)
            persist(root/'source-evidence.json',evidence)
            validate_source(c,evidence,result['turns'])
            result['source_complete']=True
        result['cleanup_confirmed']=True
        result['complete']=True
    finally:
        if client is not None:client.close()
        persist(root/'result.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--directory',type=Path)
    args=p.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(scope='offline',gpu_started=False,endpoint_updated=False,device=False,transport='webrtc-exact-v1')));return
    if args.directory is None:raise SystemExit('--directory required')
    def interrupted(*_):raise InterruptedError('finite RTC acceptance interrupted')
    old=signal.signal(signal.SIGTERM,interrupted);alarm=signal.signal(signal.SIGALRM,interrupted);signal.alarm(1500)
    try:
        result=run(args.directory)
        print(json.dumps(result,ensure_ascii=False))
    finally:signal.alarm(0);signal.signal(signal.SIGTERM,old);signal.signal(signal.SIGALRM,alarm)

if __name__=='__main__':main()
