"""Actual remote RTC client in isolated environment. No device/playback access."""
import asyncio
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import time


def candidate_summary(description):
    result={'count':0,'global_host':0,'private_host':0,'srflx':0,'relay':0}
    for line in description['sdp'].splitlines():
        if not line.startswith('a=candidate:'):continue
        fields=line.split();result['count']+=1
        if len(fields)<8:raise ValueError('candidate syntax')
        kind=fields[7]
        if kind=='host':
            try:global_ip=ipaddress.ip_address(fields[4]).is_global
            except ValueError:global_ip=False
            result['global_host' if global_ip else 'private_host']+=1
        elif kind in ('srflx','relay'):result[kind]+=1
    return result


def request(host,path,token,key,deadline,value=None,*,session_id=None):
    left=deadline-time.monotonic()
    if left<=0:raise TimeoutError('signaling deadline')
    c=http.client.HTTPSConnection(host,timeout=min(10,left))
    try:
        body=None if value is None else json.dumps(value,allow_nan=False).encode()
        headers={'Authorization':'Bearer '+token,'X-H3-RTC-Key':key,
                 'Content-Type':'application/json','Accept-Encoding':'identity',
                 'Content-Length':str(0 if body is None else len(body))}
        if session_id is not None:headers['X-H3-Session']=session_id
        c.request('GET' if value is None else 'POST',path,body=body,headers=headers)
        response=c.getresponse();body=response.read(128*1024+1)
        if response.status!=200 or len(body)>128*1024 or time.monotonic()>deadline:raise RuntimeError('signaling response failed')
        return json.loads(body)
    finally:c.close()


async def run(config,token,key,*,turn_config=None):
    from .webrtc_live_peer import Session
    from .webrtc_ice import HOST_ONLY, ice_urls
    profile=config.get('ice_profile',HOST_ONLY);urls=ice_urls(profile)
    job=config['job_id']
    if not re.fullmatch('[0-9a-f]{24}',job):raise ValueError('exact owned Job required')
    host=job+'--7860.hf.jobs';deadline=time.monotonic()+min(45,config['expires_at']-time.time()-10)
    result=dict(schema='h3-rtc-wan-connect-v1',connected=False,media_received=False,device_access=False,
        private_media_used=False,playout_verified=False,source_tail_flushed=False,
        ice_profile=profile,external_ice_servers=urls)
    tasks=[];counts={kind:{'frames':0,'samples':0} for kind in ('audio','video')}
    async def consume(track):
        while True:
            frame=await track.recv()
            counts[track.kind]['frames']+=1
            if track.kind=='audio':counts[track.kind]['samples']+=frame.samples
    try:
        async with Session(deadline=deadline,receive_audio=True,receive_video=True,control=True,ice_profile=profile,turn_config=turn_config) as peer:
            result['stage']='offer'
            offer=await peer.create_offer();result['local_candidates']=candidate_summary(offer)
            answer=await asyncio.to_thread(request,host,'/offer',token,key,deadline,
                dict(session=config['session'],boot=config['boot'],description=offer))
            if answer.get('session')!=config['session'] or answer.get('boot')!=config['boot']:raise ValueError('answer binding')
            result['remote_candidates']=candidate_summary(answer['description'])
            await peer.set_answer(answer['description']);result['stage']='ice_dtls'
            async with asyncio.timeout(30):
                await peer.wait_connected();await peer.wait_control()
            result['connected']=True;result['stage']='media'
            for _ in range(2):
                track=await peer.receive_track();tasks.append(asyncio.create_task(consume(track)))
            await peer.send('start')
            if await peer.receive()!='started':raise ValueError('control start')
            await asyncio.sleep(4.5)
            result['received']=counts
            if any(counts[kind]['frames']==0 for kind in counts):raise ValueError('both media tracks required')
            result['media_received']=True
            await peer.send('stop');result['stage']='complete'
    except Exception as error:
        result['error_type']=type(error).__name__
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());token=os.environ.pop('H3_RTC_HF_TOKEN');key=os.environ.pop('H3_RTC_KEY')
    from .webrtc_ice import HF_CLOUDFLARE_TURN_TLS
    from .webrtc_turn import validate_turn_config
    turn=(validate_turn_config(json.loads(os.environ.pop('H3_RTC_ICE_CREDENTIALS')))
          if config.get('ice_profile')==HF_CLOUDFLARE_TURN_TLS else None)
    result=asyncio.run(run(config,token,key,turn_config=turn));path=Path(sys.argv[1]).with_name('connection.json')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False),flush=True)
    return 0 if result['connected'] and result['media_received'] and result.get('stage')=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
