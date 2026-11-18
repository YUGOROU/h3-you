"""Attended RTC output candidate. No capture, app launch, or HTTP media here."""
import json
import time
from pathlib import Path
from .live_relay_commands import Commands
from .webrtc_browser_proxy import BrowserProxy
from .webrtc_connect_client import request
from .webrtc_turn import fetch_turn_credentials
from ..run_h3_relay_conversation import proposal_reply, _asr_input, MODEL


def collect_delivery(api,job,session,deadline):
    while time.monotonic()<deadline:
        for n,line in enumerate(api.fetch_job_logs(job_id=job.id,namespace='YUGOROU')):
            if n>=10000 or time.monotonic()>=deadline:raise TimeoutError('RTC delivery receipt bound')
            if len(line)>65536:raise ValueError('RTC log bound')
            try:row=json.loads(line)
            except (ValueError,TypeError):continue
            if type(row) is dict and row.get('type')=='h3_rtc_delivery':
                fields={'source_complete','media_complete','complete','closing_tail_unverified','local_ipc_source_verified','remote_original_pcm_verified'}
                if set(row)!=fields|{'type','session'} or row['session']!=session or any(type(row[k]) is not bool for k in fields):
                    raise ValueError('RTC receipt binding')
                if not row['source_complete'] or not row['local_ipc_source_verified'] or row['remote_original_pcm_verified']:
                    raise ValueError('RTC source/codec receipt contradiction')
                if row['complete']!=row['media_complete'] or row['closing_tail_unverified']==row['media_complete']:
                    raise ValueError('RTC completion contradiction')
                return row
        time.sleep(.2)
    raise TimeoutError('RTC delivery receipt unavailable')


def run_conversation(consumer,binding,root,turns,director,*,ready,token,key,job_id,
                     on_browser,completion,input_source=None):
    """Caller owns attestation/resources and explicitly authorizes UI/capture.

    on_browser receives the ephemeral URL privately. input_source is the existing
    fresh-source callable, never an implicit microphone or historical replay.
    HTTP is text/control/signaling only on this output path; upstream ASR/TTS
    media remains separate and this must not be called all-media WebRTC.
    """
    if not callable(on_browser) or not callable(completion):raise ValueError('explicit operator and completion callbacks required')
    if input_source is None:
        if type(turns) not in (list,tuple) or not 1<=len(turns)<=2 or any(type(t) is not str or not t.strip() or len(t.encode())>2000 for t in turns):
            raise ValueError('fresh bounded typed turns required')
    elif turns is not None:raise ValueError('ASR cannot reuse typed turns')
    deadline=time.monotonic()+min(300,ready['expires_at']-time.time()-10)
    if deadline<=time.monotonic():raise TimeoutError('RTC lease margin')
    host=job_id+'--7860.hf.jobs'
    status=request(host,'/rtc/status',token,key,min(deadline,time.monotonic()+10),session_id=binding.session)
    if (status.get('session')!=binding.session or status.get('boot')!=ready['rtc_boot']
            or status.get('ice_profile')!=ready['ice_profile']
            or status.get('expires_at')!=ready['expires_at']):raise ValueError('RTC gateway identity mismatch')
    turn=fetch_turn_credentials(token,ttl=600)
    config=dict(job_id=job_id,session=binding.session,boot=ready['rtc_boot'],
                expires_at=ready['expires_at'],ice_profile=ready['ice_profile'])
    commands=Commands(consumer,binding.session)
    def guard(proxy):
        if time.monotonic()>=deadline:raise TimeoutError('finite RTC conversation expired')
        if any(e['event'] in ('error','stop') for e in proxy.receipts):raise RuntimeError('receiver stopped; no replay')
    with BrowserProxy(config,hf_token=token,app_key=key,owner_attested=True,
                      turn_config=turn,media_binding=binding,signaling_path='/rtc/offer') as proxy:
        on_browser(proxy.url)
        connect_deadline=min(deadline,time.monotonic()+90)
        while not any(e['event']=='frame-presented' for e in proxy.receipts):
            guard(proxy)
            if time.monotonic()>=connect_deadline:raise TimeoutError('browser presentation timeout')
            time.sleep(.05)
        director.deadline=min(director.deadline,deadline)
        count=0
        for index in range(2 if input_source is not None else len(turns)):
            guard(proxy)
            provenance=dict(schema='h3-relay-typed-turn-v1',origin='fresh_typed_text',asr_used=False)
            if input_source is not None:
                item=input_source(context=dict(session=binding.session,epoch=index+1),deadline=deadline)
                guard(proxy)
                if item is None:break
                from .asr_source_contract import asr_origin
                text,asr=_asr_input(item)
                provenance=dict(schema='h3-relay-asr-turn-v1',origin=asr_origin(asr),asr_used=True,asr=asr)
            else:text=turns[index]
            reply=proposal_reply(director.turn(text));guard(proxy)
            commands.send('turn',index+1,dict(**provenance,director_model=MODEL,reasoning_effort='none',reply=reply,previous_audio_reused=False))
            count+=1
            note=getattr(input_source,'note_reply_published',None)
            if callable(note):note()
        commands.send('stop',count,{})
        delivery=completion(deadline)
        result=dict(delivery,turns=count,input_kind='fresh_asr' if input_source else 'fresh_typed_text',
                    browser_events=list(proxy.receipts),all_media_webrtc=False,audibility_verified=False)
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        (root/'rtc-result.json').write_text(json.dumps(result,allow_nan=False,indent=2))
        return result
