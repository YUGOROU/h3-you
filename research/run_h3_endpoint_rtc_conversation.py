"""Explicit finite Endpoint→H3 RTC composition; CLI inspection only.

Browser/capture are per-operation approvals supplied by the live API caller.
The accepted public-source probe does not authorize this attended operation.
"""
import json
import os
from pathlib import Path
import time
from research.h3_personal.voxcpm_rtc_lease import lease,PRIVATE
from research.run_h3_irodori_endpoint import persist

FENCE=PRIVATE/'.network-tts-unresolved.json'  # shared exclusion with old pair owner


def run(directory,*,on_browser,turns=None,capture_comet=False,asr_parent=None):
    if not callable(on_browser):raise ValueError('explicit approved browser callback required')
    if type(capture_comet) is not bool or capture_comet!=(asr_parent is not None):raise ValueError('paired fresh capture and ASR parent required')
    if capture_comet:
        if turns is not None:raise ValueError('fresh ASR forbids typed replay')
        from research.h3_personal.asr_rtc_factory import ASRRTCFactory
        factory=ASRRTCFactory()
    else:
        if type(turns) not in (list,tuple) or len(turns)!=2 or any(type(t) is not str or not t.strip() or len(t.encode())>2000 for t in turns):
            raise ValueError('two fresh typed turns or explicit ASR capture required')
        factory=None
    root=Path(os.path.abspath(directory))
    if root.exists() or not root.is_relative_to(PRIVATE):raise ValueError('fresh private directory required')
    root.mkdir(mode=0o700)
    from research.launch_h3_direct_conversation import run as direct
    from research.run_h3_rtc_conversation import validate_evidence
    from research.h3_personal.webrtc_upstream_adapter import RemoteVoxCPMRTC
    result=dict(schema='h3-endpoint-rtc-conversation-v1',complete=False,cleanup_confirmed=False,
        speech_input=capture_comet,transport='endpoint-rtc-v3',h3_launched=False)
    endpoint_closed=False;guarded=False;control_client=None
    try:
        with lease(root/'tts',min_remaining=600) as owner:
            c=owner['config']
            persist(FENCE,dict(directory=str(root),endpoint='h3-voxcpm-l4-a474fd64f740',state='endpoint-rtc-conversation-owned'))
            guarded=True
            def forbidden(*_):raise RuntimeError('control client cannot carry media')
            control_client=RemoteVoxCPMRTC(c['url'],open_stream=forbidden,token=owner['token'],key=owner['key'],
                session=c['session'],boot=c['boot'],expires_at=c['expires_at'])
            control_client.read('tts_ready',time.monotonic()+30)
            os.environ.update(H3_NATIVE_TTS_HF_TOKEN=owner['token'],H3_NATIVE_TTS_KEY=owner['key'])
            try:
                result['h3_result']=direct(root/'h3',turns,tts_mode='native_voxcpm',native_tts=c,
                    conversation_anchor=True,overlap_send=True,http_keepalive=True,five_minute_test=True,
                    media_transport='webrtc-rtp-v1',on_browser=on_browser,capture_comet=capture_comet,asr_parent=asr_parent,
                    **({'asr_rtc_factory':factory} if factory is not None else {}))
            finally:
                os.environ.pop('H3_NATIVE_TTS_HF_TOKEN',None);os.environ.pop('H3_NATIVE_TTS_KEY',None)
            evidence=control_client._request('GET','/evidence',None,time.monotonic()+15)
            persist(root/'source-evidence.json',evidence)
            result['validation']=validate_evidence(root,c,evidence,result['h3_result'])
            result['complete']=result['h3_result']['media_complete']
        endpoint_closed=True
        return result
    finally:
        if control_client is not None:control_client.close()
        h3=root/'h3';safe_h3=False
        if not (h3/'submission_guard.json').exists() and not (h3/'submission.json').exists():safe_h3=True
        elif (h3/'submission.json').exists() and (h3/'rtc-job-terminal.json').exists():
            try:
                submitted=json.loads((h3/'submission.json').read_text());terminal=json.loads((h3/'rtc-job-terminal.json').read_text())
                safe_h3=terminal['job_id']==submitted['job_id'] and terminal['status'] in ('CANCELED','ERROR','COMPLETED','DELETED')
                result['h3_launched']=True
            except (ValueError,KeyError,OSError):pass
        asr=h3/'asr';safe_asr=not asr.exists()
        if asr.exists():
            try:safe_asr=json.loads((asr/'result.json').read_text()).get('terminal_state')=='paused'
            except (ValueError,OSError):pass
        # On a primary failure, lease may still have paused successfully.
        if (root/'tts/terminal.json').exists():
            try:endpoint_closed=json.loads((root/'tts/terminal.json').read_text()).get('independent_confirmation') is True
            except (ValueError,OSError):pass
        result['cleanup_confirmed']=endpoint_closed and safe_h3 and safe_asr
        if guarded and result['cleanup_confirmed'] and FENCE.exists() and json.loads(FENCE.read_text()).get('directory')==str(root):FENCE.unlink()
        persist(root/'result.json',result)


if __name__=='__main__':
    print(json.dumps(dict(scope='offline',gpu_started=False,endpoint_updated=False,browser_opened=False,capture_started=False,
        tts='existing Vox L4 Endpoint exact RTC',h3='original H200x2 RTC downlink',attended_acceptance=False)))
