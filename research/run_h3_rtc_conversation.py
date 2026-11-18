"""Finite real H3→browser RTC candidate. Offline by default; no implicit capture.

Upstream ASR/Vox audio is still HTTP: this is NOT all-media WebRTC acceptance.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
from research.h3_personal.voxcpm_endpoint_protocol import PINS


def validate_evidence(root,config,evidence,result):
    if result.get('source_complete') is not True or result.get('turns')!=2:
        raise ValueError('two completed fresh source turns required')
    if result.get('remote_original_pcm_verified') is not False:raise ValueError('lossy RTC is not original PCM')
    if evidence.get('binding')!=dict(config['pins'],session=config['session'],boot=config['boot'],expires_at=config['expires_at'],epoch=0):
        raise ValueError('native endpoint binding')
    proof=evidence['evidence'];runtime=proof['runtime']
    if (runtime['device_name']!='NVIDIA L4' or runtime['device_count']!=1 or runtime['optimize'] is not False
            or runtime['sample_rate']!=48000 or not runtime['torch'].startswith('2.8.0')):
        raise ValueError('native runtime proof')
    remote=proof['completions'];local=json.loads((Path(root)/'h3/native-worker-evidence.json').read_text())['receipts']
    if len(remote)!=2 or len(local)!=2:raise ValueError('two native completions required')
    for epoch,(native,receipt) in enumerate(zip(remote,local),1):
        c=native['completion']
        if (native['epoch']!=epoch or receipt['epoch']!=epoch or receipt['pins']!=PINS
                or native['native_float_sha256']!=receipt['native_sha256'] or native['samples']!=receipt['native_samples']):
            raise ValueError('native source provenance mismatch')
        if (c['reason']!='natural_stop' or c['natural_stop'] is not True or c['stream_exhausted'] is not True
                or type(c['stop_index']) is not int or type(c['effective_max_len']) is not int
                or not 0<=c['stop_index']<c['effective_max_len']-1 or c['decision_monotonic_ns'] is None):
            raise ValueError('natural model stop not proved')
    return dict(native_natural_stop_verified=True,source_receipts_matched=True,
                remote_original_pcm_verified=False,media_complete=result['media_complete'],
                voice_identity_verified=False,spoken_words_verified=False,all_media_webrtc=False)


def run(directory,*,on_browser,turns=None,capture_comet=False,asr_parent=None):
    if not callable(on_browser):raise ValueError('private operator browser callback required')
    if bool(capture_comet)!=(asr_parent is not None):raise ValueError('explicit ASR parent and capture operation required')
    if capture_comet:
        if turns is not None:raise ValueError('no typed replay in ASR mode')
    elif type(turns) not in (list,tuple) or len(turns)!=2 or any(type(t) is not str or not t.strip() or len(t.encode())>2000 for t in turns):
        raise ValueError('two fresh typed turns or explicit ASR required')
    from research.run_h3_native_coupled import run as coupled
    from research.launch_h3_direct_conversation import run as direct
    def conversation(root,config):
        return direct(root/'h3',turns,tts_mode='native_voxcpm',native_tts=config,
            conversation_anchor=True,overlap_send=True,http_keepalive=True,five_minute_test=True,
            media_transport='webrtc-rtp-v1',on_browser=on_browser,
            capture_comet=capture_comet,asr_parent=asr_parent)
    return coupled(directory,rtc_conversation=conversation)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--directory',type=Path)
    p.add_argument('--open-browser',action='store_true',help='Explicit separately approved Mac browser/output operation')
    p.add_argument('--capture-comet',action='store_true',help='Explicit separately approved fresh application-audio capture')
    p.add_argument('--asr-parent',type=Path)
    a=p.parse_args(argv)
    if not a.run:
        print(json.dumps(dict(scope='offline',ready_for_attended_trial=True,live_acceptance=False,
            media='H3 downlink RTC H264/Opus',all_media_webrtc=False,turns=2,max_media_seconds=300,
            gpu_started=False,capture_started=False,browser_opened=False)))
        return 0
    if a.directory is None or not a.open_browser or sys.platform!='darwin':p.error('fresh directory and explicit --open-browser on Mac required')
    if a.capture_comet!=(a.asr_parent is not None):p.error('--capture-comet requires --asr-parent, mutually paired')
    turns=None if a.capture_comet else [input('Fresh turn 1: '),input('Fresh turn 2: ')]
    def open_browser(url):
        subprocess.run(['/usr/bin/open',url],check=True,timeout=10,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def interrupted(*_):raise InterruptedError('finite RTC owner interrupted')
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,1500)
    try:run(a.directory,on_browser=open_browser,turns=turns,capture_comet=a.capture_comet,asr_parent=a.asr_parent)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in previous.items():signal.signal(s,handler)
    return 0

if __name__=='__main__':raise SystemExit(main())
