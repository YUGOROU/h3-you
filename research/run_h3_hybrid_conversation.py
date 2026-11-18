"""Hybrid conversation composition. CLI is inspection-only, no implicit live run.

Live API requires explicit browser and independent source placement callbacks.
Mac↔ASR/H3 is WebRTC; native Vox L4 Job→H3 uses Network Group TLS/TCP.
"""
import json
from pathlib import Path


def run(directory,*,on_browser,source_placement_gate,turns=None,capture_comet=False,asr_parent=None):
    if not callable(on_browser) or not callable(source_placement_gate):raise ValueError('explicit attended output and placement gates required')
    if type(capture_comet) is not bool or capture_comet!=(asr_parent is not None):raise ValueError('explicit paired fresh capture/ASR parent required')
    if capture_comet:
        if turns is not None:raise ValueError('no replay/typed input in fresh ASR mode')
        from research.h3_personal.asr_rtc_factory import ASRRTCFactory
        factory=ASRRTCFactory()  # Existing isolated environment; no install/capture.
    else:
        if type(turns) not in (list,tuple) or len(turns)!=2 or any(type(t) is not str or not t.strip() or len(t.encode())>2000 for t in turns):
            raise ValueError('two fresh typed turns or explicit fresh ASR required')
        factory=None
    from research.run_h3_network_pair import run as pair
    from research.launch_h3_direct_conversation import run as direct
    def conversation(root,config):
        return direct(Path(root)/'h3',turns,tts_mode='native_voxcpm',native_tts=config,
            conversation_anchor=True,overlap_send=True,http_keepalive=True,five_minute_test=True,
            media_transport='webrtc-rtp-v1',on_browser=on_browser,capture_comet=capture_comet,asr_parent=asr_parent,
            **({'asr_rtc_factory':factory} if factory is not None else {}))
    return pair(directory,conversation=conversation,source_placement_gate=source_placement_gate)

if __name__=='__main__':
    print(json.dumps(dict(scope='offline',gpu_started=False,capture_started=False,browser_opened=False,
        live_acceptance=False,mac_asr='WebRTC exact DataChannel',tts_h3='Network Group TLS/native float32',
        h3_mac='existing WebRTC H264/Opus',requires='explicit placement and per-operation attended gates')))
