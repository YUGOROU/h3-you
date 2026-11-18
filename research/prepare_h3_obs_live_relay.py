"""Fresh generated-only UDP scene/profile. No launch, capture or global routes."""
import json
from pathlib import Path
import uuid

ID='H3E2ELiveRelay20261025'
NAME='H3 E2E Live Relay 20261025'
SCENE='Generated AV - HF relay - NO SENSORS'
PORT=48124


def prepare(root):
    root=Path(root)
    profile=root/'basic/profiles'/ID
    scene_path=root/'basic/scenes'/f'{ID}.json'
    if profile.exists() or scene_path.exists():raise ValueError('fresh profile/collection required')
    profile.mkdir(mode=0o700)
    (profile/'basic.ini').write_text(f'[General]\nName={NAME}\n\n[Video]\nBaseCX=252\nBaseCY=448\nOutputCX=252\nOutputCY=448\nFPSType=1\nFPSCommon=24\nFPSInt=24\nFPSNum=24\nFPSDen=1\n\n[Audio]\nSampleRate=48000\nChannelSetup=Stereo\nMonitoringDeviceName=BlackHole 2ch\nMonitoringDeviceId=BlackHole2ch_UID\n\n[Output]\nMode=Simple\n')
    sid=str(uuid.uuid4())
    source=dict(name='Generated paired AV only',uuid=sid,id='ffmpeg_source',versioned_id='ffmpeg_source',
        muted=False,volume=1.0,mixers=0,monitoring_type=1,
        settings=dict(is_local_file=False,input=f'udp://127.0.0.1:{PORT}?fifo_size=4096&overrun_nonfatal=0',
            input_format='mpegts',restart_on_activate=False,close_when_inactive=False,
            clear_on_media_end=False,hw_decode=False,buffering_mb=2,reconnect_delay_sec=1))
    scene=dict(name=SCENE,uuid=str(uuid.uuid4()),id='scene',versioned_id='scene',settings=dict(id_counter=1,
        items=[dict(name=source['name'],source_uuid=sid,id=1,visible=True,locked=True,
            pos=dict(x=0.0,y=0.0),scale=dict(x=1.0,y=1.0),rot=0.0,align=5,
            bounds_type=0,bounds_align=0,bounds=dict(x=0.0,y=0.0))]))
    data=dict(name=NAME,current_scene=SCENE,current_program_scene=SCENE,sources=[source,scene],
        scene_order=[dict(name=SCENE)],groups=[],transitions=[],quick_transitions=[],transition_duration=0,modules={},saved_projectors=[])
    for key in ('DesktopAudioDevice1','DesktopAudioDevice2','AuxAudioDevice1','AuxAudioDevice2','AuxAudioDevice3','AuxAudioDevice4'):data[key]=None
    scene_path.write_text(json.dumps(data,indent=2)+'\n')
    scene_path.chmod(0o600);(profile/'basic.ini').chmod(0o600)
    return dict(profile=NAME,collection=NAME,scene=SCENE,port=PORT,physical_sources=False)
