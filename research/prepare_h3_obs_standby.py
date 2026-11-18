"""Prepare a NEW empty OBS standby scene/profile; no launch/capture/playback.

Existing collections/profiles/global credentials are never overwritten. Launch
is a separate, explicitly approved operator operation. No UDP input/audio monitor
or virtual camera is configured by this preparation.
"""
import json
from pathlib import Path
import uuid

PROFILE_ID='H3E2EStandby20261025'
NAME='H3 E2E Standby 20261025'
SCENE='E2E standby - NO INPUT OR PLAYBACK'
AUDIO_KEYS=('DesktopAudioDevice1','DesktopAudioDevice2','AuxAudioDevice1',
            'AuxAudioDevice2','AuxAudioDevice3','AuxAudioDevice4')


def collection():
    scene=dict(name=SCENE,uuid=str(uuid.uuid4()),id='scene',versioned_id='scene',
               settings=dict(id_counter=0,items=[]),muted=True,volume=0.,mixers=0,monitoring_type=0)
    value=dict(name=NAME,current_scene=SCENE,current_program_scene=SCENE,sources=[scene],
               scene_order=[dict(name=SCENE)],groups=[],transitions=[],quick_transitions=[],
               transition_duration=0,modules={},saved_projectors=[])
    value.update({key:None for key in AUDIO_KEYS})
    return value


def launch_arguments():
    return ['--safe-mode','--profile',NAME,'--collection',NAME,'--scene',SCENE]


def prepare(root):
    root=Path(root).expanduser().resolve(strict=True)
    profile=root/'basic/profiles'/PROFILE_ID
    scene=root/'basic/scenes'/f'{PROFILE_ID}.json'
    if profile.exists() or profile.is_symlink() or scene.exists() or scene.is_symlink():
        raise ValueError('refuse to overwrite existing standby configuration')
    if not scene.parent.is_dir() or not profile.parent.is_dir():
        raise ValueError('existing OBS configuration layout required')
    profile.mkdir(mode=0o700)
    config=(f'[General]\nName={NAME}\n\n[Video]\nBaseCX=252\nBaseCY=448\n'
            'OutputCX=252\nOutputCY=448\nFPSType=1\nFPSCommon=24\nFPSInt=24\nFPSNum=24\nFPSDen=1\n'
            '\n[Audio]\nSampleRate=48000\nChannelSetup=Stereo\n\n[Output]\nMode=Simple\n')
    files=[(profile/'basic.ini',config),(scene,json.dumps(collection(),indent=2)+'\n')]
    for path,text in files:
        with path.open('x') as f:f.write(text)
        path.chmod(0o600)
    return dict(profile=NAME,collection=NAME,scene=SCENE,source_types=['scene'],
                global_audio_sources_disabled=True,media_playback=False,virtual_camera_started=False,
                launched=False,launch_arguments=launch_arguments())
