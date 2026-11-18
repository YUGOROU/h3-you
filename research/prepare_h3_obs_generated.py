"""Prepare a fresh OBS profile/scene with generated media only. Never launches OBS.

OBS 32.2.2: absent/null global audio objects make LoadAudioDevice return;
General.FirstRun=true avoids CreateFirstRunSources on fallback. No capture sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

SHA='385f3a24e578403b8255f0f42e8504e290bd1af2ddfc39d8e64e1501df141253'
COLLECTION='H3 Generated Only'
PROFILE='H3 Sensors Off'
SCENE='Verified generated clip - SENSORS OFF - not live inference'
AUDIO_KEYS=('DesktopAudioDevice1','DesktopAudioDevice2','AuxAudioDevice1','AuxAudioDevice2','AuxAudioDevice3','AuxAudioDevice4')


def collection(media):
    source_id=str(uuid.uuid4());scene_id=str(uuid.uuid4())
    source=dict(name='Verified silent generated video',uuid=source_id,id='ffmpeg_source',versioned_id='ffmpeg_source',
        muted=True,volume=0.0,mixers=0,monitoring_type=0,
        settings=dict(is_local_file=True,local_file=str(media),looping=False,restart_on_activate=True,
                      clear_on_media_end=False,close_when_inactive=True,hw_decode=False,speed_percent=100))
    scene=dict(name=SCENE,uuid=scene_id,id='scene',versioned_id='scene',settings=dict(id_counter=1,
        items=[dict(name=source['name'],source_uuid=source_id,id=1,visible=True,locked=True,
                    pos=dict(x=0.0,y=0.0),scale=dict(x=1.0,y=1.0),rot=0.0,align=5,
                    bounds_type=0,bounds_align=0,bounds=dict(x=0.0,y=0.0))]))
    data=dict(name=COLLECTION,current_scene=SCENE,current_program_scene=SCENE,
              sources=[source,scene],scene_order=[dict(name=SCENE)],groups=[],transitions=[],
              quick_transitions=[],transition_duration=0,modules={},saved_projectors=[])
    data.update({key:None for key in AUDIO_KEYS})
    return data


def prepare(media,root):
    media=Path(media).resolve(strict=True);root=Path(root).expanduser()
    if root.exists() or root.is_symlink():raise ValueError('refuse to overwrite existing OBS configuration')
    if hashlib.sha256(media.read_bytes()).hexdigest()!=SHA:raise ValueError('accepted generated video SHA differs')
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(media)],timeout=15))['streams']
    if len(streams)!=1 or streams[0]['codec_type']!='video' or (streams[0]['width'],streams[0]['height'])!=(252,448):raise ValueError('silent portrait video only')
    if subprocess.run(['pgrep','-x','OBS'],capture_output=True).returncode==0:raise RuntimeError('OBS already running')
    root.mkdir(mode=0o700)
    scene_dir=root/'basic/scenes';profile_dir=root/'basic/profiles/H3SensorsOff'
    scene_dir.mkdir(parents=True);profile_dir.mkdir(parents=True)
    files={root/'global.ini':'[General]\nMacOSPermissionsDialogLastShown=1\nEnableAutoUpdates=false\n',
        root/'user.ini':f'[General]\nFirstRun=true\n\n[Basic]\nProfile={PROFILE}\nProfileDir=H3SensorsOff\nSceneCollection={COLLECTION}\nSceneCollectionFile=H3GeneratedOnly\n\n[BasicWindow]\nPreviewEnabled=true\nStudioMode=false\n',
        profile_dir/'basic.ini':f'[General]\nName={PROFILE}\n\n[Video]\nBaseCX=252\nBaseCY=448\nOutputCX=252\nOutputCY=448\nFPSType=1\nFPSCommon=24\nFPSInt=24\nFPSNum=24\nFPSDen=1\n\n[Audio]\nSampleRate=48000\nChannelSetup=Stereo\n\n[Output]\nMode=Simple\n',
        scene_dir/'H3GeneratedOnly.json':json.dumps(collection(media),indent=2)+'\n'}
    for path,text in files.items():path.write_text(text);path.chmod(0o600)
    print(json.dumps(dict(obs_config=str(root),generated_sha256=SHA,source_types=['ffmpeg_source','scene'],global_audio_sources_disabled=True,launched=False)))


BLOCKED_MODULES=('mac-virtualcam','mac-avcapture','mac-avcapture-legacy','mac-capture','mac-syphon','aja','aja-output-ui','decklink','decklink-output-ui')


def disable_capture_modules(root):
    """32.2.2 module loader honors disabled_modules even for bundled modules.

    UI does not offer this for core modules; verified source-level config mechanism.
    No bundle modification, TCC change, or extension activation/deactivation.
    """
    if subprocess.run(['pgrep','-x','OBS'],capture_output=True).returncode==0:raise RuntimeError('stop OBS before editing module configuration')
    path=Path(root)/'plugin_manager/modules.json';path.parent.mkdir(parents=True,exist_ok=True)
    rows=json.loads(path.read_text()) if path.exists() else []
    backup=path.with_name('modules.before-sensors-off.json')
    if backup.exists():raise ValueError('module safety backup already exists')
    backup.write_text(json.dumps(rows,indent=2)+'\n');backup.chmod(0o600)
    mapping={row['module_name']:row for row in rows}
    for name in BLOCKED_MODULES:
        row=mapping.setdefault(name,dict(display_name=name,module_name=name,id='',version='',sources=[],outputs=[],encoders=[],services=[]))
        row['enabled']=False
    path.write_text(json.dumps(list(mapping.values()),indent=2)+'\n');path.chmod(0o600)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--media',type=Path,required=True);p.add_argument('--config',type=Path,default=Path.home()/'Library/Application Support/obs-studio');a=p.parse_args();prepare(a.media,a.config)

if __name__=='__main__':main()
