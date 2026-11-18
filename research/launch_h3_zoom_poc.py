"""Attended Zoom PoC: reuse direct HTTP/unified OBS+BlackHole. Offline default.

No Zoom/OBS UI automation/device routing. --run is NOT permission for agent to
perform those operations; operator must approve each capture/playback operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()
DEFAULT_REFERENCE=PRIVATE/'gpu-hf-irodori-diagnostic-v1/reference.wav'


def build_input(*,model_dir,reference=DEFAULT_REFERENCE,reference_offset=0,
                all_participants_consented=False,confirmation=None):
    """Local file/ONNX preflight only; never creates capture/endpoint/Job."""
    from research.h3_personal.director_tts_job import REFERENCE_SHA
    from research.run_h3_zoom_preflight import file_sha,convert
    from research.h3_personal.zoom_turn_input import build_native
    source=ROOT/'research/stream_h3_comet_pcm.swift'
    if file_sha(Path(reference))!=REFERENCE_SHA:
        raise ValueError('reviewed original reference required; no silent enrollment substitution')
    pcm=convert(reference,reference_offset,3,'f32le')
    kwargs={} if confirmation is None else {'confirmation':confirmation}
    return build_native(model_dir=model_dir,reference=reference,reference_offset=reference_offset,
        reference_file_sha256=REFERENCE_SHA,reference_pcm_sha256=hashlib.sha256(pcm).hexdigest(),
        capture_source=source,capture_source_sha256=file_sha(source),
        all_participants_consented=all_participants_consented,**kwargs)


def run(*,directory,asr_parent,model_dir,reference=DEFAULT_REFERENCE,reference_offset=0,
        all_participants_consented=False,capture_approved=False,allow_obs=False,obs_port=None,
        capture_seconds=8,five_minute_test=False,start_gate=None):
    if all_participants_consented is not True or capture_approved is not True:
        raise PermissionError('explicit all-party consent and attended capture approval required')
    if allow_obs is not True or type(obs_port) is not int or not 1024<=obs_port<=65535:
        raise PermissionError('existing unified OBS playback approval/port required')
    if type(capture_seconds) is not int or not 8<=capture_seconds<=15:
        raise ValueError('short PoC capture8..15s required')
    if not Path(model_dir).is_dir() or not Path(asr_parent).is_dir():
        raise ValueError('existing local model and reviewed ASR parent required')
    # No measured remote echo-tail implementation/lifecycle binding exists yet.
    # Refuse BEFORE native enrollment, direct launcher import, or any ASR/GPU
    # allocation. Do not replace this with a flag or an always-True callback.
    raise PermissionError('Zoom live launch unavailable: measured remote echo-tail validator required before allocation')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--check-local',action='store_true')
    p.add_argument('--directory',type=Path);p.add_argument('--asr-parent',type=Path)
    p.add_argument('--model-dir',type=Path);p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    p.add_argument('--reference-offset',type=float,default=0)
    p.add_argument('--all-participants-consented',action='store_true')
    p.add_argument('--capture-approved',action='store_true');p.add_argument('--allow-obs',action='store_true')
    p.add_argument('--obs-port',type=int);p.add_argument('--capture-seconds',type=int,default=8)
    p.add_argument('--five-minute-test',action='store_true');p.add_argument('--start-gate',type=Path)
    a=p.parse_args(argv)
    if a.run and a.check_local:p.error('--run and --check-local are mutually exclusive')
    if not a.run and not a.check_local:
        print(json.dumps(dict(offline=True,meeting_client='Zoom Desktop (operator managed)',
            input='Zoom mix -> local VoxLock -> existing ASR -> existing Pi/H3',
            output='existing HTTP -> unified OBS H264/AAC -> Virtual Camera/BlackHole2ch',
            default_capture_seconds=8,max_turns=2,per_turn_confirmation='ZOOM-IDLE',
            zoom_launched=False,devices_accessed=False,provider_started=False,
            echo_tail_gate='measured validator required before live arm; operator TTY confirmation alone is refused',
            capture_loss_policy='transient capture loss rejects the turn and keeps the finite ASR lease',
            tts='unchanged existing runtime; no model selection/transport exploration')));return 0
    if a.model_dir is None:p.error('explicit --model-dir required')
    if a.check_local:
        pipeline=build_input(model_dir=a.model_dir,reference=a.reference,reference_offset=a.reference_offset)
        print(json.dumps(dict(local_preflight=True,source=pipeline.source_contract,
            echo_policy='live launch blocked: measured remote echo-tail validator unavailable; TTY alone is insufficient',
            capture_started=False,provider_started=False)));return 0
    if a.directory is None or a.asr_parent is None:p.error('--directory and --asr-parent required')
    run(directory=a.directory,asr_parent=a.asr_parent,model_dir=a.model_dir,reference=a.reference,
        reference_offset=a.reference_offset,all_participants_consented=a.all_participants_consented,
        capture_approved=a.capture_approved,allow_obs=a.allow_obs,obs_port=a.obs_port,
        capture_seconds=a.capture_seconds,five_minute_test=a.five_minute_test,start_gate=a.start_gate)
    return 0

if __name__=='__main__':raise SystemExit(main())
