"""Offline Zoom/VoxLock fixture probe; never launches/captures/joins a meeting."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import os


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def convert(path,offset,seconds,encoding):
    # Existing explicit local file only. One FFmpeg conversion per source,
    # never an independent resampler for each streaming chunk.
    path=Path(path).resolve(strict=True)
    if not path.is_file():raise ValueError('existing local file required')
    p=subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(offset),'-i',str(path),'-t',str(seconds),
        '-vn','-ac','1','-ar','16000','-f',encoding,'pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
        timeout=60,check=True)
    expected=round(seconds*16000)*(4 if encoding=='f32le' else 2)
    if len(p.stdout)!=expected:raise ValueError('exact finite fixture duration required')
    return p.stdout


def run(args):
    import numpy as np
    from research.h3_personal.voxlock_streaming import load_cpu,VoxLockStream
    from research.h3_personal.zoom_input import EchoGuard,ZoomPCMReader,source_contract
    os.umask(0o077)
    root=args.directory.absolute()
    repo=Path(__file__).resolve().parents[1]
    if root.exists() or root.is_relative_to(repo) or any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('fresh external private result directory required')
    root.mkdir(mode=0o700)
    reference=convert(args.reference,args.reference_offset,3,'f32le')
    raw=convert(args.input,args.input_offset,args.seconds,'s16le')
    before=time.monotonic();enroll,chunk=load_cpu(args.model_dir,threads=4);loaded=time.monotonic()
    stream=VoxLockStream(enroll,chunk,np.frombuffer(reference,dtype='<f4').copy(),batch_frames=8);enrolled=time.monotonic()
    guard=EchoGuard(tail_seconds=1,tail_validated=False);guard.initial_idle()
    epoch=guard.arm(deadline=time.monotonic()+45,manual_human_turn=True,all_participants_consented=True,offline=True)
    reader=ZoomPCMReader(io.BytesIO(raw),stream,guard,epoch)
    total=energy=peak=0;first=None;start=time.monotonic()
    while data:=reader.read(640):
        if first is None:first=time.monotonic()-start
        values=np.frombuffer(data,dtype='<i2').astype(np.float64)/32768
        total+=len(values);energy+=float(np.dot(values,values));peak=max(peak,float(np.abs(values).max()))
    elapsed=time.monotonic()-start
    original=np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768
    capture_sha=file_sha(repo/'research/stream_h3_comet_pcm.swift')
    result=dict(schema='h3-zoom-offline-preflight-v1',fixture_label=args.label,
        source_contract=source_contract(reference_sha256=hashlib.sha256(reference).hexdigest(),capture_source_sha256=capture_sha),
        reference_original_sha256=file_sha(args.reference),reference_offset_seconds=args.reference_offset,
        input_sha256=file_sha(args.input),input_offset_seconds=args.input_offset,
        seconds=args.seconds,model_load_seconds=loaded-before,enrollment_seconds=enrolled-loaded,
        processing_seconds=elapsed,processing_rtf=elapsed/args.seconds,first_output_seconds=first,
        input_rms=float(np.sqrt(np.mean(original**2))),extracted_rms=(energy/total)**.5,extracted_peak=peak,
        receipt=reader.receipt(),samples_exact=total==round(args.seconds*16000),
        live_capture=False,meeting_joined=False,asr_called=False,raw_audio_saved=False,
        human_identity_verified=False,remote_echo_tail_validated=False,
        note='TSE signal preservation does not prove human-vs-clone identity; no remote ASR or media upload')
    with (root/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('fixture_label','seconds','model_load_seconds','enrollment_seconds','processing_seconds','processing_rtf','input_rms','extracted_rms','samples_exact','live_capture','asr_called')},allow_nan=False))


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run-offline',action='store_true')
    for k in ('model-dir','reference','input','directory'):p.add_argument('--'+k,type=Path)
    p.add_argument('--reference-offset',type=float,default=0);p.add_argument('--input-offset',type=float,default=0)
    p.add_argument('--seconds',type=int,default=3);p.add_argument('--label',default='explicit-local-fixture')
    a=p.parse_args(argv)
    if not a.run_offline:
        print(json.dumps(dict(scope='offline description only',capture_app_candidate='zoom',bundle_candidate='us.zoom.xos',
            runtime_verified=False,devices_accessed=False,asr_started=False,transport='existing HTTP unchanged; WebRTC experimental',
            safety='VoxLock is extraction, not proof human Yugo versus cloned AI; input initially closed')));return 0
    if any(getattr(a,k) is None for k in ('model_dir','reference','input','directory')):p.error('explicit local model/reference/input/fresh directory required')
    if not 1<=a.seconds<=10 or not 0<=a.reference_offset<=27 or not 0<=a.input_offset<=3600:p.error('finite fixture bounds required')
    run(a);return 0

if __name__=='__main__':raise SystemExit(main())
