"""Isolated finite FastH3 visual PoC. No personal media, baseline or audio input.

Run only from a reviewed pinned deployment. Segment-level output, NOT chunk-major
causal diffusion. Publication is private Hub storage, not Mac playback proof.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import struct
import time
from research.h3_personal.realtime_first import RealtimeFirstConfig,SegmentMeasurement,evaluate_segments
from research.h3_personal.realtime_segments import SegmentPipeline,segment_timing


def encode_command(path, timing, width=448, height=256):
    return ['ffmpeg','-v','error','-nostdin','-copyts','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{width}x{height}','-framerate','24','-i','pipe:0','-an',
        '-vsync','0','-filter_threads','1','-c:v','libx264','-threads','2','-video_track_timescale','12288',
        '-preset','ultrafast','-tune','zerolatency','-bf','0','-g','6','-pix_fmt','yuv420p',
        '-movflags','frag_keyframe+empty_moov+default_base_moof','-frag_duration','250000',str(path)]


def shift_fragment_decode_times(path,timing):
    """Place independent silent fMP4 segments on one24Hz presentation timeline.

    empty_moov starts each encode at0. Shift ONLY parsed moof/traf/tfdt fields;
    no media payload searching, box resizing, transcoding or paced playback.
    Track timescale is explicitly12288 above, B-frames are disabled.
    """
    data=bytearray(path.read_bytes());offset=timing.pts_start*512;seen=[]
    def visit(start,end):
        pos=start
        while pos<end:
            if end-pos<8:raise ValueError('truncated MP4 box')
            size,kind=struct.unpack_from('>I4s',data,pos);header=8
            if size==1:
                if end-pos<16:raise ValueError('truncated extended box')
                size=struct.unpack_from('>Q',data,pos+8)[0];header=16
            elif size==0:size=end-pos
            if size<header or pos+size>end:raise ValueError('invalid MP4 box size')
            if kind in (b'moof',b'traf'):visit(pos+header,pos+size)
            elif kind==b'tfdt':
                payload=pos+header
                if pos+size-payload<8:raise ValueError('truncated tfdt')
                version=data[payload]
                if version not in (0,1):raise ValueError('unsupported tfdt version')
                fmt='>Q' if version==1 else '>I';width=8 if version else 4
                if payload+4+width>pos+size:raise ValueError('truncated decode time')
                current=struct.unpack_from(fmt,data,payload+4)[0];seen.append(current)
                struct.pack_into(fmt,data,payload+4,current+offset)
            pos+=size
    visit(0,len(data))
    if not seen or seen[0]!=0:raise ValueError('expected zero-based fragmented encoding')
    path.write_bytes(data)


def validate_packets(path,timing):
    from fractions import Fraction
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_packets','-show_streams','-of','json',str(path)],timeout=20))
    packets=p['packets'];tb=Fraction(p['streams'][0]['time_base'])
    expected=[Fraction(timing.pts_start+i,24) for i in range(timing.newly_emitted_frames)]
    actual=[int(packet['pts'])*tb for packet in packets]
    if actual!=expected:raise ValueError('actual fMP4 packet PTS/count differ from planned unique frames')


def encode_rgb_segment(raw,path,timing,*,width=448,height=256,cancel=None):
    """Shared early probe/production path, compatible with image FFmpeg4.4.2."""
    errors_path=path.with_suffix('.stderr')
    with errors_path.open('wb') as errors:
        process=subprocess.Popen(encode_command(path,timing,width,height),stdin=subprocess.PIPE,stderr=errors)
        try:
            process.communicate(raw,timeout=45)
        except BaseException:
            if process.poll() is None:process.kill()
            process.wait(timeout=5)
            raise
    if process.returncode:
        tail=errors_path.read_bytes()[-8192:].decode('utf-8',errors='replace')
        print('FFMPEG_ENCODER_ERROR',process.returncode,tail,flush=True)
        raise RuntimeError(f'segment FFmpeg encoding failed (exit={process.returncode}): {tail}')
    if cancel is not None and cancel.is_set():raise RuntimeError('cancelled')
    shift_fragment_decode_times(path,timing);validate_packets(path,timing)


def encoder_preflight(directory):
    """Run actual shared encode+PTS+decode before model downloads/CUDA work."""
    directory.mkdir(parents=True,exist_ok=False)
    version=subprocess.check_output(['ffmpeg','-version'],timeout=10).decode().splitlines()[0]
    probe_version=subprocess.check_output(['ffprobe','-version'],timeout=10).decode().splitlines()[0]
    for index in (0,1):
        timing=segment_timing(index);path=directory/f'probe-{index}.mp4'
        encode_rgb_segment(bytes(32*32*3*timing.newly_emitted_frames),path,timing,width=32,height=32)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,timeout=15)
    result=dict(ffmpeg=version,ffprobe=probe_version,encode_packet_pts_full_decode=True,source='shared_production_encoder')
    print('REALTIME_ENCODER_PREFLIGHT',json.dumps(result),flush=True)
    return result


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--spec',type=Path,required=True)
    a.add_argument('--base',type=Path,required=True);a.add_argument('--adapter',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args(argv)
    spec=json.loads(args.spec.read_text());config=RealtimeFirstConfig.from_dict(spec['config'])
    job_started=time.monotonic();aout=args.output;aout.mkdir(mode=0o700,parents=True,exist_ok=False)
    for package,version in {'torch':'2.12.0','fastvideo-kernel':'0.3.5','quack-kernels':'0.5.3'}.items():
        actual=importlib.metadata.version(package).split('+')[0]
        if actual!=version:raise RuntimeError(f'{package}: expected {version}, got {actual}')
    import torch
    import numpy as np
    from PIL import Image
    from huggingface_hub import HfApi
    from fastvideo import VideoGenerator
    from fastvideo.api import (GeneratorConfig,PipelineSelection,ComponentConfig,EngineConfig,
        ParallelismConfig,OffloadConfig,CompileConfig,GenerationRequest,InputConfig,SamplingConfig,OutputConfig)
    if torch.cuda.device_count()!=2:raise RuntimeError('requires exactly two visible GPUs')
    if any('H200' not in torch.cuda.get_device_name(i) for i in range(2)):raise RuntimeError('wrong GPU SKU')
    api=HfApi(token=os.environ['HF_TOKEN']);repo=spec['publication_repo']
    if not api.repo_info(repo,repo_type='dataset').private:raise RuntimeError('publication repository must be private')
    generator=None;pipeline=None;generated=[]
    def encode(job,cancel):
        index,frames=job;timing=segment_timing(index);path=aout/f'segment-{index:03d}.mp4'
        data=np.stack(frames[timing.head_trim_frames:])
        if data.dtype!=np.uint8 or data.shape!=(timing.newly_emitted_frames,256,448,3):raise ValueError('frame shape/dtype differs')
        encode_rgb_segment(data.tobytes(),path,timing,cancel=cancel)
        return {'path':str(path),'video_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def publish(result,cancel):
        if cancel.is_set():raise RuntimeError('cancelled')
        path=Path(result['path']);receipt=api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=str(path),
            path_in_repo='outputs/'+path.name,commit_message='Generated dummy visual segment; not private input')
        return dict(result,publication_commit=receipt.oid)
    try:
        generator=VideoGenerator.from_config(GeneratorConfig(model_path=str(args.base),
            pipeline=PipelineSelection(components=ComponentConfig(lora_path=str(args.adapter),lora_strength=1.0),
                experimental={'attention_backend':'VIDEO_SPARSE_ATTN_H3','inference_torch_compile':False,
                    'VSA_sparsity':0.9,'VSA_tile_size':64,'vae_parallel_decode':True,'vae_parallel_decode_strategy':'gather'}),
            engine=EngineConfig(num_gpus=2,parallelism=ParallelismConfig(tp_size=1,sp_size=2,dist_timeout=300),
                offload=OffloadConfig(dit=False,dit_layerwise=False,text_encoder=True,image_encoder=True,vae=True,pin_cpu_memory=True),
                compile=CompileConfig(enabled=False,vae_enabled=False,audio_vae_enabled=False))))
        personal_receipts=[]
        if config.personalized:
            from research.build_h3_personal_fast_adapter import validate_personal_spec
            validate_personal_spec(spec)
            folder=Path(os.environ['H3_PERSONAL_COVERAGE_DIR'])
            for rank in (0,1):
                receipt=json.loads((folder/f'coverage-rank-{rank}.json').read_text())
                if receipt!=dict(identity_modules=208,total_modules=362,unmatched=0,rank=rank):
                    raise RuntimeError('strict personal coverage receipt differs')
                personal_receipts.append(receipt)
        previous=None;pipeline=SegmentPipeline(encode,publish=publish,join_timeout=180)
        for index in range(config.segments):
            started=time.monotonic()
            result=generator.generate(GenerationRequest(prompt=config.prompt,negative_prompt='',
                inputs=InputConfig(pil_image=previous),sampling=SamplingConfig(height=256,width=448,num_frames=124,
                    fps=24,num_inference_steps=5,guidance_scale=1.0,batch_cfg=False,seed=314159),
                output=OutputConfig(save_video=False,return_frames=True)))
            torch.cuda.synchronize();finished=time.monotonic()
            if isinstance(result,list) or len(result.frames)!=124:raise ValueError('requires124actual generated frames')
            frames=[np.asarray(frame).copy() for frame in result.frames]
            if previous is not None:previous.close()
            previous=Image.fromarray(frames[-1]).convert('RGB')
            generated.append((started,finished));pipeline.submit((index,frames),segment_id=index)
            print('REALTIME_SEGMENT_GENERATED',index,finished-started,flush=True)
        reports=pipeline.close();measurements=[]
        for report,(start,end) in zip(reports,generated):
            t=report['timing'];measurements.append(SegmentMeasurement(t.segment_id,start,end,
                report['encode_finished_at'],report['publication_finished_at'],t.pts_start,t.newly_emitted_frames,t.head_trim_frames))
        metrics=evaluate_segments(config,measurements,job_started=job_started)
        payload=dict(schema='h3-realtime-first-result-v1',config=asdict(config),metrics=metrics,
            segments=[dict(r,timing=asdict(r['timing'])) for r in reports],
            gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
            producer_blocked_seconds=pipeline.producer_blocked_seconds,external_audio_conditioning=False,
            generated_audio_discarded=True,visual_only=True,chunk_major_diffusion=False,quality_verified=False,
            original_models_touched=False,fp8_enabled=False,personalized=config.personalized,
            personal_loaded_coverage=personal_receipts,
            derived_adapter_receipt=json.loads(args.adapter.with_suffix('.json').read_text()) if config.personalized else None)
        path=aout/'profile.json';path.write_text(json.dumps(payload,indent=2)+'\n')
        api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=str(path),path_in_repo='outputs/profile.json')
        print('REALTIME_RESULT',json.dumps(metrics),flush=True)
    finally:
        try:
            if pipeline is not None and not pipeline._closed:
                pipeline.cancel()
                from research.h3_personal.realtime_segments import PipelineCancelled
                try:pipeline.close()
                except PipelineCancelled:pass
        finally:
            if generator is not None:generator.shutdown()
    return 0

if __name__=='__main__':raise SystemExit(main())
