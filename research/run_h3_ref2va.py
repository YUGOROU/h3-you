"""Explicit native Ref2VA+original identity pilot. Default: full offline preflight only.

No Hub publication, provider allocation, automatic retries or T2VA FastH3
fallback. Supplied-speech conditioning exists only through explicit
--locked-speech/--locked-speech-sha256 on the SMD path (see
research/h3_personal/external_audio.py and docs/h3-smd4-locked-audio.md).
--run needs a provisioned pinned CUDA runtime and local assets.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from research.h3_personal.ref2va_runtime import (Ref2VAConfig,SCOPE,IDENTITY_SHA,sha,
    verify_model,request,generator_config,install_ref_guard,validate_identity_header)

ARCHIVE_SHA='6ef5b1ebb373cc4d09e8e5de1851d8acf6c23c83cbe5b60c2e86eafaeec3f1c7'
FASTVIDEO_REV='ba371e33ed080e51f1884ae166baf7c4447a66fe'


def speech_duration(path):
    import wave
    with wave.open(str(path),'rb') as handle:
        rate=handle.getframerate();frames=handle.getnframes()
    if rate<=0 or frames<=0:raise ValueError('usable supplied speech WAV required')
    return frames/rate


def locked_audio_plan(a):
    """Offline plan for the supplied speech; never decodes, plays or truncates it."""
    from research.h3_personal.external_audio import plan_windows,sha256
    if (a.locked_speech is None)!=(a.locked_speech_sha256 is None):raise ValueError('locked speech and its SHA256 are required together')
    if a.locked_speech is None:
        if a.locked_speech_sha256 is not None:raise ValueError('locked speech SHA without media')
        return None
    if not a.fast_ref_smd:raise ValueError('locked external speech requires the explicit SMD Ref2VA path')
    if a.runtime_policy=='baseline':raise ValueError('locked speech requires resident visual-only decoding so generated audio is never used')
    if sha256(a.locked_speech)!=a.locked_speech_sha256:raise ValueError('locked speech SHA differs')
    return plan_windows(speech_duration(a.locked_speech),wav=a.locked_speech,sha256_value=a.locked_speech_sha256)


def verify_inputs(a):
    from research.h3_personal.personal_fast_adapter import validate_identity_config
    from research.h3_personal.adapter_header import header_length
    if sha(a.identity)!=IDENTITY_SHA:raise ValueError('original64 SHA differs')
    validate_identity_config(json.loads(a.identity_config.read_text()))
    if len(a.reference_sha256)!=64 or sha(a.reference)!=a.reference_sha256:raise ValueError('reference SHA differs')
    if sha(a.runtime_archive)!=ARCHIVE_SHA:raise ValueError('verified FastVideo source archive differs')
    # Structural CPU inspection without loading Torch or 83MB into tensor storage.
    with a.identity.open('rb') as f:
        size=header_length(f.read(8));header=json.loads(f.read(size))
    validate_identity_header(header)
    if a.fast_ref_smd:
        from research.h3_personal.fast_ref2va import WEIGHT_SHA
        if a.smd is None or sha(a.smd)!=WEIGHT_SHA:raise ValueError('pinned Ref2VA SMD4 weights required')
    elif a.smd is not None:raise ValueError('SMD weights require explicit fast Ref2VA mode')
    from PIL import Image,ImageOps
    with Image.open(a.reference) as image:
        image.load();image=ImageOps.exif_transpose(image);w,h=image.size
    if min(w,h)<=0 or max(w,h)>4*min(w,h):raise ValueError('reference aspect ratio unsupported')
    crop_proof=None
    if getattr(a,'face_crop',None) is not None:
        from research.h3_personal.ref2va_reference_assets import verify
        crop_proof=verify(a.reference,a.face_crop,a.face_crop_receipt)
    return dict(face_crop_proof=crop_proof,identity_sha256=IDENTITY_SHA,identity_modules=208,identity_alpha_over_rank=1,
        reference_sha256=a.reference_sha256,reference_oriented_width=w,reference_oriented_height=h,
        model=verify_model(a.model,json.loads(a.model_manifest.read_text())),runtime_archive_sha256=ARCHIVE_SHA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','model-manifest','identity','identity-config','reference','runtime-archive','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-sha256',required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--fast-ref-smd',action='store_true');p.add_argument('--smd',type=Path)
    p.add_argument('--regional-compile',action='store_true',help='Explicit SMD-only regional DiT reduce-overhead experiment')
    p.add_argument('--attention-mode',choices=('fa2','fa4','fa4-packed'),default='fa2')
    p.add_argument('--live-publication-repo')
    p.add_argument('--layout',choices=('landscape','portrait'),default='landscape')
    p.add_argument('--runtime-policy',choices=('baseline','resident-visual','resident-visual-cached'),default='baseline')
    p.add_argument('--face-crop',type=Path);p.add_argument('--face-crop-receipt',type=Path)
    p.add_argument('--locked-speech',type=Path,help='Supplied speech WAV locked onto every SMD denoising forward; visual-only output')
    p.add_argument('--locked-speech-sha256',help='Exact SHA256 of the supplied speech WAV')
    p.add_argument('--profile-warm-stages',action='store_true',help='SMD-only diagnostic; adds CUDA synchronization, not a comparable RT benchmark')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    if a.attention_mode!='fa2' and (not a.fast_ref_smd or a.regional_compile):raise ValueError('FA4 experiment requires eager SMD')
    if (a.layout!='landscape' or a.runtime_policy!='baseline') and (not a.fast_ref_smd or a.regional_compile):raise ValueError('runtime/layout experiments require eager SMD')
    if a.regional_compile and (not a.fast_ref_smd or a.profile_warm_stages):raise ValueError('regional compile requires SMD non-diagnostic benchmark')
    if (a.face_crop is None)!=(a.face_crop_receipt is None):raise ValueError('crop and provenance receipt required together')
    if a.face_crop is not None and not a.fast_ref_smd:raise ValueError('dual reference requires explicit SMD experiment')
    if a.profile_warm_stages and not a.fast_ref_smd:raise ValueError('warm profiling requires explicit SMD Ref2VA')
    config=Ref2VAConfig().validate()
    if a.fast_ref_smd:
        from research.h3_personal.fast_ref2va import config_for_layout
        config=config_for_layout(a.layout)
    from research.h3_personal.ref2va_runtime_options import export_geometry
    geometry=export_geometry(a.layout)
    locked=locked_audio_plan(a)
    proof=verify_inputs(a)
    if not a.run:
        print(json.dumps(dict(schema=SCOPE+('-smd4' if a.fast_ref_smd else ''),config=asdict(config),preflight=proof,
            locked_audio=locked,external_audio_conditioning=locked is not None,gpu_run=False),indent=2));return 0
    root=Path(__file__).resolve().parents[1]
    if a.output.is_relative_to(root):raise ValueError('private output must remain outside repository')
    source=a.output/'runtime'/('FastVideo-'+FASTVIDEO_REV)
    if not a.worker:
        a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
        from research.prepare_h3_realtime_first import validate_archive
        validate_archive(a.runtime_archive)
        with tarfile.open(a.runtime_archive) as archive:archive.extractall(a.output/'runtime',filter='data')
        if a.fast_ref_smd:
            from research.h3_personal.fast_ref2va import install
            overlay=install(source)
            if locked is not None:
                from research.h3_personal.external_audio import install as install_locked_audio
                overlay['locked_audio']=install_locked_audio(source)
        else:overlay=install_ref_guard(source/'fastvideo/pipelines/lora_pipeline.py')
        (a.output/'loader-overlay.json').write_text(json.dumps(overlay,indent=2)+'\n')
        env=dict(os.environ,PYTHONPATH=str(source)+os.pathsep+str(root),
            H3_REF2VA_COVERAGE_DIR=str(a.output/'coverage'),H3_SMD_REF2VA='1' if a.fast_ref_smd else '0',
            H3_SMD_LOCKED_AUDIO='1' if locked is not None else '0',
            FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=('2' if a.face_crop else '1') if a.fast_ref_smd else '0')
        from research.h3_personal.ref2va_attention import configure_environment
        configure_environment(a.attention_mode,env)
        if locked is not None:
            env['H3_SMD_LOCKED_AUDIO_WINDOW_FILE']=str(a.output/'coverage/current-locked-window.json')
        else:
            env.pop('H3_SMD_LOCKED_AUDIO_WINDOW_FILE',None)
        env['H3_REF2VA_RUNTIME_POLICY']=a.runtime_policy
        env['H3_REF2VA_REGIONAL_COMPILE']='1' if a.regional_compile else '0'
        env['TORCHDYNAMO_SUPPRESS_ERRORS']='0'
        if a.profile_warm_stages:env['H3_REF2VA_STAGE_PROFILE_DIR']=str(a.output/'stage-profile')
        else:env.pop('H3_REF2VA_STAGE_PROFILE_DIR',None)
        (a.output/'locked-audio-plan.json').write_text(json.dumps(locked,indent=2)+'\n') if locked else None
        os.execvpe(sys.executable,[sys.executable,'-m','research.run_h3_ref2va',*sys.argv[1:],'--worker'],env)
    if os.environ.get('H3_REF2VA_COVERAGE_DIR')!=str(a.output/'coverage'):raise ValueError('missing isolated worker setup')
    import fastvideo
    if Path(fastvideo.__file__).resolve().parent!=source/'fastvideo':raise RuntimeError('wrong FastVideo source imported')
    import torch
    for package,version in {'torch':'2.12.0','fastvideo-kernel':'0.3.5','quack-kernels':'0.5.3'}.items():
        if importlib.metadata.version(package).split('+')[0]!=version:raise RuntimeError(f'wrong pinned dependency: {package}')
    if torch.cuda.device_count()!=2 or any('H200' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('requires exactly2H200; no hardware/FP8 fallback')
    import fastvideo.api as api
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference
    from research.run_h3_realtime_first import encoder_preflight,encode_rgb_segment
    from research.h3_personal.realtime_segments import segment_timing
    import numpy as np
    encoder_preflight(a.output/'encoder-probe')
    effective_adapter=a.identity;derived=None
    if a.fast_ref_smd:
        from research.h3_personal.fast_ref2va import build_adapter
        torch.set_num_threads(4)
        effective_adapter=a.output/'derived-private/adapter_model.safetensors'
        derived=build_adapter(a.smd,a.identity,a.identity_config,effective_adapter)
    generator=None
    try:
        started=time.monotonic()
        configure_generator=generator_config
        if a.fast_ref_smd:
            from research.h3_personal.fast_ref2va import generator_config as configure_generator
        generation_config=(configure_generator(api,a.model,effective_adapter,regional_compile=a.regional_compile,runtime_policy=a.runtime_policy)
                           if a.fast_ref_smd else configure_generator(api,a.model,effective_adapter))
        generator=fastvideo.VideoGenerator.from_config(generation_config)
        def read_coverage():
            values=[]
            for rank in (0,1):
                receipt=json.loads((a.output/'coverage'/f'coverage-rank-{rank}.json').read_text())
                expected=dict(identity_modules=208,total_modules=208,unmatched=0,rank=16,partition='transformer_ref',worker_rank=rank)
                if a.fast_ref_smd:expected.update(total_modules=312,rank='mixed128-144')
                if receipt!=expected:raise RuntimeError('both Ref2VA identity receipts required')
                values.append(receipt)
            return values
        receipts=[] if a.fast_ref_smd else read_coverage()
        if a.fast_ref_smd:
            from research.h3_personal.fast_ref2va import segment_timing as ref_timing,evaluate_warm
            from research.h3_personal.realtime_segments import SegmentPipeline
            def encode(job,cancel):
                index,frames=job;path=a.output/f'segment-{index:03d}.mp4'
                encode_rgb_segment(frames.tobytes(),path,ref_timing(index),width=geometry['width'],height=geometry['height'],cancel=cancel)
                return dict(path=str(path),video_sha256=sha(path))
            publish=None
            if a.live_publication_repo:
                from research.h3_personal.ref2va_live_transfer import publisher
                publish=publisher(a.live_publication_repo)
            from research.h3_personal.external_audio import verify_receipts,window_manifest
            segments=locked['windows_required'] if locked else 4
            pipeline=SegmentPipeline(encode,publish=publish,join_timeout=180,timing_factory=ref_timing);generations=[]
            try:
                for index in range(segments):
                    if locked:
                        # Each request locks its own fixed window of the supplied speech.
                        binding=json.dumps(window_manifest(locked,index))
                        window_file=Path(os.environ['H3_SMD_LOCKED_AUDIO_WINDOW_FILE'])
                        window_file.parent.mkdir(parents=True,exist_ok=True)
                        temporary=window_file.with_suffix('.tmp')
                        temporary.write_text(binding);temporary.replace(window_file)
                        os.environ['H3_SMD_LOCKED_AUDIO_WINDOW']=binding
                    req=request(api,MiniMaxH3Reference,a.reference,config,face_crop=a.face_crop);req.sampling.seed+=index
                    begin=time.monotonic();result=generator.generate(req);torch.cuda.synchronize();end=time.monotonic()
                    if index==0:receipts=read_coverage()  # Deferred adapter guard runs inside load, before denoising.
                    frames=np.stack([np.asarray(frame).copy() for frame in result.frames])
                    if frames.shape!=(124,config.height,config.width,3) or frames.dtype!=np.uint8:raise ValueError('SMD actual output shape differs')
                    if geometry['crop_left']:frames=frames[:,:,geometry['crop_left']:-geometry['crop_right'],:]
                    generations.append((begin,end));pipeline.submit((index,frames),segment_id=index)
                reports=pipeline.close()
            finally:
                if not pipeline._closed:
                    pipeline.cancel()
                    from research.h3_personal.realtime_segments import PipelineCancelled
                    try:pipeline.close()
                    except PipelineCancelled:pass
            metrics=evaluate_warm(generations,reports,segments=segments)
            locked_receipts=verify_receipts(a.output/'coverage',segments) if locked else []
            # One short speech window has no warm/cache-hit request. Do not
            # require or fabricate diagnostics that only warm_forward writes.
            cache_receipts=[json.loads((a.output/'coverage'/f'cache-rank-{rank}.json').read_text()) for rank in (0,1)] if segments>1 else []
            if any(r!={'hits':segments-1} for r in cache_receipts):raise RuntimeError('one actual warm condition-cache hit per extra segment per rank required')
            merge_receipts=[json.loads((a.output/'coverage'/f'merge-rank-{rank}.json').read_text()) for rank in (0,1)]
            if any(r!={'active':312,'merged':312,'dynamic':0} for r in merge_receipts):raise RuntimeError('static merge receipt differs')
            compiled_receipts=[]
            if a.regional_compile:
                compiled_receipts=[json.loads((a.output/'coverage'/f'compile-rank-{rank}.json').read_text()) for rank in (0,1)]
                if any(r.get('wrapped_regions')!=52 for r in compiled_receipts):raise RuntimeError('52 regional wrappers per rank required')
            phase_profiles=[]
            if a.profile_warm_stages:
                phase_profiles=[json.loads((a.output/'stage-profile'/f'rank-{rank}-warm-{hit}.json').read_text())
                                for rank in (0,1) for hit in range(1,segments)]
            attention_receipts=[]
            if a.attention_mode!='fa2':
                attention_receipts=[json.loads((a.output/'coverage'/f'attention-rank-{rank}.json').read_text()) for rank in (0,1)]
                if any(r.get('backend')!='4' or not r.get('cuda_kernel_events') for r in attention_receipts):raise RuntimeError('both-rank FA4 CUDA smoke evidence required')
            packed_receipts=[]
            if a.attention_mode=='fa4-packed' and segments>1:
                packed_receipts=[json.loads((a.output/'coverage'/f'packed-rank-{rank}.json').read_text()) for rank in (0,1)]
                if any(not r['warm_counter_removed'] for r in packed_receipts):raise RuntimeError('cold counter must be removed before warm timing')
            residency_receipts=[]
            if a.runtime_policy!='baseline' and segments>1:
                residency_receipts=[json.loads((a.output/'coverage'/f'residency-rank-{rank}.json').read_text()) for rank in (0,1)]
            profile=dict(incremental_generated_transfer=bool(a.live_publication_repo),layout=a.layout,export_geometry=geometry,runtime_policy=a.runtime_policy,residency_receipts=residency_receipts,warm_audio_decode_skipped=a.runtime_policy!='baseline',packed_route_receipts=packed_receipts,attention_mode=a.attention_mode,attention_receipts=attention_receipts,regional_compile_requested=a.regional_compile,compiled_region_receipts=compiled_receipts,
                warm_stage_profiles=phase_profiles,static_merge_receipts=merge_receipts,benchmark_comparable=not a.profile_warm_stages,
                schema=SCOPE+'-smd4',config=asdict(config),input_proof=proof,loaded_coverage=receipts,
                derived_adapter_receipt=derived,metrics=metrics,generation_intervals=generations,
                cold_first_local_availability_seconds=reports[0]['publication_finished_at']-started,
                producer_blocked_seconds=pipeline.producer_blocked_seconds,
                segments=[dict(r,timing=asdict(r['timing'])) for r in reports],
                expected_forwards_per_segment=4,seeds=[config.seed+i for i in range(segments)],unique_frames=124*segments,
                conditioning_memo_entries=2 if a.face_crop else 1,conditioning_cache_receipts=cache_receipts,
                independent_ref2va_clips=True,shared_head_trim=False,reference_count=2 if a.face_crop else 1,
                latency_scope='server local encoded availability; no network/client evidence',
                visual_only=True,generated_audio_discarded=True,external_audio_conditioning=locked is not None,
                locked_audio_plan=locked,locked_audio_receipts=locked_receipts,
                supplied_speech_played_by_caller_only=locked is not None,
                identity_quality_verified=False,realtime_complete=False,originals_modified=False,uploaded=False)
            (a.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
            print('FAST_REF2VA_RESULT',json.dumps(metrics),flush=True)
            return 0
        generation_start=time.monotonic()
        result=generator.generate(request(api,MiniMaxH3Reference,a.reference,config))
        torch.cuda.synchronize();generation_seconds=time.monotonic()-generation_start
        if isinstance(result,list):raise ValueError('single Ref2VA result required')
        frames=np.stack([np.asarray(f) for f in result.frames])
        if frames.dtype!=np.uint8 or frames.shape!=(124,256,448,3):raise ValueError('actual reference output geometry differs')
        video=a.output/'review.mp4';encode_rgb_segment(frames.tobytes(),video,segment_timing(0))
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True,timeout=120)
        profile=dict(schema=SCOPE+('-smd4' if a.fast_ref_smd else ''),config=asdict(config),input_proof=proof,loaded_coverage=receipts,
            derived_adapter_receipt=derived,ref2va_four_forward_candidate=a.fast_ref_smd,
            generation_seconds=generation_seconds,runner_seconds=time.monotonic()-started,
            expected_transformer_forwards=4 if a.fast_ref_smd else 50,video_sha256=sha(video),visual_only=True,generated_audio_discarded=True,
            external_audio_conditioning=False,identity_quality_verified=False,realtime_complete=False,
            realtime_speed_verified=False,originals_modified=False,uploaded=False)
        (a.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        print('REF2VA_RESULT',json.dumps(profile),flush=True)
    finally:
        if generator is not None:generator.shutdown()
    return 0

if __name__=='__main__':raise SystemExit(main())
