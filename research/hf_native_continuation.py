"""Offline, phased native continuation worker. No network/upload/job launch."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('reference','text','audio','generate','decode'))
    p.add_argument('--request',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--conditioning-sha256')
    a=p.parse_args(argv);os.umask(0o077)
    if any(os.environ.get(k)!='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')):
        raise ValueError('offline worker required')
    from research import hf_identity_train as train
    from research.hf_identity_encode import dump
    from research.h3_personal.continuation_contract import validate,AUDIO_SHA
    request=train.read_json(a.request);case=train.read_json(a.case);native=validate(case['native']);root=Path(case['output'])
    experiment=case.get('acceleration_experiment')
    window_frames=277
    if experiment is not None:
        from research.h3_personal.acceleration_experiment import validate as validate_experiment,TORCHAO_VERSION
        experiment=validate_experiment(experiment);window_frames=experiment['window_frames']
        if experiment['quantization']!='none':
            import importlib.metadata
            if importlib.metadata.version('torchao')!=TORCHAO_VERSION:raise ValueError('pinned experimental runtime required')
    from research.h3_personal.continuation_export import recorded_wav_info
    if train.sha256(case['audio_path'])!=AUDIO_SHA:raise ValueError('original recorded audio SHA differs')
    if recorded_wav_info(case['audio_path'])!=(1364312,48000,1,3):
        raise ValueError('original WAV sample/format identity differs')
    import torch
    from safetensors.torch import load_file,save_file
    if (torch.__version__!='2.8.0+cu128' or torch.cuda.device_count()!=2 or
        any('H200' not in torch.cuda.get_device_name(i) for i in range(2))):
        raise ValueError('exact2H200/Torch2.8cu128 runtime required')
    start=time.monotonic()
    for i in range(2):torch.cuda.reset_peak_memory_stats(i)
    config=train.load_runtime(request)
    base_identity=dict(pins=request['pins'],model_inventory=request['model_inventory'],reference_policy='correct-hw-v2')
    if a.phase in ('reference','text','audio'):
        from h3_trainer.preprocessing.encoders import H3Encoders
        from research.h3_personal.continuation_backend import CorrectedBackend
        from research.h3_personal import continuation_conditioning as binding
        enc=H3Encoders(request['model'],need_video=a.phase=='reference',need_audio=a.phase=='audio',
                       need_text=a.phase=='text',**({'text_device_map':'auto'} if a.phase=='text' else {}))
        try:
            backend=CorrectedBackend(enc,Path(request['work'])/'runtime');ref=case['reference_image']
            if a.phase=='reference':
                tensors,receipt=binding.reference(backend,ref['path'],ref['sha256'],base_identity=base_identity)
                if receipt['prepared_size_wh']!=[2048,3648]:raise ValueError('fixed corrected reference geometry required')
                save_file(tensors,str(root/'reference.safetensors'));dump(root/'reference-binding.json',receipt)
            elif a.phase=='text':
                tensors,receipt=binding.text(backend,ref['path'],ref['sha256'],load_file(str(root/'reference.safetensors')),
                    train.read_json(root/'reference-binding.json'),case['prompt'],base_identity=base_identity)
                save_file(tensors,str(root/'text.safetensors'));dump(root/'text-binding.json',receipt)
            else:
                tensors=backend.audio({'sources':{'audio':{'path':case['audio_path']}},'settings':{'num_frames':685}})
                if tensors['latents'].shape!=(2284,32):raise ValueError('full audio VAE lattice differs')
                save_file(tensors,str(root/'audio.safetensors'));dump(root/'audio-report.json',dict(
                    schema='h3-recorded-audio-conditioning-v1',source_sha256=AUDIO_SHA,samples=1364312,
                    model_frames=685,latents_sha256=binding.tensor_sha(tensors['latents']),
                    original_wav_unchanged=train.sha256(case['audio_path'])==AUDIO_SHA,
                    offline_full_audio_encoding=True,causal_verified=False))
        finally:enc.unload()
        dump(root/('phase-'+a.phase+'-profile.json'),dict(phase=a.phase,seconds=time.monotonic()-start,
             gpu_peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)]))
        return 0
    if a.phase=='generate':
        import torch.distributed as dist
        from research.benchmark_h3_context_parallel import ORIGINAL64_ADAPTER_SHA256,pinned_json
        from research.h3_personal.component_cache import canonical
        from research.h3_personal.continuation_conditioning import tensor_sha
        from research.h3_personal.continuation_layout import verify_modules,build_layout
        from research.h3_personal.continuation_plan import plan
        from research.run_h3_acceleration_matrix import SOURCE_DIGESTS
        from research.h3_personal.context_parallel_plan import DIFFUSERS_REVISION,verify_source_files
        from research.h3_personal.context_parallel_runtime import install
        if a.checkpoint is None or not a.conditioning_sha256:raise ValueError('pinned conditioning/checkpoint required')
        meta=pinned_json(root/'private-conditioning.json',a.conditioning_sha256)
        if (meta['schema']!='h3-private-native-conditioning-v1' or not meta['private_conditioning_do_not_upload'] or
            meta['request_sha256']!=train.sha256(a.request) or meta['case_sha256']!=train.sha256(a.case) or
            meta['checkpoint_export_sha256']!=train.sha256(a.checkpoint/'EXPORTED.json') or
            train.sha256(a.checkpoint/'adapter_model.safetensors')!=ORIGINAL64_ADAPTER_SHA256):
            raise ValueError('native request/checkpoint binding differs')
        expected={'reference.safetensors','reference-binding.json','text.safetensors','text-binding.json','audio.safetensors','audio-report.json'}
        if set(meta['files'])!=expected or any(train.sha256(root/n)!=sha for n,sha in meta['files'].items()):
            raise ValueError('private conditioning files changed')
        ref=load_file(str(root/'reference.safetensors'));text=load_file(str(root/'text.safetensors'));audio=load_file(str(root/'audio.safetensors'))['latents']
        rb=train.read_json(root/'reference-binding.json');tb=train.read_json(root/'text-binding.json');ab=train.read_json(root/'audio-report.json')
        for report in (rb,tb):
            if hashlib.sha256(canonical({k:v for k,v in report.items() if k!='key'})).hexdigest()!=report['key']:
                raise ValueError('conditioning key differs')
        if (rb['base_identity']!=base_identity or rb['geometry_policy']!='correct-hw-v2' or
            tb['reference_key']!=rb['key'] or rb['prepared_size_wh']!=[2048,3648] or
            tensor_sha(ref['latents'])!=rb['reference_sha256'] or tensor_sha(ref['geometry'])!=rb['geometry_sha256'] or
            tensor_sha(text['prompt_embeds'])!=tb['text_sha256'] or tensor_sha(text['text_token_tags'])!=tb['tags_sha256'] or
            tensor_sha(audio)!=ab['latents_sha256'] or ab['source_sha256']!=AUDIO_SHA):
            raise ValueError('corrected reference/text/audio tensors not consistently bound')
        rank=int(os.environ['LOCAL_RANK'])
        if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('exact2rank torchrun required')
        torch.cuda.set_device(rank);device=f'cuda:{rank}'
        dist.init_process_group('nccl',device_id=torch.device(device),timeout=datetime.timedelta(seconds=300))
        try:
            import diffusers
            from diffusers.modular_pipelines.minimax_h3 import packing,packing_ref2va
            from diffusers import MiniMaxH3Scheduler
            from h3_trainer.packing import prepared_references_from_cache
            from h3_trainer.model_loader import load_transformer
            from h3_trainer.lora import set_trainable
            from research.h3_personal.turbo_adapter import compose
            from research.h3_personal.continuation_runtime import run
            verify_modules(packing,packing_ref2va)
            source=Path(diffusers.__file__).resolve().parent
            receipts=[dict(file=n,sha256=h,revision=DIFFUSERS_REVISION) for n,h in SOURCE_DIGESTS.items()]
            verify_source_files(source,receipts)
            if rank==0:train.validate_weights(request)
            dist.barrier()
            image=prepared_references_from_cache([('image',ref['latents'],ref['geometry'])])[0]
            layouts=[build_layout(packing,packing_ref2va,text_tags=text['text_token_tags'],image=image,
                     window=w,height=84,width=48,ranks=2) for w in plan(685,window_frames,22)]
            rows=tuple(dict.fromkeys(x.sequence_length for x in layouts))
            model=load_transformer(request['model'],variant='ref2va',dtype=torch.bfloat16,device=torch.device(device),quantization='none')
            set_trainable(model,'lora',config.lora)
            from research.h3_personal.continuation_checkpoint import rank_checkpoint
            local_checkpoint=rank_checkpoint(a.checkpoint,root,rank)
            train.helper().strict_reload(model,local_checkpoint,freshly_constructed=True,expected_pins=request['pins'])
            model.eval().requires_grad_(False)
            turbo_path=Path(case['local_turbo'])
            if turbo_path!=root/'turbo-local.safetensors':raise ValueError('private local Turbo path required')
            turbo=compose(model,turbo_path)
            acceleration=None
            if experiment is not None:
                from research.h3_personal.acceleration_experiment import apply
                acceleration=apply(model,experiment)
            cp=install(model,sequence_rows=rows[0],allowed_sequence_rows=rows,source_root=source,source_receipts=receipts)
            loaded=time.monotonic()-start
            def progress(value):
                if rank==0:print('NATIVE_WINDOW_COMPLETE',json.dumps(value),flush=True)
            video,locked,profile=run(model,lambda:MiniMaxH3Scheduler(shift=12),packing,packing_ref2va,
                text=text['prompt_embeds'],text_tags=text['text_token_tags'],image=image,image_rows=ref['latents'],
                source_audio=audio,total_frames=685,height=84,width=48,seed=314159,device=device,ranks=2,
                window_frames=window_frames,
                on_window=progress,set_sequence_rows=cp['set_sequence_rows'])
            profile.update(precision='bf16',quantization='none',turbo=turbo,model_load_and_gate_seconds=loaded,
                cp_shape_asserted_forwards=len(cp['observed_blocks'])//2,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                identity_quality_verified=False,realtime_verified=False)
            if experiment is not None:
                profile.update(acceleration_experiment=experiment,acceleration=acceleration,
                    precision='mixed-bf16-fp8' if experiment['quantization']!='none' else 'bf16',
                    quantization=experiment['quantization'])
            peaks=[None,None];dist.all_gather_object(peaks,profile['gpu_peak_allocated_bytes'])
            if rank==0:
                profile['gpu_peak_allocated_bytes']=peaks
                save_file({'latents':video},str(root/'video-latents.safetensors'))
                dump(root/'continuation-profile.json',profile)
            dist.barrier()
        finally:dist.destroy_process_group()
        return 0
    # Independent fresh decoder process: DiT/Qwen are not resident.
    from h3_trainer.preprocessing.encoders import H3Encoders
    from h3_trainer.preprocessing.media import write_video_with_audio
    from research.h3_personal.continuation_export import mux
    enc=H3Encoders(request['model'],need_video=True,need_audio=False,need_text=False)
    try:
        rows=load_file(str(root/'video-latents.safetensors'))['latents']
        if rows.shape!=(202*42*24,96):raise ValueError('cumulative video lattice mismatch')
        incremental_report=None
        if experiment is not None and experiment['incremental_decode']:
            from research.h3_personal.incremental_decode import write_incremental_video
            incremental_report=write_incremental_video(enc,rows,root/'video-only.mp4')
        else:
            frames=enc.decode_video(rows,202,84,48)
            if len(frames)!=685:raise ValueError('native decoder frame count differs')
            write_video_with_audio(frames[:683],root/'video-only.mp4',waveform=None)
    finally:enc.unload()
    report=mux(root/'video-only.mp4',case['audio_path'],root/'review.mp4',expected_sha=AUDIO_SHA)
    report.update(model_frames=685,export_frames=683,trimmed_video_only_tail_frames=2,
        seconds=time.monotonic()-start,gpu_peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
        geometry_policy='correct-hw-v2',identity_quality_verified=False,realtime_verified=False)
    if experiment is not None:
        report.update(acceleration_experiment=experiment,incremental_decode=incremental_report,
                      denoise_decode_overlap=False,live_transport=False)
    dump(root/'decode-profile.json',report)
    return 0


if __name__=='__main__':raise SystemExit(main())
