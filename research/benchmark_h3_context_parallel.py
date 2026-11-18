"""Opt-in torchrun worker: original-adapter single GPU vs two-GPU H3 CP.

Consumes PRIVATE local packed conditioning from hf_identity_preview pack-comparison.
No network, uploads, launches, TTS or step cuts. Decode/review are separate stages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ORIGINAL64_ADAPTER_SHA256 = '43fd1f8c6e4097611ba027dcf42503b4c889cfd5ef5e45d91abcb08c482ef151'


def pinned_json(path, sha):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:
        raise ValueError('pinned JSON bytes differ')
    return json.loads(data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--packed',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--metadata-sha256',required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--source-receipts',type=Path,required=True)
    p.add_argument('--source-receipts-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-trajectory',action='store_true')
    p.add_argument('--turbo',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(plan_only=True,full_trajectory=a.full_trajectory,
                             schedule_points=9 if a.turbo else 50,
                             denoiser_forwards=8 if a.turbo else 49,quality_verified=False)))
        return 0
    os.umask(0o077)
    if any(os.environ.get(k)!='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')):
        raise ValueError('offline model worker required')
    from research.h3_personal import component_cache as cache
    from research.h3_personal.context_parallel_plan import verify_source_files
    from research.h3_personal.context_parallel_runtime import install
    receipts=pinned_json(a.source_receipts,a.source_receipts_sha256)
    required={'transformer_minimax_h3.py','modeling_utils.py','attention_dispatch.py',
              '_modeling_parallel.py','context_parallel.py'}
    if len(receipts)!=5 or {Path(x['file']).name for x in receipts}!=required:
        raise ValueError('complete five-file pinned source audit required')
    verify_source_files(a.source_root,receipts)
    meta=pinned_json(a.metadata,a.metadata_sha256)
    request=json.loads(a.request.read_bytes())
    if (meta['schema']!='h3-private-packed-comparison-v1' or
        not meta['private_conditioning_do_not_upload'] or meta['schedule_points']!=50 or
        meta['denoiser_forwards']!=49 or meta['precision']!='bf16' or meta['quantization']!='none' or
        meta['request_sha256']!=hashlib.sha256(cache.canonical(request)).hexdigest() or
        meta['sha256']!=cache.sha256(a.packed) or
        meta['checkpoint_export_sha256']!=cache.sha256(a.checkpoint/'EXPORTED.json')):
        raise ValueError('packed/checkpoint/request identity differs')
    if cache.sha256(a.checkpoint/'adapter_model.safetensors') != ORIGINAL64_ADAPTER_SHA256:
        raise ValueError('immutable original64 adapter bytes required')
    import datetime
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file,save_file
    from research import hf_identity_train as train
    rank=int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count()!=2 or int(os.environ['WORLD_SIZE'])!=2:
        raise ValueError('exactly two visible GPUs/ranks required')
    torch.cuda.set_device(rank)
    if 'H200' not in torch.cuda.get_device_name(rank) or torch.__version__!='2.8.0+cu128':
        raise ValueError('exact H200/Torch baseline runtime required')
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=1800))
    try:
        if rank==0:a.output.mkdir(mode=0o700,exist_ok=False)
        dist.barrier()
        config=train.load_runtime(request)
        if rank==0:train.validate_weights(request)
        dist.barrier()
        import diffusers
        if Path(diffusers.__file__).resolve().parent != a.source_root.resolve():
            raise ValueError('audited source root is not the imported Diffusers package')
        from h3_trainer.model_loader import load_transformer
        from h3_trainer.lora import set_trainable
        from diffusers import MiniMaxH3Scheduler
        device=f'cuda:{rank}'
        model=load_transformer(request['model'],variant='ref2va',dtype=torch.bfloat16,
                               device=torch.device(device),quantization='none')
        set_trainable(model,'lora',config.lora)
        train.helper().strict_reload(model,a.checkpoint,freshly_constructed=True,
                                     expected_pins=request['pins'])
        model.eval().requires_grad_(False)
        model.set_attention_backend('_native_flash')
        turbo_receipt=None
        if a.turbo:
            from research.h3_personal.turbo_adapter import compose
            if meta['video_shift']!=12.0:raise ValueError('author shared video shift12 required')
            turbo_receipt=compose(model,a.turbo)
        packed=load_file(str(a.packed),device=device)
        static={k[7:]:v for k,v in packed.items() if k.startswith('static.')}
        if packed['row_timesteps'].shape[0]!=49 or packed['timestep_indices'].numel()!=meta['sequence_rows']:
            raise ValueError('packed schedule/row count differs')
        points=9 if a.turbo else 50
        nfe=points-1
        row_times=packed['turbo_row_timesteps'] if a.turbo else packed['row_timesteps']
        if row_times.shape[0]!=nfe:raise ValueError('row schedule NFE differs')
        def forward(x,i):
            return model(hidden_states=torch.cat([packed['reference'],x.to(torch.bfloat16)],dim=1),
                audio_hidden_states=packed['audio'],encoder_hidden_states=packed['text'],
                timestep=row_times[i],timestep_indices=packed['timestep_indices'],
                attention_kwargs=None,return_dict=False,**static)[0]
        def timed_first():
            torch.cuda.synchronize();start=time.monotonic()
            result=forward(packed['initial_video'],0)
            torch.cuda.synchronize();return result,time.monotonic()-start
        def trajectory(label):
            scheduler=MiniMaxH3Scheduler(shift=meta['video_shift'])
            scheduler.set_timesteps(points,device=device)
            if len(scheduler.timesteps)!=nfe:raise ValueError('NFE differs')
            x=packed['initial_video'].clone()
            torch.cuda.synchronize();start=time.monotonic()
            for i,t in enumerate(scheduler.timesteps):
                velocity=forward(x,i)
                x=scheduler.step(velocity[:,meta['condition_video_rows']:].float(),t,x,return_dict=False)[0]
                if not bool(torch.isfinite(x).all()):raise ValueError('nonfinite trajectory')
                if rank==0:print(label,i+1,nfe,flush=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if rank==0:save_file({'latents':x[0].cpu().contiguous()},str(a.output/(label+'-latents.safetensors')))
            return x,elapsed
        baseline_seconds=[];baseline_latents=None
        with torch.no_grad():
            if rank==0:
                expected,_=timed_first()
                expected,seconds=timed_first();baseline_seconds.append(seconds)
                if a.full_trajectory:baseline_latents,baseline_denoise=trajectory('single-gpu')
            dist.barrier()
            state=install(model,sequence_rows=meta['sequence_rows'],source_root=a.source_root,
                          source_receipts=receipts)
            actual,cold_seconds=timed_first()
            actual,warm_seconds=timed_first()
            passed=torch.tensor([1],device=device,dtype=torch.int32)
            errors=None
            if rank==0:
                errors=float((actual.float()-expected.float()).abs().max())
                if not bool(torch.isfinite(actual).all()) or not torch.allclose(actual,expected,rtol=.01,atol=.001):
                    passed.zero_()
            dist.broadcast(passed,src=0)
            if not passed.item():raise ValueError('production first-forward parity failed; no CP trajectory')
            if a.full_trajectory:cp_latents,cp_denoise=trajectory('two-gpu-cp')
        if rank==0:
            report=dict(schema='h3-original64-two-gpu-comparison-v1',precision='bf16',quantization='none',
                schedule_points=points,schedule_denoiser_forwards=nfe,full_trajectory=a.full_trajectory,
                turbo=turbo_receipt,
                single_gpu_executed_forwards=nfe+2 if a.full_trajectory else 2,
                cp_executed_forwards=nfe+2 if a.full_trajectory else 2,
                adapter_export_sha256=meta['checkpoint_export_sha256'],
                packed_input_sha256=meta['sha256'],first_forward_parity=True,
                first_forward_max_abs_error=errors,first_forward_rtol=.01,first_forward_atol=.001,
                first_forward_bitwise_equal=bool(torch.equal(actual,expected)),
                single_gpu_warm_forward_seconds=baseline_seconds[0],
                cp_cold_forward_seconds=cold_seconds,cp_warm_forward_seconds=warm_seconds,
                warm_forward_speedup=baseline_seconds[0]/warm_seconds,
                shape_asserted_forward_calls=len(state['observed_blocks'])//2,
                identity_quality_verified=False,realtime_verified=False,
                scope='DiT only; excludes pack/prepare/model load/VAE decode/transfer')
            if a.full_trajectory:
                report.update(single_gpu_denoise_seconds=baseline_denoise,cp_denoise_seconds=cp_denoise,
                    denoise_speedup=baseline_denoise/cp_denoise,
                    final_latent_max_abs_error=float((cp_latents-baseline_latents).abs().max()),
                    final_latent_bitwise_equal=bool(torch.equal(cp_latents,baseline_latents)))
            (a.output/'comparison-report.json').write_text(json.dumps(report,indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main())
