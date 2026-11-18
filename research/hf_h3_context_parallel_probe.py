"""Two-H200 actual H3-class CP gate with tiny random weights, no personal assets.

No pretrained H3/identity/voice downloads. Not production quality or latency proof.
Requires a pinned, preinstalled Diffusers runtime and the adjacent plan module.
"""
import argparse
import json
import os
from pathlib import Path


def worker(rank, root, port):
    import datetime
    import time
    import torch
    import torch.distributed as dist
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.models._modeling_parallel import ContextParallelConfig
    from context_parallel_plan import block_plan, materialize_plan
    torch.cuda.set_device(rank)
    if 'H200' not in torch.cuda.get_device_name(rank):
        raise ValueError('H200 only')
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',world_size=2,rank=rank,
                            timeout=datetime.timedelta(seconds=120))
    try:
        torch.manual_seed(314159)
        model=MiniMaxH3Transformer3DModel(num_attention_heads=4,attention_head_dim=32,
            hidden_size=128,num_layers=2,num_refiner_layers=1,ffn_dim=256,
            in_channels=4,audio_in_channels=8,patch_size=(1,1,1),text_dim=64,
            freq_dim=32,time_embed_hidden_dim=128,time_embed_dim=64,rope_freq_dim=4)
        model=model.to(f'cuda:{rank}',torch.bfloat16).eval().requires_grad_(False)
        for name in ('proj_in','audio_proj_in','time_embedder','proj_out','audio_proj_out','rope'):
            getattr(model,name).float()
        model.set_attention_backend('_native_flash')
        device=f'cuda:{rank}'
        def inputs(seed):
            rng=torch.Generator(device='cpu').manual_seed(seed)
            def normal(shape):return torch.randn(shape,generator=rng).to(device,torch.bfloat16)
            return dict(hidden_states=normal((1,16,4)),audio_hidden_states=normal((1,8,8)),
                encoder_hidden_states=normal((1,8,64)),
                timestep=torch.tensor([0.0,0.5,1.0],device=device),
                timestep_indices=torch.tensor([0]*8+[1]*8+[2]*16,device=device),
                token_tags=torch.tensor([1]*8+[2]*8+[0]*16,device=device),
                position_ids=torch.arange(32,device=device).float()[:,None].repeat(1,3),
                text_indices=torch.arange(8,device=device),
                audio_indices=torch.arange(8,16,device=device),
                video_indices=torch.arange(16,32,device=device),return_dict=False)
        seeds=[314159,42,7]
        with torch.no_grad():reference=[model(**inputs(seed)) for seed in seeds]
        plan=block_plan(layers=2,sequence_rows=32,heads=4)
        model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2,ring_degree=1),
                                 cp_plan=materialize_plan(plan))
        # Generic enable_parallelism touches replicated text refinement too.
        for module in model.token_refiner.modules():
            processor=getattr(module,'processor',None)
            if processor is not None and hasattr(processor,'_parallel_config'):
                processor._parallel_config=None
        shapes=[]
        handles=[model.transformer_blocks[i].register_forward_hook(
            lambda module,args,output,i=i: shapes.append([i,list(output.shape)])) for i in range(2)]
        cases=[]
        with torch.no_grad():
            for seed,expected in zip(seeds,reference):
                torch.cuda.synchronize();start=time.monotonic();actual=model(**inputs(seed))
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                errors=[]
                for a,b in zip(actual,expected):
                    if a.shape!=b.shape or not bool(torch.isfinite(a).all()) or not torch.allclose(a,b,rtol=.03,atol=.005):
                        raise ValueError('tiny H3 context parallel numeric parity failed')
                    errors.append(float((a.float()-b.float()).abs().max()))
                cases.append(dict(seed=seed,cp_forward_seconds=elapsed,max_abs_errors=errors,
                                  bitwise_equal=all(torch.equal(a,b) for a,b in zip(actual,expected))))
        for handle in handles:handle.remove()
        if shapes != [[0,[1,16,128]],[1,[1,32,128]]]*3:
            raise ValueError(f'local/gather block shape differs: {shapes}')
        report=dict(rank=rank,gpu=torch.cuda.get_device_name(rank),torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,local_and_gather_shapes=shapes,cases=cases,
                    numeric_parity=True,rtol=.03,atol=.005,backend='_native_flash',
                    algorithm='ulysses_degree2',replicated_refiner_cp_disabled=True)
        (Path(root)/f'rank-{rank}.json').write_text(json.dumps(report,indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(plan_only=True,pretrained_weights=False,identity_verified=False)));return 0
    os.umask(0o077)
    import socket
    import torch
    import torch.multiprocessing as mp
    if torch.cuda.device_count()!=2 or torch.__version__!='2.8.0+cu128':
        raise ValueError('exact two GPU/Torch baseline runtime required')
    args.output.mkdir(mode=0o700,exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(str(args.output),port),nprocs=2,join=True)
    report=dict(schema='h3-two-h200-tiny-model-cp-probe-v1',
        ranks=[json.loads((args.output/f'rank-{i}.json').read_text()) for i in range(2)],
        pretrained_weights=False,production_geometry=False,identity_quality_verified=False,
        production_speed_verified=False,realtime_verified=False)
    (args.output/'probe-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return 0


if __name__=='__main__':raise SystemExit(main())
