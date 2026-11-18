"""Finite two-H200 primitive gate: NCCL, BF16 attention, gather parity.

No H3 models, identity media, training, fallback or real-time claims.
This is NOT whole-model context parallel validation or a Job launcher.
"""
import argparse
import json
import os
from pathlib import Path
import time


def rank_worker(rank, root, port):
    import datetime
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=120))
    try:
        device=f'cuda:{rank}'
        if 'H200' not in torch.cuda.get_device_name(rank):
            raise ValueError('exact H200 required; no fallback')
        rng=torch.Generator(device='cpu').manual_seed(314159)
        q,k,v=[torch.randn((1,56,1024,128),generator=rng,dtype=torch.float32).to(device,torch.bfloat16) for _ in range(3)]
        torch.cuda.synchronize();start=time.monotonic()
        with torch.no_grad(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected=F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=False)
            local=F.scaled_dot_product_attention(q[:,:,rank*512:(rank+1)*512].contiguous(),k,v,
                                                dropout_p=0,is_causal=False)
        torch.cuda.synchronize();attention_seconds=time.monotonic()-start
        shards=[torch.empty_like(local) for _ in range(2)]
        start=time.monotonic();dist.all_gather(shards,local);torch.cuda.synchronize()
        gather_seconds=time.monotonic()-start
        actual=torch.cat(shards,dim=2)
        error=(actual.float()-expected.float()).abs().max().item()
        finite=bool(torch.isfinite(actual).all())
        passed=finite and torch.allclose(actual,expected,rtol=0.01,atol=0.001)
        if not passed:
            raise ValueError(f'BF16 attention parity failed maxabs={error}')
        (Path(root)/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            gpu=torch.cuda.get_device_name(rank), capability=list(torch.cuda.get_device_capability(rank)),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,
            nccl_version=list(torch.cuda.nccl.version()),attention_seconds=attention_seconds,
            gather_seconds=gather_seconds,max_absolute_error=error,
            bf16_attention_parity=True,bitwise_equal=bool(torch.equal(actual,expected)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(plan_only=True,world_size=2,h3_model_loaded=False,
                             whole_model_cp_verified=False)));return 0
    os.umask(0o077)
    import socket
    import torch
    import torch.multiprocessing as mp
    if torch.cuda.device_count()!=2:
        raise ValueError('exactly two visible GPUs required')
    args.output.mkdir(mode=0o700,exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(rank_worker,args=(str(args.output),port),nprocs=2,join=True)
    report=dict(schema='h3-two-h200-primitive-probe-v1',
        ranks=[json.loads((args.output/f'rank-{i}.json').read_text()) for i in range(2)],
        h3_model_loaded=False,whole_model_cp_verified=False,identity_quality_verified=False,
        realtime_verified=False,
        scope='replicated full K/V, split Q and NCCL gather; not Ulysses/Ring or H3 stack')
    (args.output/'probe-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report));return 0


if __name__=='__main__':raise SystemExit(main())
