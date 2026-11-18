"""Explicit BF16 FA4 selection and cold-only, per-rank CUDA smoke evidence."""
import os


def configure_environment(mode, env=None):
    if mode not in ('fa2', 'fa4', 'fa4-packed'):
        raise ValueError('unsupported attention mode')
    env = os.environ if env is None else env
    env.update(FASTVIDEO_ATTENTION_BACKEND='FLASH_ATTN',
               FASTVIDEO_FA4='0' if mode == 'fa2' else '1',
               FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN='1' if mode == 'fa4-packed' else '0',
               FASTVIDEO_NVFP4_FA4='0', H3_REF2VA_ATTENTION_MODE=mode)
    return env


def cold_receipt():
    """No RNG changes; profiler/synchronization never run on the warm path.

    Proves an actual call through the installed wrapper, not every H3 attention
    invocation. CUDA event names are evidence for review, not a name heuristic.
    """
    import torch
    from fastvideo.attention.utils.flash_attn_default import fa_version, flash_attn_func_compilable
    from fastvideo.attention.utils.flash_attn_no_pad import _FA_VARLEN_VERSION
    if str(fa_version) != '4' or str(_FA_VARLEN_VERSION) != '4':
        raise RuntimeError('FA4 requested but regular/varlen dispatcher differs')
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError('this experiment requires H200 SM90')
    with torch.inference_mode():
        q = (torch.arange(128*4*128, device=device, dtype=torch.float32) % 71 / 71).reshape(1,128,4,128).to(torch.bfloat16)
        # Compile/warm CuTe outside profiler; then record a real second invocation.
        flash_attn_func_compilable(q,q,q,causal=False)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            result = flash_attn_func_compilable(q,q,q,causal=False)
            torch.cuda.synchronize()
        kernels = sorted({e.name for e in prof.events() if str(e.device_type).split('.')[-1] == 'CUDA'})
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),q.transpose(1,2),q.transpose(1,2)).transpose(1,2)
        if not kernels or result.dtype != torch.bfloat16 or not torch.isfinite(result).all():
            raise RuntimeError('missing CUDA evidence or invalid BF16 FA4 output')
        torch.testing.assert_close(result,expected,atol=.02,rtol=.02)
    return dict(backend=str(fa_version),varlen_backend=str(_FA_VARLEN_VERSION),
                capability=list(capability),dtype='bfloat16',cuda_kernel_events=kernels,
                cold_smoke_only=True,full_model_kernel_trace=False,
                smoke_math_tolerance=dict(atol=.02,rtol=.02),packed_varlen=False,nvfp4=False)


def start_packed_counter(backend):
    """Count only the cold native packed branch's module-local callable."""
    if hasattr(backend,'_h3_cold_packed_counter'):raise RuntimeError('counter already installed')
    original=backend.flash_attn_varlen_func_compilable
    if backend.fa_version!='4' or original is None:raise RuntimeError('missing FA4 packed callable')
    state={'calls':0,'original':original,'active':True}
    def counted(*args,**kwargs):
        state['calls']+=1
        return original(*args,**kwargs)
    state['wrapper']=counted
    backend.flash_attn_varlen_func_compilable=counted
    backend._h3_cold_packed_counter=state


def finish_packed_counter(backend):
    state=backend._h3_cold_packed_counter
    if state['active']:
        if backend.flash_attn_varlen_func_compilable is not state['wrapper']:raise RuntimeError('packed probe callable changed')
        backend.flash_attn_varlen_func_compilable=state['original']
        state['active']=False
    return dict(cold_packed_calls=state['calls'],warm_counter_removed=True,
                scope='native backend module packed branch, first clip only',
                packed_route_executed=state['calls']>0)

