"""Explicit experimental branch; never changes the native BF16 default."""
SCHEMA='h3-fp8-streaming-experiment-v1'
TORCHAO_VERSION='0.13.0'
WHEEL_NAME='torchao-0.13.0-1-cp39-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl'
WHEEL_SHA='718568b158af481fdd6220165d3e336b7789ad8900ac40a07bf2212505e84491'
WHEEL_URL='https://files.pythonhosted.org/packages/f5/f3/5f652941cbc7d07f1d40bcfb7263919a4994f318e1cf7be45cf2b6107d46/'+WHEEL_NAME


def validate(value):
    if not isinstance(value,dict) or set(value)!={'schema','quantization','window_frames','incremental_decode','fast_accum'}:
        raise ValueError('explicit exact acceleration experiment configuration required')
    if value['schema']!=SCHEMA or value['quantization'] not in ('none','fp8-ffn','fp8-blocks'):
        raise ValueError('unknown acceleration experiment')
    if type(value['window_frames']) is not int or value['window_frames'] not in (277,192,158):
        raise ValueError('audited native window plan required')
    if any(type(value[k]) is not bool for k in ('incremental_decode','fast_accum')):
        raise ValueError('explicit decoder and accumulation flags required')
    from .continuation_plan import plan
    plan(685,value['window_frames'],22)
    return dict(value)


def selected(name,mode):
    parts=name.split('.')
    if any(p.startswith('lora_') or p=='modules_to_save' for p in parts):return False
    if 'transformer_blocks' not in parts:return False
    i=parts.index('transformer_blocks')
    if tuple(parts[:i]) not in ((),('base_model','model'),('model',)):return False
    if i+2>=len(parts) or not parts[i+1].isdigit():return False
    return parts[i+2]=='ff' or (mode=='fp8-blocks' and parts[i+2]=='attn')


def adapter_digest(model):
    import hashlib
    import torch
    digest=hashlib.sha256();count=0
    for name,p in sorted(model.named_parameters()):
        if any(part.startswith('lora_') for part in name.split('.')):
            if p.dtype!=torch.bfloat16:raise ValueError('experimental LoRA must remain BF16')
            digest.update(name.encode());digest.update(str(tuple(p.shape)).encode())
            digest.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());count+=1
    if not count:raise ValueError('named adapters required before quantization')
    return digest.hexdigest(),count


def fp8_recipe(fast_accum):
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.quantize_.common import KernelPreference
    # TorchAO0.13 v2 otherwise maps all-zero activations to scale0, then0/0.
    return Float8DynamicActivationFloat8WeightConfig(version=2,set_inductor_config=False,
        activation_value_lb=1e-12,kernel_preference=KernelPreference.TORCH,
        mm_config=Float8MMConfig(use_fast_accum=fast_accum))


def probe_fp8_linear(sample):
    import torch
    probes={}
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        for name,value in (('zero',torch.zeros((32,sample.in_features),dtype=torch.bfloat16,device=sample.weight.device)),
                           ('nonzero',torch.linspace(-1,1,32*sample.in_features,dtype=torch.bfloat16,device=sample.weight.device).reshape(32,-1))):
            result=sample(value);torch.cuda.synchronize()
            probes[name]=bool(torch.isfinite(result).all())
    ops=[e.key for e in profiler.key_averages()]
    if not all(probes.values()):raise ValueError('FP8 probe nonfinite output: '+str(probes))
    if not any('_scaled_mm' in k for k in ops):
        raise ValueError('native FP8 scaled-mm probe not observed; operators='+str(ops))
    return dict(probe_finite=probes,probe_operators=ops,native_scaled_mm_probe=True)


def apply(model,config):
    """Quantize selected base Linear weights/activations, not LoRA or VAE."""
    config=validate(config)
    if config['quantization']=='none':return dict(quantization='none',experimental=True)
    import importlib.metadata
    import time
    import torch
    if (torch.__version__!='2.8.0+cu128' or not torch.cuda.is_available() or
        torch.cuda.get_device_capability()!=(9,0) or 'H200' not in torch.cuda.get_device_name() or
        importlib.metadata.version('torchao')!=TORCHAO_VERSION):
        raise ValueError('pinned Torch2.8cu128/torchao0.13/H200 required')
    from torchao.quantization import quantize_
    from torchao.quantization.quantize_.workflows import Float8Tensor
    before,count=adapter_digest(model);start=time.monotonic()
    candidates={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and selected(n,config['quantization'])}
    if not candidates:raise ValueError('no supported base Linear modules selected')
    if any(any(d%16 for d in m.weight.shape) or m.weight.dtype!=torch.bfloat16 for m in candidates.values()):
        raise ValueError('FP8 base shape/dtype gate failed; no silent skip')
    recipe=fp8_recipe(config['fast_accum'])
    quantize_(model,recipe,filter_fn=lambda module,name:name in candidates)
    if any(not isinstance(m.weight,Float8Tensor) for m in candidates.values()):
        raise ValueError('torchao silently skipped requested FP8 weight conversion')
    after,after_count=adapter_digest(model)
    if (before,count)!=(after,after_count):raise ValueError('quantization changed named adapters')
    # Confirm actual scaled-mm execution, not only a low-bit storage label.
    sample=next(iter(candidates.values()))
    probe=probe_fp8_linear(sample)
    return dict(experimental=True,quantization=config['quantization'],torchao_version=TORCHAO_VERSION,
        wheel_sha256=WHEEL_SHA,linear_modules=len(candidates),adapter_tensors=count,
        adapter_sha256=after,adapters_unchanged=True,**probe,
        fast_accum=config['fast_accum'],activation_value_lb=1e-12,kernel_preference='TORCH',
        seconds=time.monotonic()-start,quality_verified=False)
