"""Experimental named LightX2V Ref2VA8 adapter composition; never fuse identity."""
from .adapter_header import header_length, inspect_header
from .component_cache import sha256

TURBO_SHA256='9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2'
TURBO_NAME='h3_ref2va_turbo8'


def validate_contract(report):
    modules=report['peft_modules']
    if (report['metadata'].get('alpha')!='8' or report['other_tensor_names'] or
        len(modules)!=312 or report['tensor_count']!=624 or
        {m['rank'] for m in modules}!={128}):
        raise ValueError('fixed Ref2VA8 rank128/alpha8/312-module contract required')
    return modules


def compose(model,path):
    import torch
    from peft import LoraConfig
    from safetensors.torch import load_file
    if sha256(path)!=TURBO_SHA256:raise ValueError('exact immutable Turbo weight SHA required')
    with open(path,'rb') as f:
        prefix=f.read(8);report=inspect_header(prefix,f.read(header_length(prefix)))
    modules=validate_contract(report)
    if model.active_adapters()!=['default'] or TURBO_NAME in model.peft_config:
        raise ValueError('only existing original default identity adapter allowed')
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if '.default.' in k}
    if not before:raise ValueError('identity adapter missing')
    named=dict(model.named_modules())
    for m in modules:
        layer=named[m['module']]
        if (layer.in_features,layer.out_features)!=(m['input_features'],m['output_features']):
            raise ValueError('Turbo/base module shape mismatch')
    model.add_adapter(LoraConfig(r=128,lora_alpha=8,lora_dropout=0,
        target_modules=[m['module'] for m in modules],init_lora_weights=False,use_rslora=False),
        adapter_name=TURBO_NAME)
    state={k.replace('.default.','.'+TURBO_NAME+'.'):v for k,v in load_file(str(path)).items()}
    expected={k:v for k,v in model.state_dict().items() if '.'+TURBO_NAME+'.' in k}
    if set(expected)!=set(state) or any(expected[k].shape!=v.shape for k,v in state.items()):
        raise ValueError('exact named Turbo tensor consumption required')
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys or any('.'+TURBO_NAME+'.' in k for k in result.missing_keys):
        raise ValueError('Turbo tensor reload incomplete')
    model.set_adapters(['default',TURBO_NAME],weights=[1.0,1.0])
    after=model.state_dict()
    if any(not torch.equal(value,after[k].detach().cpu()) for k,value in before.items()):
        raise ValueError('identity adapter changed during composition')
    if model.active_adapters()!=['default',TURBO_NAME]:raise ValueError('both adapters must remain active')
    model.eval().requires_grad_(False)
    return dict(name=TURBO_NAME,sha256=TURBO_SHA256,rank=128,alpha=8,scale=1.0,
                effective_scale=.0625,identity_active=True,identity_weights_unchanged=True,fused=False)
