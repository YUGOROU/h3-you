"""Opt-in installer for the GPU-tested H3 block-boundary Ulysses plan.

Call only after exact source verification. Does not authorize launches or validate
pretrained weights, whole-trajectory equivalence, identity, or deployment region.
"""
from .context_parallel_plan import block_plan, materialize_plan, verify_source_files


def install(model, *, sequence_rows, source_root, source_receipts, allowed_sequence_rows=None):
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ContextParallelConfig
    verify_source_files(source_root, source_receipts)
    if not dist.is_initialized() or dist.get_world_size()!=2:
        raise ValueError('initialized exactly-two-rank process group required')
    if getattr(model, '_h3_cp_installed', False):
        raise ValueError('refuse duplicate installation')
    layers=len(model.transformer_blocks)
    if layers!=model.config.num_layers:
        raise ValueError('block count differs from config')
    allowed=(sequence_rows,) if allowed_sequence_rows is None else tuple(allowed_sequence_rows)
    if not 1<=len(allowed)<=16 or sequence_rows not in allowed:
        raise ValueError('finite prevalidated CP row whitelist required')
    for rows in allowed:
        block_plan(layers=layers,sequence_rows=rows,heads=model.config.num_attention_heads)
    descriptors=block_plan(layers=layers,sequence_rows=sequence_rows,
                           heads=model.config.num_attention_heads)
    model.set_attention_backend('_native_flash')
    model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2,ring_degree=1),
                             cp_plan=materialize_plan(descriptors))
    refiners=0
    for module in model.token_refiner.modules():
        processor=getattr(module,'processor',None)
        if processor is not None and hasattr(processor,'_parallel_config'):
            processor._parallel_config=None
            refiners+=1
    if refiners!=model.config.num_refiner_layers:
        raise ValueError('unexpected replicated refiner processor count')
    observed=[]
    width=model.config.hidden_size
    state={'sequence_rows':sequence_rows,'allowed_sequence_rows':allowed}
    def set_sequence_rows(rows):
        if rows not in allowed or len(observed)%2:
            raise ValueError('unapproved row change or CP forward in progress')
        state['sequence_rows']=rows
    def check(index, divisor):
        def hook(module,args,output):
            if tuple(output.shape)!=(1,state['sequence_rows']//divisor,width):
                raise ValueError('CP local/gather shape differs')
            observed.append(index)
        return hook
    handles=[model.transformer_blocks[0].register_forward_hook(check(0,2)),
             model.transformer_blocks[-1].register_forward_hook(check(layers-1,1))]
    model._h3_cp_installed=True
    return dict(backend='_native_flash',ulysses_degree=2,ring_degree=1,
                replicated_refiner_count=refiners,observed_blocks=observed,hook_handles=handles,
                sequence_state=state,set_sequence_rows=set_sequence_rows)
