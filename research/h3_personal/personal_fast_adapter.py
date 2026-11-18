"""Derived constructor adapter, NOT mutation/fusion of original checkpoint files.

Pinned FastVideo defaults missing alpha to rank. Both accepted source adapters
must have alpha/rank=1, no rsLoRA/DoRA/patterns, strength1. Concatenating rank axes
then gives B_fast A_fast + B_identity A_identity without a new scale convention.
Dense FastH3 payloads pass through unchanged. GPU quality/identity NOT proven.
"""

def validate_identity_config(config):
    if config.get('r')!=16 or config.get('lora_alpha')!=16:
        raise ValueError('requires original rank16/alpha16 identity adapter')
    if any(config.get(k) for k in ('rank_pattern','alpha_pattern','use_rslora','use_dora','modules_to_save')):
        raise ValueError('unsupported identity scale/module convention')
    if config.get('bias','none')!='none':raise ValueError('identity bias unsupported')


def pairs(state,*,allow_dense):
    result={};other={}
    for name,value in state.items():
        if '.lora_A.' in name or '.lora_B.' in name:
            module,tail=name.split('.lora_',1);side,suffix=tail.split('.',1)
            if suffix not in ('weight','default.weight') or side not in ('A','B'):
                raise ValueError('unsupported LoRA key')
            group=result.setdefault(module,{})
            if side in group:raise ValueError('duplicate normalized LoRA side')
            group[side]=(name,value)
        elif allow_dense and name.endswith(('.diff','.diff_b','.set_weight')):other[name]=value
        else:raise ValueError('unexpected tensor key: '+name)
    for module,pair in result.items():
        if set(pair)!={'A','B'}:raise ValueError('incomplete LoRA pair: '+module)
        a,b=pair['A'][1],pair['B'][1]
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1]:raise ValueError('invalid rank axes')
    return result,other


def assert_loaded_coverage(state,consumed,adapted_count):
    missing=set(state)-set(consumed)
    personal=[k for k,v in state.items() if k in consumed and k.endswith('.lora_A') and len(v.shape)==2 and v.shape[0]==80]
    all_a=[k for k in state if k.endswith('.lora_A')]
    complete=all(k[:-len('.lora_A')]+'.lora_B' in state for k in all_a)
    if missing or len(personal)!=208 or adapted_count!=362 or len(all_a)!=362 or not complete:
        raise RuntimeError(f'personal loaded coverage failed: unmatched={len(missing)}, rank80={len(personal)}, adapted={adapted_count}, pairs={len(all_a)}')
    report={'identity_modules':len(personal),'total_modules':adapted_count,'unmatched':0}
    import os,json
    folder=os.environ.get('H3_PERSONAL_COVERAGE_DIR')
    if folder:
        import torch
        from pathlib import Path
        if not torch.distributed.is_initialized():raise RuntimeError('distributed personal coverage required')
        rank=torch.distributed.get_rank();report['rank']=rank
        Path(folder).mkdir(parents=True,exist_ok=True)
        Path(folder,f'coverage-rank-{rank}.json').write_text(json.dumps(report)+'\n')
    print('PERSONAL_LORA_LOADED',report,flush=True)


def install_coverage_guard(source,*,expected_sha='9db2b62897bc5f14e8cf117c5467875c41498856a6d1893c6e15a41ecab0652c'):
    """Explicit overlay of the immutable source in the isolated Job copy only."""
    source=__import__('pathlib').Path(source)
    text=source.read_text()
    marker='        # The loop above reports model layers the adapter has nothing for. This is the\n'
    if text.count(marker)!=1 or 'assert_loaded_coverage(self.lora_adapters' in text:
        raise ValueError('pinned LoRA loader guard anchor differs or overlay already installed')
    hook=("        from research.h3_personal.personal_fast_adapter import assert_loaded_coverage\n"
          "        assert_loaded_coverage(self.lora_adapters[lora_nickname], consumed, adapted_count)\n")
    original_sha=__import__('hashlib').sha256(source.read_bytes()).hexdigest()
    if original_sha!=expected_sha:raise ValueError('pinned LoRA source SHA mismatch')
    source.write_text(text.replace(marker,hook+marker))
    return dict(original_sha256=original_sha,patched_sha256=__import__('hashlib').sha256(source.read_bytes()).hexdigest())


def compose(fast,identity,identity_config):
    """CPU tensors only; no in-place edits. Caller verifies full file hashes first.

    Explicitly supports only the audited implicit FastH3 alpha=rank convention;
    any serialized alpha/unknown key is rejected rather than silently ignored.
    """
    import torch
    validate_identity_config(identity_config)
    fast_pairs,dense=pairs(fast,allow_dense=True);identity_pairs,_=pairs(identity,allow_dense=False)
    if not identity_pairs or not set(identity_pairs)<=set(fast_pairs):raise ValueError('identity target absent from FastH3 adapter')
    result=dict(fast);ranks={}
    for module,personal in identity_pairs.items():
        existing=fast_pairs[module];a,b=existing['A'][1],existing['B'][1];ia,ib=personal['A'][1],personal['B'][1]
        if any(t.device.type!='cpu' for t in (a,b,ia,ib)):raise ValueError('CPU derivation only')
        if any(t.dtype!=a.dtype for t in (b,ia,ib)):raise ValueError('no silent dtype conversion')
        if a.shape[1]!=ia.shape[1] or b.shape[0]!=ib.shape[0] or ia.shape[0]!=16:
            raise ValueError('identity/base dimensions differ')
        result[existing['A'][0]]=torch.cat((a,ia),dim=0)
        result[existing['B'][0]]=torch.cat((b,ib),dim=1)
        ranks[module]=a.shape[0]+ia.shape[0]
    return result,dict(identity_modules=len(identity_pairs),total_modules=len(fast_pairs),
        dense_tensors_preserved=len(dense),combined_ranks=ranks,constructor_only=True,
        originals_modified=False,identity_quality_verified=False,gpu_parity_verified=False)
