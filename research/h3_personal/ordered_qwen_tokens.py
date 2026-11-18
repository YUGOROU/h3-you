"""Actual ordered Qwen presentation admission; no tokenizer/model imports.

Called with the SAME native processor and presentation outputs before encoder.
Proof consistency is not independent execution authentication or GPU fitness.
"""
from copy import deepcopy
import hashlib
import json
import re

SCHEMA='h3-ordered-qwen-presentation-v1'
PROOF_KEY='_h3_ordered_qwen_tokens'
MAX_MEMO_PROOFS=8


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('canonical token proof SHA256 required')
    return value


def ordered_token_policy(edge):
    """Immutable OPERATIONAL bounds, not native architecture/memory capacity.

    Packed envelope is separately selected finite admission, not extrapolated
    from VAE R or training70000. Parent checks actual native sequence_length.
    """
    from .conditioning_experiment import ordered_admission_bounds
    limits=ordered_admission_bounds(edge)
    return dict(schema='h3-ordered-token-admission-v1',short_edge=edge,
                max_qwen_tokens=limits['max_tokens'],max_reference_rows=limits['max_reference_rows'],
                max_text_overhead=511,max_packed_rows=32768 if edge==1024 else 16384,
                admission_kind='operational-not-native-capacity')


def _native_list(value):
    # Native tensor metadata is read without torch dependency or pixel transfer.
    if hasattr(value,'tolist'):value=value.tolist()
    return value


def _ints(value,label,maximum):
    value=_native_list(value)
    if (type(value) is not list or len(value)>maximum
            or any(type(n) is not int or n<0 for n in value)):
        raise ValueError(label+' requires bounded nonnegative native integers')
    return value


def _runs(tags):
    runs=[]
    for tag in tags:
        if not runs or runs[-1][0]!=tag:runs.append([tag,1])
        else:runs[-1][1]+=1
    return runs


def _context_sha(context):
    from .ordered_references import content_identity,content_sha256
    if (type(context) is not dict or 'ordered_references' not in context
            or context.get('identity')!=content_identity(context['ordered_references'])
            or context.get('identity_sha256')!=content_sha256(context['ordered_references'])):
        raise ValueError('bound ordered context required')
    return _sha(context['identity_sha256'])


def validate_proof(proof,context,edge,prompt_sha256,expected_T=None):
    """Strict metadata roundtrip; never tokenizes or infers Qwen counts from VAE."""
    policy=ordered_token_policy(edge);identity_sha=_context_sha(context)
    _sha(prompt_sha256)
    fields={'schema','identity_sha256','prompt_sha256','policy','policy_sha256','merge_size',
            'image_grid_thw','image_token_counts','segments','total_tokens','text_overhead',
            'token_ids_sha256','token_tags_sha256','proof_sha256'}
    if type(proof) is not dict or set(proof)!=fields or proof['schema']!=SCHEMA:
        raise ValueError('exact actual Qwen token proof required')
    if (proof['identity_sha256']!=identity_sha or proof['prompt_sha256']!=prompt_sha256
            or _json(proof['policy'])!=_json(policy) or proof['policy_sha256']!=_digest(policy)):
        raise ValueError('Qwen context/prompt/admission drift')
    merge=proof['merge_size'];grids=proof['image_grid_thw'];counts=proof['image_token_counts']
    if type(merge) is not int or not 1<=merge<=64:
        raise ValueError('actual positive bounded merge size required')
    image_count=len(context['identity']['references'])
    if type(grids) is not list or len(grids)!=image_count or type(counts) is not list or len(counts)!=image_count:
        raise ValueError('actual Qwen image grids/counts must match bound reference identity')
    for grid,count in zip(grids,counts):
        if (type(grid) is not list or len(grid)!=3
                or any(type(n) is not int or not 0<n<=262144 for n in grid)
                or type(count) is not int or count<=0):
            raise ValueError('actual Qwen grid/count shape required')
        product=grid[0]*grid[1]*grid[2]
        if product%(merge*merge) or product//(merge*merge)!=count:
            raise ValueError('actual processor grid/merge/token count disagreement')
    total=proof['total_tokens'];overhead=proof['text_overhead']
    if (type(total) is not int or not 0<total<=policy['max_qwen_tokens']
            or type(overhead) is not int or not 0<=overhead<=policy['max_text_overhead']
            or overhead!=total-sum(counts)):
        raise ValueError('Qwen presentation/overhead admission exceeded')
    if expected_T is not None and (type(expected_T) is not int or total!=expected_T):
        raise ValueError('cached/native actual total T mismatch')
    segments=proof['segments']
    kinds=tuple(kind for _ in range(image_count) for kind in ('label','vision'))+('prompt',)
    if type(segments) is not list or len(segments)!=len(kinds):
        raise ValueError('ordered label/vision/prompt token segments required')
    for index,(segment,kind) in enumerate(zip(segments,kinds)):
        if (type(segment) is not dict or set(segment)!={'kind','tokens','token_ids_sha256','image_index'}
                or segment['kind']!=kind or type(segment['tokens']) is not int or segment['tokens']<=0
                or type(segment['image_index']) is not int
                or segment['image_index']!=(index//2 if index<2*image_count else -1)):
            raise ValueError('ordered actual presentation segment drift')
        _sha(segment['token_ids_sha256'])
    if ([segments[2*i+1]['tokens'] for i in range(image_count)]!=[n+2 for n in counts]
            or sum(s['tokens'] for s in segments)!=total):
        raise ValueError('vision boundaries/counts/total disagreement')
    for key in ('token_ids_sha256','token_tags_sha256','proof_sha256'):_sha(proof[key])
    if proof['token_tags_sha256']!=_digest([tag for tag,s in zip(tuple(tag for _ in range(image_count) for tag in (1,0))+(1,),segments) for _ in range(s['tokens'])]):
        raise ValueError('actual token tag runs mismatch')
    if proof['proof_sha256']!=_digest({k:v for k,v in proof.items() if k!='proof_sha256'}):
        raise ValueError('Qwen token proof digest mismatch')
    return json.loads(_json(proof))


def capture_qwen_presentation(stage,batch,image_grid_thw,merge_size,image_token_counts,token_ids,token_tags):
    """Native hook AFTER builder, BEFORE _encode_tokens; no retokenization."""
    from .conversation_conditioning import read_ordered_context
    from .conditioning_experiment import reference_short_edge
    context=read_ordered_context()
    if context is None:return None
    edge=reference_short_edge();policy=ordered_token_policy(edge)
    references=list(batch.references or [])
    image_count=len(context['identity']['references'])
    if (len(references)!=image_count or any(getattr(r,'media_type',None)!='image'
            or getattr(r,'has_audio',False) for r in references)):
        raise ValueError('actual Qwen image-only references must match bound identity')
    if type(batch.prompt) is not str or not batch.prompt:
        raise ValueError('actual nonempty prompt required')
    ids=_ints(token_ids,'actual token ids',policy['max_qwen_tokens'])
    tags=_ints(token_tags,'actual token tags',policy['max_qwen_tokens'])
    if len(ids)!=len(tags) or any(tag not in (0,1) for tag in tags):
        raise ValueError('actual token/tag sequence mismatch')
    runs=_runs(tags)
    if [r[0] for r in runs]!=[tag for _ in range(image_count) for tag in (1,0)]+[1]:
        raise ValueError('native candidate must contain exact label/vision pairs followed by prompt')
    grids=_native_list(image_grid_thw)
    counts=_ints(image_token_counts,'actual image token counts',image_count)
    segments=[];offset=0
    kinds=tuple(kind for _ in range(image_count) for kind in ('label','vision'))+('prompt',)
    for index,((_,length),kind) in enumerate(zip(runs,kinds)):
        segments.append(dict(kind=kind,tokens=length,token_ids_sha256=_digest(ids[offset:offset+length]),
                             image_index=index//2 if index<2*image_count else -1))
        offset+=length
    proof=dict(schema=SCHEMA,identity_sha256=_context_sha(context),
        prompt_sha256=hashlib.sha256(batch.prompt.encode()).hexdigest(),policy=policy,policy_sha256=_digest(policy),
        merge_size=merge_size,image_grid_thw=grids,image_token_counts=counts,segments=segments,
        total_tokens=len(ids),text_overhead=len(ids)-sum(counts),
        token_ids_sha256=_digest(ids),token_tags_sha256=_digest(tags))
    proof['proof_sha256']=_digest(proof)
    proof=validate_proof(proof,context,edge,proof['prompt_sha256'],expected_T=len(ids))
    batch.extra[PROOF_KEY]=deepcopy(proof)
    stage._h3_ordered_qwen_last_proof=deepcopy(proof)
    return proof


def validate_cached_tags(proof,tags):
    """Compare actual cached native tags, not a claimed count or inferred grid."""
    if type(proof) is not dict or type(proof.get('total_tokens')) is not int:
        raise ValueError('validated Qwen proof required for cached tags')
    actual=_ints(tags,'actual cached native token tags',proof['total_tokens'])
    if (len(actual)!=proof['total_tokens'] or any(tag not in (0,1) for tag in actual)
            or _digest(actual)!=proof.get('token_tags_sha256')):
        raise ValueError('actual cached native token tags differ from Qwen proof')
    return actual


def restore_role_proof(stage,batch,key,expected_T):
    """Restore only the current bounded role-bank proof, never infer grids."""
    from .conversation_conditioning import read_ordered_context
    from .conditioning_experiment import reference_short_edge
    context=read_ordered_context()
    if context is None:return None
    batch.extra.pop(PROOF_KEY,None)
    bank=getattr(stage,'_h3_role_condition_proofs',None)
    if type(bank) is not dict or not 0<len(bank)<=MAX_MEMO_PROOFS:
        raise ValueError('bounded current role proof bank required')
    record=bank.get(key)
    if type(record) is not dict:
        raise ValueError('current native role proof missing')
    proof=validate_proof(record.get('qwen_tokens'),context,reference_short_edge(),
                         hashlib.sha256(batch.prompt.encode()).hexdigest(),expected_T=expected_T)
    validate_cached_tags(proof,batch.extra.get('minimax_h3_text_token_tags'))
    batch.extra[PROOF_KEY]=deepcopy(proof)
    return proof


def memo_presentation(stage,batch,key,compute):
    """Wrap existing native memo, restoring only matching bounded proof on hit.

    Native memo must be enabled by caller exactly as upstream. A memo hit with
    missing proof is rejected, never fabricated/re-encoded. Both outputs and
    metadata remain owned by the existing stage.
    """
    from .conversation_conditioning import read_ordered_context
    from .conditioning_experiment import reference_short_edge
    context=read_ordered_context()
    if context is None:return stage._ref2va_memo.get_or_compute(key,compute)
    edge=reference_short_edge();prompt_sha=hashlib.sha256(batch.prompt.encode()).hexdigest()
    cache=getattr(stage,'_h3_ordered_qwen_memo_proofs',None)
    if cache is None:cache={};stage._h3_ordered_qwen_memo_proofs=cache
    if type(cache) is not dict or len(cache)>MAX_MEMO_PROOFS:
        raise ValueError('bounded native Qwen proof bank required')
    called=False
    batch.extra.pop(PROOF_KEY,None)
    def owned_compute():
        nonlocal called
        called=True
        if key not in cache and len(cache)>=MAX_MEMO_PROOFS:
            raise ValueError('native Qwen proof bank exhausted')
        value=compute()
        proof=validate_proof(batch.extra.get(PROOF_KEY),context,edge,prompt_sha)
        cache[key]=deepcopy(proof)
        return value
    result=stage._ref2va_memo.get_or_compute(key,owned_compute)
    if not called:
        proof=validate_proof(cache.get(key),context,edge,prompt_sha)
        batch.extra[PROOF_KEY]=deepcopy(proof)
    return result
