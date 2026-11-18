"""Fresh corrected reference/text preparation; NEVER reads or writes715cache.

Run reference and text phases with separately owned/unloaded backends: this API
does not instantiate/co-reside Qwen/VAE/DiT or assume available GPU memory.
"""
import hashlib
from pathlib import Path
from .component_cache import canonical
from .reference_geometry import CORRECTED


def _hash(value):return hashlib.sha256(canonical(value)).hexdigest()


def tensor_sha(tensor):
    import torch
    if tensor.device.type!='cpu':raise ValueError('private conditioning must be CPU before binding')
    t=tensor.detach().contiguous()
    h=hashlib.sha256(canonical(dict(dtype=str(t.dtype),shape=list(t.shape))))
    h.update(t.view(torch.uint8).numpy().tobytes());return h.hexdigest()


def source(path,expected):
    path=Path(path).resolve(strict=True)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    if h.hexdigest()!=expected:raise ValueError('reference source SHA mismatch')
    return path


def reference(backend,path,sha256,*,base_identity):
    import torch
    if not isinstance(base_identity,dict) or not base_identity:
        raise ValueError('explicit pinned source/model identity required')
    path=source(path,sha256);image=backend.image({'sources':{'image':{'path':str(path)}}},geometry_policy=CORRECTED)
    tensors=backend.reference(None,image)
    geometry=tensors['geometry'];latents=tensors['latents']
    expected=[0,1,image.height//16,image.width//16,0]
    if image.width%32 or image.height%32 or geometry.shape!=(1,5) or geometry[0].tolist()!=expected:
        raise ValueError('corrected reference pixel/latent geometry mismatch')
    if latents.shape!=((image.height//32)*(image.width//32),96) or not bool(torch.isfinite(latents).all()):
        raise ValueError('reference latent shape/nonfinite mismatch')
    binding=dict(schema='h3-corrected-reference-v2',geometry_policy=CORRECTED,
        source_sha256=sha256,prepared_size_wh=[image.width,image.height],geometry=expected,
        reference_sha256=tensor_sha(latents),geometry_sha256=tensor_sha(geometry),
        base_identity=base_identity,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        private_conditioning_do_not_upload=True)
    binding['key']=_hash(binding)
    return tensors,binding


def text(backend,path,sha256,reference_tensors,binding,prompt,*,base_identity):
    import torch
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('nonempty fixed prompt required')
    if binding.get('geometry_policy')!=CORRECTED or binding.get('base_identity')!=base_identity:
        raise ValueError('legacy/mismatched reference binding refused')
    if _hash({k:v for k,v in binding.items() if k!='key'})!=binding.get('key'):
        raise ValueError('reference binding changed')
    if binding.get('source_sha256')!=sha256:raise ValueError('reference source changed between phases')
    if (tensor_sha(reference_tensors['latents'])!=binding['reference_sha256'] or
        tensor_sha(reference_tensors['geometry'])!=binding['geometry_sha256']):
        raise ValueError('reference components changed between phases')
    path=source(path,sha256);image=backend.image({'sources':{'image':{'path':str(path)}}},geometry_policy=CORRECTED)
    if [image.width,image.height]!=binding['prepared_size_wh']:raise ValueError('prepared image changed')
    refs=backend.presentation(image,binding['geometry'])
    tensors=backend.text(prompt,refs)
    embeds=tensors['prompt_embeds'];tags=tensors['text_token_tags']
    if embeds.ndim!=2 or tags.shape!=(embeds.shape[0],) or tags.dtype!=torch.long or not bool(torch.isfinite(embeds).all()):
        raise ValueError('invalid fixed text conditioning')
    if not bool(((tags==0)|(tags==1)).all()):raise ValueError('invalid text modality tags')
    if tags.numel()%2:raise ValueError('odd text sequence cannot use current padless2rank CP; no implicit padding')
    report=dict(schema='h3-corrected-text-v2',geometry_policy=CORRECTED,reference_key=binding['key'],
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),text_sha256=tensor_sha(embeds),
        tags_sha256=tensor_sha(tags),private_conditioning_do_not_upload=True)
    report['key']=_hash(report)
    return tensors,report
