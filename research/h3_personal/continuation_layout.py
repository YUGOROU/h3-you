"""Bridge native latent-tail guides onto the exact pinned Diffusers layout.

The guide/discard/append/time contract was checked against vLLM-Omni
139a47a578e01eb7cd4a0c3d35a02eba4b4ded6a (Apache-2.0). This bridge does not
change Diffusers files or use its RGB/video conditioner for the dynamic guide.
"""
from dataclasses import replace
import hashlib
from pathlib import Path

PACKING_SHA='969d667d1c0316ed931d1675d474220045390e9a566cc885e3bd9cc6147b3e5b'
REFERENCE_SHA='1f025b68af3f5c4316e2b89b026d4976c5b85286dafcc8b5c1d8ceb186c9bc01'
FIELDS=('position_ids','token_tags','video_indices','audio_indices','text_indices')


def verify_modules(packing,references):
    for module,expected in ((packing,PACKING_SHA),(references,REFERENCE_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise ValueError('continuation requires exact pinned Diffusers source')
    if references.MiniMaxH3PackedSequence is not packing.MiniMaxH3PackedSequence:
        raise ValueError('mixed packing import roots')
    if references._temporal_position_grid is not packing._temporal_position_grid:
        raise ValueError('mixed rotary helpers')


def build_layout(packing,references,*,text_tags,image,window,height,width,ranks=2):
    """Image-only fixed reference + optional final AV guide. No target-prefix mask.

    Native video blocks normally ADVANCE the target clock. A latent guide must
    instead SHARE its origin with the target. Build with pinned video semantics,
    then undo ONLY that last guide's advance for target rows. Static image/text
    clocks stay fixed; all dynamic audio/guide/target clocks receive global time.
    """
    import torch
    from .continuation_plan import video_t,integer
    integer(ranks,'ranks',1)
    if ranks not in (1,2):raise ValueError('only1/2rank contracts supported')
    if image.kind!='image' or image.num_latent_frames!=1 or image.num_audio_latents:
        raise ValueError('one fixed image reference without audio required')
    if text_tags.ndim!=1 or text_tags.dtype!=torch.long or not text_tags.numel():
        raise ValueError('nonempty int64 text modality tags required')
    if not bool(((text_tags==0)|(text_tags==1)).all()):raise ValueError('invalid text tags')
    for n in (height,width):
        integer(n,'latent canvas',2)
        if n%2:raise ValueError('even latent canvas required')
    vt=video_t(window.frames);at=window.audio_end-window.audio_start
    refs=[image]
    if window.overlap:
        if window.guide_video_t>=vt or window.guide_audio_t>=at:
            raise ValueError('guide must be shorter than new target')
        refs.append(references.MiniMaxH3PreparedReference(kind='video',
            num_latent_frames=window.guide_video_t,latent_height=height,latent_width=width,
            num_audio_latents=window.guide_audio_t))
    layout=references.build_ref2va_packed_sequence(text_tags.cpu(),refs,vt,height,width,at,(1,2,2))
    if layout.sequence_length%ranks:raise ValueError('padless CP sequence is not divisible by ranks; no dummy padding')
    pos=layout.position_ids.clone()
    target_v=layout.video_indices[layout.num_condition_video_rows:]
    target_a=layout.audio_indices[layout.num_condition_audio_rows:]
    if window.overlap:
        advance=max(float(window.guide_audio_t),references._temporal_position_span(window.guide_video_t))
        pos[target_v,0]-=advance;pos[target_a,0]-=advance
    offset=float(window.temporal_offset)
    pos[layout.audio_indices,0]+=offset
    # Fixed image remains unmoved. The optional guide and target video move.
    pos[layout.video_indices[image.num_video_rows:],0]+=offset
    if not bool(torch.isfinite(pos).all()):raise ValueError('nonfinite rotary positions')
    return replace(layout,position_ids=pos)


def condition_video(packing,clean_rows,shapes,target_t,seed):
    """Native dependent-noise RF augmentation (.999), CPU fp32, fresh seed/block.

    Clean cache/previous generated latents are never modified. Matches upstream
    condition_noise.py draw shape/order; does NOT use one flat random row draw.
    """
    import torch
    if clean_rows.ndim!=2 or clean_rows.shape[1]!=96 or not shapes:
        raise ValueError('native96-channel packed condition rows required')
    expected=sum(t*(h//2)*(w//2) for t,h,w in shapes)
    if clean_rows.shape[0]!=expected:raise ValueError('condition row/shape mismatch')
    out=[];cursor=0
    for t,h,w in shapes:
        if min(t,h,w)<=0 or h%2 or w%2:raise ValueError('invalid condition shape')
        rng=torch.Generator(device='cpu').manual_seed(seed)
        noise=torch.randn((1,24,max(target_t+len(shapes),t),h,w),generator=rng,dtype=torch.float32)[:,:,:t]
        rows=packing.patchify_video_latents(noise,(1,2,2));count=rows.shape[0]
        clean=clean_rows[cursor:cursor+count].detach().to(device='cpu',dtype=torch.float32)
        timestep=torch.tensor(.999,dtype=torch.float32)
        out.append(timestep*clean+(1-timestep)*rows);cursor+=count
    return torch.cat(out).contiguous()
