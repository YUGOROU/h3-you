"""Injected-model offline continuation core. No model loading, decode or upload.

Caller owns the verified original64+Turbo8 model/2rank CP lifecycle. This core
keeps that instance across windows and reuses fixed text/image conditions.
It is NOT a GPU launcher, a causal server, or an incremental VAE implementation.
"""
import time
from .continuation_plan import plan,video_t,audio_boundary,integer
from .continuation_layout import build_layout,condition_video,verify_modules,FIELDS


class ContinuationInterrupted(RuntimeError):pass


def run(model,scheduler_factory,packing,references,*,text,text_tags,image,image_rows,
        source_audio,total_frames,height,width,seed,device,ranks=2,
        window_frames=277,overlap_frames=22,is_current=lambda:True,on_window=None,set_sequence_rows=None):
    """Rows are unbatched, clean, normalized CPU tensors (audio channel-major).

    Audio target rows are locked, never integrated from the model's audio output.
    Fixed image + AV guide use .999 visual condition timestep/augmentation;
    guide audio/target audio use1.0. Final audio rows must equal source_audio.
    callback receives scalar metadata only; no automatic private tensor transfer.
    """
    import torch
    verify_modules(packing,references)
    integer(seed,'seed')
    windows=plan(total_frames,window_frames,overlap_frames)
    if text.ndim!=2 or text.shape[0]!=text_tags.numel():raise ValueError('text row/tag mismatch')
    if image_rows.shape!=(image.num_video_rows,96):raise ValueError('image rows/descriptor mismatch')
    if source_audio.shape!=(2*audio_boundary(total_frames),32):raise ValueError('full source audio lattice mismatch')
    for tensor in (text,image_rows,source_audio):
        if tensor.device.type!='cpu' or not bool(torch.isfinite(tensor).all()):
            raise ValueError('finite CPU conditioning required')
    # Verify ALL shapes/CP parity BEFORE a forward; do not fail only on window3.
    layouts=[build_layout(packing,references,text_tags=text_tags,image=image,window=w,
                          height=height,width=width,ranks=ranks) for w in windows]
    full_audio=source_audio.detach().float().reshape(2,-1,32)
    video=None;audio=None;profiles=[]
    def check():
        current=bool(is_current())
        # CP ranks must agree before entering the next forward/collective.
        if ranks==2 and torch.distributed.is_initialized():
            if torch.distributed.get_world_size()!=2:raise ValueError('exact2rank group required')
            status=torch.tensor(int(current),dtype=torch.int32,device=device)
            torch.distributed.all_reduce(status,op=torch.distributed.ReduceOp.MIN)
            current=bool(status.item())
        if not current:raise ContinuationInterrupted('stale continuation turn')
    def sync():
        if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
    check()
    with torch.inference_mode():
        for index,(window,layout) in enumerate(zip(windows,layouts)):
            check()
            if set_sequence_rows:set_sequence_rows(layout.sequence_length)
            start=time.monotonic();vt=video_t(window.frames)
            locked=full_audio[:,window.audio_start:window.audio_end].reshape(-1,32).contiguous()
            conditions=[image_rows];shapes=[(1,image.latent_height,image.latent_width)]
            guide_audio=locked[:0]
            if window.overlap:
                per_frame=(height//2)*(width//2)
                conditions.append(video[-window.guide_video_t*per_frame:])
                shapes.append((window.guide_video_t,height,width))
                guide_audio=audio[:,-window.guide_audio_t:].reshape(-1,32).contiguous()
            visual=condition_video(packing,torch.cat(conditions),shapes,vt,seed).to(device,dtype=torch.bfloat16)[None]
            packed_audio=torch.cat((guide_audio,locked)).to(device,dtype=torch.bfloat16)[None]
            rng=torch.Generator(device='cpu').manual_seed(seed)
            x=torch.randn((1,vt*(height//2)*(width//2),96),generator=rng,dtype=torch.float32).to(device)
            scheduler=scheduler_factory()
            if getattr(scheduler.config,'shift',None)!=12:raise ValueError('video shift12 required, no silent recipe switch')
            scheduler.set_timesteps(9,device=device)
            if len(scheduler.timesteps)!=8:raise ValueError('Turbo9grid must execute8forwards')
            static={key:getattr(layout,key).to(device) for key in FIELDS}
            prompt=text.to(device,dtype=torch.bfloat16)[None]
            for t in scheduler.timesteps:
                check()
                times,indices=packing.build_row_timesteps(layout,float(t),1.0,max(float(t),.999),1.0)
                velocity,_=model(hidden_states=torch.cat((visual,x.to(torch.bfloat16)),dim=1),
                    audio_hidden_states=packed_audio,encoder_hidden_states=prompt,
                    timestep=times.to(device),timestep_indices=indices.to(device),
                    attention_kwargs=None,return_dict=False,**static)
                target=velocity[:,layout.num_condition_video_rows:]
                if target.shape!=x.shape:raise ValueError('model target shape mismatch')
                x=scheduler.step(target.float(),t,x,return_dict=False)[0]
                if not bool(torch.isfinite(x).all()):raise ValueError('nonfinite continuation target')
            check();sync()
            sampled=x[0].detach().cpu().float()
            at=window.audio_end-window.audio_start;sampled_audio=locked.reshape(2,at,32)
            if video is None:video=sampled;audio=sampled_audio.clone()
            else:
                discard=window.guide_video_t*(height//2)*(width//2)
                video=torch.cat((video,sampled[discard:]),dim=0)
                audio=torch.cat((audio,sampled_audio[:,window.guide_audio_t:]),dim=1)
            profile=dict(window=index,**window.record(),denoiser_forwards=8,
                         generation_seconds=time.monotonic()-start,sequence_rows=layout.sequence_length,
                         cumulative_video_rows=video.shape[0],cumulative_audio_t=audio.shape[1])
            check();profiles.append(profile)
            if on_window:on_window(dict(profile))
    check()
    expected=video_t(total_frames)*(height//2)*(width//2)
    if video.shape!=(expected,96) or not torch.equal(audio,full_audio):
        raise ValueError('cumulative video/audio invariant failed')
    return video.contiguous(),audio.reshape(-1,32).contiguous(),dict(
        schema='h3-native-tail-result-v1',windows=profiles,total_frames=total_frames,
        audio_rows_unchanged=True,video_shift=12,visual_condition_noise_level=.999,
        visual_condition_time_policy='max(video_timestep,.999)',
        audio_condition_timestep=1.0,offline=True,causal_verified=False,decode_performed=False,
        private_latents_do_not_upload=True)
