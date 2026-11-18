# Temporal chunk protocol adapted from Diffusers abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modifications: bounded incremental state, explicit final-grid and lifecycle gates.
"""Native H3 temporal chunk decode with bounded state, no RGB history re-encode.

Matches the pinned _decode for valid final H3 latent grids (5n+2, n>=1).
A chunk needs two future latent frames: this is not zero-lookahead causality.
Caller handles latent denormalization, CUDA autocast, RGB conversion and output.
"""
import hashlib
import inspect
from pathlib import Path

VAE_SHA='e59dc54caebff9ecb6d88c957b1ff189bf6a88419ed3551f84d3f68c20ce3192'


def verify_source(vae):
    path=Path(inspect.getfile(type(vae)))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=VAE_SHA:
        raise ValueError('pinned native video VAE source required')


class _DecodeStream:
    """Common bounded temporal algorithm; subclasses MUST verify their source."""
    def __init__(self,vae):
        if (vae.tokens_chunk_size,vae.config.token_drop,vae.temporal_compression_ratio,
                vae.token_overlap,vae.frame_pre_padding,vae.frame_overlap)!=(5,3,4,2,3,5):
            raise ValueError('audited native decode geometry required')
        self.vae=vae;self.buffer=None;self.overlap=None;self.received=0;self.closed=False
        self.emitted_frames=0;self.clips=0;self.busy=False

    def clear(self):
        self.buffer=self.overlap=None
        hook=getattr(self.vae,'clear_decode_wave',None)
        if hook is not None:hook()

    def push(self,z,*,final=False):
        if self.busy:raise ValueError('exhaust prior decoder iterator before pushing')
        self.busy=True
        try:
            yield from self._push(z,final=final)
        except BaseException:
            self.closed=True
            if getattr(self.vae,'clear_decode_wave',None) is not None:self.clear()
            raise
        finally:self.busy=False

    def _push(self,z,*,final=False):
        """Yield decoded [1,C,T,H,W] chunks. Exhaust iterator before next push."""
        import torch
        if self.closed:raise ValueError('decoder already finalized')
        if z.ndim!=5 or z.shape[0]!=1 or z.shape[2]<1 or z.shape[2]>256:
            raise ValueError('bounded single-batch positive latent input required')
        total=self.received+z.shape[2]
        if final and (total<7 or total%5!=2):
            raise ValueError('final native5n+2 grid required; no speculative padding')
        if self.buffer is not None and (z.shape[:2]+z.shape[3:]!=self.buffer.shape[:2]+self.buffer.shape[3:] or
                z.device!=self.buffer.device or z.dtype!=self.buffer.dtype):
            raise ValueError('decode stream geometry/device/dtype drift')
        self.buffer=z if self.buffer is None else torch.cat((self.buffer,z),dim=2)
        self.received=total
        while self.buffer.shape[2]>=7:
            prepare=getattr(self.vae,'prepare_decode_wave',None)
            if prepare is not None:prepare(self.buffer)
            clip=self.vae._decode_clip(self.buffer[:,:,:7])
            chunk=clip[:,:,:20][:,:,3:]
            if self.overlap is not None:chunk=self.vae._blend(self.overlap,chunk,5,dim=-3)
            self.overlap=clip[:,:,20:40][:,:,3:].clone()
            self.buffer=self.buffer[:,:,5:]
            self.clips+=1;self.emitted_frames+=chunk.shape[2]
            yield chunk
        self.buffer=self.buffer.clone()  # Release backing storage of earlier input chunks.
        if final:
            if self.buffer.shape[2]!=2 or self.overlap is None:raise ValueError('native decoder tail mismatch')
            tail=self.overlap;self.emitted_frames+=tail.shape[2]
            self.buffer=None;self.overlap=None;self.closed=True
            yield tail


class NativeDecodeStream(_DecodeStream):
    def __init__(self,vae):
        verify_source(vae)
        super().__init__(vae)


def write_incremental_video(enc,rows,path):
    """Same native chunk math, emit/encode RGB before full VAE decode finishes.

    This worker still starts AFTER denoise and writes a final MP4, not a live
    transport. No overlap with DiT, future-audio gate or physical flush claim.
    """
    import time
    import torch
    import av
    from diffusers.modular_pipelines.minimax_h3.packing import unpatchify_video_tokens
    from h3_trainer.constants import PATCH_SIZE
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    latents=unpatchify_video_tokens(rows,num_latent_frames=202,latent_height=84,latent_width=48,
        channels=int(enc.vae.config.latent_channels),patch_size=PATCH_SIZE)
    state=NativeDecodeStream(enc.vae);start=time.monotonic();first=None;first_packet=None
    raw=0;written=0;digest=hashlib.sha256();retained=0
    with torch.inference_mode(),av.open(str(path),mode='w') as container:
        stream=container.add_stream('libx264',rate=24);stream.width=768;stream.height=1344
        stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
        for offset in range(0,202,5):
            z=latents[:,:,offset:offset+5]*enc._latents_std+enc._latents_mean
            z=z.to(enc.device,torch.float32)
            for pixels in state.push(z,final=offset+5>=202):
                pixels=pixels*enc._pixel_std+enc._pixel_mean
                rgb=pixels.clamp(0,1).mul(255).to(torch.uint8).cpu()[0].permute(1,2,3,0).numpy()
                if first is None:first=time.monotonic()-start
                raw+=len(rgb);digest.update(rgb.tobytes())
                for frame in rgb:
                    if written>=683:continue
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):
                        if first_packet is None:first_packet=time.monotonic()-start
                        container.mux(packet)
                    written+=1
            if state.buffer is not None:retained=max(retained,state.buffer.shape[2])
        for packet in stream.encode():
            if first_packet is None:first_packet=time.monotonic()-start
            container.mux(packet)
    if raw!=685 or written!=683 or not state.closed:
        raise ValueError('incremental native frame count/finalization differs')
    return dict(schema='h3-incremental-native-decode-v1',decoded_frames=raw,export_frames=written,
        native_clips=state.clips,first_decoded_rgb_seconds=first,first_encoded_packet_seconds=first_packet,
        raw_rgb_sha256=digest.hexdigest(),retained_latent_frames_max=retained,
        input_batch_latent_frames=5,final_mp4_only=True,denoise_decode_overlap=False,
        physical_flush_verified=False,causal_verified=False,gpu_pixel_parity_verified=False)
