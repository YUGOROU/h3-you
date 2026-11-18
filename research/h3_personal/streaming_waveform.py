"""Prepare exact fixed native waveform windows from immutable PCM prefixes.

No VAE/model/network/device operations. Native resampling callback is injected;
its source/runtime must be verified before production use. This is not a growing
Session adapter. Prefix allocation is bounded by PCMIngress.max_bytes (with
transient copies for join/float conversion); waveform output is fixed165600.
"""
from dataclasses import dataclass, field
import hashlib
from .conversation_audio import WaveHistory, TorchWaveOps
from .streaming_windows import WindowAudioOffer
from .continuation_plan import audio_boundary


@dataclass(frozen=True)
class PreparedWaveformWindow:
    offer: WindowAudioOffer
    waveform: object = field(repr=False)
    history: WaveHistory = field(repr=False)
    padding_native_samples: int


def assemble_window(resampled_prefix, offer, previous_history, ops):
    """Build the same waveform slice as batch window-context encoding.

    resampled_prefix must have a verified stable prefix through native_stop;
    this function never declares a resampling/VAE backend causal.
    """
    if type(offer) is not WindowAudioOffer or offer.status!='ready':
        raise ValueError('ready waveform offer required')
    shape=ops.shape(resampled_prefix)
    if len(shape)!=2 or shape[0]!=2 or shape[1]<=0:
        raise ValueError('nonempty stereo native waveform required')
    start=(offer.window.audio_start-audio_boundary(offer.onset_frame))*800
    stop=offer.required_native_samples
    pieces=[]
    if start < 0:
        if (not isinstance(previous_history,WaveHistory)
                or previous_history.end_frame!=offer.onset_frame
                or ops.shape(previous_history.tail)!=(2,-start)):
            raise ValueError('matching preceding waveform history required')
        pieces.append(ops.copy(previous_history.tail))
    if not offer.eos and shape[1]<stop:
        raise ValueError('missing future PCM is not EOS padding')
    pieces.append(ops.copy(ops.slice(resampled_prefix,max(0,start),min(stop,shape[1]))))
    target=(offer.window.audio_end-offer.window.audio_start)*800
    padding=target-sum(ops.shape(piece)[1] for piece in pieces)
    if padding<0 or padding!=offer.eos_padding_native_samples or (padding and not offer.eos):
        raise ValueError('window/EOS padding drift')
    pieces.append(ops.zeros(resampled_prefix,padding))
    wave=ops.cat(pieces)
    if ops.shape(wave)!=(2,target):
        raise ValueError('native fixed waveform shape mismatch')
    keep=offer.window.guide_audio_t*800
    history=WaveHistory(offer.window.end,ops.copy(ops.slice(wave,target-keep,target)))
    return PreparedWaveformWindow(offer,wave,history,padding)


def read_original_prefix(stream, offer):
    if (type(offer) is not WindowAudioOffer or offer.status!='ready'
            or offer.binding!=stream.binding):
        raise ValueError('matching immutable source window required')
    count=offer.required_source_samples
    if stream.prefix_digest(count)!=offer.prefix_sha256:
        raise ValueError('original PCM prefix identity drift')
    step=stream.max_read_bytes//stream.binding.stride
    parts=[]
    for start in range(0,count,step):
        block=stream.read(start,min(start+step,count))
        if block.status!='ready':
            raise ValueError('immutable prefix no longer available')
        parts.append(block.pcm)
    pcm=b''.join(parts)
    if hashlib.sha256(pcm).hexdigest()!=offer.prefix_sha256:
        raise ValueError('copied original prefix hash mismatch')
    return pcm


def prepare_native_window(stream, offer, previous_history, prepare_reference_waveform,
                          *, reference_path):
    """CPU Torch/native-resampler bridge, explicit injected pinned callback.

    No audio VAE call here. Caller must verify runtime provenance/context policy;
    full native latent parity and native owner integration remain separate.
    """
    import inspect
    from pathlib import Path
    import numpy as np
    import torch
    from .streaming_windows import REFERENCE_SHA256, verify_native_context
    path=Path(reference_path).resolve(strict=True)
    if (Path(inspect.getfile(prepare_reference_waveform)).resolve()!=path
            or hashlib.sha256(path.read_bytes()).hexdigest()!=REFERENCE_SHA256):
        raise ValueError('pinned native reference callback source required')
    if offer.policy.mode=='native-48k32k-prefix128' and verify_native_context(path)!=offer.policy:
        raise ValueError('native resampler runtime/context drift')
    pcm=read_original_prefix(stream,offer)
    b=stream.binding
    values=np.frombuffer(pcm,dtype='<i2').reshape(-1,b.channels).T.copy()
    wave=torch.from_numpy(values).to(torch.float32)/32768.0
    native=prepare_reference_waveform(wave,b.sample_rate,32000,
                                      offer.required_source_samples/b.sample_rate+1.0)
    if native.device.type!='cpu' or not bool(torch.isfinite(native).all()):
        raise ValueError('finite native CPU waveform required')
    return assemble_window(native,offer,previous_history,TorchWaveOps())
