"""CPU waveform-window readiness, NOT native VAE/model readiness.

Keep native124/22 geometry and original PCM clock. Future PCM may stabilize
resampling but must never be played early or replace a source's missing EOS.
"""
from dataclasses import dataclass
from .continuation_plan import Window, audio_boundary
from .streaming_pcm import PCMIngress, PCMStreamBinding

REFERENCE_SHA256 = '103c3e31011b943fbeb42470bbd3b798e71016cf911fe7dbf0eb93de05b85a08'
NATIVE_TORCH = '2.12.0+cu126'
NATIVE_TORCHAUDIO = '2.11.0+cu126'


@dataclass(frozen=True)
class ResamplingContext:
    mode: str = 'eos-only'
    torch_version: str = ''
    torchaudio_version: str = ''
    reference_sha256: str = ''

    def __post_init__(self):
        if self.mode not in ('eos-only','same-rate-32k','native-48k32k-prefix128'):
            raise ValueError('explicit resampling context required')
        if self.mode == 'native-48k32k-prefix128' and (
            self.torch_version, self.torchaudio_version, self.reference_sha256
        ) != (NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256):
            raise ValueError('pinned measured resampler environment required')
        if self.mode != 'native-48k32k-prefix128' and any((self.torch_version,
                self.torchaudio_version, self.reference_sha256)):
            raise ValueError('unexpected resampler evidence')

    def context(self, sample_rate):
        if self.mode == 'same-rate-32k':
            if sample_rate != 32000:
                raise ValueError('same-rate context cannot resample')
            return 0
        if self.mode == 'native-48k32k-prefix128':
            if sample_rate != 48000:
                raise ValueError('measured context is only48k→32k')
            return 128
        return None


def verify_native_context(reference_path):
    """Verify actual CPU resampler provenance and conservative filter support."""
    import hashlib
    from pathlib import Path
    import torch
    import torchaudio
    context=ResamplingContext('native-48k32k-prefix128',torch.__version__,
                              torchaudio.__version__,
                              hashlib.sha256(Path(reference_path).read_bytes()).hexdigest())
    resampler=torchaudio.transforms.Resample(48000,32000)
    support=int(resampler.kernel.shape[-1])+resampler.orig_freq//resampler.gcd
    if (resampler.kernel.device.type!='cpu' or support>128
            or not bool(torch.isfinite(resampler.kernel).all())):
        raise ValueError('unverified resampler support/device')
    return context


def ceil_ratio(value, numerator, denominator):
    return (value*numerator+denominator-1)//denominator


@dataclass(frozen=True)
class WindowAudioOffer:
    status: str  # ready / wait / end
    binding: PCMStreamBinding
    window: Window
    index: int
    onset_frame: int
    policy: ResamplingContext
    required_source_samples: int
    available_source_samples: int
    required_native_samples: int
    prefix_sha256: str | None
    eos: bool
    final_source_window: bool
    eos_padding_native_samples: int
    reason: str


def inspect_audio_window(stream, *, onset_frame, index, context, long_session=False):
    """Inspect a fixed window without consuming PCM or changing any watermark.

    The native encoder must separately validate this policy against its actual
    runtime and verify unchanged prefix bytes before use. No latent state here.
    """
    if type(stream) is not PCMIngress or type(context) is not ResamplingContext:
        raise ValueError('PCM ingress and explicit resampling context required')
    if type(onset_frame) is not int or onset_frame < 124 or (onset_frame-22)%17:
        raise ValueError('native committed source onset required')
    if type(long_session) is not bool:
        raise ValueError('explicit long_session selection required')
    if type(index) is not int or not 0 <= index < (2542 if long_session else 144):
        raise ValueError('bounded source window index required')
    window=Window(onset_frame-22+index*102,onset_frame+(index+1)*102,22)
    # These are speech-only offsets. The preceding37latents are waveform
    # history owned by conversation_audio, not extra playback PCM.
    native_stop=(window.audio_end-audio_boundary(onset_frame))*800
    native_committed=(audio_boundary(onset_frame+index*102)-audio_boundary(onset_frame))*800
    rate=stream.binding.sample_rate
    source_stop=ceil_ratio(native_stop,rate,32000)
    lookahead=context.context(rate)
    snapshot=stream.snapshot()
    received=snapshot['received_samples']; eos=snapshot['state']=='sealed'
    available_native=ceil_ratio(received,32000,rate)
    status='wait'; reason='speech-pcm-incomplete'; required=source_stop
    padding=0; final=False; digest=None
    if eos and available_native <= native_committed:
        status='end'; reason='sealed-source-already-covered'; required=received
    elif eos:
        status='ready'; reason='explicit-eos'
        required=received if lookahead is None else min(source_stop+lookahead,received)
        padding=max(0,native_stop-available_native)
        final=available_native <= native_stop
        digest=stream.prefix_digest(required)
    elif lookahead is None:
        reason='resampler-context-unverified-wait-eos'
    else:
        required=source_stop+lookahead
        if received >= required:
            status='ready'; reason='fixed-waveform-window-available'
            digest=stream.prefix_digest(required)
    # Do not bind readiness to total received bytes: later append may proceed
    # without invalidating an immutable prefix. EOS/finality remains explicit.
    return WindowAudioOffer(status,stream.binding,window,index,onset_frame,context,
        required, min(received,required),native_stop,digest,eos,final,padding,reason)
