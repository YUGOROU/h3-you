"""Native source-origin contract and bounded waveform history; no device imports.

Preroll is previous *waveform*, encoded together with the new source by the VAE.
Original playback PCM is never changed. The owner and encoder share this plan.
"""
from dataclasses import dataclass
from .continuation_plan import audio_boundary, video_t


def audio_origin(onset_frame, total_frames):
    if type(onset_frame) is not int or onset_frame < 0:
        raise ValueError('integer source onset required')
    video_t(total_frames)
    base = onset_frame-22 if onset_frame else 0
    if base < 0 or (onset_frame and base == 0) or base % 17:
        raise ValueError('native source origin lattice required')
    prefix = audio_boundary(onset_frame)-audio_boundary(base)
    if audio_boundary(base+total_frames)-audio_boundary(base) != audio_boundary(total_frames):
        raise ValueError('unsupported global/local audio phase')
    return dict(schema='h3-conversation-audio-origin-v1', onset_frame=onset_frame,
                model_start_frame=base, total_frames=total_frames,
                preroll_latents=prefix, preroll_samples_32000=prefix*800,
                capacity_samples_32000=audio_boundary(total_frames)*800)


@dataclass(frozen=True)
class WaveHistory:
    end_frame: int
    tail: object


def prepare_waveform(speech, origin, history, ops):
    """Allocate prefix+whole source+padding and next bounded history on CPU.

    ops has shape/slice/copy/zeros/cat methods; separate from latent TensorOps.
    No mutation of caller-owned speech/history, no latent-zero substitution.
    """
    if origin != audio_origin(origin['onset_frame'], origin['total_frames']):
        raise ValueError('audio origin drift')
    shape = ops.shape(speech)
    if len(shape) != 2 or shape[0] != 2 or shape[1] <= 0:
        raise ValueError('nonempty stereo waveform required')
    prefix = origin['preroll_samples_32000']
    capacity = origin['capacity_samples_32000']
    padding = capacity-prefix-shape[1]
    if padding < 0:
        raise ValueError('whole source plus preroll exceeds capacity; never truncate')
    pieces = []
    if prefix:
        if (not isinstance(history, WaveHistory) or
                history.end_frame != origin['onset_frame'] or
                ops.shape(history.tail) != (2, prefix)):
            raise ValueError('matching preceding waveform history required')
        pieces.append(ops.copy(history.tail))
    elif history is not None:
        raise ValueError('initial source refuses stale waveform history')
    pieces.extend((ops.copy(speech), ops.zeros(speech, padding)))
    padded = ops.cat(pieces)
    if ops.shape(padded) != (2, capacity):
        raise ValueError('padded waveform shape drift')
    end = origin['model_start_frame']+origin['total_frames']
    keep = (audio_boundary(end)-audio_boundary(end-22))*800
    tail = ops.copy(ops.slice(padded, capacity-keep, capacity))
    return padded, WaveHistory(end, tail), padding


class TorchWaveOps:
    def shape(self, t):
        return tuple(t.shape)

    def slice(self, t, start, end):
        return t[:, start:end]

    def copy(self, t):
        return t.detach().cpu().clone().contiguous()

    def zeros(self, like, count):
        return like.new_zeros((2, count))

    def cat(self, pieces):
        import torch
        return torch.cat(pieces, dim=-1)
