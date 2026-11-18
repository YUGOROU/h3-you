"""Opt-in window-local audio conditioning; no model or device operations."""
from .continuation_plan import audio_boundary

_SCHEMA = 'h3-continuation-candidate-v1'
_AUDIO = ('full-source-v1', 'window-context-canonical-v1')
_PROGRESS = ('per-step-v1', 'bind-and-commit-v1')


def normalize(options):
    if options is None:
        return None
    if type(options) is not dict or set(options) != {'schema', 'audio_context', 'progress_receipts'}:
        raise ValueError('invalid continuation candidate keys/type')
    for key, allowed in (('schema', (_SCHEMA,)), ('audio_context', _AUDIO),
                         ('progress_receipts', _PROGRESS)):
        if type(options[key]) is not str or options[key] not in allowed:
            raise ValueError(f'invalid continuation candidate {key}')
    return dict(options)


def make_options(*, window_audio_context=False, defer_progress_receipts=False):
    if type(window_audio_context) is not bool or type(defer_progress_receipts) is not bool:
        raise ValueError('candidate switches must be bool')
    if not window_audio_context and not defer_progress_receipts:
        return None
    return dict(schema=_SCHEMA, audio_context=_AUDIO[int(window_audio_context)],
                progress_receipts=_PROGRESS[int(defer_progress_receipts)])


def encode_window_context(padded_speech, windows, encode):
    """Return (canonical CPU audio, scalar receipt), retaining earlier overlap.

    The callback owns a cloned waveform and its result is copied immediately;
    neither callback input mutation nor reused output storage can change history.
    No RNG is used here, and encoding is invoked exactly once per window.
    """
    import torch

    windows = tuple(windows)
    if not windows or not callable(encode):
        raise ValueError('nonempty windows and encoder required')
    cursor = 0
    for i, w in enumerate(windows):
        start, end, guide = w.audio_start, w.audio_end, w.guide_audio_t
        if any(type(v) is not int for v in (start, end, guide)):
            raise ValueError('integer audio boundaries required')
        if (start != audio_boundary(w.start) or end != audio_boundary(w.end)
                or guide != audio_boundary(w.start + w.overlap) - start
                or start < 0 or end <= start or guide < 0 or guide >= end - start
                or start + guide != cursor or (i == 0 and (start != 0 or guide != 0))):
            raise ValueError('noncontiguous audio boundaries')
        cursor = end

    def validate(t, shape):
        if (not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                or tuple(t.shape) != shape or not t.is_floating_point()
                or not bool(torch.isfinite(t).all())):
            raise ValueError('expected finite CPU floating tensor with exact shape')

    validate(padded_speech, (2, cursor * 800))
    pieces = []
    for w in windows:
        local = encode(padded_speech[:, w.audio_start * 800:w.audio_end * 800].clone())
        validate(local, (2, w.audio_end - w.audio_start, 32))
        pieces.append(local[:, w.guide_audio_t:].clone())
    canonical = torch.cat(pieces, dim=1)
    validate(canonical, (2, cursor, 32))
    return canonical, dict(source_window_encodes=len(windows),
                           audio_overlap_policy='prior-canonical-wins')
