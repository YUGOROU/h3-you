"""Audited external-speech locking for the SMD Ref2VA four-forward path.

Ports one invariant of the old native continuation runtime onto the accepted
SMD4/ref2va path: the supplied speech rows are fixed on every denoising forward,
their row timestep is held at the clean value1.0, the model's audio velocity is
discarded and only the video rows are integrated.

Hard rules (fail closed):

- Speech is never truncated, time-stretched, denoised or resampled to a different
  duration. A window that would cut the speech is rejected instead; only trailing
  silence inside the planner's fixed latent window may be added.
- The locked target audio rows must equal ``audio_latent_frames(num_frames)``
  exactly, so a shape disagreement stops the request.
- Image-only references only. Any native condition-audio row (reference audio)
  makes the row arithmetic ambiguous and is rejected.
- The pinned audio-VAE geometry (32 kHz, hop800,32 channels,40 latents/s) and the
  pinned FastVideo packing constants are re-verified at runtime and by test.
- No generated audio is used for playback: the caller keeps the supplied WAV.
  This module does not mux, publish, upload or play anything.
"""

import hashlib
import json
import math
import os
from pathlib import Path

# Pinned FastVideo packing constants (fastvideo/pipelines/basic/minimax_h3/packing.py).
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
# Pinned MiniMax-H3 audio VAE geometry (configs/models/vaes/minimax_h3_audio.py).
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENT_CHANNELS = 32
HOP_LENGTH = 800

# Pinned batch key for the packed layout (stages/minimax_h3_latent_preparation.py).
LAYOUT_KEY = 'minimax_h3_layout'

WINDOW_FRAMES = 124
MAX_WINDOWS = 4
SCHEMA = 'h3-smd-locked-audio-v1'
ENV_ENABLE = 'H3_SMD_LOCKED_AUDIO'
ENV_WINDOW = 'H3_SMD_LOCKED_AUDIO_WINDOW'


def sha256(path):
    """File SHA256 used for every speech, source and overlay binding."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_frames(num_frames):
    if type(num_frames) is not int or num_frames < 1 or num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f'num_frames must be of the pinned 17*n+5 grid, got {num_frames!r}')
    return num_frames


def audio_latent_frames(num_frames):
    return int(round(aligned_frames(num_frames) / FPS * AUDIO_LATENTS_PER_SECOND))


def window_latents(num_frames=WINDOW_FRAMES):
    return audio_latent_frames(num_frames)


def expected_audio_rows(num_frames=WINDOW_FRAMES):
    return window_latents(num_frames) * AUDIO_CHANNELS


def window_seconds(num_frames=WINDOW_FRAMES):
    return window_latents(num_frames) / AUDIO_LATENTS_PER_SECOND


def plan_windows(speech_seconds, *, wav, sha256_value, frames_per_window=WINDOW_FRAMES, max_windows=MAX_WINDOWS):
    """Fixed integer audio windows covering the whole supplied speech.

    Rejects a speech that does not fit ``max_windows`` instead of shortening it.
    """
    if not isinstance(speech_seconds, (int, float)) or not math.isfinite(speech_seconds) or speech_seconds <= 0:
        raise ValueError('positive finite speech duration required')
    if not isinstance(sha256_value, str) or len(sha256_value) != 64:
        raise ValueError('speech SHA256 required')
    latents = window_latents(frames_per_window)
    span = latents / AUDIO_LATENTS_PER_SECOND
    windows = max(1, math.ceil(speech_seconds / span))
    if windows > max_windows:
        raise ValueError(f'supplied speech needs {windows} windows; {max_windows} is the finite limit — '
                         'shorten the reply instead of truncating the audio')
    rows = [dict(index=index, latent_start=index * latents, latent_count=latents,
                 start_seconds=index * span, end_seconds=(index + 1) * span,
                 expected_rows=latents * AUDIO_CHANNELS) for index in range(windows)]
    return dict(schema=SCHEMA + '-plan', wav=str(wav), sha256=sha256_value,
                speech_seconds=round(float(speech_seconds), 6), frames_per_window=frames_per_window,
                window_latents=latents, window_seconds=span, windows=rows, windows_required=windows,
                truncation=False, time_stretch=False, denoise_or_gain_change=False,
                trailing_silence_seconds=round(windows * span - float(speech_seconds), 6),
                source='supplied_external_speech_not_generated')


def window_manifest(plan, index):
    """Exact per-request binding handed to the worker through the environment."""
    if plan.get('schema') != SCHEMA + '-plan':
        raise ValueError('locked-audio plan required')
    rows = [row for row in plan['windows'] if row['index'] == index]
    if len(rows) != 1:
        raise ValueError('unknown locked-audio window index')
    row = rows[0]
    if row['expected_rows'] != row['latent_count'] * AUDIO_CHANNELS:
        raise ValueError('window row arithmetic differs')
    return dict(schema=SCHEMA, index=index, wav=plan['wav'], sha256=plan['sha256'],
                latent_start=row['latent_start'], latent_count=row['latent_count'],
                expected_rows=row['expected_rows'], start_seconds=row['start_seconds'],
                planned_latent_total=plan['windows_required'] * plan['window_latents'])


def window_bounds(speech_samples, latent_start, latent_count, planned_latent_total):
    """Pure integer window arithmetic: slice offset, length and trailing silence.

    ``speech_samples`` is the supplied speech length at the audio-VAE rate after
    resampling. A speech that does not fit the planned windows is rejected; it is
    never clipped.
    """
    for name, value in (('speech_samples', speech_samples), ('latent_start', latent_start),
                        ('latent_count', latent_count), ('planned_latent_total', planned_latent_total)):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    if latent_count <= 0 or planned_latent_total < latent_start + latent_count:
        raise ValueError('planned locked-audio windows do not cover this window')
    if speech_samples > planned_latent_total * HOP_LENGTH:
        raise ValueError('supplied speech would be truncated by the planned windows')
    start = latent_start * HOP_LENGTH
    count = latent_count * HOP_LENGTH
    available = max(0, min(count, speech_samples - start))
    return dict(start=start, count=count, available=available, padded=count - available)


def load_manifest():
    # Persistent spawned GPU workers do not see later parent env updates.
    # One ordered request at a time; parent atomically replaces this shared file.
    shared = os.environ.get('H3_SMD_LOCKED_AUDIO_WINDOW_FILE')
    raw = Path(shared).read_text() if shared else os.environ.get(ENV_WINDOW)
    if not raw:
        raise ValueError('locked-audio window binding missing')
    manifest = json.loads(raw)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('locked-audio window schema differs')
    if manifest['expected_rows'] != manifest['latent_count'] * AUDIO_CHANNELS:
        raise ValueError('locked-audio row arithmetic differs')
    if any(type(manifest[key]) is not int or manifest[key] < 0
           for key in ('index', 'latent_start', 'latent_count', 'expected_rows', 'planned_latent_total')):
        raise ValueError('locked-audio window fields must be non-negative integers')
    return manifest


def receipt_path(index, rank):
    folder = os.environ.get('H3_REF2VA_COVERAGE_DIR')
    if not folder:
        raise RuntimeError('isolated coverage folder required for locked-audio receipts')
    return Path(folder) / f'locked-audio-rank-{rank}-window-{index}.json'


def _rank():
    import torch
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _write(receipt, index):
    path = receipt_path(index, _rank())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt) + '\n')


def lock_target_audio_rows(stage, batch, fastvideo_args):
    """Replace the sampled target-audio noise with the encoded supplied speech."""
    import torch
    from fastvideo.models import pinned_offload
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_audio, prepare_reference_waveform
    manifest = load_manifest()
    source = Path(manifest['wav'])
    if sha256(source) != manifest['sha256']:
        raise ValueError('locked speech bytes differ from the bound SHA256')
    vae = stage.audio_vae
    if (int(vae.hop_length), int(vae.sampling_rate), int(vae.latent_channels)) != (
            HOP_LENGTH, AUDIO_SAMPLE_RATE, AUDIO_LATENT_CHANNELS):
        raise ValueError('audio VAE geometry differs from the pinned locked-audio contract')
    layout = batch.extra.get(LAYOUT_KEY)
    if layout is None:
        raise ValueError('packed layout must exist before locked audio is applied')
    rows = int(layout.num_audio_latents) * AUDIO_CHANNELS
    if rows != manifest['expected_rows'] or int(layout.num_condition_audio_rows) != 0:
        raise ValueError('locked audio requires image-only references and the planned row count')
    waveform, source_rate = decode_reference_audio(str(source))
    if not isinstance(source_rate, int) or source_rate <= 0:
        raise ValueError('decoded speech sample rate differs')
    speech = prepare_reference_waveform(waveform, int(source_rate), AUDIO_SAMPLE_RATE,
                                        max_duration=float(waveform.shape[-1]) / float(source_rate) + 1.0)
    bounds = window_bounds(int(speech.shape[-1]), manifest['latent_start'], manifest['latent_count'],
                           manifest['planned_latent_total'])
    window = speech[:, bounds['start']:bounds['start'] + bounds['available']]
    padded = bounds['padded']
    if padded:
        window = torch.cat((window, torch.zeros((AUDIO_CHANNELS, padded), dtype=window.dtype)), dim=-1)
    device = batch.audio_latents.device
    # Mirror upstream offload handling: never silently move weights, only load the
    # isolated VAE onto the packed device for this encode.
    loaded = any(parameter.device != device for parameter in vae.parameters())
    if loaded:
        pinned_offload.load(vae, device, pin=bool(getattr(fastvideo_args, 'pin_cpu_memory', False)))
    try:
        with torch.no_grad():
            latent_dist = vae.encode(window.to(device)[:, None]).latent_dist
            latents = vae.normalize_latents(latent_dist.mode().float()).transpose(1, 2).reshape(-1, vae.latent_channels)
        latents = latents.contiguous()
    finally:
        if loaded and bool(getattr(fastvideo_args, 'vae_cpu_offload', False)):
            pinned_offload.unload(vae)
    if tuple(latents.shape) != (rows, AUDIO_LATENT_CHANNELS) or not torch.isfinite(latents).all().item():
        raise ValueError('locked speech encoding does not match the packed audio rows')
    locked = latents.to(dtype=batch.audio_latents.dtype)
    batch.audio_latents = torch.cat((batch.audio_latents[:int(layout.num_condition_audio_rows)], locked))
    batch.extra['_smd_locked_audio_rows'] = locked.clone()
    receipt = dict(schema=SCHEMA, window_index=manifest['index'], wav=str(source), wav_sha256=manifest['sha256'],
                   latent_start=manifest['latent_start'], latent_count=manifest['latent_count'],
                   expected_rows=manifest['expected_rows'], encoded_rows=int(locked.shape[0]),
                   padded_samples=padded, window_seconds=bounds['count'] / AUDIO_SAMPLE_RATE,
                   audio_sample_rate=AUDIO_SAMPLE_RATE, hop_length=HOP_LENGTH,
                   audio_timestep_policy='fixed_clean_1.0', audio_velocity_discarded=True,
                   audio_scheduler_step_skipped=True, rows_unchanged=True, steps_held=0,
                   truncation=False, time_stretch=False, supplied_speech_not_generated=True,
                   generated_audio_decode_skipped=True, watermark=None, worker_rank=_rank())
    batch.extra['_smd_locked_audio_receipt'] = receipt
    _write(receipt, manifest['index'])
    return batch


def locked_audio_timestep(audio_timestep):
    """Hold the locked rows at the clean condition timestep instead of the schedule."""
    if not isinstance(audio_timestep, float) or not math.isfinite(audio_timestep):
        raise ValueError('finite audio timestep required')
    return 1.0


def hold_locked_audio_rows(stage, batch, audio_velocity):
    """Discard audio velocity and assert the locked rows are untouched."""
    del stage
    import torch
    layout = batch.extra.get(LAYOUT_KEY)
    locked = batch.extra.get('_smd_locked_audio_rows')
    receipt = batch.extra.get('_smd_locked_audio_receipt')
    if layout is None or locked is None or receipt is None or audio_velocity is None:
        raise ValueError('locked-audio denoising state missing')
    current = batch.audio_latents[int(layout.num_condition_audio_rows):]
    if tuple(current.shape) != tuple(locked.shape) or not torch.equal(current, locked):
        raise ValueError('locked audio rows changed during denoising')
    receipt['steps_held'] = int(receipt.get('steps_held', 0)) + 1
    _write(receipt, receipt['window_index'])


def verify_receipts(directory, windows, steps=4):
    """Runner-side check of both-rank locked-audio receipts."""
    receipts = []
    for index in range(windows):
        for rank in (0, 1):
            path = Path(directory) / f'locked-audio-rank-{rank}-window-{index}.json'
            receipt = json.loads(path.read_text())
            expected = dict(schema=SCHEMA, window_index=index, audio_timestep_policy='fixed_clean_1.0',
                            audio_scheduler_step_skipped=True, steps_held=steps, rows_unchanged=True,
                            truncation=False, time_stretch=False, encoded_rows=receipt['expected_rows'],
                            worker_rank=rank)
            for key, value in expected.items():
                if receipt.get(key) != value:
                    raise RuntimeError(f'locked-audio receipt differs on window {index} rank {rank}: {key}')
            receipts.append(receipt)
    return receipts


def install(source):
    """Patch the isolated pinned FastVideo copy; never touches the shared archive."""
    source = Path(source)
    latent = source / 'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py'
    denoise = source / 'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py'
    report = {}
    original = sha256(latent)
    text = latent.read_text()
    # The packed layout must already be on the batch, so patch after it is written.
    anchor = '        batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout\n'
    hook = ("        if __import__('os').environ.get('H3_SMD_LOCKED_AUDIO') == '1':\n"
            "            from research.h3_personal.external_audio import lock_target_audio_rows\n"
            "            batch = lock_target_audio_rows(self, batch, fastvideo_args)\n")
    if text.count(anchor) != 1 or 'lock_target_audio_rows' in text:
        raise ValueError('locked-audio latent-preparation anchor differs')
    latent.write_text(text.replace(anchor, anchor + hook))
    report.update(latent_preparation_original_sha256=original, latent_preparation_patched_sha256=sha256(latent))
    original = sha256(denoise)
    text = denoise.read_text()
    timestep_anchor = ('            unique, inverse = build_row_timesteps(\n'
                       '                layout,\n'
                       '                video_timestep=video_value,\n')
    timestep_hook = ("            if __import__('os').environ.get('H3_SMD_LOCKED_AUDIO') == '1':\n"
                     "                from research.h3_personal.external_audio import locked_audio_timestep\n"
                     "                audio_value = locked_audio_timestep(audio_value)\n")
    step_anchor = ('                    batch.audio_latents[audio_start:] = self.audio_scheduler.step(\n'
                   '                        audio_velocity[0, audio_start:].float(),\n'
                   '                        audio_timestep,\n'
                   '                        batch.audio_latents[audio_start:],\n'
                   '                        return_dict=False,\n'
                   '                    )[0]\n')
    step_hook = ("                    if __import__('os').environ.get('H3_SMD_LOCKED_AUDIO') == '1':\n"
                 "                        from research.h3_personal.external_audio import hold_locked_audio_rows\n"
                 "                        hold_locked_audio_rows(self, batch, audio_velocity)\n"
                 "                    else:\n" + ''.join('    ' + line for line in step_anchor.splitlines(True)))
    if (text.count(timestep_anchor) != 1 or text.count(step_anchor) != 1
            or 'locked_audio_timestep' in text or 'hold_locked_audio_rows' in text):
        raise ValueError('locked-audio denoising anchor differs')
    text = text.replace(timestep_anchor, timestep_hook + timestep_anchor).replace(step_anchor, step_hook)
    denoise.write_text(text)
    report.update(denoising_original_sha256=original, denoising_patched_sha256=sha256(denoise))
    return report
