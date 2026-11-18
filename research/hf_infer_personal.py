"""Single private VoxCPM2 -> audio-frozen Ref2VA review, no Agent/server.

Run only inside a bounded GPU job. Stages run in separate processes to release
VRAM. Uses the pinned Trainer packing/encoding and released H3 scheduler; raw
Trainer adapter keys/shapes must match exactly. Does not assume vLLM support.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

TRAINER_PIN = '7a1563a267a17da898df3396acbd6a9d24fcb142'
MODEL_PIN = '42ed227ee7df40d41602854ae760620d6eb651fe'
VOX_PIN = '32279effe8c19989596f05d353d1447f51d9e915'
CHECKPOINT = 'runs/20261011-182118/checkpoint-0000120'
OUTPUT_REV = '78e0a753755b32138ee2ed48515a8865091bc257'
TEXT = 'おはようございます。声と口の動きを確認します。'
PROMPT = ('A selfie video of the same person, speaking naturally to the camera, '
          'with lip movements matching the driving speech, subtle facial expressions '
          'and restrained head movements.')


def frame_count(spoken: float) -> int:
    """Nearest legal bucket covering ALL speech plus 0.25s of tail silence."""
    if not math.isfinite(spoken) or spoken <= 0:
        raise ValueError('Invalid TTS duration')
    for frames in range(124, 361, 17):
        if frames / 24 >= spoken + .25 and frames / 24 <= 15:
            return frames
    raise ValueError('Speech exceeds legal <=15s bucket; refusing truncation')


def packed_video_channels(in_channels: int, patch_size) -> int:
    """Transformer config describes RAW channels; row width includes patches."""
    if in_channels <= 0 or len(patch_size) != 3 or any(p <= 0 for p in patch_size):
        raise ValueError('Invalid video patch geometry')
    return in_channels * math.prod(patch_size)


def comparison_cases():
    """One factor at a time; every case retains the trained adapter."""
    return [
        {'name': 'reference-scale1-50', 'lora_scale': 1.0, 'schedule_points': 50},
        {'name': 'reference-scale05-50', 'lora_scale': 0.5, 'schedule_points': 50},
        {'name': 'reference-scale15-50', 'lora_scale': 1.5, 'schedule_points': 50},
        {'name': 'reference-scale1-100', 'lora_scale': 1.0, 'schedule_points': 100},
    ]


def set_adapter_strength(model, scale):
    if not math.isfinite(scale) or not 0 < scale <= 2:
        raise ValueError('Comparison requires a nonzero, finite adapter strength <=2')
    # Pinned diffusers uses `weights`, NOT pipeline-level `adapter_weights`.
    model.set_adapters('default', weights=scale)
    actual = [m.scaling['default'] for m in model.modules()
              if hasattr(m, 'lora_A') and 'default' in m.lora_A]
    expected = scale * model.peft_config['default'].lora_alpha / model.peft_config['default'].r
    if not actual or any(not math.isclose(v, expected) for v in actual):
        raise ValueError('Adapter strength not applied uniformly')
    return len(actual)


def command(args):
    subprocess.run([str(a) for a in args], check=True)


def tts(root):
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from voxcpm import VoxCPM
    path = snapshot_download('openbmb/VoxCPM2', revision=VOX_PIN)
    model = VoxCPM.from_pretrained(path, load_denoiser=False,
                                  device='cuda:0', optimize=False)
    wav = model.generate(text=TEXT, reference_wav_path=str(root / 'voice-reference.wav'),
                         cfg_value=2.0, inference_timesteps=10, normalize=False,
                         denoise=False, max_len=93, retry_badcase=False, seed=42)
    rate = model.tts_model.sample_rate
    spoken = len(wav) / rate
    frames = frame_count(spoken)
    sf.write(root / 'tts.wav', wav, rate, subtype='PCM_16')
    (root / 'request.json').write_text(json.dumps({
        'text': TEXT, 'prompt': PROMPT, 'spoken_seconds': spoken,
        'frames': frames, 'duration': frames / 24, 'width': 768,
        'height': 1344, 'seed': 42, 'schedule_points': 50,
        'tts_model_revision': VOX_PIN, 'voice_reference_quality_verified': False,
        'tts_sentence_completeness_verified': False}, ensure_ascii=False, indent=2))
    command(['ffmpeg', '-y', '-v', 'error', '-i', root / 'tts.wav', '-af', 'apad',
             '-t', frames / 24, '-ar', '32000', '-ac', '2', '-c:a', 'pcm_s16le',
             root / 'driving.wav'])
    print('TTS_READY', json.dumps({'spoken_seconds': spoken, 'frames': frames}), flush=True)


def condition(root):
    from safetensors.torch import save_file
    from h3_trainer.constants import Geometry
    from h3_trainer.inference import GenerationRequest, H3Pipeline
    from h3_trainer.preprocessing.encoders import H3Encoders
    from h3_trainer.preprocessing.media import extract_audio
    request = json.loads((root / 'request.json').read_text())
    g = Geometry.create(768, 1344, request['frames']).require_generatable()
    helper = H3Pipeline('/work/model', variant='ref2va', placement='bf16')
    cond = helper.encode_conditioning(GenerationRequest(
        prompt=PROMPT, geometry=g, seed=42,
        references=[{'image': str(root / 'portrait.png')}]))
    waveform = extract_audio(root / 'driving.wav', g.num_frames)
    if waveform is None:
        raise ValueError('No usable source speech')
    enc = H3Encoders('/work/model', need_video=False, need_audio=True, need_text=False)
    audio = enc.encode_audio(waveform)
    enc.unload()
    if audio.shape[0] != g.audio_rows or cond.audio_condition_latents is not None:
        raise ValueError('Source-audio or reference layout mismatch')
    save_file({'prompt': cond.prompt_embeds.contiguous(),
               'tags': cond.text_token_tags.contiguous(),
               'reference': cond.condition_latents.contiguous(),
               'source_audio': audio.contiguous()}, str(root / 'conditioning.safetensors'))
    fields = ('kind', 'has_audio', 'num_latent_frames', 'latent_height', 'latent_width', 'num_audio_latents')
    (root / 'references.json').write_text(json.dumps([
        {key: getattr(ref, key) for key in fields} for ref in cond.prepared_references]))
    print('CONDITIONING_READY', flush=True)


def infer(root):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from safetensors.torch import load_file, save_file
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3PreparedReference
    from h3_trainer.config import LoraConfig
    from h3_trainer.constants import Geometry
    from h3_trainer.lora import apply_lora
    from h3_trainer.model_loader import load_transformer
    from h3_trainer.packing import build_layout, layout_kwargs, row_timesteps, check_reference_rows
    request = json.loads((root / 'request.json').read_text())
    g = Geometry.create(768, 1344, request['frames']).require_generatable()
    checkpoint = Path('/work/checkpoint') / CHECKPOINT
    config = json.loads((checkpoint / 'config.json').read_text())
    if (config['model']['variant'] != 'ref2va' or
        config['model']['training_mode'] != 'lora' or
        config['training_strategy']['audio']['is_generated'] or
        not config['training_strategy']['video']['is_generated']):
        raise ValueError('Checkpoint is not the intended audio-frozen Ref2VA adapter')
    device = 'cuda:0'
    dtype = torch.bfloat16
    model = load_transformer('/work/model', variant='ref2va', dtype=dtype, device=device)
    apply_lora(model, LoraConfig(**config['lora']))
    weights = load_file(str(checkpoint / 'adapter.safetensors'))
    expected = {k: p.shape for k, p in model.named_parameters() if '.lora_' in k}
    if set(expected) != set(weights) or any(expected[k] != weights[k].shape for k in expected):
        raise ValueError('Adapter keys/shapes mismatch; refusing partial application')
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any('.lora_' in k for k in missing):
        raise ValueError('Incomplete adapter load')
    del weights
    scale = float(request.get('lora_scale', 1.0))
    scaled_layers = set_adapter_strength(model, scale)
    model.eval().requires_grad_(False)
    report = {'checkpoint': CHECKPOINT, 'checkpoint_revision': OUTPUT_REV,
              'adapter_keys_loaded': len(expected), 'rank': config['lora']['rank'],
              'alpha': config['lora']['alpha'], 'model_revision': MODEL_PIN,
              'trainer_revision': TRAINER_PIN, 'audio_mode': 'frozen clean target rows, t=1',
              'scheduler': 'released MiniMaxH3Scheduler', 'personal_lora_loaded': True,
              'lora_scale': scale, 'scaled_layers': scaled_layers}
    (root / 'adapter-load.json').write_text(json.dumps(report, indent=2))
    print('ADAPTER_LOADED', json.dumps(report), flush=True)
    tensors = load_file(str(root / 'conditioning.safetensors'))
    refs = [MiniMaxH3PreparedReference(**r) for r in json.loads((root / 'references.json').read_text())]
    check_reference_rows(refs, tensors['reference'], None)
    layout = build_layout(tensors['prompt'].shape[0], g.latent_frames,
                          g.latent_height, g.latent_width, g.audio_latents,
                          references=refs, tags=tensors['tags'])
    static = layout_kwargs(layout, device)
    text = tensors['prompt'][None].to(device, dtype)
    ref = tensors['reference'][None].to(device, dtype)
    audio = tensors['source_audio'][None].to(device, dtype)
    rng = torch.Generator(device='cpu').manual_seed(request['seed'])
    channels = packed_video_channels(model.config.in_channels, model.config.patch_size)
    if ref.shape[-1] != channels or model.proj_in.in_features != channels:
        raise ValueError('Packed video channel width mismatch')
    if audio.shape[-1] != model.audio_proj_in.in_features:
        raise ValueError('Packed source-audio channel width mismatch')
    x = torch.randn((1, g.video_rows, channels),
                    generator=rng, dtype=torch.float32).to(device)
    scheduler = MiniMaxH3Scheduler(shift=config['flow_matching']['video_shift'])
    scheduler.set_timesteps(request['schedule_points'], device=device)
    begin = time.monotonic()
    with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for i, t in enumerate(scheduler.timesteps):
            timestep, indices = row_timesteps(layout, float(t), 1.0, device)
            video_velocity, _ = model(hidden_states=torch.cat([ref, x.to(dtype)], dim=1),
                audio_hidden_states=audio, encoder_hidden_states=text,
                timestep=timestep, timestep_indices=indices, attention_kwargs=None,
                return_dict=False, **static)
            x = scheduler.step(video_velocity[:, layout.num_condition_video_rows:].float(),
                               t, x, return_dict=False)[0]
            if not bool(torch.isfinite(x).all()):
                raise ValueError('Non-finite video latents')
            # Source audio and visual anchors are never updated.
            print('DENOISE', i + 1, '/', len(scheduler.timesteps),
                  'elapsed_s', round(time.monotonic() - begin, 1), flush=True)
    save_file({'latents': x[0].cpu().contiguous()}, str(root / 'video-latents.safetensors'))
    report.update(sequence_rows=layout.sequence_length,
                  denoiser_evaluations=len(scheduler.timesteps),
                  denoise_seconds=time.monotonic() - begin)
    (root / 'adapter-load.json').write_text(json.dumps(report, indent=2))


def decode(root):
    from safetensors.torch import load_file
    from h3_trainer.constants import Geometry
    from h3_trainer.preprocessing.encoders import H3Encoders
    from h3_trainer.preprocessing.media import write_video_with_audio
    request = json.loads((root / 'request.json').read_text())
    g = Geometry.create(768, 1344, request['frames'])
    enc = H3Encoders('/work/model', need_video=True, need_audio=False, need_text=False)
    rows = load_file(str(root / 'video-latents.safetensors'))['latents']
    frames = enc.decode_video(rows, g.latent_frames, g.latent_height, g.latent_width)
    enc.unload()
    if len(frames) != g.num_frames:
        raise ValueError('Decoded frame count mismatch')
    write_video_with_audio(frames, root / 'h3-video-only.mp4', waveform=None)
    # Use the original VoxCPM2 speech, NOT H3-generated/reconstructed audio.
    command(['ffmpeg', '-y', '-v', 'error', '-i', root / 'h3-video-only.mp4',
             '-i', root / 'tts.wav', '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
             '-c:a', 'aac', '-b:a', '192k', '-af', 'apad', '-t', g.duration,
             '-movflags', '+faststart', root / 'review.mp4'])
    command(['ffmpeg', '-v', 'error', '-xerror', '-i', root / 'review.mp4', '-f', 'null', '-'])
    command(['ffmpeg', '-y', '-v', 'error', '-i', root / 'review.mp4', '-frames:v', '1', root / 'preview.png'])


def orchestrate():
    import shutil
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    repo = os.environ['H3_OUTPUT_REPO']
    if not api.repo_info(repo).private:
        raise ValueError('Output must stay private')
    root = Path('/work/review')
    root.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ['H3_REVIEW_INPUT'])
    for name in ('portrait.png', 'voice-reference.wav', 'voice-provenance.json'):
        shutil.copyfile(source / name, root / name)
    for name in ('tts.wav', 'driving.wav', 'request.json', 'comparison.json', 'reference-provenance.json'):
        if (source / name).is_file():
            shutil.copyfile(source / name, root / name)
    trainer = Path('/work/trainer')
    command(['git', 'clone', '--quiet', 'https://github.com/AviadDahan/MiniMax-H3-Trainer', trainer])
    command(['git', '-C', trainer, 'checkout', '--quiet', TRAINER_PIN])
    env = dict(os.environ, PYTHONPATH=str(trainer / 'src'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    case_defs = comparison_cases() if (root / 'comparison.json').is_file() else []
    allowance = 82 * 60 if case_defs else 110 * 60
    state = {'status': 'running', 'agent': False, 'background_service': False,
             'gpu': 'h200', 'rate_usd_hour': 5, 'job_max_usd': 7.5 if case_defs else 10,
             'comparison': bool(case_defs), 'completed_cases': [],
             'model_revision': MODEL_PIN, 'checkpoint_revision': OUTPUT_REV,
             'checkpoint': CHECKPOINT, 'ai_generated': True,
             'lip_sync_quality_verified': False, 'identity_quality_verified': False}
    prefix = os.environ['H3_REVIEW_OUTPUT_PREFIX']
    def persist():
        state['elapsed_seconds'] = time.monotonic() - start
        state['estimated_gpu_usd'] = state['elapsed_seconds'] / 3600 * 5
        (root / 'review-status.json').write_text(json.dumps(state, indent=2))
        api.upload_folder(repo_id=repo, folder_path=root, path_in_repo=prefix,
                          ignore_patterns=['*.safetensors'], commit_message='Private single-utterance inference review')
    try:
        state['phase'] = 'voxcpm2_zero_shot'
        if all((root / name).is_file() for name in ('tts.wav', 'driving.wav', 'request.json')):
            print('TTS_REUSED: previously generated original speech', flush=True)
        else:
            subprocess.run([sys.executable, __file__, '--stage', 'tts'], env=env, check=True, timeout=900)
        persist()  # Preserve original TTS even if H3 fails later.
        state['phase'] = 'model_download'
        snapshot_download('MiniMaxAI/MiniMax-H3', revision=MODEL_PIN, local_dir='/work/model',
                          ignore_patterns=['FL2VA/*', 'Ref2VA/*', 'transformer/*', 'assets/*'])
        snapshot_download(repo, revision=OUTPUT_REV, local_dir='/work/checkpoint',
                          allow_patterns=[CHECKPOINT + '/*'])
        def run_stage(stage, stage_root):
            state['phase'] = stage
            print('PHASE:', stage, 'ROOT:', stage_root.name, flush=True)
            remaining = allowance - (time.monotonic() - start)
            if remaining < 60:
                raise TimeoutError('Inference time allowance exhausted')
            subprocess.run([sys.executable, __file__, '--stage', stage,
                            '--root', str(stage_root)], env=env, check=True, timeout=remaining)
            persist()
        run_stage('condition', root)
        if case_defs:
            base_request = json.loads((root / 'request.json').read_text())
            for case in case_defs:
                folder = root / 'cases' / case['name']
                folder.mkdir(parents=True)
                for name in ('conditioning.safetensors', 'references.json', 'tts.wav', 'portrait.png'):
                    os.link(root / name, folder / name)
                case_request = dict(base_request, **case)
                (folder / 'request.json').write_text(json.dumps(case_request, ensure_ascii=False, indent=2))
                state['active_case'] = case['name']
                run_stage('infer', folder)
                run_stage('decode', folder)
                state['completed_cases'].append(case)
                persist()  # Preserve each finished video before starting the next.
        else:
            run_stage('infer', root)
            run_stage('decode', root)
        state.update(status='completed_awaiting_visual_review', phase='complete')
    except BaseException as exc:
        state.update(status='failed', error=type(exc).__name__)
        raise
    finally:
        persist()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['tts', 'condition', 'infer', 'decode'])
    p.add_argument('--root', type=Path, default=Path('/work/review'))
    args = p.parse_args()
    if args.stage:
        globals()[args.stage](args.root)
    else:
        orchestrate()
