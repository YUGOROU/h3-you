"""Guided video targets for the exact Aviad pin; torch is imported only on use.

Not an identity objective, teacher, training adapter, or permission to train.
The installer wires these functions into real source paths. Gate C remains closed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path

AVIAD_REVISION = "7a1563a267a17da898df3396acbd6a9d24fcb142"
CACHE_SCHEMA = "h3-ref2va-text-uncond-v1"
PROBE_PROMPT = " "  # Musubi's single-space text probe; reference presentation retained.
C_GATE_ENABLED = False


def file_sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_settings(scale, sigma_min):
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("guidance scale must be finite and >= 1")
    if not math.isfinite(sigma_min) or not 0 <= sigma_min <= 1:
        raise ValueError("guidance sigma_min must be in [0, 1]")


def validate_training_config(config):
    """Deliberately narrow subset; reject unsupported backends before model loading."""
    strategy = config.training_strategy
    if strategy.identity_guidance_scale is None:
        return
    check_settings(strategy.identity_guidance_scale, strategy.identity_guidance_sigma_min)
    conditions = [*strategy.video.conditions, *strategy.audio.conditions]
    if (config.model.variant != "ref2va" or config.model.training_mode != "lora"
            or not strategy.video.is_generated or strategy.audio.is_generated
            or len(conditions) != 1 or conditions[0].type != "reference"
            or conditions[0].modality != "image" or conditions[0].probability != 1.0):
        raise ValueError("guidance requires Ref2VA LoRA, one always-on image reference, generated video and clean audio")
    if (config.optimization.batch_size != 1 or config.acceleration.strategy != "ddp"
            or int(os.environ.get("WORLD_SIZE", "1")) != 1
            or config.acceleration.quantization != "none"
            or config.acceleration.mixed_precision_mode not in ("no", "bf16")
            or config.lora.dropout != 0):
        raise ValueError("guidance subset requires single-process unquantized ddp, batch 1, no/bf16 and zero LoRA dropout")


def encode_pair(text_pass, record, row):
    """Use actual pinned Ref2VA pixel/label presentation, not a global text cache."""
    if not text_pass.options.references or text_pass.options.keyframes:
        raise ValueError("guidance text pass requires --references and no keyframes")
    geometry = record.get("reference_geometry", [])
    if len(geometry) != 1 or int(geometry[0][0]) != 0:
        raise ValueError("guidance cache subset requires exactly one image reference")
    if not row:
        raise ValueError("guidance cache requires the original manifest row")
    refs = text_pass._rebuild_reference_presentation(record, row)
    embeds, tags = text_pass.encoders.encode_ref2va_prompt(record.get("caption", ""), refs)
    uncond, uncond_tags = text_pass.encoders.encode_ref2va_prompt(PROBE_PROMPT, refs)
    return embeds, tags, uncond, uncond_tags


def write_uncond_cache(root, sample_id, conditional_path, embeds, tags):
    from safetensors.torch import save_file
    root = Path(root)
    reference_path = root / "reference_latents" / f"{sample_id}.safetensors"
    metadata = {
        "schema": CACHE_SCHEMA, "aviad_revision": AVIAD_REVISION,
        "sample_id": sample_id, "probe_prompt": PROBE_PROMPT,
        "conditional_sha256": file_sha256(conditional_path),
        "reference_sha256": file_sha256(reference_path),
    }
    directory = root / "identity_uncond"
    directory.mkdir(parents=True, exist_ok=True)
    save_file({"prompt_embeds": embeds, "text_token_tags": tags},
              str(directory / f"{sample_id}.safetensors"), metadata=metadata)


def validate_cache_metadata(metadata, sample_id, conditional_path, reference_path):
    expected = {
        "schema": CACHE_SCHEMA, "aviad_revision": AVIAD_REVISION,
        "sample_id": sample_id, "probe_prompt": PROBE_PROMPT,
        "conditional_sha256": file_sha256(conditional_path),
        "reference_sha256": file_sha256(reference_path),
    }
    if metadata != expected:
        raise ValueError(f"Missing/incompatible guidance cache provenance for {sample_id}")


def load_uncond_cache(dataset, record, text_payload):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    root = dataset.root
    path = root / dataset.data_sources["identity_uncond"] / f"{record.id}.safetensors"
    audio_payload = dataset._load_tensor("audio", record.id, required=True)
    if not record.has_audio or audio_payload is None:
        raise ValueError("guidance requires real clean target audio, not placeholder silence")
    audio = audio_payload["latents"]
    if audio.ndim != 2 or audio.shape != (record.audio_rows, 32) or not bool(torch.isfinite(audio).all()):
        raise ValueError("incompatible clean target audio cache")
    if "reference" not in dataset.data_sources:
        raise ValueError("guidance requires a reference source")
    reference = dataset._load_tensor("reference", record.id, required=True)
    geometry = reference["geometry"]
    if (geometry.shape != (1, 5) or int(geometry[0, 0]) != 0
            or int(geometry[0, 4]) != 0 or "audio_latents" in reference):
        raise ValueError("guidance subset requires one image reference without reference audio")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    validate_cache_metadata(metadata, record.id,
                            root / dataset.data_sources["text"] / f"{record.id}.safetensors",
                            root / dataset.data_sources["reference"] / f"{record.id}.safetensors")
    payload = load_file(str(path))
    if set(payload) != {"prompt_embeds", "text_token_tags"}:
        raise ValueError("guidance cache keys are incompatible")
    embeds, tags = payload["prompt_embeds"], payload["text_token_tags"]
    cond, cond_tags = text_payload["prompt_embeds"], text_payload.get("text_token_tags")
    if (embeds.ndim != 2 or cond.ndim != 2 or embeds.shape[1] != cond.shape[1]
            or not embeds.is_floating_point() or not bool(torch.isfinite(embeds).all())
            or tags.ndim != 1 or tags.dtype != torch.int64 or tags.numel() != embeds.shape[0]
            or cond_tags is None):
        raise ValueError("guidance cache embeddings/tags are incompatible")
    from h3_trainer.constants import MINIMAX_H3_VIDEO_TAG, MINIMAX_H3_TEXT_TAG
    if (not bool(((tags == MINIMAX_H3_TEXT_TAG) | (tags == MINIMAX_H3_VIDEO_TAG)).all())
            or int((tags == MINIMAX_H3_VIDEO_TAG).sum()) == 0
            or int((tags == MINIMAX_H3_VIDEO_TAG).sum()) != int((cond_tags == MINIMAX_H3_VIDEO_TAG).sum())):
        raise ValueError("guidance cache must retain the matching reference vision block")
    return {"identity_uncond_text": embeds, "identity_uncond_tags": tags}


def attach_probe(packed, batch, sigmas, config, references, geometry, device):
    """Rebuild only text-dependent indices/timesteps; never assemble/noise twice."""
    if config.identity_guidance_scale is None:
        return packed
    import torch
    from h3_trainer.packing import build_layout, layout_kwargs, loss_masks, row_timesteps
    if not references or batch.get("reference_video_rows") is None:
        raise ValueError("guidance cannot fall back to reference-free packing")
    if config.audio.is_generated or not config.video.is_generated:
        raise ValueError("guidance cannot change the clean-audio/video-only contract")
    if not torch.equal(packed.audio, packed.clean_audio.to(device=device, dtype=packed.audio.dtype)):
        raise ValueError("guidance requires unchanged clean target audio")
    text = batch["identity_uncond_text"]
    if text.ndim not in (2, 3) or (text.ndim == 3 and text.shape[0] != 1):
        raise ValueError("guidance supports exactly one packed sample")
    layout = build_layout(
        num_text_rows=text.shape[-2], latent_frames=geometry.latent_frames,
        latent_height=geometry.latent_height, latent_width=geometry.latent_width,
        audio_latents=geometry.audio_latents, references=references,
        tags=batch["identity_uncond_tags"],
    )
    video_mask, audio_mask = loss_masks(layout, device)
    if not torch.equal(video_mask, packed.video_loss_mask) or not torch.equal(audio_mask, packed.audio_loss_mask):
        raise ValueError("probe changed modality row layout")
    probe_layout_kwargs = layout_kwargs(layout, device)
    # Ref2VA starts the common audio/video rotary clock after text. Only this
    # common temporal translation is allowed; spatial grids/order must not change.
    delta = text.shape[-2] - packed.text.shape[-2]
    for name in ("video_indices", "audio_indices"):
        old_positions = packed.layout_kwargs["position_ids"].index_select(0, packed.layout_kwargs[name])
        new_positions = probe_layout_kwargs["position_ids"].index_select(0, probe_layout_kwargs[name])
        expected = old_positions.clone()
        expected[:, 0] += delta
        if not torch.allclose(new_positions, expected, rtol=0, atol=1e-9):
            raise ValueError("probe changed spatial layout or relative audio/video clock")
    timestep, indices = row_timesteps(layout, 1 - sigmas.video, 1.0, device)
    text = text.unsqueeze(0) if text.ndim == 2 else text
    probe = replace(packed, text=text.to(device=device, dtype=packed.text.dtype),
                    timestep=timestep, timestep_indices=indices,
                    layout_kwargs=probe_layout_kwargs,
                    sequence_length=int(layout.sequence_length), metadata={})
    # Video/audio and their augmented reference rows remain the exact same objects.
    packed.metadata.update(identity_probe=probe, identity_base_sigma=float(sigmas.u))
    return packed


def guided_target(clean, noise, uncond, scale):
    import torch
    check_settings(scale, 0)
    with torch.no_grad():
        raw = clean.detach().float() - noise.detach().float()
        uncond = uncond.detach().float()
        if raw.shape != uncond.shape:
            raise ValueError("probe target shape mismatch (no broadcasting permitted)")
        return uncond + scale * (raw - uncond)


def training_forward(forward, packed, config):
    """Called by actual train AND held-out loss paths, before conditional forward."""
    if config.identity_guidance_scale is None:
        return forward(packed)
    import torch
    check_settings(config.identity_guidance_scale, config.identity_guidance_sigma_min)
    probe = packed.metadata["identity_probe"]  # absent cache/layout: fail closed
    base_sigma = packed.metadata["identity_base_sigma"]
    if not math.isfinite(base_sigma) or not 0 <= base_sigma <= 1:
        raise ValueError("invalid base sigma")
    raw = (packed.clean_video.float() - packed.noise_video.float()).detach()
    target = raw
    applied = base_sigma >= config.identity_guidance_sigma_min
    if applied:
        # Same model, same LoRA state, same train/eval mode. No adapter disabling.
        with torch.no_grad():
            uncond_video, _ = forward(probe)
        uncond_video = uncond_video[..., packed.video_loss_mask, :]
        target = guided_target(packed.clean_video, packed.noise_video,
                               uncond_video, config.identity_guidance_scale)
    packed.metadata.update(identity_video_target=target, identity_guidance_applied=float(applied))
    return forward(packed)


def video_loss(prediction, packed, config):
    import torch
    if config.identity_guidance_scale is None:
        from h3_trainer.flow_matching import flow_matching_loss
        return flow_matching_loss(prediction, packed.clean_video, packed.noise_video,
                                  row_mask=packed.video_loss_mask)
    target = packed.metadata["identity_video_target"].detach().float()
    pred = prediction[..., packed.video_loss_mask, :].float()
    if pred.shape != target.shape or pred.numel() == 0:
        raise ValueError("guided video loss shape mismatch/empty target")
    return torch.nn.functional.mse_loss(pred, target)
