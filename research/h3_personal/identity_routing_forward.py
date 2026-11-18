"""Opt-in training dispatch for the exact patched H3 forward; no downloads."""


def validate_routing_config(config):
    strategy = config.training_strategy
    ratio = strategy.identity_tread_ratio
    if not ratio:
        return
    import os
    conditions = [*strategy.video.conditions, *strategy.audio.conditions]
    if (len(conditions) != 1 or conditions[0].type != "reference"
            or conditions[0].modality != "image" or conditions[0].probability != 1.0
            or not 0 <= strategy.identity_tread_start < strategy.identity_tread_end <= 50):
        raise ValueError("TREAD requires one always-on image reference and a valid block window")
    if (strategy.identity_guidance_scale is not None or config.model.variant != "ref2va"
            or config.model.training_mode != "lora" or not strategy.video.is_generated
            or strategy.audio.is_generated or not strategy.uses_references
            or config.optimization.batch_size != 1
            or config.optimization.gradient_accumulation_steps != 1
            or config.acceleration.strategy != "ddp"
            or int(os.environ.get("WORLD_SIZE", "1")) != 1
            or config.acceleration.quantization != "none"
            or config.acceleration.mixed_precision_mode not in ("no", "bf16")
            or config.lora.dropout != 0):
        raise ValueError("TREAD requires Ref2VA image-conditioned, clean-audio singleton LoRA without guidance")


def routing_forward(model, packed, *, ratio, seed, start_block, end_block):
    import torch
    from .identity_routing import target_video_routing_plan
    kwargs = packed.to_model_kwargs()
    if not ratio or not torch.is_grad_enabled():
        return model(**kwargs)  # no routing argument: pristine inference path
    if packed.video.shape[0] != 1 or packed.video_loss_mask.dtype != torch.bool:
        raise ValueError("TREAD requires singleton target mask")
    if (packed.audio_loss_mask.any().item()
            or not torch.equal(packed.audio, packed.clean_audio.to(packed.audio))):
        raise ValueError("TREAD cannot change clean-audio/video-only contract")
    audio_indices = packed.layout_kwargs["audio_indices"]
    audio_times = packed.timestep.index_select(0, packed.timestep_indices.index_select(0, audio_indices))
    if not torch.equal(audio_times, torch.ones_like(audio_times)):
        raise ValueError("TREAD requires clean audio timestep1")
    indices = packed.layout_kwargs["video_indices"]
    mask = packed.video_loss_mask
    if indices.ndim != 1 or mask.shape != indices.shape:
        raise ValueError("target mask must match packed video indices")
    targets = indices[mask]
    # Target-frame geometry is encoded in positions; reference rows are excluded.
    times = packed.layout_kwargs["position_ids"].index_select(0, targets)[:, 0]
    latent_frames = int(torch.unique(times).numel())
    plan = target_video_routing_plan(sequence_length=packed.sequence_length,
        video_indices=indices.tolist(), video_loss_mask=mask.tolist(),
        bypass_ratio=ratio, seed=seed, start_block=start_block, end_block=end_block,
        training=True, latent_frames=latent_frames)
    if "identity_routing_plan" in kwargs:
        raise ValueError("routing plan already present")
    return model(**kwargs, identity_routing_plan=plan)
