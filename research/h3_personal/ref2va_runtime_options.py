"""Opt-in orchestration changes; never change DiT weights, steps or conditioning."""


def export_geometry(layout):
    if layout=='landscape':return dict(canvas_width=448,canvas_height=256,width=448,height=256,crop_left=0,crop_right=0)
    if layout=='portrait':return dict(canvas_width=256,canvas_height=448,width=252,height=448,crop_left=2,crop_right=2)
    raise ValueError('unsupported fixed layout')


def finish_visual_only(batch,stage,torch):
    if '_h3_continuation' in batch.extra:
        from research.h3_personal.fast_continuation_overlay import finish_batch
        batch = finish_batch(batch)
    # Same cleanup and empty-waveform convention as upstream non-output ranks.
    # Joint audio/video DiT denoising is NOT skipped or changed.
    batch.extra['audio']=torch.empty((0,2),device='cpu',dtype=torch.float32)
    batch.extra['audio_sample_rate']=stage.audio_vae.sampling_rate
    stage._clear_runtime(batch)
    return batch


def prepare_images_cached(stage,references,num_frames,sample_rate,prepare_reference):
    from copy import copy
    from research.h3_personal.ref2va_runtime import sha
    if any(r.media_type!='image' for r in references):raise ValueError('prepared-image cache only supports explicit still references')
    key=(num_frames,sample_rate,tuple(sha(r.source) for r in references))
    from research.h3_personal.conditioning_experiment import read_policy, policy_key
    policy = read_policy()
    if policy is not None:
        key = (*key, policy_key())
    cache=getattr(stage,'_smd_prepared_images',None)
    if cache is not None and cache[0]==key:
        stage._smd_prepared_image_hits=getattr(stage,'_smd_prepared_image_hits',0)+1
        return [copy(r) for r in cache[1]]
    if policy is not None:
        from research.h3_personal.conditioning_experiment import prepare_reference as experimental_prepare
        prepared=[experimental_prepare(r,num_frames,sample_rate,native_prepare=prepare_reference) for r in references]
    else:
        prepared=[prepare_reference(r,num_frames,sample_rate) for r in references]
    # Keep clean PIL pixels; fresh metadata objects and native RNG on every call.
    stage._smd_prepared_images=(key,[copy(r) for r in prepared])
    return prepared


def residency_receipt(pipeline,torch,policy='resident-visual'):
    modules={name:pipeline.get_module(name) for name in ('vae','audio_vae')}
    devices={name:sorted({str(t.device) for t in list(module.parameters())+list(module.buffers())})
             for name,module in modules.items()}
    if any(not rows or any(not d.startswith('cuda:') for d in rows) for rows in devices.values()):
        raise RuntimeError('resident VAE policy not actually resident on CUDA')
    return dict(policy=policy,vae_devices=devices,audio_decode_skipped=True,
                prepared_image_cache_hits=getattr(pipeline._stage_name_mapping['input_preparation_stage'],'_smd_prepared_image_hits',0),
                joint_audio_denoising_unchanged=True,
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                device_total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
