"""Same-Job short Ref2VA previews from validated cache and fresh canonical adapter.

No new weights/downloads/TTS; uses clean recorded audio from the frozen sample.
Generated preview is not independent/generalization/likeness acceptance.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from research import hf_identity_train as train
from research import hf_identity_encode as encode
from research.h3_personal import component_cache as cache


def case_inputs(index, sample_id):
    matches = [s for s in index['samples'] if s['id'] == sample_id]
    if len(matches) != 1:
        raise ValueError('preview ID must be in validated fixed cache')
    sample = matches[0]
    if sample['num_frames'] != 124:
        raise ValueError('preview requires fixed124-frame short geometry')
    return sample


def conditioning_identity(request, case, reference_entry, geometry):
    """Bind conditioning to bytes/implementation, never seed/audio/adapter weights."""
    import hashlib
    from research.h3_personal.component_cache import canonical
    from research.h3_personal import component_encode
    return {'schema': 'h3-ref2va-prompt-identity-v1',
            'prompt_sha256': hashlib.sha256(case['prompt'].encode()).hexdigest(),
            'reference_image_sha256': case['reference_image']['sha256'],
            'reference_component_sha256': reference_entry['sha256'],
            'geometry': geometry, 'model_pins': request['pins'],
            'model_inventory_sha256': hashlib.sha256(canonical(request['model_inventory'])).hexdigest(),
            'runtime_overlay_sha256': {k: v for k, v in request.get('runtime_overlay', {}).items()
                                      if k.endswith('sha256')},
            'preview_implementation_sha256': cache.sha256(__file__),
            'conditioning_implementation_sha256': cache.sha256(component_encode.__file__)}


def validate_prompt_tensors(tensors):
    import torch
    if (set(tensors) != {'prompt_embeds', 'text_token_tags'} or
            tensors['prompt_embeds'].ndim != 2 or tensors['text_token_tags'].ndim != 1 or
            tensors['prompt_embeds'].shape[0] != tensors['text_token_tags'].shape[0] or
            min(tensors['prompt_embeds'].shape) <= 0 or
            tensors['prompt_embeds'].dtype != torch.bfloat16 or
            tensors['text_token_tags'].dtype != torch.long or
            not bool(torch.isfinite(tensors['prompt_embeds']).all())):
        raise ValueError('cached prompt tensor layout/nonfinite mismatch')
    return tensors


def auxiliary_identity(request, width, height, frames):
    """Bind measured envelopes to source/model/runtime/geometry, not variable speech."""
    from research.h3_personal.component_cache import canonical
    return {'models':request['pins'],
        'inventory_sha256':__import__('hashlib').sha256(canonical(request['model_inventory'])).hexdigest(),
        'runtime_overlay':request.get('runtime_overlay',{}),
        'preview_source_sha256':cache.sha256(__file__),
        'geometry':[width,height,frames], 'audio_format':'fixed-stereo32kHz'}


def create_resident_session(*, allow_measured_auxiliary=False):
    from research.h3_personal.resident_session import ResidentSession
    def release():
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return ResidentSession(release,allow_measured_auxiliary=allow_measured_auxiliary)


class TurnInterrupted(RuntimeError):
    """Cooperative turn cancellation, not a GPU Job timeout/cancellation."""


def generate(request, checkpoint, case, *, session=None, is_current=None, packed_only=False):
    options = {'packed_only': True} if packed_only else {}
    if session is None:
        return _generate(request, checkpoint, case, is_current=is_current, **options)
    with session.turn():
        return _generate(request, checkpoint, case, session=session, is_current=is_current, **options)


def _generate(request, checkpoint, case, *, session=None, is_current=None, packed_only=False):
    def check_turn():
        if is_current is not None and not is_current():
            raise TurnInterrupted('discard interrupted turn; no media published')
    check_turn()
    generation_started = time.monotonic()
    phases = {}
    phase_started = generation_started
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from safetensors.torch import load_file, save_file
    from diffusers import MiniMaxH3Scheduler
    config = train.load_runtime(request)
    from h3_trainer.model_loader import load_transformer
    from h3_trainer.lora import set_trainable
    from h3_trainer.constants import Geometry
    from h3_trainer.packing import (prepared_references_from_cache, check_reference_rows,
                                   build_layout, layout_kwargs, row_timesteps)
    from research.hf_infer_personal import packed_video_channels
    phases['runtime_load_seconds'] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    index = train.read_json(Path(request['cache']) / 'index.json')
    sample = case_inputs(index, case['sample_id'])
    tensors = {}
    for kind in ('text', 'reference', 'audio'):
        entry = index['components'][sample['components'][kind]]
        path = cache.contained_path(request['cache'], entry['path'])
        if cache.sha256(path) != entry['sha256']:
            raise ValueError('preview component byte mismatch')
        tensors[kind] = load_file(str(path), device='cpu')
    if 'driving_audio' in case:
        # Fresh generated TTS replaces frozen audio latents; never lip-sync old audio.
        check_turn()
        aux_identity=auxiliary_identity(request,sample['width'],sample['height'],sample['num_frames'])
        if session is not None:
            session.auxiliary_available('audio_vae',aux_identity,int(torch.cuda.mem_get_info()[0]))
        torch.cuda.reset_peak_memory_stats()
        import gc
        import wave
        from h3_trainer.preprocessing.encoders import H3Encoders
        from research.h3_personal.component_encode import PinnedBackend
        driving = case['driving_audio']
        if cache.sha256(driving['path']) != driving['sha256']:
            raise ValueError('generated TTS byte pin mismatch')
        with wave.open(driving['path'], 'rb') as wav:
            if wav.getnframes()/wav.getframerate() > sample['num_frames']/24:
                raise ValueError('generated TTS would be truncated by fixed preview geometry')
        enc = H3Encoders(request['model'], need_video=False, need_audio=True, need_text=False)
        backend = PinnedBackend(enc, Path(request['work']) / 'runtime')
        tensors['audio'] = {k:v.detach().cpu().contiguous() for k,v in backend.audio({
            'sources': {'audio': {'path': driving['path']}},
            'settings': {'num_frames': sample['num_frames']}}).items()}
        audio_peak=int(torch.cuda.max_memory_allocated())
        phases['audio_vae_peak_allocated_bytes']=audio_peak
        if session is not None and not session.resident:
            session.record_auxiliary_peak('audio_vae',aux_identity,audio_peak)
        enc.unload()
        del backend, enc
        gc.collect(); torch.cuda.empty_cache()
    phases['component_verify_load_seconds'] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    condition_cache_hit = None
    if 'prompt' in case:
        # A verified hit skips Qwen entirely; misses retain baseline encoding/unload order.
        import gc
        from research.h3_personal import condition_cache
        ref = case['reference_image']
        if cache.sha256(ref['path']) != ref['sha256']:
            raise ValueError('new prompt reference image pin mismatch')
        identity = conditioning_identity(request, case,
            index['components'][sample['components']['reference']],
            tensors['reference']['geometry'][0].tolist())
        root = Path(request['work']) / 'condition-cache'
        cached = condition_cache.load(root, identity,
            lambda p: validate_prompt_tensors(load_file(str(p), device='cpu')))
        condition_cache_hit = cached is not None
        if cached is not None:
            tensors['text'] = cached
        else:
            if session is not None:
                session.require_conditioner_available()
            from h3_trainer.preprocessing.encoders import H3Encoders
            from research.h3_personal.component_encode import PinnedBackend
            enc = H3Encoders(request['model'], need_video=False, need_audio=False, need_text=True,
                             text_device_map='auto')
            backend = PinnedBackend(enc, Path(request['work']) / 'runtime')
            image = backend.image({'sources': {'image': {'path': ref['path']}}})
            references = backend.presentation(image, tensors['reference']['geometry'][0].tolist())
            tensors['text'] = validate_prompt_tensors({k: v.detach().cpu().contiguous() for k, v in
                               backend.text(case['prompt'], references).items()})
            condition_cache.store(root, identity,
                                  lambda p: save_file(tensors['text'], str(p)))
            del backend, enc, references, image
            gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize()
    phases['text_conditioning_seconds'] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    check_turn()
    def load_model():
        model = load_transformer(request['model'], variant='ref2va', dtype=torch.bfloat16,
                                 device=torch.device('cuda:0'), quantization='none')
        set_trainable(model, 'lora', config.lora)
        train.helper().strict_reload(model, checkpoint, freshly_constructed=True,
                                     expected_pins=request['pins'])
        return model.eval().requires_grad_(False)
    model_resident_hit = session.resident if session is not None else False
    if session is None:
        model = load_model()
    else:
        identity = {'pins': request['pins'], 'inventory': request['model_inventory'],
                    'overlay': request.get('runtime_overlay', {}),
                    'implementation_sha256': cache.sha256(__file__),
                    'checkpoint': {name: train.sha256(Path(checkpoint)/name) for name in
                        ('EXPORTED.json', 'adapter_config.json', 'adapter_model.safetensors')},
                    'precision': 'bf16', 'quantization': 'none'}
        model = session.get(identity, load_model)
    torch.cuda.synchronize()
    phases['transformer_load_adapter_reload_seconds'] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    reference = tensors['reference']['latents']
    refs = prepared_references_from_cache([('image', reference, tensors['reference']['geometry'])])
    check_reference_rows(refs, reference, None)
    g = Geometry.create(sample['width'], sample['height'], sample['num_frames']).require_generatable()
    text = tensors['text']['prompt_embeds']
    layout = build_layout(text.shape[0], g.latent_frames, g.latent_height, g.latent_width,
                          g.audio_latents, references=refs, tags=tensors['text']['text_token_tags'])
    device = 'cuda:0'; dtype = torch.bfloat16
    static = layout_kwargs(layout, device)
    text = text[None].to(device, dtype); reference = reference[None].to(device, dtype)
    audio = tensors['audio']['latents'][None].to(device, dtype)
    channels = packed_video_channels(model.config.in_channels, model.config.patch_size)
    if reference.shape[-1] != channels or audio.shape[-1] != model.audio_proj_in.in_features:
        raise ValueError('preview packed channel mismatch')
    rng = torch.Generator(device='cpu').manual_seed(case['seed'])
    x = torch.randn((1, g.video_rows, channels), generator=rng, dtype=torch.float32).to(device)
    scheduler = MiniMaxH3Scheduler(shift=config.flow_matching.video_shift)
    scheduler.set_timesteps(case['schedule_points'], device=device)
    torch.cuda.synchronize()
    phases['packing_seconds'] = time.monotonic() - phase_started
    if packed_only:
        # Private local intermediate ONLY: contains conditioning, never publish it.
        if case['schedule_points'] != 50 or len(scheduler.timesteps) != 49:
            raise ValueError('production comparison requires unchanged 50points/49forwards')
        timesteps=[]; indices=None
        for t in scheduler.timesteps:
            row_t, row_indices=row_timesteps(layout,float(t),1.0,device)
            if indices is not None and not torch.equal(indices,row_indices):
                raise ValueError('unexpected varying row timestep map')
            indices=row_indices;timesteps.append(row_t)
        turbo_scheduler=MiniMaxH3Scheduler(shift=config.flow_matching.video_shift)
        turbo_scheduler.set_timesteps(9,device=device)  # author CLI: 8 NFE = 9 sigma grid points
        turbo_rows=[]
        for t in turbo_scheduler.timesteps:
            row_t,row_indices=row_timesteps(layout,float(t),1.0,device)
            if not torch.equal(indices,row_indices):raise ValueError('Turbo row map differs')
            turbo_rows.append(row_t)
        packed={'turbo_row_timesteps':torch.stack(turbo_rows),
                'initial_video':x,'reference':reference,'audio':audio,'text':text,
                'row_timesteps':torch.stack(timesteps),'timestep_indices':indices,
                **{'static.'+k:v for k,v in static.items()}}
        root=Path(case['output'])
        save_file({k:v.detach().cpu().contiguous() for k,v in packed.items()},
                  str(root/'private-packed.safetensors'))
        encode.dump(root/'private-packed.json',{
            'schema':'h3-private-packed-comparison-v1','private_conditioning_do_not_upload':True,
            'sha256':cache.sha256(root/'private-packed.safetensors'),
            'request_sha256':__import__('hashlib').sha256(cache.canonical(request)).hexdigest(),
            'checkpoint_export_sha256':train.sha256(Path(checkpoint)/'EXPORTED.json'),
            'case':case,'video_shift':config.flow_matching.video_shift,
            'condition_video_rows':layout.num_condition_video_rows,
            'sequence_rows':int(indices.numel()),'schedule_points':50,'denoiser_forwards':49,
            'phases':phases,'precision':'bf16','quantization':'none'})
        return
    start = time.monotonic()
    with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for i, t in enumerate(scheduler.timesteps):
            check_turn()
            timestep, indices = row_timesteps(layout, float(t), 1.0, device)
            velocity, _ = model(hidden_states=torch.cat([reference, x.to(dtype)], dim=1),
                audio_hidden_states=audio, encoder_hidden_states=text, timestep=timestep,
                timestep_indices=indices, attention_kwargs=None, return_dict=False, **static)
            x = scheduler.step(velocity[:, layout.num_condition_video_rows:].float(), t, x, return_dict=False)[0]
            if not bool(torch.isfinite(x).all()):
                raise ValueError('nonfinite preview denoise')
            print('PREVIEW_DENOISE', i + 1, '/', len(scheduler.timesteps), flush=True)
    check_turn()
    torch.cuda.synchronize()
    denoise_seconds = time.monotonic() - start
    save_file({'latents': x[0].cpu().contiguous()}, str(Path(case['output']) / 'video-latents.safetensors'))
    encode.dump(Path(case['output']) / 'inference-profile.json', {
        'schema': 'h3-inference-profile-v1', 'phase': 'generate', 'phases': phases,
        'denoise_seconds': denoise_seconds, 'generation_seconds': time.monotonic()-generation_started,
        'denoiser_forwards': len(scheduler.timesteps), 'video_seconds': g.num_frames/24,
        'video_seconds_per_denoise_second': (g.num_frames/24)/denoise_seconds,
        'condition_cache_hit': condition_cache_hit, 'model_resident_hit': model_resident_hit,
        'max_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'precision': 'bf16', 'quantization': 'none', 'quality_verified': False,
        'scope': 'generate process only; not prepare, decode, upload or end-to-end TTFV'})
    encode.dump(Path(case['output']) / 'generation.json', {'generated': True, 'sample_id':sample['id'],
        'seed':case['seed'], 'frames':g.num_frames, 'schedule_points':case['schedule_points'],
        'denoise_seconds':denoise_seconds, 'audio_mode':'generated zero-shot TTS' if 'driving_audio' in case else 'frozen clean recorded target',
        'driving_audio_sha256':case.get('driving_audio',{}).get('sha256'),
        'adapter_export_sha256':train.sha256(Path(checkpoint)/'EXPORTED.json'),
        'prompt_override': 'prompt' in case,
        'prompt_sha256': __import__('hashlib').sha256(case['prompt'].encode()).hexdigest() if 'prompt' in case else None,
        'reference_image_sha256': case.get('reference_image', {}).get('sha256'),
        'quality_verified':False, 'held_out':False, 'ai_generated':True})


def decode(request, case, *, session=None):
    if session is not None and session.resident and not session.allow_measured_auxiliary:
        raise ValueError('release resident DiT before VAE decode; VRAM overlap unverified')
    decode_started = time.monotonic()
    import torch
    from safetensors.torch import load_file
    train.load_runtime(request)
    from h3_trainer.constants import Geometry
    from h3_trainer.preprocessing.encoders import H3Encoders
    from h3_trainer.preprocessing.media import write_video_with_audio
    sample = case_inputs(train.read_json(Path(request['cache'])/'index.json'), case['sample_id'])
    g = Geometry.create(sample['width'], sample['height'], sample['num_frames'])
    aux_identity=auxiliary_identity(request,g.width,g.height,g.num_frames)
    if session is not None:
        session.auxiliary_available('video_vae',aux_identity,int(torch.cuda.mem_get_info()[0]))
    torch.cuda.reset_peak_memory_stats()
    root = Path(case['output'])
    vae_started = time.monotonic()
    enc = H3Encoders(request['model'], need_video=True, need_audio=False, need_text=False)
    torch.cuda.synchronize()
    vae_load_seconds = time.monotonic() - vae_started
    vae_started = time.monotonic()
    rows = load_file(str(root/'video-latents.safetensors'))['latents']
    frames = enc.decode_video(rows, g.latent_frames, g.latent_height, g.latent_width)
    torch.cuda.synchronize()
    vae_decode_seconds = time.monotonic() - vae_started
    vae_peak=int(torch.cuda.max_memory_allocated())
    if session is not None and not session.resident:
        session.record_auxiliary_peak('video_vae',aux_identity,vae_peak)
    enc.unload()
    del enc
    import gc
    gc.collect();torch.cuda.empty_cache()
    if len(frames) != g.num_frames:
        raise ValueError('preview decoded frame count differs')
    write_video_with_audio(frames, root/'video-only.mp4', waveform=None)
    audio = Path(case['audio_path'])
    if train.sha256(audio) != case['audio_sha256']:
        raise ValueError('preview clean audio pin differs')
    from research.prepare_identity_runtime import run
    run(['ffmpeg','-y','-v','error','-i',str(root/'video-only.mp4'),'-i',str(audio),
         '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
         '-af','apad','-t',str(g.duration),'-movflags','+faststart',str(root/'review.mp4')])
    run(['ffmpeg','-v','error','-xerror','-i',str(root/'review.mp4'),'-f','null','-'])
    encode.dump(root/'decode-profile.json', {'schema': 'h3-inference-profile-v1',
        'phase': 'decode', 'vae_peak_allocated_bytes':vae_peak,
        'dit_retained':bool(session is not None and session.resident), 'vae_load_seconds': vae_load_seconds,
        'vae_decode_seconds': vae_decode_seconds, 'decode_seconds': time.monotonic()-decode_started,
        'scope': 'decode process only; includes mux/validation, not end-to-end TTFV',
        'quality_verified': False})


def run_previews(pipeline, out, seconds):
    from huggingface_hub import HfApi, CommitOperationAdd
    request_path = out/'train-request.json'; request = train.read_json(request_path)
    checkpoint = train.read_json(out/'training/training-snapshot.json')['checkpoint']
    train.helper().inspect_verified(checkpoint, expected_pins=request['pins'])
    manifest = encode.pinned_document(pipeline['input_documents']['manifest_path']['path'],
                                     pipeline['input_documents']['manifest_path']['sha256'])
    raw = {s['id']:s for s in manifest['samples']}
    end = time.monotonic()+seconds
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    api = HfApi(token=os.environ.get('HF_TOKEN')); repo=request['output_repo']; previews=[]
    generated_audio = None
    if 'tts' in pipeline['inference']:
        tts_root=out/'conversation-tts';tts_root.mkdir(mode=0o700)
        tts_request=tts_root/'request.json';encode.dump(tts_request,pipeline['inference']['tts'])
        print('CONVERSATION_TTS_PREPARE',flush=True)
        train.bounded_process([sys.executable,'-m','research.hf_conversation_tts','prepare',
            '--request',str(tts_request),'--root',str(tts_root)],
            deadline=None if pipeline.get('operator_managed_stages') else end-120,
            env=dict(os.environ,HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',PYTHONDONTWRITEBYTECODE='1'))
        generated_audio={'path':str(tts_root/'tts.wav'),'sha256':train.sha256(tts_root/'tts.wav')}
    for i,sid in enumerate(pipeline['inference']['sample_ids']):
        root=out/'previews'/str(i);root.mkdir(parents=True,mode=0o700)
        entry=raw[sid]
        case={'sample_id':sid,'seed':pipeline['inference'].get('seed',42)+i,'schedule_points':50,'output':str(root),
              'audio_path':str(Path(pipeline['production_inputs']['media_root'])/entry['audio']),
              'audio_sha256':entry['audio_sha256']}
        if 'prompt' in pipeline['inference']:
            case.update(prompt=pipeline['inference']['prompt'],
                        reference_image=pipeline['production_inputs']['reference_images'][sid])
        if generated_audio:
            case.update(driving_audio=generated_audio,audio_path=generated_audio['path'],
                        audio_sha256=generated_audio['sha256'])
        casepath=root/'case.json';encode.dump(casepath,case)
        if pipeline['inference'].get('native_continuation'):
            from research.run_h3_native_continuation import run_native
            revision=run_native(request_path,checkpoint,casepath,pipeline,env,api)
            previews.append({'case':i,'revision':revision,'sample_id':sid,'native_continuation':True})
            continue
        if pipeline['inference'].get('context_parallel_matrix'):
            from research.run_h3_acceleration_matrix import run_matrix
            revision=run_matrix(request_path,checkpoint,casepath,pipeline,env,api)
            previews.append({'case':i,'revision':revision,'sample_id':sid,'four_condition_matrix':True})
            continue
        if pipeline['inference'].get('resident_batch'):
            import shutil
            batch=root/'resident-batch'
            train.bounded_process([sys.executable,'-m','research.render_h3_resident_batch',
                '--request',str(request_path),'--checkpoint',str(checkpoint),'--case',str(casepath),
                '--adapter-sha256',train.sha256(Path(checkpoint)/'adapter_model.safetensors'),
                '--output',str(batch),'--execute-local-models'],
                deadline=None if pipeline.get('operator_managed_stages') else end-120,env=env)
            for name in ['review.mp4','generation.json','inference-profile.json','decode-profile.json']:
                shutil.copyfile(batch/'2'/name,root/name)
            shutil.copyfile(batch/'batch-profile.json',root/'batch-profile.json')
        else:
            for phase in ('generate','decode'):
                print('PREVIEW_STAGE',i+1,phase,flush=True)
                train.bounded_process([sys.executable,'-m','research.hf_identity_preview',phase,
                    '--request',str(request_path),'--checkpoint',str(checkpoint),'--case',str(casepath)],
                    deadline=None if pipeline.get('operator_managed_stages') else end-120,env=env)
        if api.repo_info(repo,repo_type='model').private is not True:
            raise ValueError('preview output must remain private')
        # No raw media, tensor latents, local paths, or credentials uploaded.
        files=[root/'review.mp4',root/'generation.json',
               root/'inference-profile.json',root/'decode-profile.json']
        if pipeline['inference'].get('resident_batch'):files.append(root/'batch-profile.json')
        operations=[CommitOperationAdd(
            path_in_repo='identity/'+request['run_id']+'/previews/'+str(i)+'/'+p.name,
            path_or_fileobj=str(p)) for p in files]
        if generated_audio:
            from research.h3_personal.generated_transfer import H3_REVISION
            bundle={'schema':'h3-generated-playback-transfer-v1','ai_generated':True,
                'audio_source':'voxcpm2_zero_shot_generated','video_source':'h3_ref2va_identity',
                'h3_revision':H3_REVISION,'adapter_sha256':train.sha256(Path(checkpoint)/'adapter_model.safetensors'),
                'assets':{}}
            for kind,name,path in [('video','video.mp4',root/'video-only.mp4'),
                                   ('original_tts','tts.wav',Path(generated_audio['path']))]:
                bundle['assets'][kind]={'file':name,'sha256':train.sha256(path),'bytes':path.stat().st_size}
                operations.append(CommitOperationAdd(path_in_repo='generated-playback/'+request['run_id']+'-'+str(i)+'/'+name,
                                                      path_or_fileobj=str(path)))
            encode.dump(root/'manifest.json',bundle)
            operations.append(CommitOperationAdd(path_in_repo='generated-playback/'+request['run_id']+'-'+str(i)+'/manifest.json',
                                                  path_or_fileobj=str(root/'manifest.json')))
            operations.append(CommitOperationAdd(path_in_repo='identity/'+request['run_id']+'/previews/'+str(i)+'/tts-receipt.json',
                                                  path_or_fileobj=str(out/'conversation-tts/tts-receipt.json')))
        commit=api.create_commit(repo,repo_type='model',operations=operations,
                                 commit_message='Private generated preview; opt-in TTS bundle, no raw voice/latents')
        previews.append({'case':i,'revision':commit.oid,'sample_id':sid})
    encode.dump(out/'preview-report.json',{'complete':True,'previews':previews,'quality_verified':False})
    print('PREVIEWS_UPLOADED',len(previews),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('generate','decode','pack-comparison'))
    p.add_argument('--request',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True);a=p.parse_args()
    request=train.read_json(a.request);case=train.read_json(a.case)
    if a.phase in ('generate','pack-comparison'):
        generate(request,a.checkpoint,case,packed_only=a.phase=='pack-comparison')
    else:decode(request,case)

if __name__=='__main__':main()
