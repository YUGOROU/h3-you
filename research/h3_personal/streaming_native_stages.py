"""Both-rank native fixed-window hooks for bounded incremental PCM inference."""
from dataclasses import asdict
from .streaming_pcm import _id,_int,_sha,PCMStreamBinding,PCMIngress,PCMChunk
from .streaming_windows import WindowAudioOffer,ResamplingContext,inspect_audio_window
from .continuation_plan import Window
from .streaming_native_owner import StreamingWindowOwner
from .conversation_stages import common,SourceFacade

SCHEMA='h3-native-stream-window-v1'


def restore_offer(record):
    if type(record) is not dict:
        raise ValueError('serialized audio offer required')
    r=dict(record)
    r['binding']=PCMStreamBinding(**r['binding'])
    r['window']=Window(**r['window'])
    r['policy']=ResamplingContext(**r['policy'])
    return WindowAudioOffer(**r)


def validate_manifest(m):
    c=m.get('streaming_audio')
    required={'schema','id','profile_sha256','sequence','logical_source_id','logical_index',
              'role','source_final','conversation_final','committed_before','max_windows','audio_offer'}
    optional={'previous_speech_interrupted', 'visual_conditioning', 'long_session'}
    if (type(c) is not dict or not required<=set(c) or set(c)-required-optional
            or c['schema']!=SCHEMA):
        raise ValueError('exact native streaming manifest required')
    _id(c['id']);_id(c['logical_source_id']);_sha(c['profile_sha256'])
    long_session=c.get('long_session', False)
    if type(long_session) is not bool:raise ValueError('explicit long_session selection required')
    _int(c['max_windows'],1,2542 if long_session else 144);_int(c['sequence'],0,c['max_windows']-1)
    _int(c['logical_index'],0,2541)
    before=0 if c['sequence']==0 else 124+(c['sequence']-1)*102
    if (type(c['committed_before']) is not int or c['committed_before']!=before
            or c['role'] not in ('listening','speech')
            or type(c['source_final']) is not bool or type(c['conversation_final']) is not bool
            or (c['conversation_final'] and c['role']!='listening')
            or m.get('source_index')!=0 or m['index']!=0 or m['total_frames']!=124
            or m['expected_windows']!=1 or m['final'] is not True
            or m.get('candidate',{}).get('audio_context')!='window-context-canonical-v1'):
        raise ValueError('bounded immutable native window contract drift')
    if 'visual_conditioning' in c:
        from .conversation_conditioning import validate_visual_record
        record=validate_visual_record(c['visual_conditioning'])
        if record['role']!=c['role']:raise ValueError('visual conditioning role drift')
    if 'previous_speech_interrupted' in c:
        marker=c['previous_speech_interrupted']
        if (type(marker) is not dict
                or set(marker)!={'schema','source_id','committed_frame'}
                or type(marker['schema']) is not str
                or marker['schema']!='h3-speech-interruption-v1'
                or type(marker['committed_frame']) is not int
                or marker['committed_frame']!=c['committed_before']
                or c['sequence']==0 or c['logical_index']!=0):
            raise ValueError('exact previous speech interruption boundary required')
        _id(marker['source_id'])
        if marker['source_id']==c['logical_source_id']:
            raise ValueError('interruption requires a new logical source')
    if c['role']=='speech':
        offer=restore_offer(c['audio_offer'])
        window=Window(before-22,before+102,22)
        if (offer.status!='ready' or offer.window!=window or offer.index!=c['logical_index']
                or offer.binding.conversation_id!=c['id']
                or offer.binding.source_id!=c['logical_source_id']
                or offer.binding.voice_binding_sha256!=c['profile_sha256']
                or offer.binding.sample_rate!=m['source_sample_rate']
                or offer.required_source_samples!=m['source_samples']
                or c['source_final']!=offer.final_source_window):
            raise ValueError('speech window/PCM offer drift')
    elif c['audio_offer'] is not None:
        raise ValueError('listening uses explicit known silence, not missing speech')
    return c


def _encode(stage,batch,args,m,state):
    import wave
    import torch
    from fastvideo.distributed import get_local_torch_device
    from fastvideo.models import pinned_offload
    from fastvideo.pipelines.basic.minimax_h3.reference import (
        prepare_reference_waveform,decode_reference_audio)
    from .conversation_audio import audio_origin,prepare_waveform,TorchWaveOps
    from .streaming_waveform import prepare_native_window
    import inspect
    c=validate_manifest(m)
    history=getattr(stage,'_h3_conversation_audio_history',None)
    if c['role']=='speech':
        offer=restore_offer(c['audio_offer'])
        with wave.open(m['wav'],'rb') as wav:
            if (wav.getsampwidth(),wav.getnchannels(),wav.getframerate(),wav.getnframes(),wav.getcomptype())!=(
                    2,offer.binding.channels,offer.binding.sample_rate,offer.required_source_samples,'NONE'):
                raise ValueError('immutable prefix WAV metadata drift')
            pcm=wav.readframes(wav.getnframes())
        ingress=PCMIngress(offer.binding,max_bytes=16*1024*1024,max_samples=8_000_000)
        for index,start in enumerate(range(0,len(pcm),65536)):
            ingress.append(PCMChunk(offer.binding,index,start//offer.binding.stride,pcm[start:start+65536]))
        if offer.eos:
            import hashlib
            ingress.seal(total_samples=offer.required_source_samples,sha256=hashlib.sha256(pcm).hexdigest())
        actual=inspect_audio_window(ingress,onset_frame=offer.onset_frame,index=offer.index,context=offer.policy)
        if actual!=offer:
            raise ValueError('worker prefix does not reproduce parent readiness')
        prepared=prepare_native_window(ingress,offer,history,prepare_reference_waveform,
                                       reference_path=inspect.getfile(prepare_reference_waveform))
        waveform,next_history,padding=prepared.waveform,prepared.history,prepared.padding_native_samples
        ingress.cancel()
    else:
        waveform,rate=decode_reference_audio(m['wav'])
        if bool(torch.count_nonzero(waveform)):
            raise ValueError('listening source is not explicit silence')
        waveform=prepare_reference_waveform(waveform,rate,32000,100)
        waveform,next_history,padding=prepare_waveform(waveform,
            audio_origin(c['committed_before'],124),history,TorchWaveOps())
    vae=stage.audio_vae
    if (int(vae.hop_length),int(vae.sampling_rate),int(vae.latent_channels))!=(800,32000,32):
        raise ValueError('native audio VAE geometry required')
    device=get_local_torch_device()
    loaded=any(p.device!=device for p in vae.parameters())
    if loaded:pinned_offload.load(vae,device,pin=bool(getattr(args,'pin_cpu_memory',False)))
    try:
        with torch.no_grad():
            canonical=vae.normalize_latents(vae.encode(waveform.to(device)[:,None]).latent_dist.mode().float()).transpose(1,2)
            canonical=canonical.detach().cpu().clone().contiguous()
    finally:
        if loaded and bool(getattr(args,'vae_cpu_offload',False)):pinned_offload.unload(vae)
    if tuple(canonical.shape)!=(2,207,32) or not bool(torch.isfinite(canonical).all()):
        raise ValueError('native encoded immutable window drift')
    return canonical,next_history,dict(padded_samples=padding,
        waveform_samples=165600,window_context_encodes=1,
        prefix_pcm_sha256=None if c['audio_offer'] is None else c['audio_offer']['prefix_sha256'],
        received_source_samples=m['source_samples'],source_eos=(c['role']=='listening' or c['audio_offer']['eos']))


def append_conditions(stage,batch,args,references,video_rows,audio_rows,m):
    from . import fast_continuation_stages as legacy
    from fastvideo.distributed import get_local_torch_device
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
    c=validate_manifest(m);device=get_local_torch_device()
    if 'visual_conditioning' in c:
        from .conversation_conditioning import current_record, visual_enabled
        if not visual_enabled() or current_record()!=c['visual_conditioning']:
            raise ValueError('native visual conditioning binding drift')
    state=getattr(stage,'_h3_conversation_state',None)
    def prepare():
        nonlocal state
        if m.get('cancelled') or getattr(stage,'interrupt',False):
            raise ValueError('cancelled streaming preparation')
        import os
        from .idle_anchor_policy import ENVIRONMENT_VARIABLE,parse_idle_anchor_policy
        idle_anchor_interval=parse_idle_anchor_policy(os.environ.get(ENVIRONMENT_VARIABLE))
        from .visual_ancestry_policy import ENV, COMPARATOR, CANDIDATE, parse_visual_ancestry_policy
        ancestry_policy=parse_visual_ancestry_policy(m,os.environ.get(ENV))
        if ancestry_policy is not None and idle_anchor_interval is not None:
            raise ValueError('exclusive visual ancestry policy required')
        height,width=legacy._geometry(batch)
        if (not getattr(stage,'ref2va',False) or not references or audio_rows
                or len(references)!=len(video_rows) or legacy.KEY in batch.extra
                or any(r.media_type!='image' or r.has_audio or r.num_audio_latents or r.num_latent_frames!=1 for r in references)):
            raise ValueError('original fixed image references required')
        legacy._source_hash(m)
        signature=tuple((r.num_latent_frames,r.latent_height,r.latent_width,legacy.TorchOps().digest(v))
                        for r,v in zip(references,video_rows))
        binding=(c['id'],c['profile_sha256'],height,width,signature,c['max_windows'])
        owner_policy={}
        if idle_anchor_interval is not None:
            binding=binding+(idle_anchor_interval,)
            owner_policy['idle_anchor_interval']=idle_anchor_interval
        if ancestry_policy is not None:
            binding=binding+(ancestry_policy,)
            if ancestry_policy==COMPARATOR:
                owner_policy['idle_anchor_interval']=3
            elif ancestry_policy==CANDIDATE:
                owner_policy['visual_ancestry_policy']=ancestry_policy
        if state is None:
            if c['sequence']!=0:raise ValueError('first streaming window required')
            state=dict(owner=StreamingWindowOwner(c['id'],profile=c['profile_sha256'],height=height,width=width,
                enabled=True,model_healthy=getattr(stage,'_h3_conversation_model_healthy',None),max_windows=c['max_windows'],
                long_session=c.get('long_session', False),**owner_policy),
                binding=binding,epoch=-1,completed=True,ancestry_policy=ancestry_policy)
            stage._h3_conversation_state=state
        owner=state['owner']
        if (state.get('failed') or state['binding']!=binding or c['sequence']!=owner._sequence
                or owner.committedF!=c['committed_before'] or owner.active_native is not None
                or owner.emittedF!=max(0,owner.committedF-5)):
            raise ValueError('streaming owner/global clock drift')
        return owner,height,width
    owner,height,width=common(device,prepare)
    try:
        canonical,history,info=common(device,lambda:_encode(stage,batch,args,m,state))
        if owner._idle_anchor_interval is not None:
            info['idle_anchor_interval']=owner._idle_anchor_interval
        def allocate():
            if 'previous_speech_interrupted' in c:
                marker=c['previous_speech_interrupted']
                owner.interrupt_source(marker['source_id'],marker['committed_frame'])
            owner.start_window(m['session_id'],canonical_audio=canonical,
                logical_source_id=c['logical_source_id'],logical_index=c['logical_index'],role=c['role'],
                source_final=c['source_final'],profile=c['profile_sha256'],height=height,width=width)
            token=owner.begin_window()
            if state.get('ancestry_policy') is not None:
                evidence=owner.visual_ancestry_evidence
                if state['ancestry_policy']=='epoch-idle-every3-v1':
                    selected=token.model.guide_video
                    anchored=(c['role']=='listening' and c['logical_index']>=3
                              and c['logical_index']%3==0)
                    evidence=dict(policy=state['ancestry_policy'],sequence=c['sequence'],
                        kind='unconditioned' if selected is None else ('anchor' if anchored else 'latest'),
                        anchor_sha256=owner._idle_anchor_digest,
                        selected_sha256=None if selected is None else owner.ops.digest(selected),
                        anchor_sequence=(c['sequence']-c['logical_index']+1
                                         if owner._idle_anchor is not None else None))
                info['visual_ancestry_evidence']=evidence
            state.update(epoch=c['sequence'],info=info,completed=False)
            context=dict(manifest=dict(m),owner=SourceFacade(owner,token),conversation_owner=owner,
                         global_token=token,token=token.model,state=state,
                         forwards=0,steps_held=0,pending_forward=False)
            vr,ar,refs=list(video_rows),list(audio_rows),list(references)
            if token.model.guide_video is not None:
                t=token.model
                refs.append(MiniMaxH3PreparedReference(media_type='video',has_audio=True,
                    num_latent_frames=t.window.guide_video_t,latent_height=height,latent_width=width,
                    num_audio_latents=t.window.guide_audio_t))
                vr.append(t.guide_video.clone());ar.append(t.guide_audio.clone().reshape(-1,32))
            return context,vr,ar,refs
        context,vr,ar,refs=common(device,allocate)
        stage._h3_conversation_audio_history=history
        batch.extra[legacy.KEY]=context;references[:]=refs;batch.references=refs
        return vr,ar
    except Exception:
        state['failed']=True
        raise


def receipt(batch):
    import json,os,torch
    from pathlib import Path
    from . import fast_continuation_stages as legacy
    legacy._receipt(batch,committed=True)
    ctx=batch.extra[legacy.KEY];owner=ctx['conversation_owner'];c=ctx['manifest']['streaming_audio']
    row=dict(schema=SCHEMA,streaming=c,committed_frame=owner.committedF,
             native_window_id=ctx['global_token'].source_id,canonical_sha256=owner.canonical_digest,
             guide_digests=owner._digests,forwards=ctx['forwards'],steps_held=ctx['steps_held'],**ctx['state']['info'])
    path=Path(os.environ['H3_REF2VA_COVERAGE_DIR'])/f'streaming-rank-{torch.distributed.get_rank()}-window-{c["sequence"]}.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(row,f,sort_keys=True)
