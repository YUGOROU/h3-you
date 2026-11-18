"""One finite real native streaming-window trial; prerecorded producer, no devices."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import time
import wave

SCHEMA='h3-streaming-native-trial-v1'
SECTION=dict(schema=SCHEMA,profile='reference1024',expected_frames=634,max_windows=8,
             producer='prerecorded-prefix-chunks',paced=True)


def validate_spec(spec):
    from research.hf_h3_conversation_scenario import validate_spec as previous
    previous(spec)
    if (spec.get('streaming_trial')!=SECTION or spec['streaming_trial'].get('paced') is not True):
        raise ValueError('exact finite streaming trial spec required')
    if spec.get('late_listening_diagnostic') is not None:
        from research.hf_h3_late_listening_diagnostic import validate_section
        validate_section(spec)
    return spec['streaming_trial']


def validate_ordered_rank_evidence(report, context, *, expected_prompt_sha256=None, manifest=None):
    """Bind each native rank's actual geometry to candidate, not old acceptance."""
    from research.h3_personal.conditioning_experiment import read_accepted_profile,ordered_admission_bounds
    from research.h3_personal.candidate_references import build_candidate_binding,validate_candidate_binding
    from research.h3_personal.ordered_references import expected_prepared_sizes,ORIENTATION_POLICY,reference_roles
    from research.h3_personal.ordered_qwen_tokens import validate_proof,ordered_token_policy
    profile=read_accepted_profile()
    if profile is None:raise ValueError('candidate requires accepted BASE settings')
    binding=build_candidate_binding(context['ordered_references'],profile)
    if (report.get('ordered_reference_sha256')!=context['identity_sha256']
            or report.get('candidate_visual_acceptance')!='unreviewed'
            or report.get('candidate_binding_sha256')!=binding['binding_sha256']
            or validate_candidate_binding(report.get('candidate_reference_binding'))!=binding):
        raise ValueError('native candidate/base settings binding mismatch')
    geometry=report.get('ordered_geometry')
    image_count=len(reference_roles(context['ordered_references']['schema']))
    bounds=ordered_admission_bounds(profile.short_edge,references=image_count)
    if (type(geometry) is not dict or geometry.get('schema')!='h3-ordered-native-geometry-v1'
            or geometry.get('identity_sha256')!=context['identity_sha256']
            or geometry.get('candidate_visual_acceptance')!='unreviewed'
            or geometry.get('admission_kind')!='candidate-engineering-not-native-capability'
            or any(type(geometry.get(k)) is not int or geometry[k]!=v for k,v in bounds.items())
            or type(geometry.get('images')) is not list or len(geometry['images'])!=image_count):
        raise ValueError('bound actual schema-selected native image geometry required')
    expected=expected_prepared_sizes(context['ordered_references'],short_edge=profile.short_edge)['references']
    segments=geometry.get('native_layout_reference_segments')
    if (geometry.get('conditioning_and_vae_rows')!='independent-native-quantities'
            or geometry.get('reference_rows_source')!='native-layout-image-segments'
            or type(segments) is not list or len(segments)<image_count
            or type(geometry.get('guide_segments_excluded')) is not int
            or geometry['guide_segments_excluded']!=len(segments)-image_count):
        raise ValueError('native layout image segment provenance required')
    for index,segment in enumerate(segments):
        if (type(segment) is not dict or set(segment)!={'media_type','rows','latent_shape'}
                or segment['media_type'] not in (('image',) if index<image_count else ('audio','video'))
                or type(segment['rows']) is not int or segment['rows']<=0
                or type(segment['latent_shape']) is not list or len(segment['latent_shape'])!=3
                or any(type(n) is not int or n<0 for n in segment['latent_shape'])):
            raise ValueError('ordered native image/guide layout segment mismatch')
    total=0
    for index,(image,source,size) in enumerate(zip(geometry['images'],context['ordered_references']['references'],expected)):
        if type(image) is not dict:raise ValueError('per-image geometry required')
        exact=dict(role=source['role'],source_sha256=source['sha256'],
            original_size=[source['width'],source['height']],original_aspect=[source['width'],source['height']],
            stored_size=[source['width'],source['height']],oriented_size=[source['width'],source['height']],
            orientation_policy=ORIENTATION_POLICY,
            prepared_size=[size['width'],size['height']],resize_policy=profile.resize_policy,crop='none')
        if (type(image.get('exif_present')) is not bool
                or image.get('exif_orientation') is not None and type(image['exif_orientation']) is not int
                or image.get('exif_orientation') not in (None,1)
                or image.get('exif_present') is False and image.get('exif_orientation') is not None):
            raise ValueError('identity-only source orientation evidence required')
        if any(json.dumps(image.get(k),sort_keys=True)!=json.dumps(v,sort_keys=True) for k,v in exact.items()):
            raise ValueError('native ordered image source/geometry/crop mismatch')
        dims=[image.get(k) for k in ('latent_frames','latent_height','latent_width')]
        if any(type(v) is not int or v<=0 for v in dims) or dims[0]!=1 or dims[1]%2 or dims[2]%2:
            raise ValueError('actual native image latent geometry required')
        rows=dims[0]*(dims[1]//2)*(dims[2]//2)
        segment=segments[index]
        if (type(image.get('latent_rows')) is not int or image['latent_rows']!=rows
                or segment['rows']!=rows or segment['latent_shape']!=dims):
            raise ValueError('actual image patch/layout rows mismatch')
        total+=rows
    if (type(geometry.get('R')) is not int or geometry['R']!=total
            or not 0<total<=bounds['max_reference_rows']
            or type(geometry.get('T')) is not int or not 0<geometry['T']<=bounds['max_tokens']
            or type(report.get('R')) is not int or report['R']!=geometry['R']
            or type(report.get('T')) is not int or report['T']!=geometry['T']):
        raise ValueError('actual candidate R/T/admission mismatch')
    prompt=expected_prompt_sha256 or report.get('role_proof',{}).get('prompt_sha256')
    qwen=validate_proof(report.get('qwen_tokens'),context,profile.short_edge,prompt,expected_T=geometry['T'])
    packed=geometry.get('packed_rows');guide=geometry.get('guide_layout');policy=ordered_token_policy(profile.short_edge)
    if (type(packed) is not dict or packed.get('schema')!='h3-ordered-packed-rows-v1'
            or packed.get('admission_kind')!='candidate-engineering-not-native-capability'
            or packed.get('max_packed_rows')!=policy['max_packed_rows']
            or any(type(packed.get(k)) is not int or packed[k]<0 for k in ('max_packed_rows','sequence_length',
                'text_rows','reference_rows','target_video_rows','target_audio_rows','video_index_rows','audio_index_rows'))):
        raise ValueError('actual packed row admission evidence required')
    vr=sum(s['rows'] for s in segments if s['media_type']!='audio');ar=sum(s['rows'] for s in segments if s['media_type']=='audio')
    if (packed['text_rows']!=geometry['T'] or packed['reference_rows']!=vr+ar
            or packed['video_index_rows']!=vr+packed['target_video_rows']
            or packed['audio_index_rows']!=ar+packed['target_audio_rows']
            or packed['sequence_length']!=geometry['T']+vr+ar+packed['target_video_rows']+packed['target_audio_rows']
            or not 0<packed['sequence_length']<=policy['max_packed_rows']):
        raise ValueError('actual packed row decomposition mismatch')
    if (type(guide) is not dict or guide.get('image_rows')!=total
            or guide.get('guide_video_rows')!=vr-total or guide.get('guide_audio_rows')!=ar
            or any(type(guide.get(k)) is not int for k in ('image_rows','guide_video_rows','guide_audio_rows'))
            or type(guide.get('provenance')) is not dict):
        raise ValueError('actual image/guide row separation required')
    provenance=guide['provenance']
    if len(segments)==image_count:
        if vr!=total or ar or provenance.get('guide_video_frames')!=0 or provenance.get('guide_audio_latents')!=0:
            raise ValueError('unexpected guide coverage')
    elif len(segments)!=image_count+2 or [s['media_type'] for s in segments[image_count:]]!=['audio','video']:
        raise ValueError('exact native audio/video guide suffix required')
    else:
        shape=provenance.get('target_latent_shape')
        if (type(shape) is not list or len(shape)!=5 or any(type(n) is not int or n<=0 for n in shape)
                or segments[image_count+1]['latent_shape']!=[provenance.get('guide_video_frames'),*shape[3:]]
                or segments[image_count+1]['rows']!=provenance['guide_video_frames']*(shape[3]//2)*(shape[4]//2)
                or segments[image_count]['rows']!=2*provenance.get('guide_audio_latents',-1)):
            raise ValueError('actual guide latent coverage/canvas mismatch')
    if manifest is not None:
        c=manifest['streaming_audio']
        if (provenance.get('kind')!='continuation-owner-window'
                or provenance.get('owner_session_id')!=manifest['session_id']
                or type(provenance.get('window_index')) is not int or provenance['window_index']!=c['sequence']):
            raise ValueError('candidate guide owner/window manifest mismatch')
    return dict(identity_sha256=context['identity_sha256'],candidate_binding_sha256=binding['binding_sha256'],
                visual_acceptance='unreviewed',geometry=geometry,qwen_tokens=qwen)


def read_evidence(coverage,manifest):
    from research.h3_personal.streaming_native_stages import validate_manifest
    c=validate_manifest(manifest);coverage=Path(coverage);rows=[];encoded=[];refresh=[]
    from research.h3_personal.conversation_conditioning import read_ordered_context
    ordered_context=read_ordered_context()
    ordered_proofs=[];ordered_memory=[]
    visual=c.get('visual_conditioning')
    for rank in (0,1):
        path=coverage/f'streaming-rank-{rank}-window-{c["sequence"]}.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size>1<<20:
            raise ValueError('bounded native rank evidence required')
        raw=path.read_bytes();row=json.loads(raw)
        decoded=json.loads((coverage/f'h3-continuation-decode-rank-{rank}-window-0.json').read_text())
        conditioning_path=coverage/f'conditioning-experiment-{manifest["session_id"]}-window-0-rank-{rank}.json'
        if ordered_context is not None and (conditioning_path.is_symlink() or not conditioning_path.is_file()
                or conditioning_path.stat().st_size>1<<20):
            raise ValueError('bounded ordered native rank conditioning evidence required')
        conditioning_raw=conditioning_path.read_bytes()
        conditioning=json.loads(conditioning_raw)
        if ordered_context is None:
            if any(k in conditioning for k in ('ordered_reference_sha256','ordered_geometry')):
                raise ValueError('unexpected ordered reference native evidence')
        else:
            from research.h3_personal.conversation_conditioning import conditioning_sha
            ordered_proofs.append(validate_ordered_rank_evidence(conditioning,ordered_context,
                expected_prompt_sha256=visual['prompt_sha256'] if visual is not None else conditioning_sha(c['role']),
                manifest=manifest))
            memory=conditioning.get('candidate_memory')
            if (type(memory) is not dict or set(memory)!={'measurement','allocated_bytes','reserved_bytes'}
                    or memory['measurement']!='worker-current-all-models-not-reference-delta'
                    or any(type(memory[k]) is not int or memory[k]<0 for k in ('allocated_bytes','reserved_bytes'))
                    or memory['allocated_bytes']>memory['reserved_bytes']):
                raise ValueError('candidate worker memory snapshot required')
            ordered_memory.append(memory)
            encoded.append(conditioning_raw)
        expected=c['committed_before']+(124 if c['sequence']==0 else 102)
        emitted=expected-(0 if c['conversation_final'] else 5)
        if (row['schema']!=c['schema'] or row['streaming']!=c or row['native_window_id']!=manifest['session_id']
                or row['committed_frame']!=expected or row['forwards']!=4 or row['steps_held']!=4
                or row['waveform_samples']!=165600 or row['window_context_encodes']!=1
                or row['received_source_samples']!=manifest['source_samples']
                or decoded['session_id']!=manifest['session_id'] or decoded['worker_rank']!=rank
                or decoded['committed_frames']!=expected or decoded['emitted_frames']!=emitted
                or decoded['final']!=c['conversation_final']
                or conditioning['current_role']!=c['role']
                or conditioning['streaming_sequence']!=c['sequence']
                or conditioning['parameter_identity_preserved'] is not True
                or conditioning['cache_hit'] is not True):
            raise ValueError('native rank commit/decode/conditioning evidence drift')
        if visual is not None:
            if (conditioning.get('role_proof', {}).get('prompt_sha256') != visual['prompt_sha256']
                    or conditioning.get('visual_conditioning') != {k:visual[k] for k in ('target_epoch','intent_epoch','prompt_sha256')}):
                raise ValueError('native visual prompt/epoch evidence drift')
            if conditioning.get('visual_encoder_retained') is not True:
                raise ValueError('native visual encoder retention not proved')
            item=conditioning.get('visual_condition_refresh')
            import math
            if (type(item) is not dict or set(item)!={'encoded','duration_seconds'}
                    or type(item['encoded']) is not bool
                    or type(item['duration_seconds']) not in (int,float)
                    or not math.isfinite(item['duration_seconds']) or item['duration_seconds']<0):
                raise ValueError('native visual encoding provenance required')
            refresh.append(item)
        from research.h3_personal.streaming_pcm import _sha
        _sha(row['canonical_sha256'])
        if (type(row['guide_digests']) is not list or len(row['guide_digests'])!=3):
            raise ValueError('rank guide evidence required')
        for digest in row['guide_digests']:
            if digest is not None:_sha(digest)
        if c['audio_offer'] is not None:
            if (row['prefix_pcm_sha256']!=c['audio_offer']['prefix_sha256']
                    or row['source_eos']!=c['audio_offer']['eos']
                    or row['padded_samples']!=c['audio_offer']['eos_padding_native_samples']):
                raise ValueError('rank PCM prefix/EOS binding drift')
        rows.append(row);encoded.append(raw)
    if ordered_context is not None and ordered_proofs[0]!=ordered_proofs[1]:
        raise ValueError('both-rank ordered reference geometry/token disagreement')
    if (rows[0]['canonical_sha256']!=rows[1]['canonical_sha256']
            or rows[0]['guide_digests']!=rows[1]['guide_digests']):
        raise ValueError('both-rank canonical/guide disagreement')
    return dict(committed_frame=expected,emitted_frame=emitted,
        receipt_sha256=hashlib.sha256(b'\0'.join(encoded)).hexdigest(),
        evidence=dict(canonical_sha256=rows[0]['canonical_sha256'],
            prefix_pcm_sha256=rows[0]['prefix_pcm_sha256'],source_eos=rows[0]['source_eos'],
            received_source_samples=rows[0]['received_source_samples'],
            padded_samples=rows[0]['padded_samples'],both_ranks=True,forwards=4,
            resident_parameter_identity=True,role_cache_hit=True,
            **({'visual_conditioning':{k:visual[k] for k in ('target_epoch','intent_epoch','prompt_sha256')}}
               if visual is not None else {}),
            **({'visual_condition_refresh':refresh, 'visual_encoder_retained':True} if visual is not None else {}),
            **({'ordered_references':ordered_proofs[0], 'ordered_both_ranks':True,
                'ordered_worker_memory':ordered_memory} if ordered_context is not None else {})))


def run_session(root,paths,spec,deployment,deadline):
    from huggingface_hub import HfApi,CommitOperationAdd
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha
    from research.h3_personal.ref2va_runtime import sha
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime,_json
    from research.h3_personal.streaming_windows import ResamplingContext,NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256
    from research.h3_personal.streaming_trial import run_trial
    validate_spec(spec);root=Path(root);paths={k:Path(v) for k,v in paths.items()}
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('streaming trial deadline')
        return value
    api=HfApi(token=os.environ['HF_TOKEN'])
    if not api.repo_info(deployment,repo_type='dataset').private:raise ValueError('private output required')
    speech=paths['speech']/spec['locked_speech']['path']
    if sha(speech)!=spec['locked_speech']['sha256']:raise ValueError('original source WAV drift')
    with wave.open(str(speech),'rb') as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())!=(1,2,48000,577920):
            raise ValueError('full original PCM binding required')
        pcm=w.readframes(577920)
    identity=paths['personal']/spec['identity_prefix']
    profile=choose_profile(ordered_source_sha256=(sha(paths['reference']/'production-inputs/references/secondary.jpeg'),),
                           prompt_sha256=prompt_sha('speech'))
    if profile.name!='reference1024':raise ValueError('no implicit quality fallback')
    output=root.with_name(root.name+'-streaming-generated')
    runtime=LiveVoiceRuntime(model=paths['base'],model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors',identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',reference=paths['reference']/'production-inputs/references/secondary.jpeg',
        archive=root/'runtime/FastVideo.tar.gz',output=output,continuation=True,quality_profile=profile,conversation_roles=True)
    diagnostic=spec.get('late_listening_diagnostic') is not None
    if diagnostic:
        from research.h3_personal.late_listening_diagnostic import ENV
        from research.hf_h3_late_listening_diagnostic import config
        os.environ[ENV]=str(root/'late-listening-current.json')
        _json(Path(os.environ[ENV]),config(0))
    start=time.monotonic();last_role=None
    try:
        remaining();runtime.start();startup=time.monotonic()-start
        if diagnostic:
            from research.hf_h3_late_listening_diagnostic import run
            context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
            return run(runtime,root=output/'late-listening',pcm=pcm,profile=profile,context=context,
                       api=api,deployment=deployment,remaining=remaining,start=start,startup=startup)
        def backend(manifest):
            nonlocal last_role
            remaining();role=manifest['streaming_audio']['role']
            if role!=last_role:
                _json(runtime.continuation_file,dict(mode='independent'))
                runtime._switch_role(role);last_role=role
            _json(runtime.continuation_file,manifest)
            result,interval=runtime._generate(continuation=True)
            import numpy as np
            frames=np.stack([np.asarray(f).copy() for f in result.frames])
            if frames.ndim!=4 or frames.shape[1:]!=(448,256,3):raise ValueError('native pixel geometry drift')
            exported=frames[:,:,2:-2,:].copy(order='C')
            evidence=read_evidence(runtime.coverage,manifest)
            return dict(raw=exported.tobytes(),count=int(exported.shape[0]),interval=interval,**evidence)
        context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        begin=time.monotonic()
        with runtime._lock:
            trial=run_trial(root=output/'trial',speech_pcm=pcm,profile_sha256=profile.digest,
                context=context,backend=backend,timeout=int(min(300,remaining()-60)),paced=True)
        wall=time.monotonic()-begin
        review=output/'review.mp4'
        subprocess.run(['ffmpeg','-v','error','-xerror','-nostdin','-n','-i',trial['output'],
            '-map','0:v:0','-map','0:a:0','-vsync','0','-c:v','libx264','-preset','veryfast','-crf','18',
            '-pix_fmt','yuv420p','-c:a','alac','-movflags','+faststart',str(review)],check=True,timeout=int(min(120,remaining())))
        audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(review),'-map','0:a:0','-f','s16le','pipe:1'],timeout=int(min(120,remaining())))
        if (len(audio)!=1268000*2 or audio[452000*2:1029920*2]!=pcm
                or any(audio[:452000*2]) or any(audio[1029920*2:])):
            raise ValueError('review original speech/silence mismatch')
        report=dict(trial)
        report.pop('output')
        report.update(completed=True,native_backend_connected=True,profile=profile.name,
            startup_seconds=startup,wall_seconds=wall,total_seconds=time.monotonic()-start,
            review_sha256=sha(review),review_pcm_exact=True,source_wav_sha256=sha(speech),
            lip_sync_verified=False,new_live_input=False,tts_streaming=False)
        report['admission']=dict(report['admission'],native_runtime_connected=True)
        data=(json.dumps(report,sort_keys=True)+'\n').encode();video=review.read_bytes()
        if len(data)>1<<20 or not 0<len(video)<=512*1024*1024:raise ValueError('bounded output required')
        remaining()
        api.create_commit(deployment,repo_type='dataset',operations=[
            CommitOperationAdd(path_in_repo='streaming/report.json',path_or_fileobj=data),
            CommitOperationAdd(path_in_repo='streaming/review.mp4',path_or_fileobj=video)],
            commit_message='Private finite native PCM-prefix inference; not live TTS/Discord')
        return report
    finally:
        runtime.close()
        if diagnostic:os.environ.pop(ENV,None)
