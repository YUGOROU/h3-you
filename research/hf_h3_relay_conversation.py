"""Actual resident HF H3+native Irodori -> authenticated paired-AV relay.

Fresh typed/Pi input only in RAM. No rendered MP4/prompt/WAV Hub publication.
READY stdout is scalar binding metadata only, not remote reception evidence.
"""
import json
import os
from pathlib import Path
import re
import time


def validate_spec(spec):
    row=spec.get('live_relay_session')
    if (type(row) is not dict or set(row)!={'schema','session','space_id','revision','not_after','max_windows','max_frames'}
            or row['schema']!='h3-live-relay-session-v1'
            or not re.fullmatch('[a-f0-9]{32}',row['session'])
            or not re.fullmatch(r'YUGOROU/h3-live-relay-[a-z0-9-]{8,64}',row['space_id'])
            or not re.fullmatch('[a-f0-9]{40}',row['revision'])
            or type(row['not_after']) not in (int,float)
            or not 0<row['not_after']-time.time()<=1800
            or row['max_windows']!=16 or row['max_frames']!=1654
            or spec.get('locked_speech') is not None
            or spec.get('streaming_trial') is not None or spec.get('conversation_scenario') is not None
            or spec.get('live_voice_session',{}).get('max_turns')!=2):
        raise ValueError('fresh immutable finite two-turn relay session required')
    return row


def run_session(root,paths,spec,deployment,tts_future,deadline):
    from huggingface_hub import HfApi
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha,role_profile_digest
    from research.h3_personal.ref2va_runtime import sha
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.streaming_windows import ResamplingContext,NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256
    from research.h3_personal.live_relay_endpoint import attest
    from research.h3_personal.live_relay_client import SocketAdapter
    from research.h3_personal.live_relay_worker import run_worker
    from research.h3_personal.paired_av import LiveAVBinding
    from research.hf_h3_live_voice import close_tts_future
    row=validate_spec(spec);root=Path(root)
    if deadline-time.monotonic()<450:raise TimeoutError('relay lease cannot cover resident startup and conversation')
    api=HfApi(endpoint='https://huggingface.co',token=os.environ['HF_TOKEN'])
    permit=attest(api,space_id=row['space_id'],revision=row['revision'])
    reference=paths['reference']/'production-inputs/references/secondary.jpeg'
    profile=choose_profile(ordered_source_sha256=(sha(reference),),prompt_sha256=prompt_sha('speech'))
    if profile.name!='reference1024':raise ValueError('reference1024 required; no quality fallback')
    identity=paths['personal']/spec['identity_prefix'];output=root.with_name(root.name+'-live-relay')
    runtime=LiveVoiceRuntime(model=paths['base'],model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors',identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',reference=reference,
        archive=root/'runtime/FastVideo.tar.gz',output=output,continuation=True,quality_profile=profile,conversation_roles=True)
    adapter=None;tts=None
    try:
        runtime.start()
        # Rank workers remain offline; only the owner uses the approved relay.
        os.environ['HF_HUB_OFFLINE']='0'
        tts=tts_future.result(timeout=max(.001,deadline-time.monotonic()));tts.read('tts_ready',deadline)
        if min(deadline-time.monotonic(),row['not_after']-time.time())<120:
            raise TimeoutError('insufficient live session time after native preparation')
        binding=LiveAVBinding(row['session'],role_profile_digest(runtime.quality_profile,'speech'),1,
                              max_segments=row['max_windows'],max_total_frames=row['max_frames'])
        adapter=SocketAdapter(permit.base_url,session=row['session'],not_after=row['not_after'],role='producer',
            role_key=os.environ['H3_RELAY_PRODUCER_KEY'],hf_token=os.environ['HF_TOKEN'],endpoint_permit=permit)
        adapter.open_claim()
        ready=dict(type='h3_relay_ready',status='ready',session=binding.session,profile_sha256=binding.profile_sha256,
                   epoch=binding.epoch,max_segments=binding.max_segments,max_total_frames=binding.max_total_frames,
                   expires_at=min(row['not_after'],time.time()+deadline-time.monotonic()),
                   origin='fresh_typed_text',asr_used=False,tts_streaming=False)
        print(json.dumps(ready),flush=True)
        context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
        result=run_worker(runtime,tts,adapter,binding=binding,context=context,root=output/'conversation',deadline=deadline)
        print(json.dumps(dict(type='h3_relay_finished',acked=result['acked'],turns=result['turns'],
                              discord_reception_verified=False)),flush=True)
        return result
    finally:
        if adapter is not None:adapter.close()
        try:runtime.close()
        finally:close_tts_future(tts_future,tts)
