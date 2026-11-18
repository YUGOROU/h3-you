"""Finite matched global/local RoPE comparison, one resident model, no devices."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SECTION=dict(schema='h3-late-listening-diagnostic-v1',policies=['global-v1','window-local-v1'],
             phases=2,expected_frames=634,max_windows=8,decode_parity=True)


def validate_section(spec):
    value=spec.get('late_listening_diagnostic')
    if (value!=SECTION or type(value.get('phases')) is not int
            or value.get('decode_parity') is not True):
        raise ValueError('exact finite matched listening diagnostic required')


def config(phase):
    if type(phase) is not int or phase not in (0,1):raise ValueError('finite phase0/1 only')
    return dict(schema=SECTION['schema'],phase=phase,position_policy=SECTION['policies'][phase],
                conversation_id=('late-listening-global','late-listening-local')[phase],
                expected_frames=634,max_windows=8)


def layout_preflight():
    """Existing pinned packing, CPU positions only; no model mutation."""
    import torch
    from fastvideo.pipelines.basic.minimax_h3 import packing,reference
    from research.h3_personal.fast_continuation import build_layout
    from research.h3_personal.continuation_plan import Window
    images=[reference.MiniMaxH3PreparedReference('image',latent_height=2,latent_width=4)]
    kwargs=dict(text_tags=torch.tensor([1,0],dtype=torch.long),images=images,
                window=Window(510,634,22),height=2,width=2,ranks=1)
    a=build_layout(packing,reference,**kwargs)
    b=build_layout(packing,reference,**kwargs,temporal_policy='window-local-v1')
    diff=a.position_ids-b.position_ids
    expected=torch.zeros_like(diff)
    expected[a.video_indices[2:],0]=850.
    expected[a.audio_indices,0]=850.
    error=float((diff-expected).abs().max())
    tolerance=2*torch.finfo(diff.dtype).eps*float(a.position_ids.abs().max())
    if not torch.allclose(diff,expected,atol=tolerance,rtol=0):
        raise ValueError(f'experimental layout arithmetic drift: max={error}, tolerance={tolerance}')
    if not torch.equal(a.video_indices,b.video_indices) or not torch.equal(a.audio_indices,b.audio_indices):
        raise ValueError('experimental layout changed packing')
    return dict(translation_within_roundoff=True,max_abs_error=error,tolerance=tolerance,
                original_image_text_positions_unchanged=True,offset=850.)


def run(runtime,*,root,pcm,profile,context,api,deployment,remaining,start,startup):
    from huggingface_hub import CommitOperationAdd
    from research.h3_personal.live_voice_runtime import _json
    from research.h3_personal.streaming_trial import run_trial
    from research.h3_personal.conversation_control import SCHEMA as CONTROL
    from research.hf_h3_streaming_trial import read_evidence
    from research.h3_personal.late_listening_diagnostic import ENV
    import numpy as np
    root=Path(root);root.mkdir(mode=0o700,exist_ok=False)
    preflight=layout_preflight();reports=[];operations=[]
    for phase in (0,1):
        c=config(phase);_json(Path(os.environ[ENV]),c);last_role=None
        def backend(manifest):
            nonlocal last_role
            remaining();role=manifest['streaming_audio']['role']
            if role!=last_role:
                _json(runtime.continuation_file,dict(mode='independent'))
                runtime._switch_role(role);last_role=role
            _json(runtime.continuation_file,manifest)
            result,interval=runtime._generate(continuation=True)
            frames=np.stack([np.asarray(f).copy() for f in result.frames])
            if frames.ndim!=4 or frames.shape[1:]!=(448,256,3):raise ValueError('native diagnostic geometry drift')
            exported=frames[:,:,2:-2,:].copy(order='C')
            return dict(raw=exported.tobytes(),count=int(exported.shape[0]),interval=interval,
                        **read_evidence(runtime.coverage,manifest))
        begin=time.monotonic()
        with runtime._lock:
            trial=run_trial(root=root/f'phase-{phase}',speech_pcm=pcm,profile_sha256=profile.digest,
                context=context,backend=backend,timeout=int(min(300,remaining()-60)),paced=True,
                conversation_id=c['conversation_id'],native_window_prefix=f'late_phase_{phase}_window')
        wall=time.monotonic()-begin
        receipts=[]
        for rank in (0,1):
            p=runtime.coverage/f'late-listening-{phase}-rank-{rank}.json'
            if p.is_symlink() or p.stat().st_size>1<<20:raise ValueError('bounded diagnostic evidence required')
            r=json.loads(p.read_bytes())
            if (any(r.get(k)!=v for k,v in c.items()) or r['rank']!=rank or r['completed'] is not True
                    or r['latent_count']!=187 or r['committed_frames']!=634 or len(r['windows'])!=6):
                raise ValueError('diagnostic rank binding drift')
            if rank==0 and (r['frame_count']!=634 or len(r['per_frame_max'])!=634):
                raise ValueError('full native parity required')
            receipts.append(r)
        review=root/f'phase-{phase}.mp4'
        subprocess.run(['ffmpeg','-v','error','-xerror','-nostdin','-n','-i',trial['output'],
            '-map','0:v:0','-map','0:a:0','-vsync','0','-c:v','libx264','-preset','veryfast','-crf','18',
            '-pix_fmt','yuv420p','-c:a','alac','-movflags','+faststart',str(review)],check=True,
            timeout=int(min(120,remaining())))
        audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(review),'-map','0:a','-f','s16le','pipe:1'],
                                      timeout=int(min(120,remaining())))
        if len(audio)!=2536000 or audio[904000:2059840]!=pcm or any(audio[:904000]) or any(audio[2059840:]):
            raise ValueError('diagnostic PCM drift')
        trial.pop('output');trial['admission']['native_runtime_connected']=True
        trial.update(diagnostic=c,native_receipts=receipts,wall_seconds=wall,review_pcm_exact=True,
                     review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),quality_accepted=False)
        reports.append(trial)
        data=(json.dumps(trial,sort_keys=True,allow_nan=False)+'\n').encode()
        if len(data)>1<<20 or not 0<review.stat().st_size<512*1024*1024:raise ValueError('diagnostic output bound')
        operations.extend([CommitOperationAdd(path_in_repo=f'late-listening/phase-{phase}.json',path_or_fileobj=data),
                           CommitOperationAdd(path_in_repo=f'late-listening/phase-{phase}.mp4',path_or_fileobj=review.read_bytes())])
        if phase==0:
            # Retire only a fully final-ACKed conversation, never reload model.
            _json(runtime.continuation_file,dict(conversation_control=dict(schema=CONTROL,action='retire',
                conversation_id=c['conversation_id'],reason='completed')))
            runtime._generate(continuation=True)
            _json(runtime.continuation_file,dict(mode='independent'))
            archive=root/'phase-0-private-receipts';archive.mkdir(mode=0o700)
            for seq in range(6):
                for rank in (0,1):
                    p=runtime.coverage/f'streaming-rank-{rank}-window-{seq}.json'
                    if p.is_symlink():raise ValueError('receipt symlink')
                    p.rename(archive/p.name)
    summary=dict(schema=SECTION['schema'],completed=True,phases=2,one_runtime_start=True,
        startup_seconds=startup,total_seconds=time.monotonic()-start,layout_preflight=preflight,
        profile=profile.name,new_live_input=False,tts_streaming=False,device_playback_verified=False,
        realtime_complete=False,quality_accepted=False,
        native_parity=[dict(phase=i,exact=r['native_receipts'][0]['exact'],
            max=r['native_receipts'][0]['max'],mean=r['native_receipts'][0]['mean']) for i,r in enumerate(reports)])
    operations.append(CommitOperationAdd(path_in_repo='late-listening/summary.json',
        path_or_fileobj=(json.dumps(summary,sort_keys=True)+'\n').encode()))
    remaining();api.create_commit(deployment,repo_type='dataset',operations=operations,
        commit_message='Private matched late-listening diagnostic; experimental local RoPE, unchanged defaults')
    return summary
