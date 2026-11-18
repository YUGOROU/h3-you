"""Finite prerecorded-PCM chunk trial; real H3 backend is injected by HF worker.

Not ASR/TTS streaming, network transport or Discord E2E. Native workers see only
immutable prefix WAVs. Producer and final AV verifier know the full test source;
that knowledge must not enter pre-EOS speech conditioning.
"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
import wave

from .streaming_pcm import PCMIngress,PCMStreamBinding,PCMChunk
from .streaming_admission import StreamingAdmission,NativeCommitEvidence
from .streaming_native_stages import SCHEMA,validate_manifest
from .conversation_timeline import FrozenPCMSource,ConversationBinding,TimelineSpan
from .conversation_av import ConversationAVStream
from .conversation_scenario import segment_timing


def write_wav(path,pcm,rate=48000):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:
        with wave.open(f,'wb') as w:
            w.setparams((1,2,rate,0,'NONE','not compressed'));w.writeframes(pcm)
    Path(path).chmod(0o400)


def manifest_for(ticket,*,conversation_id,profile_sha256,wav,max_windows=8,
                 native_window_prefix='stream_window',long_session=False):
    from .fast_continuation_stages import SCHEMA as LEGACY
    d=ticket.decision
    with wave.open(str(wav),'rb') as w:samples=w.getnframes();rate=w.getframerate()
    logical='idle-return' if d.conversation_final else 'idle-initial'
    logical_index=0 if d.conversation_final else ticket.sequence
    if d.audio is not None:
        logical=d.audio.binding.source_id;logical_index=d.audio.index
    c=dict(schema=SCHEMA,id=conversation_id,profile_sha256=profile_sha256,
           sequence=ticket.sequence,logical_source_id=logical,logical_index=logical_index,
           role=d.role,source_final=d.conversation_final if d.audio is None else d.audio.final_source_window,
           conversation_final=d.conversation_final,
           committed_before=0 if ticket.sequence==0 else d.window.start+22,
           max_windows=max_windows,audio_offer=None if d.audio is None else asdict(d.audio),
           long_session=bool(long_session))
    m=dict(schema=LEGACY,mode='latent-tail-v1',session_id=f'{native_window_prefix}_{ticket.sequence}',
           wav=str(wav),sha256=hashlib.sha256(Path(wav).read_bytes()).hexdigest(),
           source_samples=samples,source_sample_rate=rate,
           index=0,source_index=0,total_frames=124,window_frames=124,overlap_frames=22,
           seed=314159,expected_windows=1,final=True,
           candidate=dict(schema='h3-continuation-candidate-v1',audio_context='window-context-canonical-v1',
                          progress_receipts='bind-and-commit-v1'),streaming_audio=c)
    validate_manifest(m)
    return m


def run_trial(*,root,speech_pcm,profile_sha256,context,backend,timeout=180,paced=True,
              conversation_id='streaming-trial',native_window_prefix='stream_window'):
    from research.run_h3_realtime_first import encode_rgb_segment
    root=Path(root);root.mkdir(mode=0o700,exist_ok=False)
    if type(speech_pcm) is not bytes or len(speech_pcm)!=577920*2:
        raise ValueError('trial requires existing full577920sample PCM')
    control=StreamingAdmission(conversation_id,context=context,max_windows=8,max_idle_windows=4,
                               max_turns=1,timeout=timeout)
    binding=PCMStreamBinding(conversation_id,'speech',1,profile_sha256)
    ingress=PCMIngress(binding,timeout=timeout,max_bytes=16*1024*1024,max_samples=8_000_000)
    control.attach(ingress)
    source=FrozenPCMSource('speech',speech_pcm,48000,1,hashlib.sha256(speech_pcm).hexdigest())
    # Only the finite test AV verifier knows this schedule/source upfront.
    # Native input readiness uses actual ingress, not this transport binding.
    timeline=ConversationBinding(conversation_id,(source,),(
        TimelineSpan(0,226,'listening'),TimelineSpan(226,532,'speech','speech'),
        TimelineSpan(532,634,'listening')),48000,1)
    sink=ConversationAVStream(root,timeline,width=252,height=448,timeout=timeout,paced=paced)
    silence=root/'idle.wav';write_wav(silence,bytes(48000*2))
    rows=[];events=[];cursor=0
    def feed(stop,final=False):
        received=ingress.snapshot()['received_samples']
        while received<stop:
            end=min(received+32000,stop)
            ingress.append(PCMChunk(binding,ingress.snapshot()['chunks'],received,speech_pcm[received*2:end*2]))
            received=end
        if final:ingress.seal(total_samples=stop,sha256=hashlib.sha256(speech_pcm[:stop*2]).hexdigest())
        events.append(dict(action='producer-feed',received_samples=stop,eos=final,monotonic=time.monotonic()))
    try:
        for sequence in range(6):
            if sequence==3:feed(408128)
            if sequence==4:feed(577920,True)
            if sequence==5:
                control.finish_speech();control.request_stop()
            decision=control.inspect();ticket=control.reserve(decision)
            if sequence==1:
                # Actual PCM arrival occurs AFTER this idle reservation. Drain
                # it, then recompute speech onset from the committed boundary.
                feed(204128)
                if control.inspect().status!='busy':raise ValueError('idle reservation did not drain')
            if decision.audio is None:input_wav=silence
            else:
                from .streaming_waveform import read_original_prefix
                input_wav=root/f'prefix-{sequence}.wav'
                write_wav(input_wav,read_original_prefix(ingress,decision.audio))
            manifest=manifest_for(ticket,conversation_id=conversation_id,profile_sha256=profile_sha256,
                                  wav=input_wav,native_window_prefix=native_window_prefix)
            event=dict(action='native-start',sequence=sequence,role=decision.role,
                       received_samples=ingress.snapshot()['received_samples'],
                       eos=ingress.snapshot()['state']=='sealed',monotonic=time.monotonic())
            events.append(event)
            result=backend(manifest)
            expected_end=decision.window.end-(0 if decision.conversation_final else 5)
            count=expected_end-cursor
            if (result['count']!=count or len(result['raw'])!=count*252*448*3
                    or result['committed_frame']!=decision.window.end or result['emitted_frame']!=expected_end):
                raise ValueError('actual native commit/decode output differs')
            control.commit(ticket,NativeCommitEvidence(result['committed_frame'],
                None if decision.audio is None else decision.audio.prefix_sha256,result['receipt_sha256']))
            control.ack_decoded(ticket,result['emitted_frame'])
            segment=root/f'segment-{sequence:02d}.mp4'
            encode_rgb_segment(result['raw'],segment,segment_timing(sequence,cursor,expected_end),width=252,height=448)
            segment.chmod(0o400)
            sink.accept(dict(path=str(segment),index=sequence,frame_count=count,frame_start=cursor,
                frame_stop=expected_end,time_base_numerator=1,time_base_denominator=24,
                sha256=hashlib.sha256(segment.read_bytes()).hexdigest(),video_only=True,
                committed=True,conversation_final=decision.conversation_final))
            rows.append(dict(sequence=sequence,role=decision.role,frame_start=cursor,frame_stop=expected_end,
                committed_frame=result['committed_frame'],received_samples=event['received_samples'],
                eos=event['eos'],generation_interval=list(result['interval']),
                native_evidence=result['evidence'],receipt_sha256=result['receipt_sha256']))
            cursor=expected_end
        output=sink.finish(conversation_final=True)
        if cursor!=634 or control.snapshot()['phase']!='closed':raise ValueError('finite trial incomplete')
        early=[r for r in rows if r['role']=='speech' and not r['eos']]
        if len(early)!=2:raise ValueError('first two speech windows must infer before EOS')
        destination=root/'continuous.nut'
        fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as dst,Path(output).open('rb') as src:
            while block:=src.read(1024*1024):dst.write(block)
        destination.chmod(0o400)
        return dict(schema='h3-streaming-native-trial-v1',frames=634,output=str(destination),
            segments=rows,events=events,av=sink.receipt(),admission=control.snapshot(),
            first_two_speech_before_eos=True,new_live_input=False,tts_streaming=False,
            original_pcm_sha256=source.sha256,device_playback_verified=False,realtime_complete=False)
    finally:
        ingress.cancel();sink.cancel()
