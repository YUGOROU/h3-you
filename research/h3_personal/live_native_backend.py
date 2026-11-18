"""Adapter to one ALREADY-warm H3 runtime; never starts/reloads the model.

Persistent owner/decoder spans all turns of one conversation. CPU tests inject
pixel exporter/receipt reader, not replacement native math in a real worker.
Native uncertainty is terminal for this adapter. Explicit healthy abandonment
may retire that owner without restarting weights; pipeline health gates decide.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import threading

from .streaming_native_stages import validate_manifest


def _export(result,path,start,stop,index):
    import numpy as np
    from research.run_h3_realtime_first import encode_rgb_segment
    from .conversation_scenario import segment_timing
    frames=np.stack([np.asarray(frame) for frame in result.frames])
    if frames.dtype!=np.uint8 or frames.shape!=(stop-start,448,256,3):
        raise ValueError('native pixel geometry/count/dtype differs')
    pixels=frames[:,:,2:-2,:].copy(order='C')
    encode_rgb_segment(pixels.tobytes(),path,segment_timing(index,start,stop),width=252,height=448)
    path.chmod(0o400)
    return path.read_bytes()


class LiveNativeBackend:
    def __init__(self,runtime,root,*,exporter=_export,evidence_reader=None,visual_ancestry_policy=None,stable_selfie=False):
        from .visual_ancestry_policy import CANDIDATE
        if visual_ancestry_policy is not None and (type(visual_ancestry_policy) is not str or visual_ancestry_policy != CANDIDATE):
            raise ValueError('explicit candidate visual ancestry policy required')
        if visual_ancestry_policy is not None and 'H3_IDLE_ANCHOR_DIAGNOSTIC' in os.environ:
            raise ValueError('mixed idle/conversation ancestry forbidden')
        self._visual_policy=visual_ancestry_policy
        self._visual_anchor=None
        self._intent_lock=threading.Lock()
        self._intent_epoch=0; self._intent_target=0; self._intent=None
        self.visual_intents=getattr(runtime, 'visual_intents', False)
        if type(self.visual_intents) is not bool:raise ValueError('explicit visual runtime selection required')
        if (type(stable_selfie) is not bool or (stable_selfie and
                (not self.visual_intents or visual_ancestry_policy != CANDIDATE))):
            raise ValueError('stable selfie requires visual transport and conversation ancestry')
        self.stable_selfie=stable_selfie
        self.visual_events=[]
        if (runtime._state!='ready' or runtime._continuation_enabled is not True
                or runtime.conversation_roles is not True or runtime.quality_profile.name!='reference1024'):
            raise ValueError('already-ready reference1024 conversation runtime required')
        if evidence_reader is None:
            from research.hf_h3_streaming_trial import read_evidence
            evidence_reader=read_evidence
        from .conversation_conditioning import role_profile_digest
        self._base_profile_digest=role_profile_digest(runtime.quality_profile,'speech')
        self.runtime=runtime;self.root=Path(root).resolve()
        if self.root.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('generated output must remain outside repository')
        self.root.mkdir(mode=0o700,exist_ok=False)
        self._export=exporter;self._read=evidence_reader
        self._id=None;self._sequence=0;self._emitted=0;self._committed=0
        self._final=False;self._failed=False;self._retired=False
        self._native_released=False;self._archive_created=False
        self._lock=threading.Lock();self.receipts=[]
        with runtime._lock:
            if runtime._state!='ready' or getattr(runtime,'_h3_live_native_owner',None) is not None:
                raise RuntimeError('one live native adapter per warm runtime')
            runtime._h3_live_native_owner=self

    def accept_visual_intent(self, epoch, proposal):
        """Called only after ordered current-turn admission, never from stale Pi IO.

        Speech interruption alone does not reset the active visual continuation.
        Native application snapshots this state at its next legal window boundary.
        """
        from .conversation_conditioning import resolve_visual_intent, validate_visual_intent
        if not self.visual_intents or type(epoch) is not int or not 1 <= epoch <= 1000:
            raise ValueError('explicit finite visual admission required')
        proposal=validate_visual_intent(proposal)
        if self.stable_selfie and proposal['instruction'] is not None:
            raise ValueError('stable selfie forbids camera/action instructions')
        with self._intent_lock:
            if epoch <= self._intent_target:raise ValueError('stale visual intent')
            if len(self.visual_events)>=2:raise ValueError('two-turn visual admission bound')
            self._intent=resolve_visual_intent(self._intent, proposal)
            self._intent_target=epoch
            if proposal['instruction'] is not None:self._intent_epoch=epoch
            self.visual_events.append(dict(stage='instruction_received', target_epoch=epoch,
                intent_epoch=self._intent_epoch,
                instruction_sha256=None if self._intent is None else hashlib.sha256(
                    self._intent['instruction'].encode()).hexdigest(), visual_achievement='unverified'))

    def __call__(self,manifest):
        from .live_voice_runtime import _json
        from .conversation_conditioning import role_prompt,role_profile_digest
        from .visual_ancestry_policy import MANIFEST_KEY, VALUE, ENV, parse_visual_ancestry_policy
        if 'idle_anchor_diagnostic' in manifest:
            raise ValueError('caller-provided idle diagnostics forbidden')
        if MANIFEST_KEY in manifest:
            if self._visual_policy is not None:
                raise ValueError('caller cannot override live ancestry policy')
            # Dedicated comparison harness owns an explicitly paired diagnostic
            # manifest/environment. Ordinary callers cannot activate this path.
            parse_visual_ancestry_policy(manifest,os.environ.get(ENV))
        if self.visual_intents:
            from .conversation_conditioning import role_record
            if 'visual_conditioning' in manifest.get('streaming_audio', {}):
                raise ValueError('caller cannot override owned visual conditioning')
            with self._intent_lock:
                if self.stable_selfie and (self._intent is not None or self._intent_epoch != 0):
                    raise ValueError('stable selfie cannot inherit active movement')
                record=role_record(manifest['streaming_audio']['role'], visual_intents=True,
                    active=self._intent, intent_epoch=self._intent_epoch, target_epoch=self._intent_target)
            manifest=dict(manifest, streaming_audio=dict(manifest['streaming_audio'], visual_conditioning=record))
        if not self.visual_intents and 'visual_conditioning' in manifest.get('streaming_audio', {}):
            raise ValueError('visual manifest requires explicit runtime selection')
        c=validate_manifest(manifest)
        if self._visual_policy is not None:
            manifest=dict(manifest)
            manifest[MANIFEST_KEY]=dict(schema=VALUE,policy=self._visual_policy)
        profiler=getattr(self,'_window_profiler',None)
        index=c['sequence']
        def span(stage):
            return nullcontext() if profiler is None else profiler.span(index,stage)
        with self._lock,self.runtime._lock,span('backend_total'):
            if profiler is not None:
                profiler.role(index,c['role'])
            if self._failed or self._retired or self._final:raise RuntimeError('terminal native adapter')
            if (self.runtime._state!='ready' or c['profile_sha256']!=self._base_profile_digest
                    or role_profile_digest(self.runtime.quality_profile,'speech')!=self._base_profile_digest
                    or c['sequence']!=self._sequence or c['committed_before']!=self._committed
                    or (self._id is not None and self._id!=c['id'])):
                raise ValueError('foreign/out-of-order native request')
            self._id=c['id']
            end=c['committed_before']+(124 if c['sequence']==0 else 102)
            emitted=end-(0 if c['conversation_final'] else 5)
            try:
                with span('role_switch'):
                    if self.visual_intents:
                        from .conversation_conditioning import current_record
                        if current_record()!=c['visual_conditioning']:
                            _json(self.runtime.continuation_file,dict(mode='independent'))
                            self.runtime._switch_role(c['role'], visual_binding=c['visual_conditioning'])
                    elif self.runtime.config.prompt!=role_prompt(c['role']):
                        _json(self.runtime.continuation_file,dict(mode='independent'))
                        self.runtime._switch_role(c['role'])
                with span('manifest_write'):
                    _json(self.runtime.continuation_file,manifest)
                with span('native_call'):
                    result,interval=self.runtime._generate(continuation=True)
                    if profiler is not None:
                        profiler.interval(index,'native_generate_sync',*interval)
                with span('evidence_read'):
                    proof=self._read(self.runtime.coverage,manifest)
                with span('evidence_validate'):
                    prefix=None if c['audio_offer'] is None else c['audio_offer']['prefix_sha256']
                    if (proof['committed_frame']!=end or proof['emitted_frame']!=emitted
                            or proof['evidence']['both_ranks'] is not True
                            or proof['evidence']['forwards']!=4
                            or proof['evidence']['resident_parameter_identity'] is not True
                            or proof['evidence']['role_cache_hit'] is not True
                            or proof['evidence']['prefix_pcm_sha256']!=prefix):
                        raise ValueError('native evidence clock/conditioning/owner differs')
                    candidate_sha=getattr(self.runtime,'ordered_reference_sha256',None)
                    candidate_proof=proof['evidence'].get('ordered_references')
                    if candidate_sha is not None:
                        if (proof['evidence'].get('ordered_both_ranks') is not True
                                or type(candidate_proof) is not dict
                                or candidate_proof.get('identity_sha256')!=candidate_sha
                                or candidate_proof.get('candidate_binding_sha256')!=self.runtime.candidate_binding_sha256
                                or candidate_proof.get('visual_acceptance')!='unreviewed'):
                            raise ValueError('both-rank candidate conditioning proof missing or stale')
                    elif candidate_proof is not None or 'ordered_both_ranks' in proof['evidence']:
                        raise ValueError('unexpected candidate conditioning evidence')
                with span('ancestry_read'):
                    ancestry = self._read_visual_ancestry(manifest) if self._visual_policy is not None else None
                path=self.root/f'segment-{self._sequence:03d}.mp4'
                with span('video_export'):
                    video=self._export(result,path,self._emitted,emitted,self._sequence)
                if type(video) is not bytes or not 0<len(video)<=16*1024*1024:
                    raise ValueError('bounded immutable generated video required')
                output=dict(video=video,count=emitted-self._emitted,committed_frame=end,emitted_frame=emitted,
                    conditioning_prefix_sha256=prefix,
                    receipt_sha256=proof['receipt_sha256'])
                self.receipts.append(dict(sequence=self._sequence,role=c['role'],generation_interval=tuple(interval),
                    committed_before=c['committed_before'],receipt_sha256=proof['receipt_sha256'],
                    committed_frame=end,emitted_frame=emitted,native_evidence=proof['evidence']))
                if self.visual_intents:
                    visual=c['visual_conditioning']
                    expected={k:visual[k] for k in ('target_epoch','intent_epoch','prompt_sha256')}
                    if proof['evidence'].get('visual_conditioning')!=expected:
                        raise ValueError('both-rank visual application proof missing')
                    self.receipts[-1]['visual_conditioning']=dict(expected,
                        stage='native_conditioning_applied', visual_achievement='unverified',
                        condition_refresh_by_rank=proof['evidence']['visual_condition_refresh'],
                        presentation_outcome='unverified')
                if ancestry is not None:
                    self.receipts[-1]['visual_ancestry_selection']=ancestry
                self._sequence+=1;self._committed=end;self._emitted=emitted;self._final=c['conversation_final']
                return output
            except BaseException:
                self._quarantine()
                raise

    def _read_visual_ancestry(self,manifest):
        """Selection proof only; clocks remain owned by read_evidence above."""
        seq=self._sequence; rows=[]
        for rank in (0,1):
            path=self.runtime.coverage/f'streaming-rank-{rank}-window-{seq}.json'
            if path.is_symlink() or not path.is_file() or path.stat().st_size>1<<20:
                raise ValueError('bounded ancestry rank receipt required')
            row=json.loads(path.read_bytes())
            e=row.get('visual_ancestry_evidence'); guides=row.get('guide_digests')
            if (row.get('streaming')!=manifest['streaming_audio']
                    or row.get('native_window_id')!=manifest['session_id']
                    or type(e) is not dict or set(e)!={'policy','sequence','kind','anchor_sha256','selected_sha256','anchor_sequence'}
                    or e['policy']!=self._visual_policy or type(e['sequence']) is not int or e['sequence']!=seq
                    or type(guides) is not list or len(guides)!=3 or e['selected_sha256']!=guides[0]
                    or e['kind']!=('unconditioned' if seq==0 else 'anchor' if seq%3==0 else 'latest')):
                raise ValueError('actual ancestry policy/rank/selection mismatch')
            for key in ('anchor_sha256','selected_sha256'):
                digest=e[key]
                if digest is not None and (type(digest) is not str or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest)):
                    raise ValueError('invalid ancestry digest')
            if seq==0:
                if any(e[k] is not None for k in ('anchor_sha256','selected_sha256','anchor_sequence')):
                    raise ValueError('unexpected initial ancestry')
            elif (type(e['anchor_sequence']) is not int or e['anchor_sequence']!=1
                    or e['anchor_sha256'] is None or e['selected_sha256'] is None
                    or (seq==1 and e['anchor_sha256']!=guides[0])
                    or (seq>1 and e['anchor_sha256']!=self._visual_anchor)
                    or (seq%3==0 and guides[0]!=self._visual_anchor)):
                raise ValueError('immutable first-tail/global refresh mismatch')
            rows.append(e)
        if rows[0]!=rows[1]:
            raise ValueError('both-rank ancestry disagreement')
        if seq==1:self._visual_anchor=rows[0]['anchor_sha256']
        return dict(rows[0])

    def _quarantine(self):
        self._failed=True
        if self.runtime._state=='ready':
            self.runtime._state='degraded'
            self.runtime._h3_live_native_quarantine=self

    def _finish_release(self):
        from .live_voice_runtime import _json
        if not self._native_released:raise RuntimeError('native release not acknowledged')
        archive=self.runtime.coverage/('retired-'+hashlib.sha256(self._id.encode()).hexdigest()[:24])
        if not self._archive_created:
            archive.mkdir(mode=0o700,exist_ok=False);self._archive_created=True
        if archive.is_symlink():raise ValueError('symlink archive')
        for index in range(self._sequence+1):
            for rank in (0,1):
                p=self.runtime.coverage/f'streaming-rank-{rank}-window-{index}.json'
                if p.is_symlink():raise ValueError('symlink native receipt')
                if p.exists():
                    destination=archive/p.name
                    if destination.exists():raise ValueError('archive collision')
                    p.rename(destination)
        _json(self.runtime.continuation_file,dict(mode='independent'))
        self._retired=True
        if getattr(self.runtime,'_h3_live_native_owner',None) is self:
            del self.runtime._h3_live_native_owner
        if (getattr(self.runtime,'_h3_live_native_quarantine',None) is self
                and self.runtime._state=='degraded'):
            del self.runtime._h3_live_native_quarantine
            self.runtime._state='ready'

    def _release(self,action,reason):
        from .live_voice_runtime import _json
        from .conversation_control import SCHEMA
        with self._lock,self.runtime._lock:
            if self._retired or self._id is None:raise RuntimeError('no native owner to release')
            if action=='retire' and not self._native_released and (self._failed or not self._final or self._emitted!=self._committed):
                raise RuntimeError('final native decoder ACK required')
            try:
                if not self._native_released:
                    _json(self.runtime.continuation_file,dict(conversation_control=dict(schema=SCHEMA,
                        action=action,reason=reason,conversation_id=self._id)))
                    self.runtime._generate(continuation=True)
                    self._native_released=True
                # A diagnosed local archival repair may be repeated, never the
                # already-ACKed native release RPC. Unknown RPC outcome stays gated.
                self._finish_release()
            except BaseException:
                self._quarantine()
                raise

    def close_unused(self):
        with self._lock,self.runtime._lock:
            if self._id is not None or self._failed:raise RuntimeError('explicit owner release required')
            if getattr(self.runtime,'_h3_live_native_owner',None) is self:
                del self.runtime._h3_live_native_owner
            self._retired=True

    def retire(self):self._release('retire','completed')

    def abandon(self,*,reason):
        if reason not in ('local-preparation-failure','local-sink-failure'):
            raise ValueError('explicit local abandonment reason required')
        self._release('abandon',reason)
