"""Ref2VA-trained SMD four-forward candidate, never T2VA FastH3/PDD substitution.

CPU/source compatibility does not establish identity or realtime performance.
"""
from dataclasses import dataclass,asdict
import json
from pathlib import Path
from research.h3_personal.ref2va_runtime import Ref2VAConfig,sha,IDENTITY_SHA

REPO='zjwfufu/SMD-MiniMax-H3-Ref2VA-4step'
REV='8f67bf6b8b017bf16be4fd36fdfe135eeb78715a'
WEIGHT_SHA='b8cd339e167ab29e112d4f8934d3ac7f20b7e3269895fb149761bd959336bdd7'
TRAIN_BASE_REV='939557dc319dd91227e30195a763f272ba7f8765'

@dataclass(frozen=True)
class FastRef2VAConfig(Ref2VAConfig):
    sigma_grid_points:int=5

    def validate(self):
        values=asdict(self);values.update(height=256,width=448)
        expected=asdict(FastRef2VAConfig())
        differing={k for k in expected if values[k]!=expected[k]}
        if ((self.height,self.width) not in ((256,448),(448,256)) or not differing<={'prompt'}
                or any(type(getattr(self,k)) is not int for k in ('height','width','num_frames','fps','sigma_grid_points','seed'))):
            raise ValueError('fixed reference-conditioned SMD4 pilot required')
        if differing:
            # Only the declared conversation roles may change the prompt; all
            # geometry/sampler/precision fields above stay fixed.
            from research.h3_personal.conversation_conditioning import is_declared_prompt
            if not is_declared_prompt(self.prompt):
                raise ValueError('fixed reference-conditioned SMD4 pilot required')
        return self


def config_for_layout(layout):
    from research.h3_personal.ref2va_runtime_options import export_geometry
    geometry=export_geometry(layout)
    return FastRef2VAConfig(height=geometry['canvas_height'],width=geometry['canvas_width']).validate()


def generator_config(api,base,adapter,*,regional_compile=False,runtime_policy='baseline'):
    from research.h3_personal.ref2va_runtime import generator_config as native_config
    config=native_config(api,base,adapter)
    config.pipeline.experimental['h3_sequential_load']=True
    config.engine.offload.lazy_module_load=False
    if runtime_policy not in ('baseline','resident-visual','resident-visual-cached'):raise ValueError('unsupported runtime policy')
    if runtime_policy!='baseline':config.engine.offload.vae=False
    if regional_compile:
        config.pipeline.experimental['inference_torch_compile']=True
        config.engine.compile.enabled=True
        # Native loader requires fullgraph and injects precision-preserving options;
        # it forbids mode+options. Express reduce-overhead's cudagraphs via options.
        config.engine.compile.mode=None
        config.engine.compile.extras={'options':{'triton.cudagraphs':True}}
        config.engine.compile.fullgraph=True
        config.engine.compile.dynamic=False
    return config


def segment_timing(index):
    from research.h3_personal.realtime_segments import SegmentTiming
    if type(index) is not int or index<0:raise ValueError('invalid segment id')
    return SegmentTiming(index,0,124,index*124,(index+1)*124)


def evaluate_warm(generations,reports,segments=4):
    if type(segments) is not int or not 1<=segments<=4:raise ValueError('one to four real generated clips required')
    if len(generations)!=segments or len(reports)!=segments:raise ValueError('every generated clip needs a publication receipt')
    base=reports[0]['publication_finished_at']
    lateness=[max(0,r['publication_finished_at']-base-i*124/24) for i,r in enumerate(reports)]
    if segments<2:
        # No second clip means no measured warm interval; never report a throughput.
        return dict(warm_generation_throughput=None,warm_local_availability_throughput=None,warm_consumer_throughput=None,
            availability_lateness_seconds=lateness,warm_measurement_available=False,server_gate_passed=False,
            reference_identity_verified=False,motion_continuity_verified=False,client_playback_verified=False,realtime_complete=False)
    video=(segments-1)*124/24
    generation=video/sum(end-start for start,end in generations[1:])
    consumer=video/sum(r['publication_finished_at']-r['encode_started_at'] for r in reports[1:])
    availability=video/(reports[-1]['publication_finished_at']-reports[0]['publication_finished_at'])
    return dict(warm_generation_throughput=generation,warm_local_availability_throughput=availability,
        warm_consumer_throughput=consumer,availability_lateness_seconds=lateness,warm_measurement_available=True,
        server_gate_passed=generation>=1 and consumer>=1 and availability>=1 and max(lateness)<=0.001,
        reference_identity_verified=False,motion_continuity_verified=False,client_playback_verified=False,realtime_complete=False)


def raw_sigmas():
    # Exact published algorithm, including the half-sized final raw segment.
    base=1.0/(4-1+0.5);values=[1.0]
    for step in range(4):values.append(values[-1]-base*(0.5 if step==3 else 1.0))
    values[-1]=0.0
    return values


def shifted_sigmas(shift):return [shift*s/(1+(shift-1)*s) for s in raw_sigmas()]


def configure_schedulers(video,audio,grid_points,device):
    if grid_points!=5 or float(video.shift)!=12.0 or float(audio.shift)!=3.0:raise ValueError('SMD sampler contract differs')
    import torch
    for scheduler in (video,audio):
        scheduler.set_timesteps(sigmas=torch.tensor(shifted_sigmas(float(scheduler.shift)),dtype=torch.float32,device=device),device=device)


def build_adapter(smd,identity,config,output,*,canonical_fingerprint=False):
    from safetensors.torch import load_file,save_file
    from research.h3_personal.personal_fast_adapter import compose
    import torch
    output=Path(output)
    if output.exists() or output.resolve() in (Path(smd).resolve(),Path(identity).resolve()):raise ValueError('new derived output required')
    if sha(smd)!=WEIGHT_SHA or sha(identity)!=IDENTITY_SHA:raise ValueError('source SHA differs')
    result,report=compose(load_file(str(smd),device='cpu'),load_file(str(identity),device='cpu'),json.loads(Path(config).read_text()))
    if (report['identity_modules'],report['total_modules'],report['dense_tensors_preserved'])!=(208,312,0) or set(report['combined_ranks'].values())!={144}:
        raise ValueError('SMD+identity derived coverage differs')
    if any(not torch.isfinite(v).all().item() for v in result.values()):raise ValueError('nonfinite adapter')
    output.parent.mkdir(parents=True,exist_ok=True)
    save_file(result,str(output),metadata={'composition':'SMD128+identity16; alpha/rank=1 per module','source_revision':REV})
    actual=load_file(str(output),device='cpu')
    if set(actual)!=set(result) or any(not torch.equal(v,actual[k]) for k,v in result.items()):raise ValueError('derived strict reload failed')
    if sha(smd)!=WEIGHT_SHA or sha(identity)!=IDENTITY_SHA:raise ValueError('source mutated')
    report.update(derived_sha256=sha(output),strict_cpu_reload=True,smd_repo=REPO,smd_revision=REV)
    if canonical_fingerprint:
        from research.h3_personal.adapter_fingerprint import fingerprint_file
        report.update(fingerprint_file(output))
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def load_deferred_adapter(pipeline):
    if not getattr(pipeline,'_smd_deferred_adapter',False):return
    transformer=pipeline.get_module('transformer')
    if transformer is None or pipeline.trainable_transformer_modules:raise RuntimeError('unexpected deferred transformer state')
    pipeline.trainable_transformer_modules={'transformer':transformer}
    pipeline.convert_to_lora_layers()
    pipeline._setting_constructor_adapter=True
    try:pipeline.set_lora_adapter(pipeline.lora_nickname,pipeline.lora_path,strength=pipeline.lora_strength)
    finally:pipeline._setting_constructor_adapter=False
    pipeline._smd_deferred_adapter=False


def regional_compile_receipt(transformer):
    # Pinned ComposedPipelineBase wraps submodule.forward, not the module object.
    def selected(name):
        parts=name.split('.')
        return ((len(parts)==2 and parts[0]=='transformer_blocks' and parts[1].isdigit())
                or (len(parts)==3 and parts[:2]==['token_refiner','refiner_blocks'] and parts[2].isdigit()))
    targets=[m for name,m in transformer.named_modules() if selected(name)]
    wrapped=sum(hasattr(m.forward,'_torchdynamo_orig_callable') for m in targets)
    if len(targets)!=52 or wrapped!=52:raise RuntimeError('regional forward compile wrapper coverage differs')
    return dict(wrapped_regions=wrapped,wrapper_kind='forward-callable',mode='options:triton.cudagraphs',
                dynamic=False,fullgraph=True,wrapper_coverage_not_kernel_parity=True)


def static_merge_receipt(pipeline):
    active=[layer for group in pipeline.lora_layers.values()
            for _,layers in group.lora_layers_by_block() for layer in layers.values()
            if not layer.disable_lora]
    merged=sum(bool(layer.merged) for layer in active)
    if len(active)!=312 or merged!=312:raise RuntimeError('all312 active LoRA layers must already be statically merged')
    return dict(active=312,merged=312,dynamic=0)


def assert_loaded(pipeline,state,consumed,adapted_count):
    if not getattr(pipeline,'_ref2va',False) or pipeline._extra_config_module_map.get('transformer')!='transformer_ref':raise RuntimeError('not actual Ref2VA')
    a={k:v for k,v in state.items() if k.endswith('.lora_A')}
    personal=sum(len(v.shape)==2 and v.shape[0]==144 for v in a.values())
    other=sum(len(v.shape)==2 and v.shape[0]==128 for v in a.values())
    complete=all(k[:-len('.lora_A')]+'.lora_B' in state and state[k[:-len('.lora_A')]+'.lora_B'].shape[1]==v.shape[0] for k,v in a.items())
    if len(state)!=624 or len(a)!=312 or personal!=208 or other!=104 or not complete or adapted_count!=312 or set(state)-set(consumed):raise RuntimeError('SMD Ref2VA loaded coverage differs')
    report=dict(identity_modules=208,total_modules=312,unmatched=0,rank='mixed128-144',partition='transformer_ref')
    import os
    folder=os.environ.get('H3_REF2VA_COVERAGE_DIR')
    if not folder:raise RuntimeError('isolated coverage folder required')
    import torch
    if not torch.distributed.is_initialized():raise RuntimeError('both-rank coverage required')
    rank=torch.distributed.get_rank();report['worker_rank']=rank
    Path(folder).mkdir(parents=True,exist_ok=True)
    Path(folder,f'coverage-rank-{rank}.json').write_text(json.dumps(report)+'\n')
    Path(folder,f'merge-rank-{rank}.json').write_text(json.dumps(static_merge_receipt(pipeline))+'\n')
    if os.environ.get('H3_REF2VA_ATTENTION_MODE') in ('fa4','fa4-packed'):
        from research.h3_personal.ref2va_attention import cold_receipt
        attention=cold_receipt()
        Path(folder,f'attention-rank-{rank}.json').write_text(json.dumps(attention)+'\n')
        print('FAST_REF2VA_ATTENTION',json.dumps(dict(attention,worker_rank=rank)),flush=True)
    if os.environ.get('H3_REF2VA_ATTENTION_MODE')=='fa4-packed':
        from fastvideo.attention.backends import flash_attn as backend
        from research.h3_personal.ref2va_attention import start_packed_counter
        start_packed_counter(backend)
    print('FAST_REF2VA_LOADED',json.dumps(report),flush=True)


def condition_key(batch):
    from research.h3_personal.conditioning_experiment import read_policy, policy_key, snapshot_sources
    policy = read_policy()
    if policy is not None:
        if '_h3_conditioning_sources' not in batch.extra: snapshot_sources(batch)
        if tuple(sha(r.source) for r in batch.extra['_h3_conditioning_sources']) != batch.extra['_h3_conditioning_source_sha']:
            raise RuntimeError('original conditioning source bytes drifted')
        return (batch.prompt, batch.height, batch.width, batch.num_frames, batch.fps,
                batch.extra['_h3_conditioning_source_sha'], policy_key())
    references=list(batch.references or [])
    if not 1<=len(references)<=2 or any(r.media_type!='image' or not isinstance(r.source,(str,Path)) for r in references):
        raise ValueError('ordered original image and optional explicit face crop required')
    return (batch.prompt,batch.height,batch.width,batch.num_frames,batch.fps,tuple(sha(r.source) for r in references))


def get_condition_cache(stage,key):
    from research.h3_personal.conditioning_experiment import read_policy
    if read_policy() is not None:
        cache = getattr(stage, '_h3_conditioning_cache', {})
        if cache:
            from research.h3_personal.conditioning_experiment import processor_digest
            if processor_digest(stage) != stage._h3_conditioning_processor_digest:
                raise RuntimeError('loaded processor identity drift')
        return cache.get(key)
    cache = getattr(stage, '_smd_condition_cache', None)
    return cache if cache is not None and cache[0] == key else None


def store_condition(stage,batch,fastvideo_args=None):
    from research.h3_personal.conditioning_experiment import read_policy, store_condition as store_experiment
    if read_policy() is not None:
        return store_experiment(stage, batch, fastvideo_args)
    key=batch.extra.pop('_smd_condition_key',None)
    if key is None:raise RuntimeError('condition cache binding missing')
    # Cache only encoder outputs, never sampled target noise or RF augmentation.
    stage._smd_condition_cache=(key,[t.detach().cpu().clone() for t in batch.prompt_embeds],
        batch.extra['minimax_h3_text_token_tags'].detach().cpu().clone())


def continuation_warm_enabled(batch):
    """Snapshot validated activation before execution, never during teardown.

    The on-disk mode decides: legacy independent windows may be released on
    failure, while latent-tail/conversation owners are retained as faulted.
    """
    import json
    import os
    from pathlib import Path
    path = os.environ.get('H3_SMD_CONTINUATION_WINDOW_FILE')
    if '_h3_continuation' in batch.extra:
        return True
    if path:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('manifest must be an object')
        return value.get('mode') == 'latent-tail-v1'
    from research.h3_personal.fast_continuation_stages import active
    return active()


def warm_forward(pipeline,batch,args,cache):
    """Default path imports no controller/profiler; enabled failures are terminal.

    Controller API: apply_request(pipeline,batch,args,policy) returns an owner
    exposing on_before_forward(stage,batch,index), on_hold_audio(stage,batch),
    on_commit(stage,batch), close(). apply_request owns transactional preparation,
    probe and install after the resident static merge; no eager fallback here.
    """
    import os
    from research.h3_personal.conversation_control import boundary, read_manifest, roles_active
    if roles_active():
        from fastvideo.pipelines.composed_pipeline_base import get_local_torch_device
        pipeline._h3_conversation_control_boundary = True
        try:
            if boundary(pipeline, batch, get_local_torch_device()):
                return batch
        finally:
            pipeline._h3_conversation_control_boundary = False
    manifest = read_manifest()
    try:
        trace, denoise = None, None
        if manifest.get('trace_policy') is not None or manifest.get('denoise_policy') is not None:
            from research.h3_personal.utilization_policy import policies
            trace, denoise = policies(manifest.get('trace_policy'), manifest.get('denoise_policy'))
    except ValueError:
        pipeline._h3_utilization_faulted = True
        pipeline._smd_continuation_warm_faulted = True
        raise
    if getattr(pipeline, '_h3_utilization_faulted', False):
        raise RuntimeError('utilization owner faulted; reuse forbidden')
    identity = getattr(pipeline, '_h3_utilization_identity', None)
    session = manifest.get('session_id')
    if identity is not None and identity[0] == session and identity[1:] != (trace, denoise):
        pipeline._h3_utilization_faulted = True
        pipeline._smd_continuation_warm_faulted = True
        raise ValueError('immutable utilization policy drift')
    if trace is None and denoise is None and manifest.get('measure_memory') is not True:
        return _guarded_forward(pipeline,batch,args,cache)
    from research.h3_personal.fast_continuation_stages import _manifest
    if trace is not None or denoise is not None:
        _manifest()  # Reject mode/policy mixes before any device work.
    if getattr(pipeline, '_h3_utilization_faulted', False):
        raise RuntimeError('utilization owner faulted; reuse forbidden')
    pipeline._h3_utilization_identity = (session, trace, denoise)
    import time
    started = time.monotonic()
    controller = None
    result = None
    pipeline._h3_window_trace_report = None
    succeeded = False
    try:
        if denoise is not None:
            static_merge_receipt(pipeline)
            # Lazy import deliberately fails closed until controller integration.
            from research.h3_personal.denoise_execution import apply_request
            controller = apply_request(pipeline,batch,args,denoise)
            if controller is None:
                raise RuntimeError('denoise controller integration required')
            batch.extra['_h3_denoise_execution'] = controller
        if trace is None:
            result = _guarded_forward(pipeline,batch,args,cache)
        else:
            result = _trace_warm_forward(pipeline,batch,args,cache,manifest,trace)
        succeeded = True
        return result
    except BaseException:
        pipeline._h3_utilization_faulted = True
        pipeline._smd_continuation_warm_faulted = True
        raise
    finally:
        batch.extra.pop('_h3_denoise_execution', None)
        if result is not None and isinstance(getattr(result, 'extra', None), dict):
            result.extra.pop('_h3_denoise_execution', None)
        if controller is not None:
            try:
                controller.close()
            except BaseException:
                pipeline._h3_utilization_faulted = True
                pipeline._smd_continuation_warm_faulted = True
                raise
        if manifest.get('mode') == 'latent-tail-v1' and os.environ.get('H3_REF2VA_COVERAGE_DIR'):
            _write_utilization(pipeline, manifest, trace, denoise, controller, started,
                               time.monotonic(), succeeded)


def utilization_name(session, index, rank):
    # Hash, never interpolate caller-controlled session strings into paths.
    import hashlib
    if type(session) is not str or not session or type(index) is not int or index < 0:
        raise ValueError('invalid utilization identity')
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError('invalid utilization rank')
    return f'utilization-{hashlib.sha256(session.encode()).hexdigest()}-{index}-{rank}.json'


def _write_utilization(pipeline, manifest, trace, denoise, controller, start, end, success):
    import os
    import torch
    rank = torch.distributed.get_rank()
    report = getattr(pipeline, '_h3_window_trace_report', None)
    row = dict(schema='h3-window-utilization-v1', session_id=manifest['session_id'],
        window_index=manifest['index'], rank=rank, trace_policy=trace, denoise_policy=denoise,
        worker_monotonic_start=start, worker_monotonic_end=end,
        faulted=not success or bool(getattr(pipeline, '_h3_utilization_faulted', False)),
        trace_available=False, coverage_proven=False, truncated=False,
        event_count=0, trace_bytes=0, trace_summary=None,
        denoise_installed=False, candidate_rejected=False,
        parameter_identity_preserved=False, representative_preflight_complete=False,
        backend_compiles=0, compile_seconds=0., regions=0)
    if manifest.get('measure_memory') is True:
        # Fresh, own-device post-request snapshot; free bytes are not a peak.
        device = torch.cuda.current_device() if callable(getattr(torch.cuda, 'current_device', None)) else None
        row['logical_device'] = device
        for key, name in (('peak_allocated_bytes', 'max_memory_allocated'),
                          ('peak_reserved_bytes', 'max_memory_reserved')):
            fn = getattr(torch.cuda, name, None)
            row[key] = int(fn(device)) if device is not None and callable(fn) else None
        info = getattr(torch.cuda, 'mem_get_info', None)
        values = info(device) if device is not None and callable(info) else (None, None)
        row['post_request_free_bytes'], row['post_request_total_bytes'] = values
    if controller is not None:
        # Adapter close has already produced this scalar-only local receipt.
        metrics = json.loads((Path(os.environ['H3_REF2VA_COVERAGE_DIR']) /
            ('controller-' + utilization_name(manifest['session_id'], manifest['index'], rank))).read_text())
        # Never bind a stale controller receipt to a fresh warm-window identity.
        bound = all(type(metrics.get(k)) is type(v) and metrics[k] == v for k, v in
                    dict(session_id=manifest['session_id'], window_index=manifest['index'],
                         rank=rank, denoise_policy=denoise).items())
        for key in ('denoise_installed', 'candidate_rejected', 'parameter_identity_preserved',
                    'representative_preflight_complete', 'backend_compiles', 'compile_seconds',
                    'regions', 'extra_status_collectives', 'extra_status_seconds',
                    'state', 'owner_terminal', 'representative_metrics', 'rejection_kind',
                    'prepare_error_code', 'source_pinned_verified', 'main50_verified',
                    'parameter_dtype_histogram', 'native_type_hashes'):
            if bound and key in metrics:
                row[key] = metrics[key]
    if report is not None:
        from research.h3_personal.device_activity import serialize
        row.update(trace_available=bool(report['devices']['cuda']['events']),
                   truncated=report['truncated'], event_count=report['events_seen'],
                   trace_bytes=len(serialize(report)), trace_summary=report)
    if trace is not None and not row['trace_available']:
        row['finding'] = 'CUDA activity unavailable or this window not selected; metadata only'
    directory = Path(os.environ['H3_REF2VA_COVERAGE_DIR']).resolve()
    if directory.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('private coverage directory required')
    target = directory / utilization_name(manifest['session_id'], manifest['index'], rank)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(row, stream, allow_nan=False)


def _trace_warm_forward(pipeline,batch,args,cache,manifest,policy):
    import os
    import torch
    from research.h3_personal.device_activity import Capture, Policy, capability, analyze, serialize, bound_report
    key = manifest['session_id']
    owner = getattr(pipeline, '_h3_trace_owner', None)
    if owner is None or owner[0] != key:
        owner = (key, dict(policy), Capture(Policy(enabled=True,
            selected_window=policy['selected_window'], max_events=policy['max_events'],
            max_bytes=policy['max_bytes'])))
        pipeline._h3_trace_owner = owner
    if owner[1] != policy:
        raise ValueError('immutable trace policy drift')
    def factory(*, activities, **settings):
        enum = torch.profiler.ProfilerActivity
        return torch.profiler.profile(activities=[getattr(enum, a.upper()) for a in activities],
                                      **settings)
    capture = owner[2]
    # A later unselected window must never retain the selected window's report.
    pipeline._h3_window_trace_report = None
    caps = capability(torch.profiler)
    caps['cuda'] = caps.get('cuda', False) and torch.cuda.is_available()
    shapes = []
    model = pipeline.get_module('transformer')
    selected = manifest['index'] == policy['selected_window']
    had_forward = 'forward' in vars(model)
    original_override = vars(model).get('forward')
    original_forward = model.forward
    def measured_forward(*a, **kw):
        if len(shapes) >= 4:
            raise ValueError('expected exactly four native forwards')
        logical = int(kw['position_ids'].shape[0])
        video = int(kw['video_indices'].numel())
        audio = int(kw['audio_indices'].numel())
        text = int(kw['text_indices'].numel())
        guide = 784 if manifest['index'] > 0 else 0
        guide_audio = 74 if guide else 0
        reference = video - 4144 - guide
        if reference <= 0 or text <= 0 or audio < guide_audio or logical != video + audio + text:
            raise ValueError('actual native source counts differ')
        local = (logical + 1) // 2
        shapes.append(dict(window_index=manifest['index'], forward_index=len(shapes),
            reference_rows=reference, text_rows=text, target_video_rows=4144,
            guide_video_rows=guide, target_audio_rows=audio-guide_audio,
            guide_audio_rows=guide_audio, logical_rows=logical, local_rows=local,
            derived_qkv_payload_bytes=3*local*56*128*2))
        return original_forward(*a, **kw)
    with capture.window(manifest['index'], factory=factory,
                        capabilities=caps) as profiler:
        pipeline._h3_trace_active = profiler is not None
        if selected:
            model.forward = measured_forward
        try:
            from contextlib import nullcontext
            from research.h3_personal.device_activity import semantic_forwards
            with (semantic_forwards(pipeline.get_module('transformer'),
                                    torch.profiler.record_function)
                  if profiler is not None else nullcontext()), (torch.profiler.record_function('h3.warm_forward')
                  if profiler is not None else nullcontext()):
                result = _guarded_forward(pipeline,batch,args,cache)
        finally:
            if selected:
                if had_forward:
                    model.forward = original_override
                else:
                    del model.forward
            pipeline._h3_trace_active = False
    if selected and len(shapes) != 4:
        raise ValueError('expected exactly four actual shape rows')
    if manifest['index'] == policy['selected_window']:
        rank = torch.distributed.get_rank()
        report = analyze(profiler.events() if profiler is not None else (), rank=rank,
                         policy=capture.policy,
                         device_types={})  # Event enum names validated without private Torch namespace.
        report['shapes'] = shapes
        bound_report(report, min(capture.policy.max_bytes, 262144))
        # BF16 (2-byte) policy assumption, derived payload, NOT actual network traffic.
        pipeline._h3_window_trace_report = report
        directory = Path(os.environ['H3_REF2VA_COVERAGE_DIR']).resolve().parent / 'utilization'
        if directory.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('private utilization output outside repository required')
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Scalar sanitized aggregates only. Never export Chrome traces or upload.
        import uuid
        target = directory / f'rank-{rank}-{uuid.uuid4().hex}.json'
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(serialize(report))
    return result


def _guarded_forward(pipeline,batch,args,cache):
    """Native forward with the conversation in-flight flag actually held.

    Host abandonment (conversation_control) is refused while this flag is set,
    and the flag is cleared on every outcome, including unknown failures.
    """
    from research.h3_personal.conversation_control import roles_active
    if not roles_active():
        return _warm_forward(pipeline,batch,args,cache)
    if getattr(pipeline, '_h3_conversation_forward_inflight', False):
        raise RuntimeError('forward still in flight')
    pipeline._h3_conversation_forward_inflight = True
    try:
        return _warm_forward(pipeline,batch,args,cache)
    finally:
        pipeline._h3_conversation_forward_inflight = False


def _warm_forward(pipeline,batch,args,cache):
    import os
    if os.environ.get('H3_REF2VA_ATTENTION_MODE')=='fa4-packed':
        from fastvideo.attention.backends import flash_attn as backend
        if backend._h3_cold_packed_counter['active']:
            from research.h3_personal.ref2va_attention import finish_packed_counter
            import torch
            receipt=finish_packed_counter(backend)
            Path(os.environ['H3_REF2VA_COVERAGE_DIR'],f'packed-rank-{torch.distributed.get_rank()}.json').write_text(json.dumps(receipt)+'\n')
    from fastvideo.pipelines.composed_pipeline_base import get_local_torch_device
    device=get_local_torch_device()
    # Optional allocator counters only, not CUDA events or a synchronization.
    # Reset per warm request on the worker's own device, never parent context.
    memory_manifest=None
    manifest_path=os.environ.get('H3_SMD_CONTINUATION_WINDOW_FILE')
    if manifest_path and os.environ.get('H3_SMD_MEASURE_MEMORY')=='1':
        manifest=json.loads(Path(manifest_path).read_text())
        if manifest.get('mode')=='latent-tail-v1' and manifest.get('measure_memory') is True:
            import torch
            torch.cuda.reset_peak_memory_stats(device)
            memory_manifest=manifest
    from research.h3_personal.conditioning_experiment import read_policy, snapshot_sources, capture_prepared_geometry
    if read_policy() is not None:
        snapshot_sources(batch)
        batch.extra['_h3_conditioning_cache_hit'] = True
    batch=pipeline._stage_name_mapping['input_preparation_stage'](batch,args)
    capture_prepared_geometry(batch)
    batch.prompt_embeds=[t.to(device).clone() for t in cache[1]]
    # Packed-sequence metadata is built on CPU; only embeddings go to the GPU.
    batch.extra['minimax_h3_text_token_tags']=cache[2].clone()
    from research.h3_personal.ordered_qwen_tokens import restore_role_proof
    restore_role_proof(pipeline._stage_name_mapping['conditioning_stage'],batch,cache[0],
                       int(cache[2].shape[0]))
    # No Qwen move/reload and no parking resident DiT on a verified cache hit.
    import os
    from research.h3_personal.ref2va_helpers import WarmStageProfile
    diagnostic=WarmStageProfile(os.environ.get('H3_REF2VA_STAGE_PROFILE_DIR'),
        max_hits=6 if os.environ.get('H3_REF2VA_COMPARISON_PROFILE')=='1' else 3)
    for name in ('latent_preparation_stage','denoising_stage','video_decoding_stage','audio_decoding_stage'):
        from contextlib import nullcontext
        label = nullcontext()
        if getattr(pipeline, '_h3_trace_active', False):
            import torch
            label = torch.profiler.record_function({
                'latent_preparation_stage': 'h3.prepare',
                'denoising_stage': 'h3.denoise',
                'video_decoding_stage': 'h3.video_decode',
                'audio_decoding_stage': 'h3.audio_decode'}[name])
        if name == 'denoising_stage':
            from research.h3_personal.conditioning_experiment import write_evidence
            write_evidence(pipeline, batch)
        with diagnostic.stage(name), label:
            if name=='audio_decoding_stage' and os.environ.get('H3_REF2VA_RUNTIME_POLICY') in ('resident-visual','resident-visual-cached'):
                import torch
                from research.h3_personal.ref2va_runtime_options import finish_visual_only
                batch=finish_visual_only(batch,pipeline._stage_name_mapping[name],torch)
            else:batch=pipeline._stage_name_mapping[name](batch,args)
    stage=pipeline._stage_name_mapping['conditioning_stage']
    stage._smd_condition_hits=getattr(stage,'_smd_condition_hits',0)+1
    import torch,os
    rank=torch.distributed.get_rank()
    diagnostic.finish(rank,stage._smd_condition_hits)
    if memory_manifest is not None:
        memory=dict(schema='h3-worker-memory-v1',worker_rank=rank,
            cache_hit=stage._smd_condition_hits,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            contains_parallel_acceptance=memory_manifest.get('parallel_acceptance') is True)
        directory=Path(os.environ['H3_REF2VA_COVERAGE_DIR']).parent/'memory'
        directory.mkdir(mode=0o700,parents=True,exist_ok=True)
        path=directory/f'rank-{rank}-request-hit{stage._smd_condition_hits}.json'
        with path.open('x') as stream:json.dump(memory,stream,allow_nan=False)
    if os.environ.get('H3_REF2VA_RUNTIME_POLICY') in ('resident-visual','resident-visual-cached'):
        from research.h3_personal.ref2va_runtime_options import residency_receipt
        Path(os.environ['H3_REF2VA_COVERAGE_DIR'],f'residency-rank-{rank}.json').write_text(json.dumps(residency_receipt(pipeline,torch,os.environ['H3_REF2VA_RUNTIME_POLICY']))+'\n')
    if os.environ.get('H3_REF2VA_REGIONAL_COMPILE')=='1':
        receipt=regional_compile_receipt(pipeline.get_module('transformer'))
        Path(os.environ['H3_REF2VA_COVERAGE_DIR'],f'compile-rank-{rank}.json').write_text(json.dumps(receipt)+'\n')
    Path(os.environ['H3_REF2VA_COVERAGE_DIR'],f'cache-rank-{rank}.json').write_text(json.dumps({'hits':stage._smd_condition_hits})+'\n')
    return batch


def install(source):
    from research.h3_personal.ref2va_runtime import install_ref_guard
    source=Path(source);loader=source/'fastvideo/pipelines/lora_pipeline.py'
    report=install_ref_guard(loader)
    text=loader.read_text().replace('from research.h3_personal.ref2va_runtime import assert_ref_loaded_coverage','from research.h3_personal.fast_ref2va import assert_loaded as assert_ref_loaded_coverage')
    anchor='        elif not self.training_mode and self.lora_path is not None:\n'
    deferred=("        elif (__import__('os').environ.get('H3_SMD_REF2VA') == '1' and not self.training_mode\n"
              "              and self.lora_path is not None and not self.trainable_transformer_modules):\n"
              "            self._smd_deferred_adapter = True\n")
    if text.count(anchor)!=1:raise ValueError('deferred LoRA constructor anchor differs')
    loader.write_text(text.replace(anchor,deferred+anchor));report['patched_sha256']=sha(loader)
    stage=source/'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py'
    original=sha(stage)
    # Source archive hash is checked before this isolated copy is made.
    marker='        elif dmd_steps is None:\n'
    text=stage.read_text()
    if text.count(marker)!=1:raise ValueError('schedule source anchor differs')
    hook=("        elif __import__('os').environ.get('H3_SMD_REF2VA') == '1':\n"
          "            if layout.num_condition_video_rows <= 0: raise ValueError('SMD requires visual reference rows')\n"
          "            from research.h3_personal.fast_ref2va import configure_schedulers\n"
          "            configure_schedulers(self.scheduler, self.audio_scheduler, batch.num_inference_steps, device)\n")
    stage.write_text(text.replace(marker,hook+marker))
    pipeline=source/'fastvideo/pipelines/basic/minimax_h3/minimax_h3_pipeline.py'
    pipeline_original=sha(pipeline);text=pipeline.read_text()
    anchor='        if h3_encoder_split_enabled(fastvideo_args):\n            try:\n'
    hook=("        if getattr(self, '_smd_continuation_warm_faulted', False):\n"
          "            raise RuntimeError('continuation warm forward faulted; pipeline reuse forbidden')\n"
          "        if __import__('os').environ.get('H3_SMD_REF2VA') == '1':\n"
          "            if h3_encoder_split_enabled(fastvideo_args): raise ValueError('SMD warm cache requires ordinary SP2 topology')\n"
          "            from research.h3_personal.fast_ref2va import condition_key, warm_forward, continuation_warm_enabled, get_condition_cache\n"
          "            from research.h3_personal.conversation_control import is_control\n"
          "            if is_control(): return warm_forward(self, batch, fastvideo_args, None)\n"
          "            key = condition_key(batch)\n"
          "            stage = self._stage_name_mapping.get('conditioning_stage')\n"
          "            cache = get_condition_cache(stage, key)\n"
          "            if self._denoise_stages_ready and cache is None:\n"
          "                from research.h3_personal.conditioning_experiment import refresh_visual_condition\n"
          "                refresh_visual_condition(self, batch, fastvideo_args)\n"
          "                cache = get_condition_cache(stage, key)\n"
          "            if self._denoise_stages_ready and cache is not None and cache[0] == key:\n"
          "                continuation = continuation_warm_enabled(batch)\n"
          "                try: return warm_forward(self, batch, fastvideo_args, cache)\n"
          "                except BaseException:\n"
          "                    if continuation:\n"
          "                        self._smd_continuation_warm_faulted = True\n"
          "                    else:\n"
          "                        self._release_all_lazy_modules()\n"
          "                    raise\n"
          "            from research.h3_personal.conditioning_experiment import read_policy\n"
          "            if self._denoise_stages_ready and read_policy() is not None:\n"
          "                raise RuntimeError('experimental warm conditioning cache miss; reload forbidden')\n"
          "            batch.extra['_smd_condition_key'] = key\n")
    if text.count(anchor)!=1:raise ValueError('pipeline cache source anchor differs')
    text=text.replace(anchor,hook+anchor)
    anchor='            self._apply_inference_compile(tuple(name for name in loaded if name in _DENOISE_MODULE_NAMES))\n'
    hook=("            if __import__('os').environ.get('H3_SMD_REF2VA') == '1':\n"
          "                from research.h3_personal.fast_ref2va import load_deferred_adapter\n"
          "                load_deferred_adapter(self)\n")
    if text.count(anchor)!=1:raise ValueError('denoise-load adapter hook differs')
    text=text.replace(anchor,hook+anchor)
    release='        self._release_text_encoder()\n        self._load_denoise_modules(fastvideo_args)\n'
    retain=('        from research.h3_personal.conditioning_experiment import retain_visual_conditioner\n'
            '        if not retain_visual_conditioner(self):\n'
            '            self._release_text_encoder()\n'
            '        self._load_denoise_modules(fastvideo_args)\n')
    if text.count(release)!=1:raise ValueError('cold conditioner release anchor differs')
    pipeline.write_text(text.replace(release,retain))
    conditioning=source/'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_conditioning.py'
    condition_original=sha(conditioning);text=conditioning.read_text()
    anchor='        batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = text_token_tags\n'
    hook=("        if __import__('os').environ.get('H3_SMD_REF2VA') == '1':\n"
          "            from research.h3_personal.fast_ref2va import store_condition\n"
          "            store_condition(self, batch, fastvideo_args)\n")
    if text.count(anchor)!=1:raise ValueError('conditioning cache source anchor differs')
    text=text.replace(anchor,anchor+hook)
    memo='            return self._ref2va_memo.get_or_compute(key, lambda: self._present_ref2va(batch, references, device))\n'
    memo_hook=('            from research.h3_personal.ordered_qwen_tokens import memo_presentation\n'
               '            return memo_presentation(self, batch, key, lambda: self._present_ref2va(batch, references, device))\n')
    if text.count(memo)!=1:raise ValueError('ordered Qwen native memo anchor differs')
    text=text.replace(memo,memo_hook)
    admission='            video_block_token_counts,\n        )\n        return self._encode_tokens(\n'
    admission_hook=('            video_block_token_counts,\n        )\n'
        '        from research.h3_personal.ordered_qwen_tokens import capture_qwen_presentation\n'
        '        capture_qwen_presentation(self, batch, image_grid_thw, int(self.processor.image_processor.merge_size),\n'
        '                                  image_token_counts, token_ids, token_tags)\n'
        '        return self._encode_tokens(\n')
    if text.count(admission)!=1:raise ValueError('ordered Qwen pre-encode anchor differs')
    conditioning.write_text(text.replace(admission,admission_hook))
    input_stage=source/'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_input_preparation.py'
    input_original=sha(input_stage);text=input_stage.read_text()
    anchor='        batch.references = [prepare_reference(reference, num_frames, target_sample_rate) for reference in references]\n'
    hook=("        from research.h3_personal.conditioning_experiment import snapshot_sources\n"
          "        snapshot_sources(batch)\n"
          "        if __import__('os').environ.get('H3_REF2VA_RUNTIME_POLICY') == 'resident-visual-cached':\n"
          "            from research.h3_personal.ref2va_runtime_options import prepare_images_cached\n"
          "            batch.references = prepare_images_cached(self, references, num_frames, target_sample_rate, prepare_reference)\n"
          "        else:\n"+'    '+anchor)
    if text.count(anchor)!=1:raise ValueError('input preparation cache anchor differs')
    input_stage.write_text(text.replace(anchor,hook))
    return dict(input_original_sha256=input_original,input_patched_sha256=sha(input_stage),loader=report,schedule_original_sha256=original,schedule_patched_sha256=sha(stage),
        pipeline_original_sha256=pipeline_original,pipeline_patched_sha256=sha(pipeline),
        conditioning_original_sha256=condition_original,conditioning_patched_sha256=sha(conditioning))
