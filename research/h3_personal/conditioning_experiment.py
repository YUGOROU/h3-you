"""Isolated short-edge conditioning; no model loading, tensor slicing or GPU probes."""
import contextvars
from copy import copy
import hashlib
import json
import os
import re
import secrets
from pathlib import Path

EXPERIMENT_ENV = 'H3_REF2VA_CONDITIONING_EXPERIMENT_FILE'
ACCEPTED_PROFILE_ENV = 'H3_REF2VA_ACCEPTED_PROFILE_FILE'
SCHEMA = 'h3-conditioning-experiment-v1'
RESIZE_POLICY = 'shortedge-lanczos-grid32-v1'
# Native portrait 1737x3088 single-reference probe expectations, not fabricated measurements.
PORTRAIT_EXPECTED_RT = {1024: {'R': 1824, 'T': 1867}, 512: {'R': 448, 'T': 491}}
_edge = contextvars.ContextVar('h3_reference_short_edge', default=None)


def validate_manifest(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'short_edges', 'active_short_edge'}:
        raise ValueError('invalid conditioning experiment fields')
    if value['schema'] != SCHEMA or value['short_edges'] != [1024, 512] or any(type(x) is not int for x in value['short_edges']):
        raise ValueError('invalid conditioning experiment policy')
    if type(value['active_short_edge']) is not int or value['active_short_edge'] not in (1024, 512):
        raise ValueError('invalid active short edge')
    return dict(value, short_edges=list(value['short_edges']))


def read_policy():
    path = os.environ.get(EXPERIMENT_ENV)
    return validate_manifest(json.loads(Path(path).read_text())) if path else None


def reference_short_edge(native=2048):
    override = _edge.get()
    policy = read_policy()
    return override if override is not None else policy['active_short_edge'] if policy else native


def reference_canvas_multiple(native=16):
    return 32 if read_policy() is not None else native


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_accepted_profile():
    path = os.environ.get(ACCEPTED_PROFILE_ENV)
    if not path:
        return None
    from .accepted_profiles import AcceptedQualityProfile
    profile = AcceptedQualityProfile.from_record(json.loads(Path(path).read_text()))
    policy = read_policy()
    if policy != profile.legacy_conditioning_manifest():
        raise ValueError('accepted profile conditioning policy mismatch')
    # Both banks are computed under request-local edge overrides.
    from dataclasses import replace
    edge = reference_short_edge()
    from .conversation_conditioning import effective_profile
    return effective_profile(replace(profile, name='reference%s' % edge))


def validate_accepted_batch(batch, stage=None):
    profile = read_accepted_profile()
    if profile is None:
        return
    from .conversation_conditioning import current_role, conditioning_sha as prompt_sha, conditioning_prompt
    role = current_role()
    if role is not None:
        # The published profile is the CURRENT source's role variant; the cold
        # bank precomputes other roles under bind_role, so compare role-locally.
        from dataclasses import replace
        profile = replace(profile, prompt_sha256=prompt_sha(role))
    from .conversation_conditioning import read_ordered_context
    candidate = read_ordered_context()
    actual_sources = tuple(_source_binding(batch))
    if candidate is not None:
        selected = tuple(r['sha256'] for r in candidate['ordered_references']['references'])
        if actual_sources != selected or batch.prompt != conditioning_prompt(role):
            raise ValueError('candidate ordered sources/actual role prompt mismatch')
        # Validate accepted BASE geometry/processor/window separately: original
        # source acceptance is not transferred to these newly selected images.
        actual_sources = profile.ordered_source_sha256
    profile.validate_binding(
        ordered_source_sha256=actual_sources,
        prompt_sha256=hashlib.sha256(batch.prompt.encode()).hexdigest(),
        processor_sha256=processor_digest(stage) if stage is not None else profile.processor_sha256,
        geometry=(batch.width, batch.height))
    if batch.num_frames != profile.window or batch.fps != profile.fps:
        raise ValueError('accepted profile window/fps mismatch')


def policy_key():
    policy = read_policy()
    profile = read_accepted_profile()
    from .conversation_conditioning import read_ordered_context
    candidate = read_ordered_context()
    return None if policy is None else digest(dict(policy, active_short_edge=reference_short_edge(), resize_policy=RESIZE_POLICY,
        **({'accepted_profile_digest': profile.digest} if profile is not None else {}),
        **({'candidate_reference_sha256': candidate['identity_sha256'],
            'candidate_visual_acceptance': 'unreviewed'} if candidate is not None else {})))


def snapshot_sources(batch):
    if read_policy() is None:
        return
    from research.h3_personal.ref2va_runtime import sha
    refs = list(batch.references or [])
    if not refs or any(r.media_type != 'image' or not isinstance(r.source, (str, Path)) for r in refs):
        raise ValueError('experiment requires original ordered still sources')
    batch.extra['_h3_conditioning_sources'] = tuple(copy(r) for r in refs)
    batch.extra['_h3_conditioning_source_sha'] = tuple(sha(r.source) for r in refs)
    validate_accepted_batch(batch)


def processor_digest(stage):
    processor = stage.processor
    def config(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        value = getattr(obj, 'config', None)
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        raise ValueError('loaded processor exact configuration unavailable')
    values = {name: config(getattr(processor, name)) for name in ('image_processor', 'video_processor') if hasattr(processor, name)}
    if not values:
        raise ValueError('loaded vision processor configuration unavailable')
    if getattr(processor, 'config', None) is not None or hasattr(processor, 'to_dict'):
        values['processor'] = config(processor)
    identity = digest(values)
    profile = read_accepted_profile()
    if profile is not None and identity != profile.processor_sha256:
        raise ValueError('accepted processor binding mismatch')
    return identity


def _cpu(batch):
    return ([t.detach().cpu().clone() for t in batch.prompt_embeds],
            batch.extra['minimax_h3_text_token_tags'].detach().cpu().clone())


def store_condition(stage, batch, fastvideo_args=None):
    if getattr(stage, '_h3_conditioning_precomputing', False):
        return
    from research.h3_personal.fast_ref2va import condition_key
    policy = read_policy()
    if policy is None:
        return
    from .conversation_conditioning import current_role
    if current_role() is not None:
        return _store_role_conditions(stage, batch, fastvideo_args)
    validate_accepted_batch(batch, stage)
    identity = processor_digest(stage)
    old = getattr(stage, '_h3_conditioning_processor_digest', identity)
    if old != identity:
        raise RuntimeError('loaded processor identity drift')
    stage._h3_conditioning_processor_digest = identity
    sources = batch.extra.get('_h3_conditioning_sources')
    if not sources:
        raise RuntimeError('original reference snapshot missing')
    cache = {}
    evidence = {}
    active = policy['active_short_edge']
    cache[condition_key(batch)] = (condition_key(batch), *_cpu(batch))
    evidence[active] = _sizes(batch)
    stage._h3_conditioning_precomputing = True
    try:
        for edge in policy['short_edges']:
            if edge == active:
                continue
            clone = copy(batch)
            clone.extra = dict(batch.extra)
            clone.prompt_embeds = []
            token = _edge.set(edge)
            try:
                rate = getattr(batch, 'audio_sample_rate', None)
                rate = rate if type(rate) is int and rate > 0 else 32000
                clone.references = [prepare_reference(copy(r), batch.num_frames, rate) for r in sources]
                clone.extra['_smd_condition_key'] = condition_key(clone)
                result = stage.forward(clone, fastvideo_args)
                if result is not None:
                    clone = result
                key = condition_key(clone)
                cache[key] = (key, *_cpu(clone))
                evidence[edge] = _sizes(clone)
            finally:
                _edge.reset(token)
    finally:
        stage._h3_conditioning_precomputing = False
    stage._h3_conditioning_cache = cache
    stage._h3_conditioning_sizes = evidence
    batch.extra['_h3_conditioning_precomputed_edges'] = sorted(evidence, reverse=True)


def _sizes(batch):
    refs = batch.references
    if not isinstance(refs, list) or not refs:
        raise RuntimeError('actual prepared still references unavailable')
    sizes = []
    for ref in refs:
        image = getattr(ref, 'image', None)
        size = getattr(image, 'size', None)
        if (getattr(ref, 'media_type', None) != 'image' or
                not isinstance(size, tuple) or len(size) != 2 or
                any(type(n) is not int or n <= 0 for n in size)):
            raise RuntimeError('invalid actual prepared still geometry')
        sizes.append(list(size))
    return sizes


def _source_binding(batch):
    from research.h3_personal.ref2va_runtime import sha
    sources = batch.extra.get('_h3_conditioning_sources')
    hashes = batch.extra.get('_h3_conditioning_source_sha')
    if not sources or not hashes or tuple(sha(r.source) for r in sources) != hashes:
        raise RuntimeError('conditioning geometry source binding drift')
    return hashes


def capture_prepared_geometry(batch):
    """Capture actual native prepared images before latent preparation clears them."""
    if read_policy() is None:
        return
    validate_accepted_batch(batch)
    sizes = tuple(tuple(size) for size in _sizes(batch))
    sources = _source_binding(batch)
    if len(sizes) != len(sources):
        raise RuntimeError('conditioning geometry source count mismatch')
    binding = (policy_key(), sources, sizes)
    batch.extra['_h3_conditioning_prepared_geometry'] = (binding, digest(binding))
    from .conversation_conditioning import read_ordered_context
    if read_ordered_context() is not None:
        # Native VAE mutates these SAME descriptor objects with actual latent
        # geometry before clearing pixels/references; no image tensor copy.
        batch.extra['_h3_ordered_prepared_references'] = tuple(batch.references)


def _receipt_sizes(batch):
    if batch.references is not None:
        return _sizes(batch)
    snapshot = batch.extra.get('_h3_conditioning_prepared_geometry')
    if not isinstance(snapshot, tuple) or len(snapshot) != 2:
        raise RuntimeError('bound prepared geometry unavailable')
    binding, checksum = snapshot
    if (not isinstance(binding, tuple) or len(binding) != 3 or
            digest(binding) != checksum or binding[0] != policy_key() or
            binding[1] != _source_binding(batch)):
        raise RuntimeError('conditioning geometry snapshot binding drift')
    return [list(size) for size in binding[2]]


def ordered_admission_bounds(edge, *, references=2):
    """Finite candidate engineering admission, NOT a native model capacity.

    Supported stills (explicit schema cardinality), supported maxaspect4 and
    round32 bound the reference-row envelope. Qwen total tokens have a SEPARATE
    engineering envelope; no T-R text count or T>=R relation is assumed. Both
    returned maxima are inclusive.
    """
    if type(edge) is not int or edge not in (1024, 512):
        raise ValueError('explicit supported candidate shortedge required')
    if type(references) is not int or references not in (1, 2):
        raise ValueError('explicit supported candidate reference count required')
    rows = references * (4 * edge // 32) * (edge // 32)
    return dict(max_reference_rows=rows, max_tokens=rows + 511)


def ordered_layout_evidence(batch):
    """Validate actual fixed images and owner-bound guide suffix, without constants.

    Pinned build_layout appends one video-with-audio reference LAST; native
    packing serializes that reference as audio THEN video. Cold requests have
    no guide. Shape/coverage comes from the actual target canvas and token.
    """
    from .continuation_plan import video_t, Window
    layout=batch.extra.get('minimax_h3_layout')
    segments=getattr(layout,'reference_segments',None)
    from .conversation_conditioning import read_ordered_context
    candidate=read_ordered_context()
    # An explicit candidate context carries its own portable identity. A falsy
    # context is the legacy two-reference default; a malformed non-empty one is
    # rejected rather than silently downgraded to the legacy cardinality.
    if candidate is None:
        image_count=2
    else:
        if (type(candidate) is not dict or type(candidate.get('identity')) is not dict
                or type(candidate['identity'].get('references')) is not list):
            raise RuntimeError('explicit ordered candidate context identity required')
        image_count=len(candidate['identity']['references'])
    if (not isinstance(segments,tuple) or len(segments)<image_count
            or any(not isinstance(s,tuple) or len(s)!=3 for s in segments)
            or [s[0] for s in segments[:image_count]]!=['image']*image_count):
        raise RuntimeError('actual native image segments must match explicit candidate identity')
    image_rows=0
    for kind,count,dims in segments[:image_count]:
        if (not isinstance(dims,tuple) or len(dims)!=3 or dims[0]!=1
                or any(type(n) is not int or n<=0 for n in dims)
                or dims[1]%2 or dims[2]%2 or type(count) is not int
                or count!=dims[0]*(dims[1]//2)*(dims[2]//2)):
            raise RuntimeError('actual native image segment shape/rows mismatch')
        image_rows+=count
    ctx=batch.extra.get('_h3_continuation')
    guide_rows=audio_rows=0
    provenance=dict(kind='cold-no-guide',owner_session_id=None,window_index=None,
                    target_latent_shape=None,guide_video_frames=0,guide_audio_latents=0)
    if ctx is not None:
        try:
            token=ctx['token'];owner=ctx['owner'];manifest=ctx['manifest'];window=token.window
            shape=batch.raw_latent_shape
            if (len(shape)!=5 or tuple(shape[:3])!=(1,24,video_t(124)) or batch.num_frames!=124
                    or any(type(n) is not int or n<=0 or n%2 for n in shape[3:])):
                raise ValueError('target latent canvas')
            height,width=shape[3:];per=(height//2)*(width//2)
            if (not isinstance(window,Window) or window.frames!=124
                    or getattr(layout,'latent_height',None)!=height or getattr(layout,'latent_width',None)!=width
                    or getattr(layout,'num_video_latent_frames',None)!=video_t(124)):
                raise ValueError('native layout/window canvas')
            if owner.session_id!=manifest['session_id'] or owner._per_frame!=per:
                raise ValueError('owner session/canvas')
            global_token=ctx.get('global_token')
            if global_token is None:
                if owner.pending_window is not token or token.session_id!=manifest['session_id'] or token.index!=manifest['index']:
                    raise ValueError('owner/window token')
            else:
                conversation=ctx['conversation_owner']
                if (global_token.model is not token or owner.token is not global_token
                        or conversation.active_native is not global_token
                        or conversation.conversation_id!=token.session_id):
                    raise ValueError('global owner token')
                control=manifest.get('streaming_audio',manifest.get('conversation',{}))
                sequence=control.get('sequence',control.get('index',token.index))
                # Streaming opens a one-window source Session each time: model
                # token.index is source-local (0), not the conversation sequence.
                if (type(sequence) is not int or sequence!=conversation._sequence
                        or control.get('id')!=token.session_id
                        or global_token.source_id!=manifest['session_id']):
                    raise ValueError('global window identity')
            if type(window.guide_video_t) is not int or type(window.guide_audio_t) is not int:
                raise ValueError('guide plan')
            expected=[]
            if window.overlap:
                guide_rows=window.guide_video_t*per;audio_rows=window.guide_audio_t*2
                if (token.guide_video is None or token.guide_audio is None
                        or tuple(token.guide_video.shape)!=(guide_rows,96)
                        or tuple(token.guide_audio.shape)!=(2,window.guide_audio_t,32)):
                    raise ValueError('guide tensor coverage')
                expected=[('audio',audio_rows,(0,0,0)),('video',guide_rows,(window.guide_video_t,height,width))]
            elif token.guide_video is not None or token.guide_audio is not None:
                raise ValueError('unexpected guide tensors')
            if list(segments[image_count:])!=expected:
                raise ValueError('guide segment count/order/shape/rows')
            provenance=dict(kind='continuation-owner-window',owner_session_id=owner.session_id,
                window_index=(token.index if global_token is None else sequence),target_latent_shape=list(shape),
                guide_video_frames=window.guide_video_t,guide_audio_latents=window.guide_audio_t)
        except (KeyError,AttributeError,TypeError,ValueError) as error:
            raise RuntimeError('actual continuation guide owner/window/canvas mismatch: '+str(error)) from error
    elif len(segments)!=image_count:
        raise RuntimeError('cold candidate cannot carry guide segments')
    if (type(getattr(layout,'num_condition_video_rows',None)) is not int
            or layout.num_condition_video_rows!=image_rows+guide_rows
            or type(getattr(layout,'num_condition_audio_rows',None)) is not int
            or layout.num_condition_audio_rows!=audio_rows):
        raise RuntimeError('actual layout condition row totals mismatch')
    return dict(image_rows=image_rows,guide_video_rows=guide_rows,guide_audio_rows=audio_rows,
                provenance=provenance)


def ordered_packed_evidence(batch):
    """Actual native sequence decomposition and independent pre-DiT cap."""
    from .ordered_qwen_tokens import ordered_token_policy
    layout=batch.extra.get('minimax_h3_layout')
    policy=ordered_token_policy(reference_short_edge())
    try:
        counts={name:getattr(layout,name).numel() for name in ('text_indices','video_indices','audio_indices')}
        if any(type(n) is not int or n<0 for n in counts.values()):raise ValueError('strict index counts')
        scalars=[getattr(layout,name) for name in ('sequence_length','num_condition_video_rows',
            'num_condition_audio_rows','num_video_latent_frames','latent_height','latent_width','num_audio_latents')]
        if any(type(n) is not int or n<0 for n in scalars):raise ValueError('strict layout integer fields')
        length,cv,ca,t,h,w,a=scalars
        if min(t,h,w,a)<=0 or h%2 or w%2:raise ValueError('actual target geometry')
        reference_rows=sum(s[1] for s in layout.reference_segments)
        if reference_rows!=cv+ca:raise ValueError('reference segment coverage')
        target_video=t*(h//2)*(w//2);target_audio=2*a
        if counts['video_indices']!=cv+target_video or counts['audio_indices']!=ca+target_audio:
            raise ValueError('target index coverage')
        actual=counts['text_indices']+reference_rows+target_video+target_audio
        if length!=actual or sum(counts.values())!=length or not 0<length<=policy['max_packed_rows']:
            raise ValueError('packed sequence sum/cap')
    except (KeyError,AttributeError,TypeError,ValueError) as error:
        raise RuntimeError('candidate actual packed admission mismatch: '+str(error)) from error
    return dict(schema='h3-ordered-packed-rows-v1',admission_kind='candidate-engineering-not-native-capability',
        max_packed_rows=policy['max_packed_rows'],sequence_length=length,text_rows=counts['text_indices'],
        reference_rows=reference_rows,target_video_rows=target_video,target_audio_rows=target_audio,
        video_index_rows=counts['video_indices'],audio_index_rows=counts['audio_indices'])


def ordered_geometry_evidence(batch, *, R, T):
    """Actual native per-image VAE geometry, never inferred from PNG sizes."""
    from .conversation_conditioning import read_ordered_context
    from .ordered_references import expected_prepared_sizes
    candidate = read_ordered_context()
    if candidate is None:
        return None
    edge = reference_short_edge()
    expected = expected_prepared_sizes(candidate['ordered_references'], short_edge=edge)['references']
    sizes = _receipt_sizes(batch)
    from .ordered_references import verify_orientation
    orientation=verify_orientation(candidate['ordered_references'])
    actual = batch.extra.get('_h3_ordered_prepared_references')
    image_count=len(expected)
    if not isinstance(actual, tuple) or len(actual) != image_count or len(sizes) != image_count:
        raise RuntimeError('actual native prepared reference cardinality must match candidate')
    layout = batch.extra.get('minimax_h3_layout')
    guide_evidence=ordered_layout_evidence(batch)
    packed_evidence=ordered_packed_evidence(batch)
    if packed_evidence['text_rows']!=T:
        raise RuntimeError('candidate actual packed text rows mismatch')
    segments=layout.reference_segments
    rows = []
    for source, resized, size, ref, segment, oriented in zip(candidate['ordered_references']['references'], expected, sizes, actual, segments[:image_count],orientation['images']):
        dims = [getattr(ref, name, None) for name in ('num_latent_frames', 'latent_height', 'latent_width')]
        if (getattr(ref, 'media_type', None) != 'image' or dims[0] != 1
                or any(type(n) is not int or n <= 0 for n in dims)
                or dims[1] % 2 or dims[2] % 2
                or size != [resized['width'], resized['height']]):
            raise RuntimeError('native per-image prepared/latent geometry mismatch')
        count = segment[1]
        if (type(count) is not int or count <= 0 or segment[2] != tuple(dims)
                or count != dims[0] * (dims[1] // 2) * (dims[2] // 2)):
            raise RuntimeError('actual layout image rows/latent descriptor mismatch')
        rows.append(dict(role=source['role'], source_sha256=source['sha256'],
            original_size=[source['width'], source['height']], original_aspect=[source['width'], source['height']],
            stored_size=oriented['stored_size'],oriented_size=oriented['oriented_size'],
            orientation_policy=orientation['orientation_policy'],exif_present=oriented['exif_present'],
            exif_orientation=oriented['exif_orientation'],
            prepared_size=size, resize_policy=RESIZE_POLICY, crop='none',
            latent_frames=dims[0], latent_height=dims[1], latent_width=dims[2], latent_rows=count))
    # Explicit ENGINEERING admission, not an upstream/model supported capacity:
    # explicit schema cardinality, allowed maxaspect4, round32, active shortedge;
    # native image rows use actual DiT patch2 above. Qwen T has an INDEPENDENT
    # engineering total-token envelope with512 headroom, not T-R text count.
    bounds = ordered_admission_bounds(edge, references=image_count)
    actual_rows = sum(r['latent_rows'] for r in rows)
    if (type(R) is not int or type(T) is not int or R != actual_rows
            or not 0 < R <= bounds['max_reference_rows'] or not 0 < T <= bounds['max_tokens']):
        raise RuntimeError('candidate actual R/T admission exceeded; no truncation/fallback')
    return dict(schema='h3-ordered-native-geometry-v1', identity_sha256=candidate['identity_sha256'],
        candidate_visual_acceptance='unreviewed', images=rows, R=R, T=T,
        admission_kind='candidate-engineering-not-native-capability', **bounds,
        conditioning_and_vae_rows='independent-native-quantities',
        reference_rows_source='native-layout-image-segments', guide_segments_excluded=len(segments)-image_count,
        guide_layout=guide_evidence, packed_rows=packed_evidence,
        native_layout_reference_segments=[dict(media_type=s[0], rows=s[1], latent_shape=list(s[2]))
                                          for s in segments])


def export_receipt(stage=None, batch=None):
    if batch is None:
        batch, stage = stage, None
    policy = read_policy()
    if policy is None:
        return None
    layout = batch.extra.get('minimax_h3_layout')
    rows = getattr(layout, 'num_condition_video_rows', None)
    ctx = batch.extra.get('_h3_continuation')
    from .conversation_conditioning import read_ordered_context
    if read_ordered_context() is not None:
        rows = ordered_layout_evidence(batch)['image_rows']
    elif rows is not None and ctx is not None:
        rows -= ctx['token'].window.guide_video_t * 112
    indices = getattr(layout, 'text_indices', None)
    text = int(indices.numel()) if indices is not None else None
    return dict(schema=SCHEMA, policy=policy, resize_policy=RESIZE_POLICY, prepared_sizes=_receipt_sizes(batch), R=rows, T=text,
                cache_hit=bool(batch.extra.get('_h3_conditioning_cache_hit', False)),
                precomputed_edges=sorted(getattr(stage, '_h3_conditioning_sizes', {}), reverse=True),
                processor_config_sha256=getattr(stage, '_h3_conditioning_processor_digest', None))


def conditioning_evidence(batch):
    return batch.extra.get('_h3_conditioning_evidence') or export_receipt(batch=batch)


def write_evidence(pipeline, batch):
    if read_policy() is None:
        return None
    import torch
    stage = pipeline._stage_name_mapping['conditioning_stage']
    validate_accepted_batch(batch, stage)
    report = export_receipt(stage, batch)
    if report['R'] is None or report['T'] is None:
        raise RuntimeError('conditioning experiment actual R/T unavailable')
    report['R'] = int(report['R'])
    edge = report['policy']['active_short_edge']
    expected_size = {1024: [1024, 1824], 512: [512, 896]}[edge]
    from .conversation_conditioning import (current_role, conditioning_sha as prompt_sha,
                                            prompt_sha as base_prompt_sha, role_profile_digest)
    role = current_role()
    expected = PORTRAIT_EXPECTED_RT[edge]
    from .conversation_conditioning import read_ordered_context
    candidate = read_ordered_context()
    if candidate is not None:
        from .candidate_references import build_candidate_binding, candidate_binding_sha256
        candidate_binding = build_candidate_binding(candidate['ordered_references'], read_accepted_profile())
        actual_geometry = ordered_geometry_evidence(batch, R=report['R'], T=report['T'])
        from .ordered_qwen_tokens import validate_proof,validate_cached_tags,PROOF_KEY
        qwen=validate_proof(batch.extra.get(PROOF_KEY),candidate,edge,
                            hashlib.sha256(batch.prompt.encode()).hexdigest(),expected_T=report['T'])
        # Native latent preparation consumes/pops the raw tag tensor. Verify
        # the actual packed text span used by DiT, not the removed extra key.
        layout=batch.extra['minimax_h3_layout']
        validate_cached_tags(qwen,layout.token_tags[layout.text_indices])
        report.update(qwen_tokens=qwen,ordered_reference_sha256=candidate['identity_sha256'],
                      candidate_binding_sha256=candidate_binding_sha256(candidate_binding),
                      candidate_reference_binding=candidate_binding,
                      candidate_memory=dict(measurement='worker-current-all-models-not-reference-delta',
                          allocated_bytes=int(torch.cuda.memory_allocated()),
                          reserved_bytes=int(torch.cuda.memory_reserved())),
                      candidate_visual_acceptance='unreviewed', ordered_geometry=actual_geometry)
        stage._h3_ordered_actual_geometry = actual_geometry
        expected_size = [image['prepared_size'] for image in actual_geometry['images']]
        expected = dict(R=actual_geometry['R'], T=actual_geometry['T'])
    if role is not None:
        from .fast_ref2va import condition_key
        key = condition_key(batch)
        proof = getattr(stage, '_h3_role_condition_proofs', {}).get(key)
        cached = getattr(stage, '_h3_conditioning_cache', {}).get(key)
        profile = read_accepted_profile()
        native_t = int(cached[2].shape[0]) if cached is not None and cached[0] == key else None
        if (profile is None or not proof or proof['role'] != role or proof['profile_digest'] != profile.digest
                or proof['prompt_sha256'] != prompt_sha(role)
                or report['T'] != proof['T'] or report['T'] != native_t
                or (candidate is None and not report['R'] < report['T'] < report['R'] + 512)):
            raise RuntimeError('conversation role cache evidence mismatch')
        if candidate is not None and (proof.get('ordered_reference_sha256') != candidate['identity_sha256']
                or proof.get('qwen_tokens')!=report['qwen_tokens']):
            raise RuntimeError('candidate role cache context/token proof drift')
        expected = dict(R=expected['R'], T=proof['T'])
        report.update(role=role, role_proof=proof, cache_bank_roles=['speech', 'listening'])
    if ({k: report[k] for k in ('R', 'T')} != expected
            or report['prepared_sizes'] != (expected_size if candidate is not None else [expected_size])
            or report['precomputed_edges'] != [1024, 512]
            or report['cache_hit'] is not True
            or not report['processor_config_sha256']):
        raise RuntimeError('conditioning experiment pre-DiT evidence mismatch')
    identity = tuple(id(p) for p in pipeline.get_module('transformer').parameters())
    previous = getattr(pipeline, '_h3_conditioning_parameter_identity', identity)
    if not identity or previous != identity:
        raise RuntimeError('experimental warm parameter identity drift')
    pipeline._h3_conditioning_parameter_identity = identity
    pipeline._h3_conditioning_cache_hits = getattr(pipeline, '_h3_conditioning_cache_hits', 0) + 1
    report.update(parameter_identity_preserved=True,
                  experimental_cache_hits=pipeline._h3_conditioning_cache_hits)
    path = os.environ.get('H3_SMD_CONTINUATION_WINDOW_FILE')
    manifest = json.loads(Path(path).read_text()) if path else {}
    report.update(session_id=None, window_index=None)
    if manifest.get('mode') == 'latent-tail-v1':
        session, index = manifest.get('session_id'), manifest.get('index')
        if (not isinstance(session, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', session)
                or type(index) is not int or index < 0):
            raise RuntimeError('invalid conditioning receipt session/window')
        ctx = batch.extra.get('_h3_continuation')
        if manifest.get('conversation') is not None or manifest.get('streaming_audio') is not None:
            from .accepted_profiles import AcceptedQualityProfile
            from dataclasses import replace
            if manifest.get('streaming_audio') is not None:
                from .streaming_native_stages import validate_manifest
                streaming = validate_manifest(manifest)
                conversation = dict(streaming, source_epoch=streaming['sequence'])
                if 'visual_conditioning' in streaming:
                    from .conversation_conditioning import current_record
                    visual=streaming['visual_conditioning']
                    if current_record()!=visual or proof['prompt_sha256']!=visual['prompt_sha256']:
                        raise RuntimeError('native conditioning visual request drift')
                    report['visual_conditioning']={k:visual[k] for k in ('target_epoch','intent_epoch','prompt_sha256')}
                    report['visual_condition_refresh']=batch.extra.get('_h3_visual_condition_refresh',dict(encoded=False,duration_seconds=0.0))
                    encoder=getattr(stage,'conditioner',None)
                    report['visual_encoder_retained']=bool(getattr(stage,'_h3_visual_conditioner_retained',False)
                        and encoder is not None and pipeline.get_module('text_encoder') is encoder
                        and tuple(id(p) for p in encoder.parameters())==stage._h3_visual_conditioner_identity)
                report.update(streaming_sequence=streaming['sequence'],
                              logical_source_id=streaming['logical_source_id'],
                              logical_window_index=streaming['logical_index'])
            else:
                from .conversation_stages import validate_manifest
                conversation = validate_manifest(manifest)
            owner = ctx.get('conversation_owner') if ctx is not None else None
            token = ctx.get('global_token') if ctx is not None else None
            profile_path = os.environ.get(ACCEPTED_PROFILE_ENV)
            base_profile = (AcceptedQualityProfile.from_record(json.loads(Path(profile_path).read_text()))
                            if profile_path else None)
            if (owner is None or token is None or base_profile is None
                    or owner.conversation_id != conversation['id']
                    or owner.active_native is not token or owner.pending_window is not ctx['token']
                    or token.model is not ctx['token']
                    or token.model.session_id != conversation['id']
                    or token.local.session_id != session or token.source_id != session
                    or index != token.local.index or index != token.model.index
                    or ctx['owner'].session_id != session
                    or ctx['owner'].owner is not owner or ctx['owner'].token is not token
                    or owner.source_session.session_id != session
                    or owner._index != index or owner._epoch != token.epoch
                    or conversation['source_epoch'] != token.epoch - 1
                    or owner._binding != (token.profile, token.canvas)
                    or token.profile != conversation['profile_sha256']
                    or token.profile not in {role_profile_digest(base_profile, r)
                                            for r in ('speech', 'listening')}
                    or role != conversation['role']
                    or ctx.get('manifest') != manifest):
                raise RuntimeError('conditioning receipt conversation context mismatch')
            report.update(conversation_id=conversation['id'], source_epoch=conversation['source_epoch'],
                          current_role=role, fixed_base_profile_digest=token.profile)
        elif (ctx is None or ctx['owner'].session_id != session
                or ctx['token'].session_id != session or ctx['token'].index != index):
            raise RuntimeError('conditioning receipt context mismatch')
        report.update(session_id=session, window_index=index)
    report.update(worker_rank=torch.distributed.get_rank(),
                  denoise_stages_ready=bool(pipeline._denoise_stages_ready))
    report['denoise_load_counters'] = {name: value for name, value in vars(pipeline).items()
        if ('load' in name and ('count' in name or 'counter' in name)) and type(value) is int}
    batch.extra['_h3_conditioning_evidence'] = report
    folder = Path(os.environ['H3_REF2VA_COVERAGE_DIR'])
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ('conditioning-experiment-rank-%s.json' % report['worker_rank'])
    if report['session_id'] is not None:
        immutable = folder / ('conditioning-experiment-%s-window-%s-rank-%s.json' %
                              (report['session_id'], report['window_index'], report['worker_rank']))
        _publish_private(immutable, report, immutable=True)
    _publish_private(target, report)
    return report


def _publish_private(target, report, immutable=False):
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise RuntimeError('conditioning receipt symlink rejected')
    if immutable and target.exists():
        raise RuntimeError('immutable conditioning receipt already exists')
    tmp = target.with_name('.conditioning-' + secrets.token_hex(16) + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if target.is_symlink() or (immutable and target.exists()):
            raise RuntimeError('conditioning receipt publication conflict')
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def experimental_resolver(width, height):
    if width <= 0 or height <= 0:
        raise ValueError('reference image must have positive size')
    if width > 4 * height or height > 4 * width:
        raise ValueError('reference image must be within 1:4 and 4:1')
    scale = reference_short_edge() / min(width, height)
    return tuple(max(32, round(size * scale / 32) * 32) for size in (height, width))


def prepare_reference(reference, num_frames, sample_rate, native_prepare=None):
    """Run pinned native image code with request-local resolver globals only."""
    from types import FunctionType
    if native_prepare is None:
        from fastvideo.pipelines.basic.minimax_h3.reference import prepare_reference as native_prepare
    if read_policy() is None:
        return native_prepare(reference, num_frames, sample_rate)
    from research.h3_personal.fast_continuation import REFERENCE_SHA
    module_name = 'fastvideo.pipelines.basic.minimax_h3.reference'
    if (not isinstance(native_prepare, FunctionType)
            or native_prepare.__module__ != module_name
            or native_prepare.__globals__.get('__name__') != module_name
            or native_prepare.__name__ != 'prepare_reference'):
        raise ValueError('requires pinned native prepare_reference function')
    source = Path(native_prepare.__globals__['__file__'])
    if (source.resolve() != Path(native_prepare.__code__.co_filename).resolve()
            or hashlib.sha256(source.read_bytes()).hexdigest() != REFERENCE_SHA):
        raise ValueError('requires pinned native reference source SHA')
    if reference.media_type != 'image' or not isinstance(reference.source, (str, Path)):
        raise ValueError('experiment requires original ordered still sources')
    clone = FunctionType(native_prepare.__code__,
                         dict(native_prepare.__globals__, resolve_reference_image_size=experimental_resolver),
                         native_prepare.__name__, native_prepare.__defaults__, native_prepare.__closure__)
    clone.__kwdefaults__ = native_prepare.__kwdefaults__
    return clone(reference, num_frames, sample_rate)


def _store_role_conditions(stage, batch, fastvideo_args):
    from .conversation_conditioning import (bind_role, current_role, conditioning_prompt as role_prompt,
                                            conditioning_sha as prompt_sha, read_ordered_context)
    from .fast_ref2va import condition_key
    validate_accepted_batch(batch, stage)
    identity = processor_digest(stage)
    if getattr(stage, '_h3_conditioning_processor_digest', identity) != identity:
        raise RuntimeError('loaded processor identity drift')
    if getattr(stage, '_h3_conditioning_cache', None):
        raise RuntimeError('conversation cold bank already populated')
    sources = batch.extra.get('_h3_conditioning_sources')
    if not sources:
        raise RuntimeError('original reference snapshot missing')
    cache, proofs, sizes = {}, {}, {}
    stage._h3_conditioning_precomputing = True
    try:
        for edge in read_policy()['short_edges']:
            token = _edge.set(edge)
            try:
                for role in ('speech', 'listening'):
                    with bind_role(role):
                        clone = copy(batch)
                        clone.extra = dict(batch.extra)
                        clone.prompt = role_prompt(role)
                        clone.prompt_embeds = []
                        rate = getattr(batch, 'audio_sample_rate', None)
                        rate = rate if type(rate) is int and rate > 0 else 32000
                        clone.references = [prepare_reference(copy(r), batch.num_frames, rate) for r in sources]
                        key = condition_key(clone)
                        clone.extra['_smd_condition_key'] = key
                        result = stage.forward(clone, fastvideo_args)
                        if result is not None:
                            clone = result
                        validate_accepted_batch(clone, stage)
                        if condition_key(clone) != key:
                            raise RuntimeError('role conditioning key drift')
                        cache[key] = (key, *_cpu(clone))
                        sizes[edge] = _sizes(clone)
                        candidate = read_ordered_context()
                        proofs[key] = dict(role=role, prompt_sha256=prompt_sha(role),
                            profile_digest=read_accepted_profile().digest,
                            T=int(clone.extra['minimax_h3_text_token_tags'].shape[0]))
                        if candidate is not None:
                            # No VAE geometry exists in this Qwen-only cold bank.
                            # Bound its actual tag count by the separate engineering
                            # envelope; exact R is checked later against native VAE.
                            bounds = ordered_admission_bounds(
                                edge, references=len(candidate['identity']['references']))
                            if not 0 < proofs[key]['T'] <= bounds['max_tokens']:
                                raise RuntimeError('candidate cold token admission exceeded; no truncation')
                            from .ordered_qwen_tokens import validate_proof,validate_cached_tags,PROOF_KEY
                            qwen=validate_proof(clone.extra.get(PROOF_KEY),candidate,edge,
                                                prompt_sha(role),expected_T=proofs[key]['T'])
                            validate_cached_tags(qwen,clone.extra['minimax_h3_text_token_tags'])
                            proofs[key].update(ordered_reference_sha256=candidate['identity_sha256'],
                                               candidate_visual_acceptance='unreviewed',qwen_tokens=qwen)
            finally:
                _edge.reset(token)
    finally:
        stage._h3_conditioning_precomputing = False
    stage._h3_conditioning_processor_digest = identity
    stage._h3_conditioning_cache = cache
    stage._h3_conditioning_sizes = sizes
    stage._h3_role_condition_proofs = proofs
    stage._h3_visual_conditioner_identity = tuple(id(p) for p in stage.conditioner.parameters()) if getattr(stage, 'conditioner', None) is not None else None
    batch.extra['_h3_conditioning_precomputed_edges'] = [1024, 512]


def refresh_visual_condition(pipeline, batch, args):
    """Encode one new visual condition with the ALREADY-owned encoder only.

    No lazy-loader, DiT parking/reload or geometry/model fallback. Cold legacy
    bank stays immutable. Two accepted turns x two roles cap new entries at four.
    The native conditioner encodes the complete ordered reference+text again;
    contextual reference embeddings are never spliced onto unrelated text.
    """
    from .conversation_conditioning import visual_enabled, current_record, current_role, conditioning_sha
    from .fast_ref2va import condition_key
    if not visual_enabled() or current_record()['visual_intent'] is None:
        return
    batch.extra['_h3_visual_condition_refresh']=dict(encoded=False,duration_seconds=0.0)
    stage = pipeline._stage_name_mapping['conditioning_stage']
    key = condition_key(batch)
    cache = getattr(stage, '_h3_conditioning_cache', {})
    if key in cache:
        return
    if read_policy()['active_short_edge'] != 1024:
        raise RuntimeError('visual continuation requires original reference1024')
    identity = processor_digest(stage)
    if identity != getattr(stage, '_h3_conditioning_processor_digest', None):
        raise RuntimeError('visual processor identity drift')
    encoder = getattr(stage, 'conditioner', None)
    if encoder is None or pipeline.get_module('text_encoder') is not encoder:
        raise RuntimeError('visual conditioner unavailable; implicit reload forbidden')
    parameters = tuple(id(p) for p in encoder.parameters())
    if not parameters or parameters != getattr(stage, '_h3_visual_conditioner_identity', None):
        raise RuntimeError('visual conditioner identity drift')
    additions = getattr(stage, '_h3_visual_condition_keys', set())
    if len(additions) >= 4:
        raise RuntimeError('finite visual condition bank exhausted')
    validate_accepted_batch(batch, stage)
    sources = batch.extra.get('_h3_conditioning_sources')
    if not sources:
        raise RuntimeError('original visual reference snapshot missing')
    clone = copy(batch)
    clone.extra = dict(batch.extra)
    clone.prompt_embeds = []
    rate = getattr(batch, 'audio_sample_rate', None)
    rate = rate if type(rate) is int and rate > 0 else 32000
    clone.references = [prepare_reference(copy(r), batch.num_frames, rate) for r in sources]
    clone.extra['_smd_condition_key'] = key
    import time
    started=time.monotonic()
    stage._h3_conditioning_precomputing = True
    try:
        result = stage.forward(clone, args)
        if result is not None:clone = result
        validate_accepted_batch(clone, stage)
        if (condition_key(clone) != key or tuple(id(p) for p in encoder.parameters()) != parameters
                or stage.conditioner is not encoder or pipeline.get_module('text_encoder') is not encoder):
            raise RuntimeError('visual conditioning binding/encoder drift')
        tokens = int(clone.extra['minimax_h3_text_token_tags'].shape[0])
        from .conversation_conditioning import read_ordered_context
        candidate = read_ordered_context()
        if candidate is None:
            reference_rows = PORTRAIT_EXPECTED_RT[1024]['R']
        else:
            actual = getattr(stage, '_h3_ordered_actual_geometry', None)
            if actual is None or actual['identity_sha256'] != candidate['identity_sha256']:
                raise RuntimeError('candidate visual refresh requires actual cold native geometry')
            reference_rows = actual['R']
        if (candidate is None and not reference_rows < tokens < reference_rows + 512
                or candidate is not None and not 0 < tokens <= ordered_admission_bounds(
                    1024, references=len(candidate['identity']['references']))['max_tokens']):
            raise RuntimeError('visual native token budget exceeded; no truncation')
        stored = (key, *_cpu(clone))
        proof = dict(role=current_role(), prompt_sha256=conditioning_sha(current_role()),
            profile_digest=read_accepted_profile().digest, T=tokens)
        if candidate is not None:
            from .ordered_qwen_tokens import validate_proof,validate_cached_tags,PROOF_KEY
            qwen=validate_proof(clone.extra.get(PROOF_KEY),candidate,1024,
                                proof['prompt_sha256'],expected_T=tokens)
            validate_cached_tags(qwen,clone.extra['minimax_h3_text_token_tags'])
            proof.update(ordered_reference_sha256=candidate['identity_sha256'],
                         candidate_visual_acceptance='unreviewed',qwen_tokens=qwen)
        stage._h3_role_condition_proofs[key] = proof
        cache[key] = stored
        stage._h3_visual_condition_keys = additions | {key}
        batch.extra['_h3_visual_condition_refresh']=dict(encoded=True,duration_seconds=time.monotonic()-started)
    finally:
        stage._h3_conditioning_precomputing = False


def retain_visual_conditioner(pipeline):
    """Opt-in retention at the ONE cold encode→denoise release call site.

    Sharded native encoders cannot use the pipeline's CPU-move path and would
    otherwise be destroyed. Keep the exact already-owned object where it is;
    no device/dtype/sharding change or second model. Ordinary release and error
    cleanup methods are not overridden. Memory fit remains an actual-GPU gate.
    """
    from .conversation_conditioning import visual_enabled
    if not visual_enabled():return False
    stage=pipeline._stage_name_mapping['conditioning_stage']
    encoder=getattr(stage,'conditioner',None)
    if encoder is None or pipeline.get_module('text_encoder') is not encoder:
        raise RuntimeError('cannot retain an unowned visual conditioner')
    identity=tuple(id(p) for p in encoder.parameters())
    if not identity or identity!=getattr(stage,'_h3_visual_conditioner_identity',None):
        raise RuntimeError('cold visual conditioner identity drift')
    stage._h3_visual_conditioner_retained=True
    return True
