"""Bounded, metadata-only projection of already-checked native rank receipts.

This is not execution authentication: receipt SHA binds the native source JSON;
no pixels, paths, prompts, tensors or raw token IDs cross this log boundary.
"""
import json
import re

SCHEMA = 'h3-ordered-metadata-v1'
MAX_BYTES = 24576


def _sha(value):
    if type(value) is not str or re.fullmatch('[a-f0-9]{64}', value) is None:
        raise ValueError('ordered metadata SHA256 required')
    return value


def _integer(value):
    if type(value) is not int or not 0 <= value <= 1 << 50:
        raise ValueError('bounded ordered metadata integer required')
    return value


def _literal(expected):
    def check(value):
        if type(value) is not type(expected) or value != expected:
            raise ValueError('ordered metadata literal mismatch')
        return value
    return check


def _one_of(*choices):
    def check(value):
        if type(value) is not str or value not in choices:
            raise ValueError('ordered metadata enum mismatch')
        return value
    return check


def _session(value):
    if value is not None and (type(value) is not str or re.fullmatch('[a-zA-Z0-9_-]{1,128}', value) is None):
        raise ValueError('bounded opaque native owner required')
    return value


def _copy(value, spec):
    if type(spec) is dict:
        if type(value) is not dict or set(value) != set(spec):
            raise ValueError('exact allowlisted ordered metadata fields required')
        return {key: _copy(value[key], child) for key, child in spec.items()}
    if type(spec) is tuple:
        child, low, high = spec
        if type(value) is not list or not low <= len(value) <= high:
            raise ValueError('bounded ordered metadata list required')
        return [_copy(item, child) for item in value]
    return spec(value)


def _optional(check):
    return lambda value: None if value is None else check(value)


_SIZE = (_integer, 2, 2)
_IMAGE = dict(role=_one_of('person', 'surroundings'),
    source_sha256=_sha, original_size=_SIZE, original_aspect=_SIZE, stored_size=_SIZE,
    oriented_size=_SIZE, prepared_size=_SIZE, orientation_policy=_literal('png-exif-identity-only-v1'),
    exif_present=lambda v: v if type(v) is bool else _literal(False)(v),
    exif_orientation=_optional(_literal(1)), resize_policy=_literal('shortedge-lanczos-grid32-v1'),
    crop=_literal('none'), latent_frames=_integer, latent_height=_integer, latent_width=_integer,
    latent_rows=_integer)
_SEGMENT = dict(media_type=_one_of('image', 'audio', 'video'),
    rows=_integer, latent_shape=(_integer, 3, 3))
_PROVENANCE = dict(kind=_one_of('cold-no-guide', 'continuation-owner-window'),
    owner_session_id=_session, window_index=_optional(_integer),
    target_latent_shape=_optional(lambda v: _copy(v, (_integer, 5, 5))),
    guide_video_frames=_integer, guide_audio_latents=_integer)
_PACKED = dict(schema=_literal('h3-ordered-packed-rows-v1'),
    admission_kind=_literal('candidate-engineering-not-native-capability'),
    **{k: _integer for k in ('max_packed_rows', 'sequence_length', 'text_rows', 'reference_rows',
        'target_video_rows', 'target_audio_rows', 'video_index_rows', 'audio_index_rows')})
_GEOMETRY = dict(schema=_literal('h3-ordered-native-geometry-v1'), identity_sha256=_sha,
    candidate_visual_acceptance=_literal('unreviewed'), images=(_IMAGE, 1, 2), R=_integer, T=_integer,
    admission_kind=_literal('candidate-engineering-not-native-capability'),
    max_reference_rows=_integer, max_tokens=_integer,
    conditioning_and_vae_rows=_literal('independent-native-quantities'),
    reference_rows_source=_literal('native-layout-image-segments'), guide_segments_excluded=_integer,
    guide_layout=dict(image_rows=_integer, guide_video_rows=_integer, guide_audio_rows=_integer,
        provenance=_PROVENANCE), packed_rows=_PACKED, native_layout_reference_segments=(_SEGMENT, 1, 4))
_MEMORY = dict(measurement=_literal('worker-current-all-models-not-reference-delta'),
    allocated_bytes=_integer, reserved_bytes=_integer)
_BINDING = dict(target_epoch=_integer, intent_epoch=_integer, prompt_sha256=_sha)


def validate_metadata(value, *, expected_identity, visual_binding, sequence):
    """Validate detached allowlisted projection against expected spec/READY content."""
    from .ordered_references import content_identity, content_sha256, expected_prepared_sizes
    from .ordered_qwen_tokens import validate_proof, ordered_token_policy
    _sha(expected_identity)
    if type(sequence) is not int or not 0 <= sequence < 2542:
        raise ValueError('bounded ordered metadata window required')
    if (type(value) is not dict or set(value) != {'schema', 'receipt_sha256', 'visual_conditioning',
            'ordered_references', 'ordered_both_ranks', 'ordered_worker_memory'}
            or value['schema'] != SCHEMA or value['ordered_both_ranks'] is not True):
        raise ValueError('exact both-rank ordered metadata required')
    if len(json.dumps(value, allow_nan=False, separators=(',', ':')).encode()) > MAX_BYTES:
        raise ValueError('ordered metadata byte bound')
    _sha(value['receipt_sha256'])
    binding = _copy(value['visual_conditioning'], _BINDING)
    if binding != {k: visual_binding[k] for k in _BINDING} or not 0 <= binding['intent_epoch'] <= binding['target_epoch'] <= 1000:
        raise ValueError('ordered metadata visual prompt/epoch drift')
    ordered = value['ordered_references']
    if type(ordered) is not dict or set(ordered) != {'identity_sha256', 'candidate_binding_sha256', 'visual_acceptance', 'geometry', 'qwen_tokens'}:
        raise ValueError('exact ordered reference metadata required')
    if ordered['identity_sha256'] != expected_identity or ordered['visual_acceptance'] != 'unreviewed':
        raise ValueError('ordered metadata candidate identity drift')
    _sha(ordered['candidate_binding_sha256'])
    geometry = _copy(ordered['geometry'], _GEOMETRY)
    if geometry['identity_sha256'] != expected_identity:
        raise ValueError('ordered geometry identity drift')
    # Reconstruct only portable content identity, never access these dummy paths.
    references = [dict(role=image['role'], path='/metadata/' + str(i) + '.png', sha256=image['source_sha256'],
        width=image['original_size'][0], height=image['original_size'][1]) for i, image in enumerate(geometry['images'])]
    from .ordered_references import SCHEMA as PAIR_SCHEMA, SINGLE_SCHEMA, reference_roles
    # The schema is chosen by explicit cardinality; the roles it declares must
    # then match the actual images, so a single-reference schema can never be
    # silently widened or narrowed into the other one.
    config = dict(schema=SINGLE_SCHEMA if len(references)==1 else PAIR_SCHEMA, references=references)
    if tuple(r['role'] for r in references) != reference_roles(config['schema']):
        raise ValueError('ordered geometry roles must match explicit schema')
    image_count=len(references)
    qwen = ordered['qwen_tokens']
    token_policy = qwen.get('policy') if type(qwen) is dict else None
    edge = token_policy.get('short_edge') if type(token_policy) is dict else None
    context = dict(ordered_references=config, identity=content_identity(config, short_edge=edge),
        identity_sha256=content_sha256(config, short_edge=edge))
    if context['identity_sha256'] != expected_identity:
        raise ValueError('ordered image sources do not match READY identity')
    qwen = validate_proof(qwen, context, edge, binding['prompt_sha256'], expected_T=geometry['T'])
    policy = ordered_token_policy(edge)
    from .conditioning_experiment import ordered_admission_bounds
    bounds = ordered_admission_bounds(edge, references=image_count)
    prepared = expected_prepared_sizes(config, short_edge=edge)['references']
    segments = geometry['native_layout_reference_segments']
    total = 0
    for image, size, segment in zip(geometry['images'], prepared, segments):
        dims = [image[k] for k in ('latent_frames', 'latent_height', 'latent_width')]
        rows = dims[0] * (dims[1] // 2) * (dims[2] // 2)
        if (dims[0] != 1 or not dims[1] or not dims[2] or dims[1] % 2 or dims[2] % 2
                or any(image[key] != image['original_size'] for key in ('original_aspect', 'stored_size', 'oriented_size'))
                or not image['exif_present'] and image['exif_orientation'] is not None
                or image['prepared_size'] != [size['width'], size['height']]
                or image['latent_rows'] != rows or segment != dict(media_type='image', rows=rows, latent_shape=dims)):
            raise ValueError('ordered metadata actual image/latent geometry drift')
        total += rows
    packed = geometry['packed_rows']; guide = geometry['guide_layout']; provenance = guide['provenance']
    vr = sum(s['rows'] for s in segments if s['media_type'] != 'audio')
    ar = sum(s['rows'] for s in segments if s['media_type'] == 'audio')
    if (len(segments) not in (image_count, image_count+2)
            or [s['media_type'] for s in segments[image_count:]] != (['audio', 'video'] if len(segments) == image_count+2 else [])
            or geometry['R'] != total or not 0 < total <= bounds['max_reference_rows']
            or geometry['max_reference_rows'] != bounds['max_reference_rows'] or geometry['max_tokens'] != bounds['max_tokens']
            or not 0 < geometry['T'] <= bounds['max_tokens']
            or geometry['guide_segments_excluded'] != len(segments) - image_count
            or guide['image_rows'] != total or guide['guide_video_rows'] != vr - total or guide['guide_audio_rows'] != ar
            or packed['max_packed_rows'] != policy['max_packed_rows'] or packed['text_rows'] != geometry['T']
            or packed['reference_rows'] != vr + ar or packed['video_index_rows'] != vr + packed['target_video_rows']
            or packed['audio_index_rows'] != ar + packed['target_audio_rows']
            or packed['sequence_length'] != geometry['T'] + vr + ar + packed['target_video_rows'] + packed['target_audio_rows']
            or not 0 < packed['sequence_length'] <= policy['max_packed_rows']
            or provenance['kind'] == 'continuation-owner-window' and provenance['window_index'] != sequence):
        raise ValueError('ordered metadata independent R/T/packed/guide admission drift')
    memory = _copy(value['ordered_worker_memory'], (_MEMORY, 2, 2))
    if any(row['allocated_bytes'] > row['reserved_bytes'] for row in memory):
        raise ValueError('ordered worker memory counters drift')
    return dict(schema=SCHEMA, receipt_sha256=value['receipt_sha256'], visual_conditioning=binding,
        ordered_references=dict(identity_sha256=expected_identity, candidate_binding_sha256=ordered['candidate_binding_sha256'],
            visual_acceptance='unreviewed', geometry=geometry, qwen_tokens=qwen),
        ordered_both_ranks=True, ordered_worker_memory=memory)


def project_metadata(receipt, *, expected_identity):
    """Project only checked proof fields; a candidate may never omit attachment."""
    evidence = receipt.get('native_evidence')
    keys = ('ordered_references', 'ordered_both_ranks', 'ordered_worker_memory')
    if expected_identity is None:
        # Ordered candidates always carry the attachment, so a noncandidate
        # receipt may legitimately omit it and stays untouched.
        if isinstance(evidence, dict) and any(k in evidence for k in keys):
            raise ValueError('unexpected ordered native metadata')
        return None
    if type(evidence) is not dict:
        raise ValueError('candidate receipt requires ordered native evidence')
    value = dict(schema=SCHEMA, receipt_sha256=receipt['receipt_sha256'],
        visual_conditioning={k: receipt['visual_conditioning'][k] for k in _BINDING},
        **{k: evidence.get(k) for k in keys})
    return validate_metadata(value, expected_identity=expected_identity,
        visual_binding=receipt['visual_conditioning'], sequence=receipt['sequence'])
