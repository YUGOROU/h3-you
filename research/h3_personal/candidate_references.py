"""Portable unreviewed image binding to accepted BASE settings, not visual acceptance.

Pure metadata validation only: no file access, upload, pixel decoding or native
execution. Actual geometry receipts must be supplied by the native owner; their
validation is consistency checking, not independent evidence of execution.
"""
import hashlib
import json
import re

from .accepted_profiles import AcceptedQualityProfile
from .ordered_references import content_identity, expected_prepared_sizes

SCHEMA = 'h3-candidate-reference-binding-v1'
GEOMETRY_SCHEMA = 'h3-candidate-reference-geometry-v1'
BASE_FIELDS = ('name', 'geometry', 'fps', 'window', 'overlap', 'precision',
               'sampler', 'forwards', 'short_edge', 'processor_sha256', 'resize_policy')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(label + ' must be a strict positive integer')
    return value


def _snapshot(profile):
    return json.loads(_json({k: getattr(profile, k) for k in BASE_FIELDS}))


def build_candidate_binding(ordered_config, base_profile):
    """Bind content to BASE settings without inheriting old source/prompt/review.

    ordered_config is structurally checked, not read. Fresh file verification
    remains mandatory at the owning staging/native boundary.
    """
    if type(base_profile) is not AcceptedQualityProfile:
        raise ValueError('accepted BASE settings profile required')
    identity = content_identity(ordered_config, short_edge=base_profile.short_edge)
    record = dict(schema=SCHEMA, base_settings=_snapshot(base_profile),
                  ordered_content=identity, ordered_content_sha256=_digest(identity),
                  content_visual_acceptance='unreviewed')
    record['binding_sha256'] = _digest(record)
    return validate_candidate_binding(record)


def _identity_config(identity, short_edge):
    # Structural surrogate paths only, never accessed or published. This reuses
    # the exact generic ordered schema without introducing portable file paths.
    if type(identity) is not dict or type(identity.get('references')) is not list:
        raise ValueError('portable ordered content identity required')
    from .ordered_references import reference_roles
    refs = identity['references']
    roles = reference_roles(identity.get('schema'))
    if len(refs) != len(roles):
        raise ValueError('candidate cardinality must match explicit reference schema')
    config = dict(schema=identity['schema'], references=[])
    for i, r in enumerate(refs):
        if type(r) is not dict or set(r) != {'role', 'sha256', 'width', 'height'}:
            raise ValueError('exact portable reference fields required')
        config['references'].append(dict(r, path='/candidate-reference-validation/input-%d.png' % i))
    expected = content_identity(config, short_edge=short_edge)
    if _json(identity) != _json(expected):
        raise ValueError('noncanonical content/order/preprocessing identity')
    return config


def validate_candidate_binding(record):
    """Strict canonical parse/roundtrip returning detached JSON-native metadata."""
    fields = {'schema', 'base_settings', 'ordered_content', 'ordered_content_sha256',
              'content_visual_acceptance', 'binding_sha256'}
    if (type(record) is not dict or set(record) != fields or record['schema'] != SCHEMA
            or record['content_visual_acceptance'] != 'unreviewed'):
        raise ValueError('exact unreviewed candidate binding required')
    base = record['base_settings']
    if type(base) is not dict or set(base) != set(BASE_FIELDS):
        raise ValueError('base settings only; source/prompt/review acceptance forbidden')
    # Revalidate all accepted fixed base settings, but never copy their old
    # source, prompt, visual review or derived singleton R/T into this binding.
    profile = AcceptedQualityProfile(base['name'], ('0' * 64,), '0' * 64)
    expected_base = _snapshot(profile)
    if (any(type(base[k]) is not type(expected_base[k]) for k in BASE_FIELDS)
            or any(type(v) is not int for v in base['geometry'])
            or _json(base) != _json(expected_base)):
        raise ValueError('base settings drift or noncanonical types')
    _identity_config(record['ordered_content'], profile.short_edge)
    for key in ('ordered_content_sha256', 'binding_sha256'):
        if type(record[key]) is not str or re.fullmatch('[0-9a-f]{64}', record[key]) is None:
            raise ValueError('canonical SHA256 required')
    if record['ordered_content_sha256'] != _digest(record['ordered_content']):
        raise ValueError('content digest mismatch')
    unsigned = {k: v for k, v in record.items() if k != 'binding_sha256'}
    if record['binding_sha256'] != _digest(unsigned):
        raise ValueError('candidate binding digest mismatch')
    return json.loads(_json(record))


def candidate_binding_sha256(record):
    return validate_candidate_binding(record)['binding_sha256']


def validate_geometry_evidence(record, binding, *, max_reference_rows, max_tokens):
    """Check supplied ACTUAL per-image geometry/rows against content and bounds.

    Bounds must be explicitly chosen by the existing native admission owner,
    not guessed globally here. R is the sum of supplied actual latent rows; no
    latent row count is inferred from PNG dimensions or resize arithmetic.
    This function does not authenticate rank/provenance or compare both ranks.
    """
    binding = validate_candidate_binding(binding)
    _positive(max_reference_rows, 'reference row bound')
    _positive(max_tokens, 'token bound')
    if (type(record) is not dict
            or set(record) != {'schema', 'candidate_sha256', 'images', 'R', 'T'}
            or record['schema'] != GEOMETRY_SCHEMA
            or record['candidate_sha256'] != binding['binding_sha256']
            or type(record['images']) is not list
            or len(record['images']) != len(binding['ordered_content']['references'])):
        raise ValueError('bound candidate actual geometry cardinality required')
    identity = binding['ordered_content']
    config = _identity_config(identity, binding['base_settings']['short_edge'])
    prepared = expected_prepared_sizes(config, short_edge=binding['base_settings']['short_edge'])
    total = 0
    for image, source, size in zip(record['images'], identity['references'], prepared['references']):
        if type(image) is not dict or set(image) != {'role', 'sha256', 'original_size', 'prepared_size', 'latent_rows'}:
            raise ValueError('exact per-image geometry fields required')
        if image['role'] != source['role'] or image['sha256'] != source['sha256']:
            raise ValueError('reference content/order mismatch')
        for key, expected in (('original_size', [source['width'], source['height']]),
                              ('prepared_size', [size['width'], size['height']])):
            if (type(image[key]) is not list or len(image[key]) != 2
                    or any(type(v) is not int or v <= 0 for v in image[key])
                    or image[key] != expected):
                raise ValueError('actual original/prepared geometry mismatch or crop')
        total += _positive(image['latent_rows'], 'actual latent rows')
    r = _positive(record['R'], 'actual R')
    t = _positive(record['T'], 'actual T')
    # Qwen presentation tokens and VAE reference rows are separate counts.
    # Their observed singleton similarity is not a native T>=R contract.
    if r != total or r > max_reference_rows or t > max_tokens:
        raise ValueError('actual R/T mismatch or admission bound exceeded')
    return json.loads(_json(record))
