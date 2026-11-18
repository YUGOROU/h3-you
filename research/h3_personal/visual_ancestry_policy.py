"""Strict opt-in manifest/environment gate for bounded ancestry comparisons."""
ENV = 'H3_VISUAL_ANCESTRY_DIAGNOSTIC'
VALUE = 'bounded-ancestry-comparison-v1'
MANIFEST_KEY = 'visual_ancestry_diagnostic'
COMPARATOR = 'epoch-idle-every3-v1'
CANDIDATE = 'conversation-tail-every3-v1'


def parse_visual_ancestry_policy(manifest, envvalue):
    """Return an explicitly paired comparison arm, or None when both absent."""
    if type(manifest) is not dict:
        raise ValueError('manifest must be a top-level dict')
    if MANIFEST_KEY not in manifest and envvalue is None:
        return None
    section = manifest.get(MANIFEST_KEY)
    if (type(envvalue) is not str or envvalue != VALUE or
            type(section) is not dict or set(section) != {'schema', 'policy'} or
            type(section['schema']) is not str or section['schema'] != VALUE or
            type(section['policy']) is not str or
            section['policy'] not in (COMPARATOR, CANDIDATE)):
        raise ValueError('invalid visual ancestry diagnostic opt-in')
    return section['policy']


def validate_selection(value, sequence, anchor_sha256=None):
    """Bounded exported scalar proof; native backend independently checks both ranks."""
    import re
    fields={'policy','sequence','kind','anchor_sha256','selected_sha256','anchor_sequence'}
    if (type(sequence) is not int or not 0 <= sequence < 2542
            or type(value) is not dict or set(value) != fields
            or value.get('policy') != CANDIDATE
            or type(value.get('sequence')) is not int or value['sequence'] != sequence
            or value.get('kind') != ('unconditioned' if sequence == 0 else
                                    'anchor' if sequence % 3 == 0 else 'latest')):
        raise ValueError('invalid ancestry selection binding')
    for key in ('anchor_sha256','selected_sha256'):
        digest=value[key]
        if digest is not None and (type(digest) is not str or not re.fullmatch('[a-f0-9]{64}',digest)):
            raise ValueError('invalid ancestry selection digest')
    if sequence == 0:
        if any(value[key] is not None for key in ('anchor_sha256','selected_sha256','anchor_sequence')):
            raise ValueError('initial ancestry must be unconditioned')
    elif (type(value['anchor_sequence']) is not int or value['anchor_sequence'] != 1
          or value['anchor_sha256'] is None or value['selected_sha256'] is None
          or (sequence == 1 and value['anchor_sha256'] != value['selected_sha256'])
          or (sequence > 1 and value['anchor_sha256'] != anchor_sha256)
          or (sequence % 3 == 0 and value['selected_sha256'] != value['anchor_sha256'])):
        raise ValueError('immutable first-tail ancestry mismatch')
    return dict(value)
