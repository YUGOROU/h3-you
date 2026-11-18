"""Approved placement gate for the paired native TTS L4 source Job.

USER decision 2026-10-29: VoxCPM2 native TTS carries **no region/territory
restriction** (neither license nor user preference), unlike H3. Therefore the
L4 source of the pair may be accepted without country evidence, and the
unverifiable region must be recorded as unknown rather than invented from a
device label, network group or Job metadata.

H3 placement is NOT covered by this gate and must not be relaxed by it: the
original H200x2 territory/license contract keeps its own gates.

The gate is TTS-source specific. It accepts only:
  - the sanitized pre-create plan passed by `run_h3_network_pair`
  - the raw control-plane Job JSON of the exact owned source Job
and returns exactly True on success (the owner's contract), raising otherwise.
"""
import json
import re
from pathlib import Path

SCHEMA = 'h3-pair-tts-placement-v1'
POLICY = ('voxcpm2 native TTS has no region/territory restriction '
          '(USER 2026-10-29); region evidence is not required and must stay '
          'unverified rather than inferred')
TTS_FLAVOR = 'l4x1'
PLAN_KEYS = {'phase', 'namespace', 'flavor', 'image', 'group'}
SERVICE_KEYS = {'group', 'alias', 'port', 'certificate_sha256', 'server_job_id'}


def _group(value):
    return type(value) is str and re.fullmatch('[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?', value)


def _record(path, row):
    path = Path(path)
    previous = []
    if path.exists():
        previous = json.loads(path.read_text())
        if type(previous) is not list:
            raise ValueError('placement attestation shape')
    previous.append(row)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(previous, indent=1))
    temporary.chmod(0o600)
    temporary.replace(path)


def _verdict(observed, spec, phase, region):
    service = spec.get('network_service')
    required = SERVICE_KEYS - {'server_job_id'} if phase == 'pre_create' else SERVICE_KEYS
    if (type(service) is not dict or set(service) != required or not _group(service.get('group'))
            or not re.fullmatch('[a-z0-9](?:[a-z0-9-]{0,32}[a-z0-9])?', str(service.get('alias')))
            or type(service.get('port')) is not int or not 1024 <= service['port'] <= 65535
            or type(service.get('certificate_sha256')) is not str or not re.fullmatch('[a-f0-9]{64}', service['certificate_sha256'])
            or (phase != 'pre_create' and (type(service.get('server_job_id')) is not str
                                            or not re.fullmatch('[a-f0-9]{24}', service['server_job_id'])))):
        raise ValueError('pair spec lacks one approved network group')
    return dict(schema=SCHEMA, phase=phase, policy=POLICY,
                tts_region_constraint=None, tts_region_evidence=region,
                h3_placement_covered=False, flavor=TTS_FLAVOR,
                image=observed.get('dockerImage') or observed.get('image'),
                group=service['group'], alias=service.get('alias'))


def approve_plan(plan, spec):
    """Sanitized pre-create plan: before artifacts, references or any Job."""
    if type(plan) is not dict or set(plan) != PLAN_KEYS:
        raise ValueError('sanitized pre-create plan required')
    if plan['phase'] != 'before_source_create' or plan['namespace'] != 'YUGOROU':
        raise ValueError('owned namespace/source phase required')
    if plan['flavor'] != TTS_FLAVOR:
        raise ValueError('TTS source gate covers only the L4 source, not H3')
    from research.h3_personal.director_tts_job import IMAGE
    if plan['image'] != IMAGE or not _group(plan['group']):
        raise ValueError('pinned image/fresh group required')
    if plan['group'] != spec['network_service']['group']:
        raise ValueError('plan/spec group mismatch')
    return _verdict(plan, spec, 'pre_create', 'unverified-region-unconstrained')


def approve_source(row, spec):
    """Raw running source Job: exact owned identity, no public media ports."""
    if type(spec.get('network_service')) is not dict or 'server_job_id' not in spec['network_service']:
        raise ValueError('running source requires the created source Job identity')
    service = spec['network_service']
    expected = dict(group=service['group'], aliases=[service['alias']])
    from research.h3_personal.director_tts_job import IMAGE
    if type(row) is not dict:
        raise ValueError('raw source Job metadata required')
    expose = row.get('expose') or {}
    if (row.get('id') != service['server_job_id'] or row.get('owner', {}).get('name') != 'YUGOROU'
            or row.get('flavor') != TTS_FLAVOR or row.get('dockerImage') != IMAGE
            or row.get('network') != expected or type(row.get('retry')) is not int or row['retry'] != 0
            or row.get('status', {}).get('stage') != 'RUNNING'
            or type(expose) is not dict or expose.get('portsPublic') or expose.get('ports')):
        raise ValueError('TTS source placement contract mismatch')
    region = row.get('provider') if type(row.get('provider')) is str else 'unverified-region-unconstrained'
    return _verdict(row, spec, 'running_source', region)


def gate(directory):
    """Return the owner callback, persisting each verdict in a private receipt."""
    path = Path(directory) / 'source-placement.json'
    def source_placement_gate(observed, spec):
        row = approve_plan(observed, spec) if type(observed) is dict and 'phase' in observed else approve_source(observed, spec)
        _record(path, row)
        return True
    return source_placement_gate
