"""Finite listening-template contract; no environment/device/model side effects.

This module alone does not enable native conditioning. Consumers must establish
cold-bank authority before READY and verify actual both-rank prompt evidence.
Numeric motion proposals are intentionally not consumed here.
"""
from dataclasses import dataclass
import hashlib
import json

from .conversation_conditioning import LISTENING_PROMPT, SPEECH_PROMPT

POLICY = 'h3-listening-templates-v1'
MANIFEST_KEY = 'listening_template_selection'
ENV = 'H3_REF2VA_LISTENING_TEMPLATE_POLICY'
CADENCE_FRAMES = 16 * 24
# Tuples/strings only: callers cannot mutate the process-wide registry.
REGISTRY = (
    ('baseline', LISTENING_PROMPT),
    ('soft-attention', LISTENING_PROMPT + ' A very subtle relaxed head inclination and natural blink; keep the original identity, clothing, background, framing and lighting fixed.'),
    ('gentle-attention', LISTENING_PROMPT + ' A very subtle shift of attentive gaze and relaxed posture, with closed lips; keep the original identity, clothing, background, framing and lighting fixed.'),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def prompt_digest(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


REGISTRY_DIGEST = _digest(dict(policy=POLICY, cadence_frames=CADENCE_FRAMES,
                               templates=REGISTRY, speech_prompt=SPEECH_PROMPT))


@dataclass(frozen=True)
class ListeningTemplatePolicy:
    version: str = POLICY
    registry_sha256: str = REGISTRY_DIGEST

    def __post_init__(self):
        if (type(self.version) is not str or self.version != POLICY
                or type(self.registry_sha256) is not str
                or self.registry_sha256 != REGISTRY_DIGEST):
            raise ValueError('unknown immutable listening registry')

    def declaration(self):
        return dict(policy=self.version, registry_sha256=self.registry_sha256)


def validate_authority(policy, roles_declaration, environment_value):
    """Require explicit config AND matching roles declaration AND environment.

    Ambient environment alone is never permission to expand a cold bank.
    Default-off roles manifests are deliberately not extended by this module.
    """
    if policy is None:
        if roles_declaration is not None or environment_value is not None:
            raise ValueError('unpaired listening template authority')
        return None
    if type(policy) is not ListeningTemplatePolicy:
        raise ValueError('explicit immutable listening template policy required')
    if roles_declaration != policy.declaration() or environment_value != policy.version:
        raise ValueError('listening template authority mismatch')
    return policy


def select(policy, role, committed_frame, sequence):
    if type(policy) is not ListeningTemplatePolicy:
        raise ValueError('explicit immutable listening template policy required')
    if role not in ('listening', 'speech'):
        raise ValueError('unknown role')
    if type(sequence) is not int or not 0 <= sequence < 144:
        raise ValueError('bounded absolute sequence required')
    expected = 0 if sequence == 0 else 124 + (sequence - 1) * 102
    if type(committed_frame) is not int or committed_frame != expected:
        raise ValueError('stale absolute committed clock')
    template_id, prompt = (REGISTRY[(committed_frame // CADENCE_FRAMES) % len(REGISTRY)]
                           if role == 'listening' else ('speech', SPEECH_PROMPT))
    return dict(**policy.declaration(), template_id=template_id,
                prompt_sha256=prompt_digest(prompt), role=role,
                committed_frame=committed_frame, sequence=sequence)


def selected_prompt(record, policy, role, committed_frame, sequence):
    expected = select(policy, role, committed_frame, sequence)
    if type(record) is not dict or record != expected:
        raise ValueError('unknown/missing/stale template selection')
    return SPEECH_PROMPT if role == 'speech' else dict(REGISTRY)[record['template_id']]


def verify_rank_evidence(selection, rows, *, policy, role, committed_frame, sequence):
    """Verify independently measured selected prompt + warm hit on EACH rank.

    This is not a substitute for the existing native clock/profile/owner proof.
    Native producers must derive actual_prompt_sha256 from batch.prompt, not
    echo a caller's selection. Require exactly two ordered rank records.
    """
    prompt = selected_prompt(selection, policy, role, committed_frame, sequence)
    if type(rows) is not list or len(rows) != 2:
        raise ValueError('both-rank template evidence required')
    for rank, row in enumerate(rows):
        if (type(row) is not dict
                or set(row) != {'rank', 'selection', 'actual_prompt_sha256', 'cache_hit'}
                or type(row['rank']) is not int or row['rank'] != rank
                or row['selection'] != selection
                or row['actual_prompt_sha256'] != prompt_digest(prompt)
                or row['cache_hit'] is not True):
            raise ValueError('actual both-rank template/cache evidence mismatch')
    return dict(selection)
