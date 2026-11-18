"""Explicit, immutable public role bindings. Listening quality is pending review."""
import contextvars
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from .ref2va_runtime import PROMPT

SPEECH_PROMPT = PROMPT
LISTENING_PROMPT = ('The person in reference image 1 calmly listening to the camera with closed lips. '
                    'Preserve the appearance of the person in the reference. Stable lighting and background.')
ROLES_ENV = 'H3_REF2VA_CONVERSATION_ROLES_FILE'
CURRENT_ROLE_ENV = 'H3_REF2VA_CONVERSATION_CURRENT_ROLE_FILE'
ORDERED_CONTEXT_ENV = 'H3_ORDERED_REFERENCE_CONTEXT_FILE'
ORDERED_CONTEXT_SCHEMA = 'h3-ordered-reference-context-v1'
SCHEMA = 'h3-conversation-roles-v1'
VISUAL_INTENT_SCHEMA = 'h3-visual-intent-v1'
_role = contextvars.ContextVar('h3_conditioning_role', default=None)


def read_ordered_context(*, check_files=True):
    """Owned candidate context, fresh by default; metadata-only prompt composition
    does not rehash megabytes recursively. Native admission/render always uses
    fresh checks, never transfers single-image visual acceptance.
    """
    if type(check_files) is not bool:
        raise ValueError('explicit candidate asset verification boolean required')
    path = os.environ.get(ORDERED_CONTEXT_ENV)
    if not path:
        return None
    from .ordered_references import (verify_local_files, validate_ordered_references,
                                     content_identity, content_sha256)
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 16384:
        raise ValueError('bounded private ordered-reference context required')
    value = json.loads(source.read_text())
    if (type(value) is not dict or set(value) != {'schema', 'ordered_references', 'identity',
            'identity_sha256', 'visual_acceptance'} or value['schema'] != ORDERED_CONTEXT_SCHEMA
            or value['visual_acceptance'] != 'unreviewed'):
        raise ValueError('exact unreviewed ordered-reference context required')
    selected = (verify_local_files(value['ordered_references']) if check_files
                else validate_ordered_references(value['ordered_references']))
    if (value['identity'] != content_identity(selected)
            or value['identity_sha256'] != content_sha256(selected)):
        raise ValueError('ordered-reference context content/policy drift')
    return dict(value, ordered_references=selected)


def ordered_context_record(selected):
    from .ordered_references import verify_local_files, content_identity, content_sha256
    selected = verify_local_files(selected)
    return dict(schema=ORDERED_CONTEXT_SCHEMA, ordered_references=selected,
                identity=content_identity(selected), identity_sha256=content_sha256(selected),
                visual_acceptance='unreviewed')


def _reference_guidance():
    context = read_ordered_context(check_files=False)
    if context is None:
        return ''
    from .ordered_references import role_prompt as reference_role_prompt
    return ' ' + reference_role_prompt(context['ordered_references'])


def is_declared_prompt(prompt):
    """Legacy prompts, or the exact currently owned versioned visual binding."""
    if type(prompt) is not str:return False
    if prompt in (SPEECH_PROMPT, LISTENING_PROMPT):
        return True
    if read_ordered_context(check_files=False) is not None and prompt in {role_prompt(r) for r in ('speech', 'listening')}:
        return True
    if not visual_enabled():
        return False
    return prompt in {conditioning_prompt(r) for r in ('speech', 'listening')}


def role_prompt(role):
    if role not in ('speech', 'listening'):
        raise ValueError('invalid conversation role')
    return (SPEECH_PROMPT if role == 'speech' else LISTENING_PROMPT) + _reference_guidance()


def prompt_sha(role):
    return hashlib.sha256(role_prompt(role).encode()).hexdigest()


def roles_manifest(*, visual_intents=False):
    if type(visual_intents) is not bool:
        raise ValueError('explicit visual role contract required')
    value = dict(schema=SCHEMA, prompt_sha256={r: prompt_sha(r) for r in ('speech', 'listening')})
    if visual_intents:
        value.update(schema='h3-conversation-roles-visual-v1', visual_intents=True)
    return value


def validate_manifest(value):
    enabled = type(value) is dict and value.get('schema') == 'h3-conversation-roles-visual-v1'
    expected = roles_manifest(visual_intents=enabled)
    if value != expected or (enabled and value.get('visual_intents') is not True):
        raise ValueError('invalid conversation roles manifest')
    return expected


def visual_enabled():
    path = os.environ.get(ROLES_ENV)
    return bool(path and validate_manifest(json.loads(Path(path).read_text())).get('visual_intents'))


def validate_visual_record(record):
    fields = {'role', 'prompt_sha256', 'visual_intent', 'intent_epoch', 'target_epoch'}
    if type(record) is not dict or set(record) != fields:
        raise ValueError('exact visual conditioning record required')
    role_prompt(record['role'])
    for name in ('intent_epoch', 'target_epoch'):
        if type(record[name]) is not int or not 0 <= record[name] <= 1000:
            raise ValueError('finite visual epoch required')
    active = record['visual_intent']
    if active is not None:
        active = validate_visual_intent(active)
        if active['instruction'] is None or not 0 < record['intent_epoch'] <= record['target_epoch']:
            raise ValueError('resolved current visual intent required')
    elif record['intent_epoch'] != 0:
        raise ValueError('unbound visual intent epoch')
    actual = hashlib.sha256(visual_prompt(record['role'], active).encode()).hexdigest()
    if record['prompt_sha256'] != actual:
        raise ValueError('visual prompt binding mismatch')
    return dict(record, visual_intent=active)


def current_record():
    path = os.environ.get(ROLES_ENV)
    if not path:
        return None
    policy = validate_manifest(json.loads(Path(path).read_text()))
    record = json.loads(Path(os.environ[CURRENT_ROLE_ENV]).read_text())
    if policy.get('visual_intents'):
        return validate_visual_record(record)
    if (type(record) is not dict or set(record) != {'role', 'prompt_sha256'}
            or record['prompt_sha256'] != prompt_sha(record['role'])):
        raise ValueError('invalid current role record')
    return record


def conditioning_prompt(role):
    record = current_record()
    return visual_prompt(role, None if record is None else record.get('visual_intent'))


def conditioning_sha(role):
    return hashlib.sha256(conditioning_prompt(role).encode()).hexdigest()


def current_role():
    record = current_record()
    return None if record is None else (_role.get() or record['role'])


def role_record(role, *, visual_intents=False, active=None, intent_epoch=0, target_epoch=0):
    if type(visual_intents) is not bool:raise ValueError('explicit visual role selection required')
    if not visual_intents:
        if active is not None or intent_epoch or target_epoch:
            raise ValueError('visual conditioning requires explicit opt-in')
        return dict(role=role, prompt_sha256=prompt_sha(role))
    return validate_visual_record(dict(role=role,
        prompt_sha256=hashlib.sha256(visual_prompt(role, active).encode()).hexdigest(),
        visual_intent=active, intent_epoch=intent_epoch, target_epoch=target_epoch))


@contextmanager
def bind_role(role):
    if current_role() is None:
        raise ValueError('roles not enabled')
    role_prompt(role)
    token = _role.set(role)
    try:
        yield
    finally:
        _role.reset(token)


def effective_profile(profile):
    role = current_role()
    if role is None:
        return profile
    # On-disk binding must itself be a declared public role, never arbitrary text.
    accepted = {prompt_sha(r) for r in ('speech', 'listening')}
    if visual_enabled():
        accepted.update(conditioning_sha(r) for r in ('speech', 'listening'))
    if profile.prompt_sha256 not in accepted:
        raise ValueError('undeclared accepted prompt')
    if _role.get() is None and profile.prompt_sha256 != conditioning_sha(role):
        raise ValueError('accepted/current role drift')
    return replace(profile, prompt_sha256=conditioning_sha(role))

def role_profile_digest(profile, role):
    """Stable legacy-normalized role identity, including the fixed PCM binding.

    Visual prompts have separate actual profile/cache/proof digests. Camera
    intent must not alter the conversation's original PCM/reference identity.
    """
    from dataclasses import replace
    role_prompt(role)
    baseline = replace(profile, prompt_sha256=prompt_sha(role)).digest
    context = read_ordered_context(check_files=False)
    if context is None:
        return baseline
    from .candidate_references import build_candidate_binding, candidate_binding_sha256
    binding = build_candidate_binding(context['ordered_references'], profile)
    return hashlib.sha256(json.dumps(dict(schema='h3-candidate-role-binding-v1',
        base_settings_profile_sha256=baseline, ordered_reference_sha256=context['identity_sha256'],
        candidate_binding_sha256=candidate_binding_sha256(binding),
        candidate_visual_acceptance='unreviewed'), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_visual_intent(value):
    """Effect-free proposed direction, not native support or visual achievement.

    Null means continue the active direction. Epoch admission/cancellation belongs
    to the existing turn owner, not to this data validator.
    """
    if (type(value) is not dict or set(value) != {'schema', 'instruction'}
            or value['schema'] != VISUAL_INTENT_SCHEMA):
        raise ValueError('versioned visual intent required')
    text = value['instruction']
    if text is not None:
        if (type(text) is not str or not text.strip() or len(text) > 256
                or any(ord(c) < 32 or ord(c) == 127 or 0xd800 <= ord(c) <= 0xdfff for c in text)
                or len(text.encode('utf-8')) > 1024):
            raise ValueError('bounded natural-language visual instruction required')
        text = text.strip()
    return dict(schema=VISUAL_INTENT_SCHEMA, instruction=text)


def resolve_visual_intent(active, proposal):
    """Resolve continue/replace only AFTER the caller's current-epoch guard.

    Reply cancellation alone must not clear person/scene/action continuation.
    Return independent records; never mutate a previous accepted proposal.
    """
    previous = None if active is None else validate_visual_intent(active)
    if previous is not None and previous['instruction'] is None:
        raise ValueError('active direction must be resolved')
    proposed = validate_visual_intent(proposal)
    return previous if proposed['instruction'] is None else proposed


def visual_prompt(role, active):
    """Explicit visual-role prompt; legacy defaults remain immutable.

    Only the versioned role binding admits it; native cache/evidence must match.
    """
    baseline = role_prompt(role)
    if active is None:
        return baseline
    direction = validate_visual_intent(active)['instruction']
    if direction is None:
        raise ValueError('resolve continuation before composing visual prompt')
    behavior = ('The person is speaking; lip movement follows the supplied speech when the face is visible. '
                if role == 'speech' else 'The person is listening without speaking. ')
    return ('The same single person in reference image 1. '
            'Preserve the appearance, clothing and identity of the reference person when visible. '
            'One continuous handheld smartphone video in the same physical setting. '
            'Maintain the same body and its current position and orientation across speaking and listening. '
            'Continue from the current view rather than reinserting the reference pose or restarting a movement. '
            'For a finite camera adjustment, move once and then hold the resulting natural composition. '
            'Keep person, surroundings and camera motion temporally consistent; '
            'the camera may turn away and the face may leave the frame. '
            + behavior + 'Current visual direction: ' + direction + _reference_guidance())
