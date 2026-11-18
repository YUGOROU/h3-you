"""CPU-only motion proposals, NOT native conditioning or a playback controller.

No environment reads, random state, mutable epoch, or runtime registration.
Time is supplied from the existing conversation frame clock, never role-local.
Numeric values are desired directions, not measured/model-enforced motion.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from .conversation_conditioning import prompt_sha
from .visual_ancestry_policy import CANDIDATE

SCHEMA = 'h3-listening-motion-proposal-v1'
MAX_FRAMES = 14710
BOUNDS = dict(gaze_degrees=2.0, head_yaw_degrees=1.5,
              head_pitch_degrees=0.8, posture_degrees=0.6,
              blink=1.0, light_fraction=0.015)


@dataclass(frozen=True)
class MotionPolicy:
    enabled: bool = False
    subtle_light: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.subtle_light) is not bool:
            raise ValueError('explicit boolean policy required')
        if self.subtle_light and not self.enabled:
            raise ValueError('light requires motion opt-in')


def _frame(frame):
    if type(frame) is not int or not 0 <= frame <= MAX_FRAMES:
        raise ValueError('bounded absolute conversation frame required')


def sample(policy, role, frame):
    """Sample smooth absolute-time targets; no listening-entry anchor/reset.

    Speech deliberately has no added motion or mouth instruction. Role boundaries
    are not interpolation commands; a future consumer must not snap these targets.
    """
    if type(policy) is not MotionPolicy:
        raise ValueError('MotionPolicy required')
    if type(role) is not str or role not in ('listening', 'speech'):
        raise ValueError('invalid role')
    _frame(frame)
    if not policy.enabled:
        return None
    t = frame / 24.0
    targets = dict.fromkeys(BOUNDS, 0.0)
    if role == 'listening':
        for name, period in (('gaze_degrees', 19), ('head_yaw_degrees', 31),
                             ('head_pitch_degrees', 23), ('posture_degrees', 47)):
            targets[name] = BOUNDS[name] * math.sin(2 * math.pi * t / period)
        # Compact smooth blink pulse every 5.5 seconds, not a window toggle.
        distance = abs((t - 2.75 + 2.75) % 5.5 - 2.75)
        targets['blink'] = ((1 + math.cos(math.pi * distance / 0.16)) / 2
                            if distance < 0.16 else 0.0)
        if policy.subtle_light:
            targets['light_fraction'] = BOUNDS['light_fraction'] * math.sin(2 * math.pi * t / 61)
    return targets


def propose_window(policy, role, sequence):
    """Return a bounded inspectable spec, never a native manifest extension.

    Frame intervals describe existing 124/102 committed windows, NOT a new clock.
    Per-frame samples are analytic directions; identity/background remain fixed.
    """
    if type(sequence) is not int or not 0 <= sequence < 144:
        raise ValueError('bounded global sequence required')
    start = 0 if sequence == 0 else 124 + (sequence - 1) * 102
    stop = 124 + sequence * 102
    first = sample(policy, role, start)
    if first is None:
        return None
    spec = dict(schema=SCHEMA, runtime_integrated=False, role=role,
                sequence=sequence, start_frame=start, stop_frame=stop,
                role_prompt_sha256=prompt_sha(role),
                ancestry_policy=CANDIDATE, anchor_reset=False,
                background='fixed', camera='fixed', image_warp=False,
                subtle_light=policy.subtle_light,
                directions=('Gradual small gaze, natural blink, gentle head and posture changes; '
                            'closed lips while listening; preserve identity, clothing and background.'
                            if role == 'listening' else ''),
                samples=[sample(policy, role, f) for f in range(start, stop)])
    payload = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(spec=spec, spec_sha256=hashlib.sha256(payload.encode()).hexdigest())
