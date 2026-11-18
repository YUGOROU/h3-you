"""Opt-in accepted validation profiles; never changes production or training defaults."""
from dataclasses import dataclass, fields
import hashlib
import json
import re

from research.h3_personal.conditioning_experiment import (
    SCHEMA, RESIZE_POLICY, validate_manifest,
)

PROCESSOR_SHA256 = '37aae0fda06bc6fa6e6f7ea20df6ff7aa8e0415bdc6ee0a47bbdb02ad77dcc03'
QUALITY_REVIEWS = (
    ('reference1024', '35bf71c005e1ca0edf42829caf001a95484380059cb0bd3539f822f39f312884'),
    ('reference512', 'fbd1bb57ba9f23b07be3fe289028c32dd302bc65920021bf1ecb2fdc55491cbd'),
)
ACCEPTANCE_SOURCE = 'plan.md: USER「1024が理想的ですが、512も良い！！！！」'


def _sha(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('expected lowercase SHA256, not raw content')
    return value


@dataclass(frozen=True)
class AcceptedQualityProfile:
    name: str
    ordered_source_sha256: tuple[str, ...]
    prompt_sha256: str
    processor_sha256: str = PROCESSOR_SHA256
    geometry: tuple[int, int] = (256, 448)
    exported_geometry: tuple[int, int] = (252, 448)
    fps: int = 24
    window: int = 124
    overlap: int = 22
    precision: str = 'BF16-with-native-FP32-exceptions'
    sampler: str = 'SMD4'
    forwards: int = 4
    resize_policy: str = RESIZE_POLICY
    resize_grid: int = 32
    training_original_immutable: bool = True
    acceptance_source: str = ACCEPTANCE_SOURCE

    def __post_init__(self):
        if type(self.name) is not str or self.name not in dict(QUALITY_REVIEWS):
            raise ValueError('explicit accepted profile required; native2048 is historical only')
        if (type(self.ordered_source_sha256) is not tuple or not self.ordered_source_sha256):
            raise ValueError('nonempty ordered source SHA tuple required')
        for value in self.ordered_source_sha256:
            _sha(value)
        _sha(self.prompt_sha256)
        if self.processor_sha256 != PROCESSOR_SHA256:
            raise ValueError('processor mismatch')
        fixed = dict(geometry=(256, 448), exported_geometry=(252, 448), fps=24,
                     window=124, overlap=22, precision='BF16-with-native-FP32-exceptions',
                     sampler='SMD4', forwards=4, resize_policy=RESIZE_POLICY, resize_grid=32,
                     training_original_immutable=True, acceptance_source=ACCEPTANCE_SOURCE)
        for key, expected in fixed.items():
            actual = getattr(self, key)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f'{key} mismatch')
            if isinstance(expected, tuple) and any(type(n) is not int for n in actual):
                raise ValueError(f'{key} requires strict integers')

    @property
    def short_edge(self):
        return 1024 if self.name == 'reference1024' else 512

    @property
    def expected_prepared_geometry(self):
        """Single portrait probe expectation, not a general source geometry assertion."""
        return (1024, 1824) if self.short_edge == 1024 else (512, 896)

    @property
    def expected_rt(self):
        return (1824, 1867) if self.short_edge == 1024 else (448, 491)

    @property
    def quality_review_sha256(self):
        return dict(QUALITY_REVIEWS)[self.name]

    def validate_binding(self, *, ordered_source_sha256, prompt_sha256,
                         processor_sha256, geometry):
        """Reject drift before cache reuse; order is significant."""
        for key, value in locals().copy().items():
            if key == 'self':
                continue
            expected = getattr(self, key)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f'{key} binding mismatch')
            if key == 'geometry' and any(type(n) is not int for n in value):
                raise ValueError('geometry requires strict integers')
        _sha(prompt_sha256)
        _sha(processor_sha256)
        return self

    def legacy_conditioning_manifest(self):
        """Explicit adapter to the existing three-field experiment policy/math."""
        return validate_manifest(dict(schema=SCHEMA, short_edges=[1024, 512],
                                      active_short_edge=self.short_edge))

    def wire_record(self):
        record = {f.name: getattr(self, f.name) for f in fields(self)}
        record.update(schema='h3-accepted-quality-profile-v1', short_edge=self.short_edge,
                      expected_prepared_geometry=self.expected_prepared_geometry,
                      expected_rt=self.expected_rt,
                      quality_review_sha256=self.quality_review_sha256)
        # Return independent JSON-native containers, never mutable aliases.
        return json.loads(json.dumps(record, allow_nan=False))

    @classmethod
    def from_record(cls, record):
        """Parse only complete canonical wire records, including derived fields."""
        if type(record) is not dict:
            raise ValueError('accepted profile record required')
        names = {f.name for f in fields(cls)}
        if not names <= record.keys():
            raise ValueError('incomplete accepted profile record')
        values = {name: record[name] for name in names}
        for name in ('ordered_source_sha256', 'geometry', 'exported_geometry'):
            if type(values[name]) is not list:
                raise ValueError('wire arrays required')
            values[name] = tuple(values[name])
        profile = cls(**values)
        if (set(record) != set(profile.wire_record()) or
                json.dumps(record, sort_keys=True, allow_nan=False) !=
                json.dumps(profile.wire_record(), sort_keys=True, allow_nan=False)):
            raise ValueError('noncanonical accepted profile record')
        return profile

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.wire_record(), sort_keys=True,
                                        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    @property
    def cache_identity(self):
        return self.digest


def choose_profile(name=None, *, ordered_source_sha256, prompt_sha256,
                   processor_sha256=PROCESSOR_SHA256):
    """Default only this opt-in factory to 1024, not any live runtime."""
    return AcceptedQualityProfile('reference1024' if name is None else name,
                                  ordered_source_sha256, prompt_sha256, processor_sha256)
