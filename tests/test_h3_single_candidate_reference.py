"""Explicit single candidate reference: schema cardinality, bounds and staging.

Offline only. No cloud, devices, pixels or native execution.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal import conditioning_experiment as cond
from research.h3_personal.ordered_references import (
    SCHEMA as PAIR_SCHEMA, SINGLE_SCHEMA, reference_roles,
    validate_ordered_references, validate_content_identity, content_identity,
    expected_prepared_sizes,
)
from research.h3_personal.private_root import base as private_base

SHA_A = 'a' * 64
SHA_B = 'b' * 64


def ref(role, sha, width=1737, height=3088):
    return dict(role=role, path='/private/' + role + '.png', sha256=sha,
                width=width, height=height)


def single(sha=SHA_A, **kw):
    return dict(schema=SINGLE_SCHEMA, references=[ref('person', sha, **kw)])


def pair():
    return dict(schema=PAIR_SCHEMA,
                references=[ref('person', SHA_A), ref('surroundings', SHA_B, 3500, 2750)])


class SchemaCardinalityTests(unittest.TestCase):
    def test_single_schema_is_exactly_one_person(self):
        self.assertEqual(reference_roles(SINGLE_SCHEMA), ('person',))
        self.assertEqual(reference_roles(PAIR_SCHEMA), ('person', 'surroundings'))

    def test_single_reference_accepts_exactly_one(self):
        self.assertEqual(validate_ordered_references(single())['schema'], SINGLE_SCHEMA)
        for bad in (dict(schema=SINGLE_SCHEMA, references=[]),
                    dict(schema=SINGLE_SCHEMA, references=[ref('person', SHA_A), ref('surroundings', SHA_B)]),
                    dict(schema=SINGLE_SCHEMA, references=[ref('surroundings', SHA_A)])):
            with self.assertRaises(ValueError):
                validate_ordered_references(bad)

    def test_pair_schema_remains_exactly_two(self):
        self.assertEqual(len(validate_ordered_references(pair())['references']), 2)
        with self.assertRaises(ValueError):
            validate_ordered_references(dict(schema=PAIR_SCHEMA, references=[ref('person', SHA_A)]))

    def test_unknown_schema_rejected(self):
        for schema in ('h3-single-reference-v2', '', None, 1):
            with self.assertRaises(ValueError):
                reference_roles(schema)

    def test_single_identity_preserves_schema(self):
        identity = content_identity(single())
        self.assertEqual(identity['schema'], SINGLE_SCHEMA)
        self.assertEqual(len(identity['references']), 1)
        self.assertEqual(identity['references'][0]['role'], 'person')
        self.assertEqual(validate_content_identity(identity), identity)

    def test_single_identity_cannot_be_widened_to_pair(self):
        identity = content_identity(single())
        widened = dict(identity, references=identity['references'] + identity['references'])
        with self.assertRaises(ValueError):
            validate_content_identity(widened)
        with self.assertRaises(ValueError):
            validate_content_identity(dict(identity, schema=PAIR_SCHEMA))

    def test_single_prepared_geometry_is_round32_no_crop(self):
        prepared = expected_prepared_sizes(single(), short_edge=1024)['references']
        self.assertEqual(len(prepared), 1)
        self.assertEqual((prepared[0]['width'], prepared[0]['height']), (1024, 1824))
        # R/T are never inferred from PNG headers.
        self.assertIsNone(expected_prepared_sizes(single())['R'])


class AdmissionBoundTests(unittest.TestCase):
    def test_bounds_scale_with_explicit_cardinality(self):
        two = cond.ordered_admission_bounds(1024)
        one = cond.ordered_admission_bounds(1024, references=1)
        self.assertEqual((two['max_reference_rows'], two['max_tokens']), (8192, 8703))
        self.assertEqual((one['max_reference_rows'], one['max_tokens']), (4096, 4607))
        self.assertEqual(two, cond.ordered_admission_bounds(1024, references=2))

    def test_bounds_reject_unsupported_cardinality_and_edge(self):
        for references in (0, 3, -1, True, None, '2'):
            with self.assertRaises(ValueError):
                cond.ordered_admission_bounds(1024, references=references)
        for edge in (256, 2048, '1024', None):
            with self.assertRaises(ValueError):
                cond.ordered_admission_bounds(edge)

    def test_token_policy_uses_single_cardinality(self):
        from research.h3_personal.ordered_qwen_tokens import ordered_token_policy
        policy = ordered_token_policy(1024)
        self.assertEqual(policy['max_reference_rows'], 8192)
        self.assertEqual(policy['admission_kind'], 'operational-not-native-capacity')


class LayoutEvidenceTests(unittest.TestCase):
    def test_single_candidate_requires_its_own_identity(self):
        # A context without a portable identity is rejected, never treated as legacy two.
        from tests.test_h3_ordered_guide_layout import batch
        with patch('research.h3_personal.conversation_conditioning.read_ordered_context',
                   return_value={}):
            with self.assertRaises(RuntimeError):
                cond.ordered_layout_evidence(batch())

    def test_malformed_identity_rejected(self):
        from tests.test_h3_ordered_guide_layout import batch
        for bad in ({'identity': None}, {'identity': {}},
                    {'identity': {'references': 'person'}}):
            with patch('research.h3_personal.conversation_conditioning.read_ordered_context',
                       return_value=bad):
                with self.assertRaises(RuntimeError):
                    cond.ordered_layout_evidence(batch())

    def test_single_candidate_layout_uses_one_image_segment(self):
        """A one-reference identity must match a one-image native layout."""
        from tests.test_h3_ordered_guide_layout import batch
        b = batch(guide=False)
        layout = b.extra['minimax_h3_layout']
        # Keep only the first image segment: a genuine single-reference layout.
        layout.reference_segments = layout.reference_segments[:1]
        layout.num_condition_video_rows = layout.reference_segments[0][1]
        candidate = {'identity': {'references': [{'role': 'person'}]}}
        with patch('research.h3_personal.conversation_conditioning.read_ordered_context',
                   return_value=candidate):
            proof = cond.ordered_layout_evidence(b)
        self.assertEqual(proof['image_rows'], 1824)

    def test_single_identity_cannot_accept_two_image_layout(self):
        from tests.test_h3_ordered_guide_layout import batch
        candidate = {'identity': {'references': [{'role': 'person'}]}}
        with patch('research.h3_personal.conversation_conditioning.read_ordered_context',
                   return_value=candidate):
            # The legacy two-image fixture has an extra image segment; the
            # single identity must not silently ignore it as a guide.
            with self.assertRaises(RuntimeError):
                cond.ordered_layout_evidence(batch(guide=False))


def rgb8_png(width, height):
    """Minimal valid RGB8 PNG header+IDAT+IEND for structural file checks."""
    import struct, zlib
    ihdr = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)
    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', zlib.crc32(kind + data)))
    raw = b''.join(b'\x00' + b'\x00' * (width * 3) for _ in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


class StagingNameTests(unittest.TestCase):
    """Staging needs a symlink-free ancestor chain; macOS /var is a symlink."""

    def setUp(self):
        import shutil
        # Resolve so the ancestor chain contains no symlink (macOS /var -> /private/var).
        self.root = Path(tempfile.mkdtemp(prefix='h3-single-')).resolve()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def _write(self, role, width=1737, height=3088):
        import hashlib
        path = self.root / (role + '.png')
        data = rgb8_png(width, height)
        path.write_bytes(data)
        return dict(role=role, path=str(path),
                    sha256=hashlib.sha256(data).hexdigest(),
                    width=width, height=height)

    def _stage(self):
        stage = self.root / 'stage'
        stage.mkdir()
        return stage

    def test_single_stages_only_person(self):
        from research import launch_h3_direct_conversation as L
        selected = dict(schema=SINGLE_SCHEMA, references=[self._write('person')])
        stage = self._stage()
        L.stage_ordered_inputs(stage, selected)
        staged = sorted(p.name for p in (stage / 'ordered-inputs').iterdir())
        self.assertEqual(staged, ['person.png'])

    def test_pair_stages_both(self):
        from research import launch_h3_direct_conversation as L
        selected = dict(schema=PAIR_SCHEMA, references=[
            self._write('person'), self._write('surroundings', 3500, 2750)])
        stage = self._stage()
        L.stage_ordered_inputs(stage, selected)
        staged = sorted(p.name for p in (stage / 'ordered-inputs').iterdir())
        self.assertEqual(staged, ['person.png', 'surroundings.png'])

    def test_staging_rejects_changed_bytes(self):
        from research import launch_h3_direct_conversation as L
        row = self._write('person')
        selected = dict(schema=SINGLE_SCHEMA, references=[row])
        (self.root / 'person.png').write_bytes(rgb8_png(1737, 3088) + b'tampered')
        stage = self._stage()
        with self.assertRaises(ValueError):
            L.stage_ordered_inputs(stage, selected)


class PreparedSingleConfigTests(unittest.TestCase):
    """The operator-prepared single config must satisfy the explicit schema."""

    def test_prepared_config_if_present(self):
        path = private_base() / 'reference-single-20261110/reference-config.json'
        if not path.is_file():
            self.skipTest('prepared single reference config not present')
        value = json.loads(path.read_text())
        self.assertEqual(value['schema'], SINGLE_SCHEMA)
        selected = validate_ordered_references(value)
        self.assertEqual([r['role'] for r in selected['references']], ['person'])
        identity = content_identity(selected)
        self.assertEqual(identity['schema'], SINGLE_SCHEMA)


if __name__ == '__main__':
    unittest.main()
