"""Approved TTS-source placement gate: no TTS region constraint, H3 untouched."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from research.h3_personal.director_tts_job import IMAGE
from research.h3_personal.pair_placement import approve_plan, approve_source, gate, PLAN_KEYS


def spec(**over):
    row = dict(network_service=dict(group='h3-pair-abc', alias='voxcpm', port=17861,
                                    certificate_sha256='a' * 64, server_job_id='6ae34aa541e90535df78cda5'))
    row.update(over)
    return row


def pre_create_spec(**over):
    """The owner's real pre-create spec has no source Job id yet."""
    row = spec(**over)
    row['network_service'] = {k: v for k, v in row['network_service'].items() if k != 'server_job_id'}
    return row


def plan(**over):
    row = dict(phase='before_source_create', namespace='YUGOROU', flavor='l4x1', image=IMAGE, group='h3-pair-abc')
    row.update(over)
    return row


def source(**over):
    row = dict(id='6ae34aa541e90535df78cda5', owner=dict(name='YUGOROU'), flavor='l4x1', dockerImage=IMAGE,
               network=dict(group='h3-pair-abc', aliases=['voxcpm']), retry=0,
               status=dict(stage='RUNNING'), expose=dict(ports=[], portsPublic=[]))
    row.update(over)
    return row


class PlanTests(unittest.TestCase):
    def test_accepts_only_the_owned_l4_source_plan(self):
        plan_row = approve_plan(plan(), pre_create_spec())
        self.assertEqual(plan_row['flavor'], 'l4x1')
        # No TTS region constraint is introduced by this change.
        self.assertIsNone(plan_row['tts_region_constraint'])
        self.assertEqual(plan_row['tts_region_evidence'], 'unverified-region-unconstrained')
        self.assertIs(plan_row['h3_placement_covered'], False)

    def test_pre_create_spec_needs_no_source_job_id_but_source_does(self):
        self.assertEqual(approve_plan(plan(), pre_create_spec())['phase'], 'pre_create')
        with self.assertRaises(ValueError):
            approve_source(source(), pre_create_spec())

    def test_rejects_h3_flavor(self):
        for flavor in ('h200x2', 'l4x4', 'cpu-basic'):
            with self.assertRaises(ValueError):
                approve_plan(plan(flavor=flavor), spec())

    def test_rejects_foreign_group_image_namespace_and_extra_fields(self):
        for over in (dict(group='other-group'), dict(image='python:3.12'),
                     dict(namespace='SOMEONE'), dict(phase='running')):
            with self.assertRaises(ValueError):
                approve_plan(plan(**over), spec())
        with self.assertRaises(ValueError):
            approve_plan({**plan(), 'country': 'us'}, spec())

    def test_rejects_spec_without_single_group(self):
        for bad in (dict(network_service=dict(group='h3-pair-abc', alias='voxcpm')), dict(network_service=dict(group='BAD_GROUP'))):
            with self.assertRaises(ValueError):
                approve_plan(plan(), bad)


class SourceTests(unittest.TestCase):
    def test_accepts_running_owned_source(self):
        row = approve_source(source(), spec())
        self.assertEqual(row['phase'], 'running_source')

    def test_region_is_never_inferred_and_not_required(self):
        self.assertEqual(approve_source(source(), spec())['tts_region_evidence'], 'unverified-region-unconstrained')
        self.assertEqual(approve_source(source(provider='aws'), spec())['tts_region_evidence'], 'aws')

    def test_rejects_public_ports_retry_foreign_owner_and_identity(self):
        for over in (dict(expose=dict(ports=[dict(port=17861)], portsPublic=[])),
                     dict(expose=dict(ports=[], portsPublic=[dict(port=17861)])),
                     dict(retry=1), dict(owner=dict(name='OTHER')), dict(id='0' * 24),
                     dict(flavor='h200x2'), dict(status=dict(stage='ERROR')),
                     dict(network=dict(group='h3-pair-abc', aliases=['other']))):
            with self.assertRaises(ValueError):
                approve_source(source(**over), spec())


class ReceiptTests(unittest.TestCase):
    def test_gate_returns_true_and_persists_private_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = gate(directory)
            self.assertIs(callback(plan(), pre_create_spec()), True)
            self.assertIs(callback(source(), spec()), True)
            path = Path(directory) / 'source-placement.json'
            rows = json.loads(path.read_text())
            self.assertEqual([r['phase'] for r in rows], ['pre_create', 'running_source'])
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_gate_propagates_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = gate(directory)
            with self.assertRaises(ValueError):
                callback(source(expose=dict(portsPublic=[dict(port=1)])), spec())
            self.assertFalse((Path(directory) / 'source-placement.json').exists())


if __name__ == '__main__':
    unittest.main()
