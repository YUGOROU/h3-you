"""Synthetic projection/log persistence tests; not native/GPU execution proof."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.ordered_metadata import MAX_BYTES, project_metadata, validate_metadata
from test_h3_ordered_rank_evidence import fixture
from research.hf_h3_streaming_trial import validate_ordered_rank_evidence


def receipt(sequence=0, target=1):
    profile, context, report = fixture()
    report['ordered_geometry']['guide_layout']['provenance']['window_index'] = sequence
    with patch('research.h3_personal.conditioning_experiment.read_accepted_profile', return_value=profile):
        ordered = validate_ordered_rank_evidence(report, context)
    visual = dict(target_epoch=target, intent_epoch=target, prompt_sha256=report['qwen_tokens']['prompt_sha256'])
    evidence = dict(ordered_references=ordered, ordered_both_ranks=True,
        ordered_worker_memory=[dict(measurement='worker-current-all-models-not-reference-delta',
            allocated_bytes=100 + rank, reserved_bytes=200 + rank) for rank in (0, 1)],
        irrelevant_private_path='/private/not-exported.png')
    return dict(sequence=sequence, receipt_sha256='c' * 64, native_evidence=evidence,
        visual_conditioning=visual), context['identity_sha256']


def events():
    rows = []
    for sequence, target in enumerate((1, 2)):
        source, identity = receipt(sequence, target)
        rows.append(dict(type='h3_visual_intent_received', session='a' * 32, target_epoch=target,
            intent_epoch=target, stage='instruction_received', instruction_sha256='d' * 64,
            visual_achievement='unverified'))
        rows.append(dict(type='h3_visual_conditioning', session='a' * 32, sequence=sequence,
            **source['visual_conditioning'], stage='native_conditioning_applied',
            presentation_epoch=sequence, presentation_outcome='unverified', visual_achievement='unverified',
            ordered_metadata=project_metadata(source, expected_identity=identity)))
    rows.append(dict(type='h3_native_tts_receipts', receipts=[dict(epoch=1), dict(epoch=2)]))
    return rows, identity


class ProjectionTests(unittest.TestCase):
    def test_actual_qwen_T_and_vae_R_survive_independently_with_rank_memory_and_sha(self):
        source, identity = receipt()
        metadata = project_metadata(source, expected_identity=identity)
        ordered = metadata['ordered_references']
        self.assertEqual(ordered['geometry']['R'], 3136)
        self.assertEqual(ordered['geometry']['T'], 3000)
        self.assertEqual(ordered['qwen_tokens']['total_tokens'], 3000)
        self.assertEqual(ordered['qwen_tokens']['image_token_counts'], [1000, 1800])
        self.assertEqual(ordered['qwen_tokens']['image_grid_thw'], [[1, 40, 100], [1, 40, 180]])
        self.assertEqual(ordered['geometry']['packed_rows']['text_rows'], 3000)
        self.assertEqual(metadata['receipt_sha256'], source['receipt_sha256'])
        self.assertEqual([m['allocated_bytes'] for m in metadata['ordered_worker_memory']], [100, 101])
        encoded = json.dumps(metadata)
        self.assertNotIn('/private/', encoded)
        self.assertNotIn('"token_ids":', encoded)
        self.assertLess(len(encoded.encode()), MAX_BYTES)
        metadata['ordered_worker_memory'][0]['allocated_bytes'] = 0
        self.assertEqual(source['native_evidence']['ordered_worker_memory'][0]['allocated_bytes'], 100)

    def test_legacy_has_no_attachment_and_unexpected_candidate_rejected(self):
        self.assertIsNone(project_metadata(dict(native_evidence={}), expected_identity=None))
        source, identity = receipt()
        with self.assertRaises(ValueError):
            project_metadata(source, expected_identity=None)
        for key in ('ordered_references', 'ordered_both_ranks', 'ordered_worker_memory'):
            wrong = copy.deepcopy(source); wrong['native_evidence'].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                project_metadata(wrong, expected_identity=identity)

    def test_no_private_extra_fields_or_raw_token_ids_are_exported(self):
        source, identity = receipt()
        for inject in (
                lambda e: e['ordered_references']['geometry'].update(path='/private/x'),
                lambda e: e['ordered_references']['geometry']['images'][0].update(prompt='secret'),
                lambda e: e['ordered_references']['qwen_tokens'].update(token_ids=[1, 2]),
                lambda e: e['ordered_worker_memory'][0].update(tensor=[1])):
            wrong = copy.deepcopy(source); inject(wrong['native_evidence'])
            with self.assertRaises(ValueError):
                project_metadata(wrong, expected_identity=identity)

    def test_identity_prompt_sha_rank_memory_and_independent_admission_fail_closed(self):
        source, identity = receipt()
        for mutate in (
                lambda m: m.update(receipt_sha256='not-a-sha'),
                lambda m: m.update(ordered_both_ranks=False),
                lambda m: m['ordered_references'].update(identity_sha256='f' * 64),
                lambda m: m['visual_conditioning'].update(prompt_sha256='f' * 64),
                lambda m: m['ordered_references']['qwen_tokens'].update(prompt_sha256='f' * 64),
                lambda m: m['ordered_references']['geometry'].update(R=3000),
                lambda m: m['ordered_references']['geometry'].update(T=3136),
                lambda m: m['ordered_references']['geometry']['packed_rows'].update(sequence_length=32769),
                lambda m: m['ordered_worker_memory'][0].update(allocated_bytes=201),
                lambda m: m['ordered_worker_memory'][0].update(allocated_bytes=True)):
            wrong = project_metadata(source, expected_identity=identity); mutate(wrong)
            with self.assertRaises(ValueError):
                validate_metadata(wrong, expected_identity=identity, visual_binding=source['visual_conditioning'], sequence=0)


class CollectorTests(unittest.TestCase):
    def collect(self, rows, identity, directory):
        from research.launch_h3_direct_conversation import collect_native_worker_receipts
        api = Mock(); api.fetch_job_logs.return_value = [json.dumps(row) for row in rows]
        return collect_native_worker_receipts(api, NS(id='synthetic'), directory, time.monotonic() + 2,
            visual_intents=True, ordered_reference_sha256=identity)

    def test_existing_visual_artifact_persists_full_bounded_metadata(self):
        rows, identity = events()
        with tempfile.TemporaryDirectory() as tmp:
            self.collect(rows, identity, tmp)
            result = json.loads((Path(tmp) / 'visual-worker-evidence.json').read_text())
        self.assertEqual([r['ordered_metadata'] for r in result['windows']],
            [rows[1]['ordered_metadata'], rows[3]['ordered_metadata']])
        self.assertEqual(result['visual_achievement'], 'unverified')

    def test_missing_duplicate_identity_and_prompt_drift_are_rejected(self):
        original, identity = events()
        mutations = (
            lambda rows: rows[1].pop('ordered_metadata'),
            lambda rows: rows.insert(2, copy.deepcopy(rows[1])),
            lambda rows: rows[3]['ordered_metadata']['ordered_references'].update(identity_sha256='f' * 64),
            lambda rows: rows[3].update(prompt_sha256='f' * 64),
            lambda rows: rows[3].update(intent_epoch=1),
            lambda rows: rows[3]['ordered_metadata']['ordered_references'].update(candidate_binding_sha256='f' * 64),
            lambda rows: rows[1].update(sequence=144))
        for mutate in mutations:
            rows = copy.deepcopy(original); mutate(rows)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.collect(rows, identity, tmp)
                self.assertFalse((Path(tmp) / 'visual-worker-evidence.json').exists())

    def test_wrong_expected_READY_identity_and_unsolicited_attachment_rejected(self):
        rows, identity = events()
        for expected in ('f' * 64, None):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                self.collect(rows, expected, tmp)

    def test_existing_worker_event_and_launcher_bind_shared_projection_to_READY(self):
        import ast
        import inspect
        from research.h3_personal import live_relay_worker
        from research import launch_h3_direct_conversation
        worker = inspect.getsource(live_relay_worker)
        launcher = inspect.getsource(launch_h3_direct_conversation)
        self.assertIn('metadata=project_metadata(backend.receipts[-1]', worker)
        # Identity binding must come from the bound backend runtime, tolerating a
        # fake/caller backend object that exposes no runtime attribute at all.
        tree = ast.parse(worker)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'project_metadata']
        self.assertTrue(calls)
        identity = [k.value for n in calls for k in n.keywords if k.arg == 'expected_identity']
        self.assertTrue(identity)
        rendered = [ast.unparse(value) for value in identity]
        self.assertTrue(any('ordered_reference_sha256' in text and 'runtime' in text
                            for text in rendered), rendered)
        self.assertIn("{'ordered_metadata':metadata}", worker)
        self.assertIn("{'ordered_reference_sha256':ready['ordered_reference_sha256']}", launcher)

    def test_metadata_and_existing_log_line_size_bounds_unchanged(self):
        for size in (MAX_BYTES + 1, 65537):
            rows, identity = events()
            rows[1]['ordered_metadata']['ordered_references']['geometry']['schema'] = 'x' * size
            with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'byte bound|log line bound'):
                self.collect(rows, identity, tmp)


if __name__ == '__main__':
    unittest.main()
