"""CPU consumer harness: real admission, profile, evidence and timing validation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import hf_h3_conditioning_exploration as worker
from research.h3_personal import continuation_candidate as candidate
from research.h3_personal import continuation_efficiency as efficiency
from test_h3_continuation_diagnostic_worker import spec_fixture
from test_h3_conditioning_exploration_worker import evidence


class EndToEndTests(unittest.TestCase):
    def exercise(self, *, fail_second=False, private=True, existing=False, bad_memory=None):
        events, commits, constructors = [], [], []
        canary = 'PRIVATE-PROMPT-MUST-NOT-LEAK'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'deployment'
            root.mkdir()
            output = root.with_name(root.name+'-conditioning-generated')
            if existing:
                output.mkdir()
            paths = {key: root/key for key in ('speech', 'personal', 'base', 'smd', 'reference')}
            spec = spec_fixture()
            spec['conditioning_exploration'] = worker.CONTRACT
            wav = paths['speech']/spec['locked_speech']['path']
            owner = self

            class FakeRuntime:
                def __init__(self, **kwargs):
                    constructors.append(kwargs)
                    self.coverage = output/'coverage'
                    self.edge = 1024
                    self.closed = 0
                def start(self):
                    owner.assertFalse(output.exists())
                    output.mkdir(mode=0o700)
                    self.coverage.mkdir(mode=0o700)
                    (output/'memory').mkdir(mode=0o700)
                    events.append('start')
                def configure_conditioning(self, **kwargs):
                    owner.assertEqual(kwargs, {'short_edge': 512})
                    owner.assertEqual(events[-1], 'progress-reference1024')
                    self.edge = 512
                    events.append('configure512')
                def close(self):
                    self.closed += 1
                    owner.assertEqual(self.closed, 1)
                    events.append('close')
                def render_continuation(self, source, sha, turn):
                    owner.assertEqual(source, wav)
                    owner.assertEqual(sha, worker.source_contract.SOURCE_SHA256)
                    owner.assertEqual(turn, output/('reference1024' if self.edge == 1024 else 'reference512'))
                    events.append('render-'+turn.name)
                    if fail_second and self.edge == 512:
                        raise RuntimeError('injected second render')
                    turn.mkdir()
                    session = 'session-'+turn.name
                    profile = dict(
                        continuation_plan=dict(
                            candidate=candidate.make_options(window_audio_context=True, defer_progress_receipts=True),
                            efficiency=efficiency.make_options(device_audio_validation=True, parallel_decode_waves=True),
                            parallel_acceptance=False),
                        unique_frames=328, speech_pcm_exact=True, full_decode_verified=True,
                        session_id=session, input_proof={'prompt': canary},
                        derived_adapter_receipt={'sha256': 'a'*64}, prompt=canary,
                        windows=[dict(seed=314159, session_id=session,
                            fixed_reference_sha256=worker.efficiency.baseworker.REFERENCE_SHA,
                            wav_sha256=sha) for _ in range(3)],
                        timing_metrics=dict(first_segment_ready_seconds=2., total_wall_seconds=9.,
                            generation_total_seconds=6., unique_video_seconds=328/24,
                            unique_throughput_total=328/9, unique_throughput_generation=328/6,
                            segment_ready_elapsed_seconds=[2., 5., 8.], unknown=canary),
                        pipeline_timing=[dict(clock='parent_monotonic', window_index=i,
                            request_start=i*10., request_end=i*10+1.,
                            encoding_start=i*10+2., encoding_end=i*10+3.,
                            validation_start=i*10+4., validation_end=i*10+5.) for i in range(3)])
                    (turn/'profile.json').write_text(json.dumps(profile))
                    # Typed receipt fixture: helper producer has separate unit coverage.
                    for rank in range(2):
                        row = dict(evidence(self.edge, rank), session_id=session, unknown=canary)
                        (self.coverage/f'conditioning-experiment-rank-{rank}.json').write_text(json.dumps(row))
                    for hit in range(1 if self.edge == 1024 else 4, 4 if self.edge == 1024 else 7):
                        for rank in range(2):
                            memory = dict(schema='h3-worker-memory-v1', worker_rank=rank,
                                cache_hit=hit, contains_parallel_acceptance=False,
                                peak_allocated_bytes=1000+hit+rank, peak_reserved_bytes=2000+hit+rank)
                            if bad_memory and hit == 1 and rank == 0:
                                if bad_memory == 'missing':
                                    continue
                                key, value = bad_memory
                                memory[key] = value
                            (output/'memory'/f'rank-{rank}-request-hit{hit}.json').write_text(json.dumps(memory))
                    owner.assertEqual(len(worker.source_worker.VIDEOS), 5)
                    for name in worker.source_worker.VIDEOS:
                        (turn/name).write_bytes(('review-'+turn.name if name == 'review-with-speech.mp4' else 'fake-video').encode())
                    events.append('completed-'+turn.name)
                    return SimpleNamespace(profile=turn/'profile.json', review_with_speech=turn/'review-with-speech.mp4')

            def preflight(source, *, timeout_seconds):
                self.assertEqual(source, wav)
                self.assertIs(type(timeout_seconds), int)
                self.assertTrue(1 <= timeout_seconds <= 3600)

            def verify(path, source, deadline):
                self.assertEqual(source, wav)
                self.assertTrue(path.is_file())
                self.assertGreater(deadline, time.monotonic())
                return dict(frames=328, pts_continuous=True)

            def commit(repo, *, repo_type, operations, commit_message):
                self.assertEqual((repo, repo_type), ('private/output', 'dataset'))
                commits.append(operations)
                names = [op['path_in_repo'] for op in operations]
                if any(name.endswith('.mp4') for name in names):
                    self.assertEqual(events[-1], 'close')
                    events.append('final')
                else:
                    self.assertEqual(len(names), 1)
                    arm = names[0].removeprefix(worker.PREFIX).removesuffix('-progress.json')
                    self.assertEqual(events[-1], 'completed-'+arm)
                    events.append('progress-'+arm)

            hub = SimpleNamespace(HfApi=lambda **kw: SimpleNamespace(
                repo_info=lambda *a, **kw: SimpleNamespace(private=private), create_commit=commit),
                CommitOperationAdd=lambda **kw: kw)
            with patch.dict(sys.modules, huggingface_hub=hub), patch.dict(os.environ, HF_TOKEN='offline'), \
                    patch.object(worker.source_contract, 'preflight', side_effect=preflight), \
                    patch.object(worker.baseworker, 'verify_media', side_effect=verify) as media, \
                    patch.object(worker, 'LiveVoiceRuntime', FakeRuntime):
                if not private or existing:
                    with self.assertRaisesRegex(ValueError, 'private deployment|fresh private output'):
                        worker.run(root, paths, spec, 'private/output', time.monotonic()+60)
                    self.assertEqual(constructors, [])
                    self.assertEqual(commits, [])
                    return
                if bad_memory:
                    with self.assertRaises(ValueError):
                        worker.run(root, paths, spec, 'private/output', time.monotonic()+60)
                    self.assertEqual(events[-1], 'close')
                    self.assertEqual(commits, [])
                    return
                if fail_second:
                    with self.assertRaisesRegex(RuntimeError, 'injected second render'):
                        worker.run(root, paths, spec, 'private/output', time.monotonic()+60)
                    self.assertEqual(events[-1], 'close')
                    self.assertEqual(len(commits), 1)
                    self.assertEqual(commits[0][0]['path_in_repo'], worker.PREFIX+'reference1024-progress.json')
                    self.assertEqual(media.call_count, 5)
                else:
                    report = worker.run(root, paths, spec, 'private/output', time.monotonic()+60)
                    self.assertTrue(report['completed'])
                    for arm, first_hit in zip(worker.ARMS, (1, 4)):
                        row = report['arms'][arm]
                        self.assertEqual(len(row['window_memory']), 6)
                        self.assertEqual([m['cache_hit'] for m in row['window_memory']],
                            [hit for hit in range(first_hit, first_hit+3) for _ in range(2)])
                        self.assertEqual(row['rank_peak_memory'], [dict(worker_rank=rank,
                            peak_allocated_bytes=1000+first_hit+2+rank,
                            peak_reserved_bytes=2000+first_hit+2+rank) for rank in range(2)])
                        self.assertEqual(row['finite_supply']['first_wait_seconds'], 2.)
                        self.assertEqual(row['finite_supply']['max_stall_seconds'], 0.)
                    self.assertFalse(report['candidate_adopted'])
                    self.assertFalse(report['auto_adopt'])
                    self.assertEqual(report['perceptual_quality'], 'USER-review-pending')
                    self.assertEqual(events, ['start', 'render-reference1024', 'completed-reference1024',
                        'progress-reference1024', 'configure512', 'render-reference512',
                        'completed-reference512', 'progress-reference512', 'close', 'final'])
                    self.assertEqual(media.call_count, 10)
                    self.assertEqual(len(commits), 3)
                    self.assertEqual([op['path_in_repo'] for op in commits[-1]],
                        [worker.PREFIX+name for name in ('reference1024-review.mp4', 'reference512-review.mp4', 'report.json')])
                    self.assertEqual(json.loads(commits[-1][-1]['path_or_fileobj']), report)
                self.assertEqual(events.count('close'), 1)
                self.assertEqual(len(constructors), 1)
                kwargs = constructors[0]
                for key in ('continuation', 'window_audio_context', 'defer_progress_receipts',
                            'device_audio_validation', 'parallel_decode_waves', 'measure_memory'):
                    self.assertIs(kwargs[key], True)
                self.assertIs(kwargs['parallel_decode_acceptance'], False)
                self.assertIs(kwargs['profile_stages'], False)
                self.assertEqual(kwargs['conditioning_experiment']['active_short_edge'], 1024)
                for operations in commits:
                    for op in operations:
                        data = op['path_or_fileobj']
                        self.assertIsInstance(data, bytes)
                        self.assertNotIn(canary.encode(), data)
                        if op['path_in_repo'].endswith('.json'):
                            self.assertLessEqual(len(data), worker.MAX_JSON)
                            self.assertNotIn(b'unknown', data)
                            json.loads(data)

    def test_success_reaches_final_publication(self):
        self.exercise()

    def test_memory_receipts_fail_closed(self):
        for corruption in ('missing', ('peak_allocated_bytes', float('nan')),
                           ('peak_reserved_bytes', float('inf')), ('cache_hit', 4), ('worker_rank', 1)):
            with self.subTest(corruption=corruption):
                self.exercise(bad_memory=corruption)

    def test_second_render_failure_closes_with_first_progress_only(self):
        self.exercise(fail_second=True)

    def test_public_repo_rejected_before_runtime_construction(self):
        self.exercise(private=False)

    def test_existing_output_rejected_before_runtime_construction(self):
        self.exercise(existing=True)


if __name__ == '__main__':
    unittest.main()
