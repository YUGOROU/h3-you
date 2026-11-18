"""Offline mocks only: no Hub, GPU, FFmpeg, private speech or devices."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import hf_h3_continuation_diagnostic as worker
from research import bootstrap_h3_fast_ref2va as bootstrap
from research.h3_personal.continuation_diagnostic import source_contract, validate_metadata
from research.h3_personal.continuation_plan import plan


def spec_fixture():
    source = source_contract()
    planned = dict(schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
        sha256=source['sha256'], source_samples=577920, source_sample_rate=48000,
        source_channels=1, total_frames=328, window_frames=124, overlap_frames=22,
        expected_windows=3, sampler='SMD4', fps=24, seed=314159,
        windows=[w.record() for w in plan(328, 124, 22)])
    return dict(config=dict(timeout_seconds=600),
        continuation_diagnostic=validate_metadata(source, planned, timeout_seconds=600),
        locked_speech=dict(repo=source['repo'], revision=source['revision'], path=source['path'],
                           sha256=source['sha256'], samples=577920, sample_rate=48000),
        layout='portrait', runtime_policy=worker.POLICY, attention_mode='fa4-packed',
        reference_sha256=worker.REFERENCE_SHA, identity_prefix='identity')


class WorkerTests(unittest.TestCase):
    def exercise(self, failure=None, comparison=False, candidate=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'deployment'
            root.mkdir()
            turn = root.with_name(root.name+'-continuation-generated')/'continuation'
            paths = {key: root/key for key in ('speech', 'personal', 'base', 'smd', 'reference')}
            runtime = Mock()
            def render(*args):
                target = args[2]
                target.mkdir(parents=True)
                for name in (*worker.VIDEOS, *worker.RECEIPTS, 'speech.wav', 'latent.pt'):
                    (target/name).write_bytes(b'generated mock')
                (target/'profile.json').write_text(json.dumps(
                    dict(derived_adapter_receipt={'sha256': 'a'*64},
                         windows=[dict(seed=314159+i) for i in range(3)])))
                if candidate is not None:
                    profile = json.loads((target/'profile.json').read_text())
                    profile['continuation_plan'] = {'candidate': copy.deepcopy(candidate)}
                    (target/'profile.json').write_text(json.dumps(profile))
                if comparison or candidate is not None:
                    stage_dir = target.parent/'stage-profile'
                    stage_dir.mkdir(exist_ok=True)
                    for rank in range(2):
                        for hit in range(1, 7 if comparison else 4):
                            timing = dict(schema='h3-ref2va-warm-stage-diagnostic-v1',
                                worker_rank=rank, cache_hit=hit, synchronization_added=True,
                                benchmark_comparable=False, stages=[dict(name=n, host_seconds=0.1,
                                    default_stream_elapsed_ms=2.0) for n in (
                                    'latent_preparation_stage', 'denoising_stage',
                                    'video_decoding_stage', 'audio_decoding_stage')])
                            path = stage_dir/f'rank-{rank}-warm-{hit}.json'
                            path.write_text(json.dumps(timing))
                    if target.name == 'continuation':
                        path = stage_dir/f'rank-1-warm-{6 if comparison else 3}.json'
                        if failure == 'missing_timing':
                            path.unlink()
                        elif failure in ('tampered_timing', 'raw_timing', 'nan_timing'):
                            timing = json.loads(path.read_text())
                            if failure == 'tampered_timing':
                                timing['cache_hit'] = 1
                            elif failure == 'raw_timing':
                                timing['raw'] = [1, 2]
                            else:
                                timing['stages'][0]['host_seconds'] = float('nan')
                            path.write_text(json.dumps(timing))
                        if failure in ('seed', 'windows'):
                            profile = json.loads((target/'profile.json').read_text())
                            if failure == 'seed':
                                profile['windows'][0]['seed'] = 42
                            else:
                                profile['windows'].pop()
                            (target/'profile.json').write_text(json.dumps(profile))
                return SimpleNamespace(profile=target/'profile.json',
                                       review_with_speech=target/'review-with-speech.mp4')
            runtime.render_continuation.side_effect = render
            runtime.render.side_effect = render
            if failure in ('start', 'render', 'render_continuation'):
                getattr(runtime, failure).side_effect = RuntimeError('injected failure')
            def close():
                if candidate is not None and failure == 'candidate_drift':
                    path = turn/'profile.json'
                    profile = json.loads(path.read_text())
                    profile['continuation_plan']['candidate']['audio_context'] = False
                    path.write_text(json.dumps(profile))
            runtime.close.side_effect = close
            api = Mock()
            api.repo_info.return_value.private = True
            api.create_commit.return_value.oid = 'mock-commit'
            hub = SimpleNamespace(HfApi=Mock(return_value=api),
                                  CommitOperationAdd=lambda **kw: SimpleNamespace(**kw))
            spec = spec_fixture()
            if candidate is not None:
                spec['continuation_candidate'] = copy.deepcopy(candidate)
            if comparison:
                spec['continuation_comparison'] = copy.deepcopy(worker.CONTINUATION_COMPARISON)
            with patch.dict(sys.modules, huggingface_hub=hub), \
                    patch.dict(os.environ, HF_TOKEN='offline-test'), \
                    patch.object(worker, 'LiveVoiceRuntime', return_value=runtime) as constructor, \
                    patch.object(worker.contract, 'preflight', return_value={'local_source_verified': True}) as preflight, \
                    patch.object(worker.subprocess, 'run') as decode:
                if failure == 'preflight':
                    preflight.side_effect = ValueError('bad source')
                if failure == 'decode':
                    decode.side_effect = RuntimeError('bad decode')
                if failure:
                    with self.assertRaises((RuntimeError, ValueError)):
                        worker.run(root, paths, spec, 'private/output', time.monotonic()+500)
                    api.create_commit.assert_not_called()
                else:
                    receipt = worker.run(root, paths, spec, 'private/output', time.monotonic()+500)
                    self.assertEqual(receipt['expected_windows'], 3)
                    self.assertTrue(all(receipt[k] >= 0 for k in
                                        ('startup_seconds', 'render_seconds', 'total_seconds')))
                    self.assertEqual(decode.call_count, 10 if comparison else 5)
                    if comparison:
                        self.assertEqual(receipt['render_calls'], 2)
                        self.assertEqual(len(receipt['comparison']['stage_timing_artifacts']), 12)
                        self.assertEqual(receipt['comparison']['independent_first_seed'], 314159)
                        self.assertEqual(receipt['comparison']['continuation_seed'], 314159)
                        self.assertTrue(receipt['comparison']['actual_adapter_receipts_equal'])
                        self.assertEqual([c[0] for c in runtime.method_calls],
                                         ['start', 'render', 'render_continuation', 'close'])
                    if candidate is not None:
                        self.assertEqual(receipt['continuation_candidate']['options'], candidate)
                        self.assertIs(receipt['continuation_candidate']['quality_verified'], False)
                        self.assertEqual(len(receipt['continuation_candidate']['stage_timing_artifacts']), 6)
                        self.assertEqual(receipt['render_calls'], 1)
                        self.assertEqual([c[0] for c in runtime.method_calls],
                                         ['start', 'render_continuation', 'close'])
                    else:
                        self.assertNotIn('continuation_candidate', receipt)
                    for call in decode.call_args_list:
                        self.assertIn('-xerror', call.args[0])
                        self.assertIn('null', call.args[0])
                        self.assertLessEqual(call.kwargs['timeout'], 120)
                    operations = api.create_commit.call_args.kwargs['operations']
                    self.assertEqual({o.path_in_repo for o in operations},
                        {worker.PREFIX+n for n in (*worker.VIDEOS, *worker.RECEIPTS, 'diagnostic-profile.json')} |
                        ({worker.PREFIX+'independent/'+n for n in (*worker.VIDEOS, *worker.RECEIPTS)}
                         if comparison else set()) |
                        ({worker.PREFIX+f'stage-profile/rank-{rank}-warm-{hit}.json'
                          for rank in range(2) for hit in range(1, 7 if comparison else 4)}
                         if comparison or candidate is not None else set()))
                    self.assertFalse(any(o.path_in_repo.endswith(('.wav', '.pt')) for o in operations))
                preflight.assert_called_once_with(paths['speech']/worker.contract.SOURCE_PATH,
                                                 timeout_seconds=600)
                if failure == 'preflight':
                    constructor.assert_not_called()
                else:
                    self.assertTrue(constructor.call_args.kwargs['continuation'])
                    for key, expected in (
                            ('window_audio_context', candidate is not None and candidate['audio_context'] == 'window-context-canonical-v1'),
                            ('defer_progress_receipts', candidate is not None and candidate['progress_receipts'] == 'bind-and-commit-v1'),
                            ('profile_stages', True)):
                        if candidate is not None:
                            self.assertIs(constructor.call_args.kwargs[key], expected)
                        else:
                            self.assertNotIn(key, constructor.call_args.kwargs)
                    if comparison:
                        self.assertIs(constructor.call_args.kwargs['comparison_diagnostics'], True)
                    else:
                        self.assertNotIn('comparison_diagnostics', constructor.call_args.kwargs)
                    runtime.start.assert_called_once_with()
                    runtime.close.assert_called_once_with()
                    self.assertEqual(runtime.render_continuation.call_count, int(failure not in ('start', 'render')))
                    if comparison and failure != 'start':
                        runtime.render.assert_called_once_with(
                            paths['speech']/worker.contract.SOURCE_PATH,
                            worker.contract.SOURCE_SHA256, turn.parent/'independent')
                    else:
                        runtime.render.assert_not_called()
                    if failure not in ('start', 'render'):
                        runtime.render_continuation.assert_called_once_with(
                            paths['speech']/worker.contract.SOURCE_PATH,
                            worker.contract.SOURCE_SHA256, turn)

    def test_candidate_admission(self):
        valid = dict(schema='h3-continuation-candidate-v1',
                     audio_context='window-context-canonical-v1',
                     progress_receipts='bind-and-commit-v1')
        for value in (None, False, {}, dict(valid, extra=True),
                      dict(valid, audio_context=False), dict(valid, progress_receipts='unknown'),
                      dict(valid, audio_context='full-source-v1', progress_receipts='per-step-v1')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.validate_spec(dict(spec_fixture(), continuation_candidate=value))
        for key, value in (('continuation_comparison', worker.CONTINUATION_COMPARISON),
                           ('decode_parity_diagnostic', worker.DECODE_PARITY_DIAGNOSTIC)):
            spec = dict(spec_fixture(), continuation_candidate=valid)
            spec[key] = value
            with self.assertRaises(ValueError):
                worker.validate_spec(spec)

    def test_candidate_toggles_and_scalar_publication(self):
        for audio, progress in ((True, False), (False, True), (True, True)):
            candidate = dict(schema='h3-continuation-candidate-v1',
                audio_context='window-context-canonical-v1' if audio else 'full-source-v1',
                progress_receipts='bind-and-commit-v1' if progress else 'per-step-v1')
            with self.subTest(audio=audio, progress=progress):
                self.exercise(candidate=candidate)
        for failure in ('candidate_drift', 'missing_timing', 'tampered_timing',
                        'raw_timing', 'nan_timing'):
            with self.subTest(failure=failure):
                self.exercise(failure, candidate=candidate)

    def test_decode_parity_contract_and_receipts(self):
        spec = spec_fixture()
        spec['decode_parity_diagnostic'] = copy.deepcopy(worker.DECODE_PARITY_DIAGNOSTIC)
        worker.validate_spec(spec)
        for key, value in [('serial', 1), ('atol', 0), ('total_frames', 328.0),
                           ('arrivals', [37, 30, True]), ('extra', True)]:
            bad = copy.deepcopy(spec)
            bad['decode_parity_diagnostic'][key] = value
            with self.assertRaises(ValueError):
                worker.validate_spec(bad)
        spec['continuation_comparison'] = worker.CONTINUATION_COMPARISON
        with self.assertRaises(ValueError):
            worker.validate_spec(spec)

    def test_decode_parity_publication_and_failures(self):
        for failure in (None, 'missing', 'tamper', 'sha'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/'deployment'
                root.mkdir()
                output = root.with_name(root.name+'-continuation-generated')
                turn = output/'continuation'
                spec = spec_fixture()
                spec['decode_parity_diagnostic'] = copy.deepcopy(worker.DECODE_PARITY_DIAGNOSTIC)
                runtime = Mock()
                def render(*args):
                    turn.mkdir(parents=True)
                    for name in (*worker.VIDEOS, *worker.RECEIPTS):
                        (turn/name).write_text('{}')
                    (turn/'profile.json').write_text(json.dumps({'session_id': 'session'}))
                    return SimpleNamespace(profile=turn/'profile.json',
                        review_with_speech=turn/'review-with-speech.mp4')
                runtime.render_continuation.side_effect = render
                def close():
                    coverage = output/'coverage'
                    coverage.mkdir()
                    for rank in range(2):
                        if failure == 'missing' and rank == 1:
                            continue
                        metric = dict(max=1.0, mean=0.1, per_frame_max=[1.0]*328,
                            per_frame_mean=[0.1]*328, exact=False, within_atol=False)
                        row = dict(schema='h3-decode-parity-v1', session_id='session',
                            worker_rank=rank, total_frames=328, arrivals=[37,30,30],
                            emitted_counts=[119,102,107], normalized_shape=[1,24,97,28,16],
                            decoded_shape=[1,3,328,448,256], inputs_unchanged=True,
                            frames_exactly_once=True, closed=True, native_parallel_verified=False,
                            diagnostic_only=True, quality_verified=False, realtime_verified=False,
                            serial=True, atol=0.0,
                            source_sha256='e3cec57d51623c8343c3bc25160e863d253997fb4158aa283bba874154f70fa1',
                            source_revision='ba371e33ed080e51f1884ae166baf7c4447a66fe',
                            cuda_fp16_autocast=True, denormalized_dtype='torch.float32',
                            parity=False, input_sha256=('b' if failure == 'sha' and rank else 'a')*64,
                            raw_float=metric, denormalized_pixels=metric, uint8=metric,
                            untrusted_raw_tensor=[1,2,3])
                        if failure == 'tamper':
                            row['raw_float']['max'] = float('nan')
                        (coverage/f'decode-parity-rank-{rank}.json').write_text(json.dumps(row))
                runtime.close.side_effect = close
                api = Mock()
                api.repo_info.return_value.private = True
                hub = SimpleNamespace(HfApi=Mock(return_value=api),
                    CommitOperationAdd=lambda **kw: SimpleNamespace(**kw))
                paths = {k: root/k for k in ('speech','personal','base','smd','reference')}
                with patch.dict(sys.modules, huggingface_hub=hub), \
                        patch.dict(os.environ, HF_TOKEN='offline'), \
                        patch.object(worker, 'LiveVoiceRuntime', return_value=runtime) as constructor, \
                        patch.object(worker.contract, 'preflight', return_value={}), \
                        patch.object(worker.subprocess, 'run'):
                    if failure:
                        with self.assertRaises(ValueError):
                            worker.run(root, paths, spec, 'private/output', time.monotonic()+500)
                        api.create_commit.assert_not_called()
                    else:
                        receipt = worker.run(root, paths, spec, 'private/output', time.monotonic()+500)
                        self.assertFalse(receipt['decode_parity']['parity'])
                        operations = api.create_commit.call_args.kwargs['operations']
                        parity = [o for o in operations if '/decode-parity/' in o.path_in_repo]
                        self.assertEqual(len(parity), 2)
                        for op in parity:
                            row = json.loads(Path(op.path_or_fileobj).read_text())
                            self.assertNotIn('untrusted_raw_tensor', row)
                            self.assertIs(row['serial'], True)
                            self.assertIs(type(row['atol']), float)
                            self.assertEqual(row['atol'], 0.0)
                            self.assertEqual(row['source_sha256'],
                                'e3cec57d51623c8343c3bc25160e863d253997fb4158aa283bba874154f70fa1')
                            self.assertEqual(row['source_revision'],
                                'ba371e33ed080e51f1884ae166baf7c4447a66fe')
                            self.assertIs(row['cuda_fp16_autocast'], True)
                            self.assertEqual(row['denormalized_dtype'], 'torch.float32')
                            worker.validate_parity(row, row['worker_rank'], 'session')
                            # Pinned H3 VAE is spatial16, not spatial8. Actual
                            # packed coverage is112rows/time (2x2 patch), not448.
                            _, _, _, h, w = row['normalized_shape']
                            self.assertEqual((h, w), (448//16, 256//16))
                            self.assertEqual((h//2)*(w//2), 112)
                            for key, value in [('normalized_shape', [1,24,97,56,32]), ('serial', 1), ('atol', 0), ('atol', 1e-6),
                                    ('source_sha256', '0'*64), ('source_revision', '0'*40),
                                    ('cuda_fp16_autocast', False),
                                    ('denormalized_dtype', 'torch.float16'), ('parity', True)]:
                                bad = copy.deepcopy(row)
                                bad[key] = value
                                with self.subTest(key=key), self.assertRaises(ValueError):
                                    worker.validate_parity(bad, row['worker_rank'], 'session')
                            for key in ('serial', 'atol', 'source_sha256', 'source_revision',
                                        'cuda_fp16_autocast', 'denormalized_dtype'):
                                bad = copy.deepcopy(row)
                                del bad[key]
                                with self.subTest(missing=key), self.assertRaises(ValueError):
                                    worker.validate_parity(bad, row['worker_rank'], 'session')
                            for name in ('raw_float', 'denormalized_pixels', 'uint8'):
                                for key, value in [('exact', True), ('within_atol', True),
                                                   ('max', 0), ('mean', 2.0)]:
                                    bad = copy.deepcopy(row)
                                    bad[name][key] = value
                                    with self.subTest(metric=name, key=key), self.assertRaises(ValueError):
                                        worker.validate_parity(bad, row['worker_rank'], 'session')
                            exact = copy.deepcopy(row)
                            for name in ('raw_float', 'denormalized_pixels', 'uint8'):
                                exact[name] = dict(max=0.0, mean=0.0,
                                    per_frame_max=[0.0]*328, per_frame_mean=[0.0]*328,
                                    exact=True, within_atol=True)
                            exact['parity'] = True
                            worker.validate_parity(exact, row['worker_rank'], 'session')
                            exact['parity'] = False
                            with self.assertRaises(ValueError):
                                worker.validate_parity(exact, row['worker_rank'], 'session')
                        self.assertFalse(any('/stage-profile/' in o.path_in_repo or
                            o.path_in_repo.endswith(('.pt','.wav')) for o in operations))
                    self.assertIs(constructor.call_args.kwargs['decode_parity'], True)
                    runtime.render.assert_not_called()
                    runtime.render_continuation.assert_called_once()
                    runtime.close.assert_called_once()

    def test_comparison_contract_rejection(self):
        valid = spec_fixture()
        valid['continuation_comparison'] = copy.deepcopy(worker.CONTINUATION_COMPARISON)
        worker.validate_spec(valid)
        for value in (None, {}, True, dict(worker.CONTINUATION_COMPARISON, full_source=1),
                      dict(worker.CONTINUATION_COMPARISON, extra=True),
                      dict(worker.CONTINUATION_COMPARISON, order=list(reversed(worker.CONTINUATION_COMPARISON['order'])))):
            bad = copy.deepcopy(valid)
            bad['continuation_comparison'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.validate_spec(bad)

    def test_comparison_warm_pair_and_failures(self):
        self.exercise(comparison=True)
        for failure in ('start', 'render', 'render_continuation', 'decode'):
            with self.subTest(failure=failure):
                self.exercise(failure, comparison=True)

    def test_comparison_rejects_bad_evidence(self):
        for failure in ('missing_timing', 'tampered_timing', 'raw_timing', 'nan_timing',
                        'seed', 'windows'):
            with self.subTest(failure=failure):
                self.exercise(failure, comparison=True)

    def test_success_allowlisted_publication(self):
        self.exercise()

    def test_failures_close_without_replay(self):
        for failure in ('start', 'render_continuation', 'decode', 'preflight'):
            with self.subTest(failure=failure):
                self.exercise(failure)

    def test_metadata_source_and_incompatible_specs(self):
        spec = spec_fixture()
        worker.validate_spec(spec)
        spec['continuation_diagnostic']['local_source_verified'] = True
        worker.validate_spec(spec)
        for key, value in [('live_voice_session', {}), ('layout', 'landscape'),
                           ('attention_mode', 'fa2'), ('regional_compile', True),
                           ('runtime_policy', 'baseline'), ('incremental_transfer', True),
                           ('continuation_diagnostic', None)]:
            bad = copy.deepcopy(spec)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_spec(bad)
        bad = copy.deepcopy(spec)
        bad['locked_speech']['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            worker.validate_spec(bad)
        for timeout in (True, 25, 3601, float('inf')):
            bad = copy.deepcopy(spec)
            bad['config']['timeout_seconds'] = timeout
            with self.assertRaises(ValueError):
                worker.validate_spec(bad)

    def test_asset_provenance(self):
        spec = spec_fixture()
        speech = spec['locked_speech']
        assets = [dict(role='speech', repo_type='dataset', repo=speech['repo'],
                       revision=speech['revision'], files=[dict(path=speech['path'], sha256=speech['sha256'])])]
        assets.extend(dict(role=role) for role in ('base', 'personal', 'reference', 'smd'))
        worker.validate_spec(spec, assets)
        assets[0]['revision'] = '0'*40
        with self.assertRaises(ValueError):
            worker.validate_spec(spec, assets)

    def test_expired_deadline_before_source_or_runtime(self):
        with patch.object(worker.contract, 'preflight') as preflight, \
                patch.object(worker, 'LiveVoiceRuntime') as runtime:
            with self.assertRaises(TimeoutError):
                worker.run(Path('/unused'), {}, spec_fixture(), 'unused/repo', time.monotonic()-1)
            preflight.assert_not_called()
            runtime.assert_not_called()

    def test_bootstrap_fixed_deadline_and_no_retry(self):
        for fail in (False, True):
            with self.subTest(timeout=fail), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch('time.monotonic', return_value=100), \
                    patch.object(bootstrap.subprocess, 'Popen') as popen, \
                    patch.object(os, 'killpg') as kill:
                process = popen.return_value
                process.pid = 12345
                process.wait.side_effect = ([subprocess.TimeoutExpired('mock', 475), 0]
                                            if fail else [0, 0])
                if fail:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        bootstrap.dispatch_continuation(Path('/staged'), spec_fixture(), 'private/output', 0)
                else:
                    bootstrap.dispatch_continuation(Path('/staged'), spec_fixture(), 'private/output', 0)
                popen.assert_called_once()
                command = popen.call_args.args[0]
                self.assertEqual(command[2], 'research.hf_h3_continuation_diagnostic')
                self.assertEqual(float(command[-1]), 575)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                self.assertEqual(process.wait.call_args_list[0].kwargs['timeout'], 475)
                kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
