import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research import hf_h3_conditioning_exploration as worker


def evidence(edge=1024, rank=0):
    return dict(schema='h3-conditioning-experiment-v1', worker_rank=rank,
        session_id='session', window_index=2, cache_hit=True, precomputed_edges=[1024, 512],
        parameter_identity_preserved=True, denoise_stages_ready=True,
        R=1824 if edge == 1024 else 448, T=1867 if edge == 1024 else 491,
        policy=dict(schema='h3-conditioning-experiment-v1', short_edges=[1024, 512], active_short_edge=edge),
        resize_policy='shortedge-lanczos-grid32-v1',
        prepared_sizes=[[1024, 1824] if edge == 1024 else [512, 896]], processor_config_sha256='a'*64)


class WorkerTests(unittest.TestCase):
    def test_finite_supply(self):
        row = worker.finite_supply([2., 8., 13.])
        self.assertEqual(row['first_wait_seconds'], 2.)
        self.assertEqual(row['middle_ready_interval_seconds'], 6.)
        self.assertEqual(row['middle_generated_supply_seconds'], 4.25)
        deadlines = [2+119/24, 2+221/24]
        self.assertEqual(row['sequential_playback_deadlines_seconds'], deadlines)
        self.assertEqual(row['lateness_seconds'], [8-deadlines[0], 13-deadlines[1]])
        self.assertEqual(row['max_stall_seconds'], 13-deadlines[1])
        self.assertEqual(row['full_source_seconds'], 12.04)
        self.assertAlmostEqual(row['padding_seconds'], 328/24-12.04)
        for key in ('causal', 'long_run_acceptance', 'perceptual_acceptance', 'padding_is_new_conversation'):
            self.assertIs(row[key], False)
        for ready in ([], [1, 2], [0, 2, 3], [1, float('nan'), 3],
                      [1, 2, float('inf')], [True, 2, 3], [3, 2, 4]):
            with self.subTest(ready=ready), self.assertRaises(ValueError):
                worker.finite_supply(ready)

    def test_receipt_fail_closed(self):
        for edge in (1024, 512):
            row = evidence(edge)
            worker.validate_evidence(row, 0, edge, 'session')
            for key, value in [('R', 1), ('T', 1), ('window_index', 1),
                               ('session_id', 'stale'), ('cache_hit', False),
                               ('parameter_identity_preserved', False),
                               ('denoise_stages_ready', False), ('prepared_sizes', [[1, 2]])]:
                with self.subTest(edge=edge, key=key), self.assertRaises(ValueError):
                    worker.validate_evidence(dict(row, **{key: value}), 0, edge, 'session')

    def test_real_spec_admission(self):
        from test_h3_continuation_diagnostic_worker import spec_fixture
        spec = spec_fixture()
        spec['conditioning_exploration'] = worker.CONTRACT
        self.assertEqual(worker.validate_spec(spec), worker.CONTRACT)
        for change in ({'extra': True}, {'auto_adopt': 0}, {'baseline': dict(worker.CONTRACT['baseline'], ref_job_ids=['other'])}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_spec(dict(spec, conditioning_exploration=dict(worker.CONTRACT, **change)))

    def test_contract_import_is_detached(self):
        from research.hf_h3_conditioning_exploration import CONTRACT
        CONTRACT['baseline']['ref_job_ids'].append('tampered')
        self.assertEqual(worker.CONTRACT['baseline']['ref_job_ids'], ['6adbfe461e112550ce7dad1c'])

    def test_exact_size_and_resize_policy(self):
        for change in ({'prepared_sizes': [[1024, 2048]]}, {'resize_policy': 'other'},
                       {'parameter_identity_preserved': 1}, {'cache_hit': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_evidence(dict(evidence(), **change), 0, 1024, 'session')

    def test_bounds_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve()/'file'
            p.write_bytes(b'ab')
            deadline = time.monotonic()+10
            self.assertEqual(worker.read_bounded(p, 2, deadline), b'ab')
            with self.assertRaises(ValueError):
                worker.read_bounded(p, 1, deadline)
            link = Path(tmp).resolve()/'link'
            link.symlink_to(p)
            with self.assertRaises(ValueError):
                worker.read_bounded(link, 10, deadline)
        with self.assertRaises(ValueError):
            worker.bounded_json({'bad': float('nan')})
        with self.assertRaises(ValueError):
            worker.bounded_json({'large': 'x'*worker.MAX_JSON})

    def test_runtime_closed_on_start_failure(self):
        import sys
        import types
        from unittest.mock import Mock
        api = Mock()
        api.repo_info.return_value.private = True
        runtime = Mock()
        runtime.start.side_effect = RuntimeError('injected')
        hub = types.SimpleNamespace(HfApi=Mock(return_value=api), CommitOperationAdd=Mock())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()/'root'
            root.mkdir()
            paths = {k: root/k for k in ('speech', 'personal', 'base', 'smd', 'reference')}
            spec = dict(locked_speech={'path': 'full.wav'}, config={'timeout_seconds': 30}, identity_prefix='id')
            with patch.dict(sys.modules, {'huggingface_hub': hub}), patch.dict('os.environ', {'HF_TOKEN': 'fake'}), \
                    patch.object(worker, 'validate_spec', return_value={}), \
                    patch.object(worker.source_contract, 'preflight'), \
                    patch.object(worker, 'LiveVoiceRuntime', return_value=runtime) as constructor:
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    worker.run(root, paths, spec, 'private/output', time.monotonic()+30)
                runtime.close.assert_called_once()
                api.create_commit.assert_not_called()
                self.assertTrue(constructor.call_args.kwargs['continuation'])
                self.assertFalse(constructor.call_args.kwargs['parallel_decode_acceptance'])
                self.assertEqual(constructor.call_args.kwargs['conditioning_experiment']['short_edges'], [1024, 512])

    def test_contract_reuses_source_validator(self):
        plan = worker.CONTRACT
        with patch.object(worker.source_worker, 'validate_spec') as validate:
            self.assertEqual(worker.validate_spec({'conditioning_exploration': plan}), plan)
            validate.assert_called_once()
            with self.assertRaises(ValueError):
                worker.validate_spec({'conditioning_exploration': dict(plan, auto_adopt=True)})


if __name__ == '__main__':
    unittest.main()
