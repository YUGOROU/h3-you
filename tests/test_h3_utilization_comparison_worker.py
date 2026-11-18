import copy
import json
import sys
import tempfile
import types
from pathlib import Path
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

from research import hf_h3_utilization_comparison as worker
from research import launch_h3_continuation_diagnostic as launcher
from research.h3_personal.utilization_policy import normalize


class UtilizationWorkerTests(unittest.TestCase):
    def test_exact_public_policy_contracts(self):
        self.assertEqual(normalize(worker.TRACE_POLICY, 'trace'), worker.TRACE_POLICY)
        self.assertEqual(normalize(worker.DENOISE_POLICY, 'denoise'), worker.DENOISE_POLICY)
        contract = worker.comparison_contract()
        self.assertEqual(contract['arms'], ['baseline', 'trace', 'candidate_warmup', 'candidate'])
        self.assertEqual(contract['total_frames'], 328)
        self.assertIsNone(contract['numerical_tolerance'])
        contract['trace_policy']['max_events'] = 1
        self.assertEqual(worker.comparison_contract()['trace_policy']['max_events'], 200000)

    def test_spec_exclusivity_and_strict_types(self):
        with patch.object(worker.base, 'validate_spec'):
            spec = {'utilization_comparison': worker.comparison_contract()}
            worker.validate_spec(spec)
            for mode in worker.EXCLUSIVE:
                with self.assertRaises(ValueError):
                    worker.validate_spec(dict(spec, **{mode: {}}))
            broken = copy.deepcopy(spec)
            broken['utilization_comparison']['total_frames'] = 328.0
            with self.assertRaises(ValueError):
                worker.validate_spec(broken)

    def test_real_base_spec_and_policy_admission(self):
        from test_h3_continuation_diagnostic_worker import spec_fixture
        spec = spec_fixture()
        spec['utilization_comparison'] = worker.comparison_contract()
        worker.validate_spec(spec)
        for field in ('total_frames',):
            broken = copy.deepcopy(spec)
            broken['utilization_comparison'][field] = float(broken['utilization_comparison'][field])
            with self.assertRaises(ValueError):
                worker.validate_spec(broken)
        broken = copy.deepcopy(spec)
        broken['layout'] = 'landscape'
        with self.assertRaises(ValueError):
            worker.validate_spec(broken)

    def test_launcher_rejects_non_strict_contract_before_staging_checks(self):
        from test_h3_continuation_diagnostic_worker import spec_fixture
        spec = spec_fixture()
        spec['utilization_comparison'] = worker.comparison_contract()
        spec['utilization_comparison']['total_frames'] = 328.0
        with patch.object(launcher, 'inspect_base', return_value=(spec, {'files': []})):
            with self.assertRaisesRegex(ValueError, 'utilization comparison'):
                launcher.inspect('/unused')

    def rejection_profile(self):
        return {'utilization': dict(owner_complete=False, owner_terminal='faulted',
            candidate_rejected=True, trace_policy=None, denoise_policy=worker.DENOISE_POLICY,
            ranks=[dict(rank=i, schema='h3-window-utilization-v1', session_id='fresh',
                window_index=0, faulted=True, trace_policy=None,
                denoise_policy=worker.DENOISE_POLICY, candidate_rejected=True,
                parameter_identity_preserved=True) for i in (0, 1)])}

    def test_actual_rejection_requires_faulted_incomplete_both_rank_receipt(self):
        profile = self.rejection_profile()
        profile['utilization']['ranks'][0]['compile_seconds'] = 181
        rows = worker.rejected_evidence(profile, 'candidate_warmup')
        self.assertTrue(rows[0]['compile_time_budget_exceeded'])
        for key, value in [('owner_complete', True), ('owner_terminal', 'complete'),
                           ('candidate_rejected', False)]:
            broken = copy.deepcopy(profile)
            broken['utilization'][key] = value
            with self.assertRaises(ValueError):
                worker.rejected_evidence(broken, 'candidate_warmup')
        for key, value in [('session_id', 'other'), ('window_index', 1),
                           ('faulted', False), ('candidate_rejected', False),
                           ('parameter_identity_preserved', False), ('denoise_policy', None),
                           ('backend_compiles', 9)]:
            broken = copy.deepcopy(profile)
            broken['utilization']['ranks'][1][key] = value
            with self.assertRaises(ValueError):
                worker.rejected_evidence(broken, 'candidate_warmup')
        profile['utilization']['ranks'].pop()
        with self.assertRaises(ValueError):
            worker.rejected_evidence(profile, 'candidate_warmup')

    def test_cpu_only_trace_is_unavailable(self):
        row = worker.trace_scalars({'devices': {'cpu': dict(events=1, busy_us=1)}})
        self.assertFalse(row['available'])
        self.assertIn('cpu', row['devices'])

    def test_packaging_newest_owned_modules(self):
        for name in ('device_activity', 'local_denoise_compile', 'denoise_execution', 'utilization_policy'):
            self.assertIn('research/h3_personal/'+name+'.py', launcher.EXTRA)
        self.assertIn('research/hf_h3_utilization_comparison.py', launcher.EXTRA)

    def test_cli_offline_exclusive_prepare_only(self):
        for flags in (['--utilization-comparison'], ['--prepare', '--utilization-comparison', '--efficiency-comparison'],
                      ['--prepare', '--utilization-comparison', '--window-audio-context']):
            with self.assertRaises(SystemExit):
                launcher.main(['--directory', '/tmp/no', '--ledger', '/tmp/no', *flags])

    def test_topology_never_copies_identifiers(self):
        text = 'host-secret GPU-uuid serial123\nGPU0 X NV18 0-31 private\nGPU1 NV18 X hostname\n'
        result = worker.parse_topology(text)
        self.assertTrue(result['available'])
        self.assertNotIn('secret', str(result))
        self.assertFalse(worker.parse_topology('GPU0 X uuid\nGPU1 SYS X')['available'])

    def test_trace_summary_allowlist(self):
        report = {'devices': {'cuda': dict(events=2, busy_us=12, secret='raw-host',
            category_union_us={'compute': 12, 'raw-name': 3}, disjoint_us={'compute': 12},
            family_event_counts={'gemm': 2, 'raw-kernel': 8})}, 'coverage_proven': True}
        clean = worker.trace_scalars(report)
        self.assertNotIn('raw', str(clean))
        self.assertFalse(clean['coverage_proven'])
        report['devices']['cuda']['busy_us'] = float('nan')
        with self.assertRaises(ValueError):
            worker.trace_scalars(report)

    def test_owner_completion_and_arm_policy_required(self):
        with patch.object(worker.efficiency, 'validate_profile'):
            with self.assertRaises(ValueError):
                worker.validate_profile({}, 'baseline')
            profile = {'utilization': dict(owner_complete=True, trace_policy=None, denoise_policy=None)}
            worker.validate_profile(profile, 'baseline')
            with self.assertRaises(ValueError):
                worker.validate_profile(profile, 'trace')

    def test_trace_unavailable_is_not_coverage(self):
        profile = {'utilization': {'ranks': [{'rank': 0}, {'rank': 1}]}}
        evidence = worker.scalar_evidence(profile, 'trace')
        self.assertFalse(evidence[0]['trace_available'])
        self.assertFalse(evidence[0]['coverage_proven'])
        profile['utilization']['ranks'][0]['coverage_proven'] = True
        with self.assertRaises(ValueError):
            worker.scalar_evidence(profile, 'trace')

    def test_candidate_no_silent_eager_fallback(self):
        profile = {'utilization': {'ranks': [{'rank': 0}, {'rank': 1}]}}
        with self.assertRaises(ValueError):
            worker.scalar_evidence(profile, 'candidate')
        for row in profile['utilization']['ranks']:
            row.update(candidate_rejected=True, secret='never-copy')
        self.assertNotIn('secret', str(worker.scalar_evidence(profile, 'candidate')))
        profile['utilization']['ranks'][0]['backend_compiles'] = 9
        with self.assertRaises(ValueError):
            worker.scalar_evidence(profile, 'candidate')

    def test_numerical_uint8_no_auto_tolerance(self):
        with patch.object(worker.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, stdout=bytes([0, 2, 255])),
                subprocess.CompletedProcess([], 0, stdout=bytes([0, 3, 253]))]):
            result = worker.compare_rgb('a', 'b', time.monotonic()+20, width=1, height=1, frames=1)
        self.assertEqual(result['max'], 2)
        self.assertEqual(result['mean'], 1)
        self.assertFalse(result['quality_accepted'])
        self.assertIsNone(result['tolerance'])

    def test_decode_wrong_frames_refused(self):
        with patch.object(worker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=b'')):
            with self.assertRaises(ValueError):
                worker.compare_rgb('a', 'b', time.monotonic()+20, width=1, height=1, frames=1)

    def exercise_runtime(self, rejected=False, fault=False, diagnosed=False):
        from research.h3_personal import live_voice_runtime
        events = []
        self_profile = self.rejection_profile()
        api = Mock()
        api.repo_info.return_value.private = True
        hub = types.SimpleNamespace(HfApi=Mock(return_value=api), CommitOperationAdd=lambda **k: k)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'root'
            root.mkdir()
            class Runtime:
                def __init__(self, **kwargs):
                    events.append(('init', kwargs))
                def start(self):
                    events.append(('start',))
                def configure_utilization(self, **kwargs):
                    events.append(('configure', kwargs))
                def close(self):
                    events.append(('close',))
                def render_continuation(self, wav, sha, turn):
                    arm = turn.name
                    events.append(('render', arm))
                    if fault and arm == 'candidate_warmup':
                        raise RuntimeError('GPU fault must propagate')
                    turn.mkdir(parents=True)
                    rows = [dict(rank=i, denoise_installed=True, parameter_identity_preserved=True,
                                 representative_preflight_complete=True,
                                 candidate_rejected=rejected and arm == 'candidate_warmup') for i in (0, 1)]
                    profile = dict(session_id=arm, input_proof={}, derived_adapter_receipt={},
                        utilization=dict(owner_complete=True,
                            trace_policy=worker.TRACE_POLICY if arm == 'trace' else None,
                            denoise_policy=worker.DENOISE_POLICY if arm.startswith('candidate') else None,
                            ranks=rows))
                    (turn/'profile.json').write_text(json.dumps(profile))
                    if diagnosed and arm == 'candidate_warmup':
                        (turn/'profile.json').write_text(json.dumps(self_profile))
                        raise worker.CandidateRejected('representative gate')
                    for name in worker.base.VIDEOS:
                        (turn/name).write_bytes(b'private-media')
                    (turn/'raw.trace').write_text('private')
                    return types.SimpleNamespace(profile=turn/'profile.json')
            def strict_preflight(path, *, timeout_seconds):
                self.assertIs(type(timeout_seconds), int)
                self.assertGreaterEqual(timeout_seconds, 1)
                self.assertLessEqual(timeout_seconds, 100)
            with patch.dict(sys.modules, huggingface_hub=hub), patch.dict('os.environ', HF_TOKEN='test'), \
                    patch.object(live_voice_runtime, 'LiveVoiceRuntime', Runtime), \
                    patch.object(worker, 'validate_spec'), patch.object(worker.source, 'preflight', side_effect=strict_preflight), \
                    patch.object(worker, 'topology', return_value={}), \
                    patch.object(worker, 'device_mapping', return_value=[]), \
                    patch.object(worker.efficiency, 'validate_profile'), \
                    patch.object(worker, 'verify_media', return_value={'frames': 328}), \
                    patch.object(worker, 'timing_scalars', return_value={'total_wall_seconds': 1}), \
                    patch.object(worker, 'compare_rgb', return_value={'exact': True}):
                args = (root, {k: root/k for k in ('speech', 'personal', 'base', 'smd', 'reference')},
                        dict(locked_speech={'path': 'fixed.wav'}, config={'timeout_seconds': 100},
                             identity_prefix='identity'), 'private/deployment', time.monotonic()+100)
                if fault or (rejected and not diagnosed):
                    with self.assertRaises((RuntimeError, ValueError)):
                        worker.run(*args)
                    self.assertEqual(api.create_commit.call_count, 2)
                    for call, arm in zip(api.create_commit.call_args_list, ('baseline', 'trace')):
                        operations = call.kwargs['operations']
                        self.assertEqual(len(operations), 1)
                        self.assertEqual(operations[0]['path_in_repo'],
                                         f'outputs/utilization/progress-{arm}.json')
                        progress = json.loads(Path(operations[0]['path_or_fileobj']).read_text())
                        self.assertFalse(progress['candidate_adopted'])
                        self.assertTrue(progress['completed_owner'])
                        self.assertEqual(progress['arm'], arm)
                else:
                    receipt = worker.run(*args)
                    self.assertEqual(receipt['candidate_measured'], not rejected)
                    operations = api.create_commit.call_args.kwargs['operations']
                    self.assertEqual(len(operations), 1)
                    self.assertTrue(operations[0]['path_in_repo'].endswith('comparison-profile.json'))
            self.assertEqual(events[-1], ('close',))
            init = events[0][1]
            self.assertIsNone(init['trace_policy'])
            self.assertIsNone(init['denoise_policy'])
            self.assertFalse(init['profile_stages'])
            arms = [e[1] for e in events if e[0] == 'render']
            self.assertEqual(arms, worker.comparison_contract()['arms'][:len(arms)])
            self.assertEqual(sum(e[0] == 'start' for e in events), 1)

    def test_same_runtime_order_and_scalar_only_publication(self):
        self.exercise_runtime()

    def test_successful_fake_owner_rejection_is_error(self):
        self.exercise_runtime(rejected=True)

    def test_actual_exception_publishes_rejection_without_missing_media(self):
        self.exercise_runtime(rejected=True, diagnosed=True)

    def test_gpu_fault_close_and_no_publication_or_fallback(self):
        self.exercise_runtime(fault=True)

    def test_deadline_not_renewed(self):
        with self.assertRaises(Exception):
            worker.compare_rgb('a', 'b', time.monotonic()-1, width=1, height=1, frames=1)


if __name__ == '__main__':
    unittest.main()
