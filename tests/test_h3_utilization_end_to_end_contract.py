"""CPU integration of actual receipt producers/consumers; no Torch/GPU required."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research.h3_personal.denoise_execution import RequestAdapter
from research.h3_personal import fast_ref2va as forward
from research.h3_personal import fast_continuation_render as renderer
from research.h3_personal.device_activity import analyze, Policy
from research import hf_h3_utilization_comparison as worker
from research import hf_h3_efficiency_comparison as efficiency


class EndToEndContractTests(unittest.TestCase):
    session = '../private-session'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.coverage = Path(self.temp.name)
        self.rank = 0
        # Only the APIs used by receipt writing exist; importing Torch is unnecessary.
        torch = NS(distributed=NS(get_rank=lambda: self.rank), cuda=NS(
            current_device=lambda: self.rank, max_memory_allocated=lambda d: 80000000000,
            max_memory_reserved=lambda d: 88000000000,
            mem_get_info=lambda d: (40000000000, 140000000000)))
        patches = [patch.dict('sys.modules', {'torch': torch}),
                   patch.dict(os.environ, {'H3_REF2VA_COVERAGE_DIR': str(self.coverage)})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def controller(self, index, rejected=False):
        adapter = RequestAdapter.__new__(RequestAdapter)
        adapter.session_id, adapter.window_index = self.session, index
        adapter.policy = dict(worker.DENOISE_POLICY)
        adapter.controller = NS(state='faulted' if rejected else 'ready',
            budget=NS(count=4, seconds=2.5), blocks=[object() for _ in range(50)],
            identity=('unchanged-parameter-ids',),
            _identity=lambda: ('unchanged-parameter-ids',),
            receipt=dict(candidate_rejected=rejected, identity_unchanged=True,
                representative_preflight_complete=not rejected, agreements=3,
                agreement_seconds=.002, representative_metrics={'exact': not rejected,
                    'max': 1.0 if rejected else 0.0, 'mean': 0.0}))
        adapter._write_receipt()
        return adapter

    def trace_report(self):
        def event(start, end, name, device):
            return NS(time_range=NS(start=start, end=end), name=name, device_type=device)
        return analyze([event(100, 150, 'h3.warm_forward', 'cpu'),
                        event(101, 140, 'h3.block.49', 'cpu'),
                        event(105, 130, 'ampere_sgemm_128x64', 'cuda')],
                       policy=Policy(enabled=True))

    def write_pair(self, index, trace=None, denoise=None, rejected=False):
        rows = []
        for rank in (0, 1):
            self.rank = rank
            controller = self.controller(index, rejected) if denoise else None
            pipeline = NS()
            if trace and index == trace['selected_window']:
                pipeline._h3_window_trace_report = self.trace_report()
            manifest = dict(session_id=self.session, index=index, measure_memory=True)
            forward._write_utilization(pipeline, manifest, trace, denoise,
                                       controller, 10.0, 12.0, not rejected)
            path = self.coverage / forward.utilization_name(self.session, index, rank)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent, self.coverage)
            rows.append(json.loads(path.read_text()))
        return rows

    def snapshots(self, trace=None, denoise=None):
        snapshots = []
        # v7's three suffixes end at 119, 221, 328, not three full windows.
        for index, (offset, count) in enumerate(((0, 119), (119, 102), (221, 107))):
            snapshots.append(dict(session_id=self.session, seed=314159,
                fixed_reference_sha256=efficiency.baseworker.REFERENCE_SHA,
                wav_sha256=efficiency.source_contract.SOURCE_SHA256,
                utilization_receipts=self.write_pair(index, trace, denoise),
                continuation_stage_receipts=[dict(worker_rank=r, committed=True,
                    final=index == 2, total_frames=328 if index == 2 else offset+count,
                    window_index=index, session_id=self.session) for r in (0, 1)],
                continuation_decode_receipts=[dict(worker_rank=r, final=index == 2,
                    frame_offset=offset, frame_count=count, window_index=index,
                    session_id=self.session) for r in (0, 1)]))
        return snapshots

    def profile(self, snapshots, trace=None, denoise=None):
        return dict(session_id=self.session, unique_frames=328, speech_pcm_exact=True,
            full_decode_verified=True, derived_adapter_receipt={'sha256': 'a'*64},
            input_proof={'reference_sha256': efficiency.baseworker.REFERENCE_SHA,
                         'source_sha256': efficiency.source_contract.SOURCE_SHA256},
            continuation_plan=dict(candidate=efficiency.make_options(
                window_audio_context=True, defer_progress_receipts=True),
                efficiency=efficiency.efficiency_options(device_audio_validation=True,
                    parallel_decode_waves=True), parallel_acceptance=False),
            windows=snapshots, utilization=renderer.profile_evidence(
                snapshots, trace, denoise, 328))

    def test_candidate_real_controller_to_worker(self):
        snapshots = self.snapshots(denoise=worker.DENOISE_POLICY)
        profile = self.profile(snapshots, denoise=worker.DENOISE_POLICY)
        self.assertIs(worker.validate_profile(profile, 'candidate'), profile)
        rows = worker.scalar_evidence(profile, 'candidate')
        self.assertTrue(all(r['denoise_installed'] and
                            r['representative_preflight_complete'] for r in rows))
        self.assertEqual([r['backend_compiles'] for r in rows], [4, 4])
        for pair in profile['utilization']['windows']:
            self.assertEqual([r['session_id'] for r in pair], [self.session]*2)

    def test_trace_selected_window_survives_final_unselected_window(self):
        snapshots = self.snapshots(trace=worker.TRACE_POLICY)
        self.assertIsNone(snapshots[-1]['utilization_receipts'][0]['trace_summary'])
        profile = self.profile(snapshots, trace=worker.TRACE_POLICY)
        worker.validate_profile(profile, 'trace')
        rows = worker.scalar_evidence(profile, 'trace')
        for row in rows:
            self.assertTrue(row['trace_available'])
            self.assertFalse(row['coverage_proven'])
            self.assertEqual(row['trace_summary']['semantic_intervals'][0]['label'],
                             'h3.warm_forward')
            self.assertEqual(row['trace_summary']['devices']['cuda']['busy_us'], 25)

    def test_memory_only_default_is_baseline_not_candidate(self):
        profile = self.profile(self.snapshots())
        worker.validate_profile(profile, 'baseline')
        for row in worker.scalar_evidence(profile, 'baseline'):
            self.assertFalse(row['denoise_installed'])
            self.assertEqual(row['peak_allocated_bytes'], 80000000000)
        self.assertIsNone(profile['utilization']['denoise_policy'])
        self.assertIsNone(profile['utilization']['trace_policy'])

    def test_faulted_controller_real_receipts_to_rejection_consumers(self):
        self.write_pair(0, denoise=worker.DENOISE_POLICY, rejected=True)
        runtime = NS(coverage=self.coverage)
        manifest = dict(session_id=self.session, index=0)
        evidence = renderer.rejection_evidence(runtime, manifest, worker.DENOISE_POLICY)
        rows = worker.rejected_evidence({'utilization': evidence}, 'candidate_warmup')
        self.assertTrue(all(r['candidate_rejected'] for r in rows))
        self.assertFalse(evidence['owner_complete'])
        path = self.coverage / forward.utilization_name(self.session, 0, 1)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            renderer.rejection_evidence(runtime, manifest, worker.DENOISE_POLICY)

    def test_render_wraps_ipc_exception_only_with_real_terminal_receipts(self):
        from research.h3_personal.denoise_execution import CandidateRejected
        mode = self.coverage / 'mode.json'
        mode.write_text(json.dumps({'mode': 'independent'}))
        wav = self.coverage / 'input.wav'
        wav.write_bytes(b'CPU staging fixture; no decoder reached')
        planned = dict(wav=str(wav), sha256='b'*64, total_frames=328,
            window_frames=124, overlap_frames=22, expected_windows=3,
            seed=314159, source_samples=577920, source_sample_rate=48000)
        runtime = NS(coverage=self.coverage, continuation_file=mode,
            _trace_policy=None, _denoise_policy=worker.DENOISE_POLICY, _requests=0)
        ipc_error = RuntimeError('worker IPC wrapped exception')

        def generate(**kwargs):
            manifest = json.loads(mode.read_text())
            self.session = manifest['session_id']
            self.write_pair(0, denoise=worker.DENOISE_POLICY, rejected=True)
            raise ipc_error

        runtime._generate = generate
        turn = self.coverage / 'turn'
        with patch.object(renderer, 'plan', return_value=planned), patch.dict(
                'sys.modules', {'numpy': NS()}):
            with self.assertRaises(CandidateRejected) as caught:
                renderer.render(runtime, wav, planned['sha256'], turn)
        self.assertIs(caught.exception.__cause__, ipc_error)
        profile = json.loads((turn / 'profile.json').read_text())
        self.assertFalse(profile['full_decode_verified'])
        self.assertFalse(profile['remote_acceptance'])
        worker.rejected_evidence(profile, 'candidate_warmup')
        self.assertEqual(json.loads(mode.read_text()), {'mode': 'independent'})

    def test_measured_normal_render_snapshots_disabled_acceptance(self):
        mode = self.coverage / 'mode.json'
        mode.write_text(json.dumps({'mode': 'independent'}))
        wav = self.coverage / 'input.wav'
        wav.write_bytes(b'CPU fixture; no decoder reached')
        planned = dict(wav=str(wav), sha256='b'*64, total_frames=328,
            window_frames=124, overlap_frames=22, expected_windows=3,
            seed=314159, source_samples=577920, source_sample_rate=48000)
        runtime = NS(coverage=self.coverage, continuation_file=mode,
            _trace_policy=None, _denoise_policy=None, _requests=0,
            _measure_memory=True, _parallel_acceptance_enabled=False,
            _parallel_acceptance_pending=False,
            _candidate=efficiency.make_options(window_audio_context=True,
                                               defer_progress_receipts=True),
            _efficiency=efficiency.efficiency_options(device_audio_validation=True,
                                                      parallel_decode_waves=True))
        def stop_after_snapshot(**kwargs):
            raise RuntimeError('stop after actual plan snapshot')
        runtime._generate = stop_after_snapshot
        turn = self.coverage / 'normal-measured'
        with patch.object(renderer, 'plan', return_value=planned), patch.dict(
                'sys.modules', {'numpy': NS()}):
            with self.assertRaisesRegex(RuntimeError, 'stop after actual plan'):
                renderer.render(runtime, wav, planned['sha256'], turn)
        actual = json.loads((turn / 'speech-plan.json').read_text())
        self.assertIs(actual['parallel_acceptance'], False)
        self.session = actual['session_id']
        profile = self.profile(self.snapshots())
        profile['continuation_plan'] = actual
        efficiency.validate_profile(profile, optimized=True, acceptance=False)

    def test_stale_controller_binding_never_promotes_candidate(self):
        self.rank = 0
        adapter = self.controller(0)
        path = self.coverage / ('controller-' + forward.utilization_name(self.session, 0, 0))
        original = json.loads(path.read_text())
        for key, value in (('session_id', 'old'), ('window_index', 1),
                           ('rank', 1), ('denoise_policy', None)):
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(original, **{key: value})))
                forward._write_utilization(NS(), dict(session_id=self.session, index=0),
                    None, worker.DENOISE_POLICY, adapter, 1., 2., True)
                target = self.coverage / forward.utilization_name(self.session, 0, 0)
                row = json.loads(target.read_text())
                self.assertFalse(row['denoise_installed'])
                with self.assertRaisesRegex(ValueError, 'installation proof'):
                    worker.scalar_evidence({'utilization': {'ranks': [row, dict(row, rank=1)]}},
                                           'candidate')
                target.unlink()


if __name__ == '__main__':
    unittest.main()
