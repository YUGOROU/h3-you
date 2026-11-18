"""CPU-only worker orchestration and publication boundary tests."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

from research.h3_personal import continuation_efficiency

# The contract is supplied by the coordinating change. Permit isolated worker
# development without modifying the shared module on disk.
with patch.object(continuation_efficiency, 'comparison_contract',
                  getattr(continuation_efficiency, 'comparison_contract',
                          lambda: {'schema': 'test-comparison'}), create=True):
    worker = importlib.import_module('research.hf_h3_efficiency_comparison')


class WorkerTests(unittest.TestCase):
    def exercise(self, parity=True, fail=False, memory_bad=False, hash_bad=False, session_bad=False,
                 window_bad=None, acceptance_bad=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'root'
            root.mkdir()
            output = root.with_name(root.name+'-efficiency-generated')
            events = []
            api = Mock()
            api.repo_info.return_value.private = True
            hub = types.SimpleNamespace(HfApi=Mock(return_value=api),
                CommitOperationAdd=lambda **kwargs: kwargs)
            spec = dict(efficiency_comparison=worker.comparison_contract(),
                        config={'timeout_seconds': 100}, locked_speech={'path': 'fixed.wav'},
                        identity_prefix='identity')
            paths = {k: root/k for k in ('speech', 'personal', 'base', 'smd', 'reference')}

            class Runtime:
                def __init__(self, **kwargs):
                    events.append(('init', kwargs))
                def start(self):
                    events.append(('start',))
                def configure_efficiency(self, **kwargs):
                    events.append(('configure', kwargs))
                def close(self):
                    events.append(('close',))
                def render_continuation(self, wav, sha, turn):
                    events.append(('render', turn.name))
                    if fail:
                        raise RuntimeError('injected render failure')
                    turn.mkdir(parents=True)
                    optimized = turn.name == 'optimized'
                    profile = dict(continuation_plan=dict(
                        candidate=worker.make_options(window_audio_context=True,
                                                      defer_progress_receipts=optimized)),
                        unique_frames=328, speech_pcm_exact=True, full_decode_verified=True,
                        windows=[{'seed': 314159}, {}, {}], session_id=turn.name,
                        derived_adapter_receipt={'sha': 'adapter'},
                        input_proof={'reference_sha256': 'reference', 'source': 'fixed'})
                    profile['continuation_plan']['parallel_acceptance'] = turn.name == 'acceptance'
                    if acceptance_bad == turn.name:
                        profile['continuation_plan']['parallel_acceptance'] = not (turn.name == 'acceptance')
                    if optimized:
                        profile['continuation_plan']['efficiency'] = worker.efficiency_options(
                            device_audio_validation=True, parallel_decode_waves=True)
                    profile['windows'] = [dict(seed=314159, session_id=turn.name,
                        fixed_reference_sha256=worker.baseworker.REFERENCE_SHA,
                        wav_sha256=worker.source_contract.SOURCE_SHA256) for _ in range(3)]
                    for window in profile['windows']:
                        window.update(continuation_stage_receipts=[{'worker_rank': 0, 'rows': 12}],
                                      continuation_decode_receipts=[{'worker_rank': 1, 'verified': True}])
                    for name in (*worker.VIDEOS, *worker.RECEIPTS):
                        value = profile if name == 'profile.json' else (
                            profile['windows'][int(name[7:10])] if name.startswith('window-') else {})
                        (turn/name).write_text(json.dumps(value))
                    # Match the renderer: immutable windows are written before metadata is attached.
                    for window in profile['windows']:
                        window['segment_metadata'] = {'frames': 112, 'start': 0}
                    if window_bad == 'changed':
                        profile['windows'][0]['continuation_stage_receipts'][0]['rows'] = 13
                    elif window_bad == 'missing':
                        del profile['windows'][0]['continuation_decode_receipts']
                    elif window_bad == 'extra':
                        profile['windows'][0]['unexpected'] = True
                    elif window_bad == 'type':
                        profile['windows'][0]['continuation_decode_receipts'][0]['verified'] = 1
                    (turn/'profile.json').write_text(json.dumps(profile))
                    # These must never appear in a commit, even beneath an arm.
                    (turn/'raw.wav').write_bytes(b'private')
                    (turn/'latents.pt').write_bytes(b'private')
                    coverage = output/'coverage'
                    coverage.mkdir(exist_ok=True)
                    for rank in range(2):
                        (coverage/f'parallel-parity-rank-{rank}.json').write_text(json.dumps(dict(
                            worker_rank=rank, session_id='wrong' if session_bad else 'acceptance',
                            input_sha256=('b' if hash_bad and rank else 'a')*64, parity=parity)))
                    memory = output/'memory'
                    memory.mkdir(exist_ok=True)
                    arm_index = ('acceptance', 'baseline', 'optimized').index(turn.name)
                    for rank in range(2):
                        for hit in range(arm_index*3+1, arm_index*3+4):
                            row = dict(schema='h3-worker-memory-v1', worker_rank=rank,
                                cache_hit=hit, contains_parallel_acceptance=hit <= 3,
                                peak_allocated_bytes=True if memory_bad else 10, peak_reserved_bytes=20)
                            (memory/f'rank-{rank}-request-hit{hit}.json').write_text(json.dumps(row))
                    return types.SimpleNamespace(profile=turn/'profile.json',
                                                 review_with_speech=turn/'review-with-speech.mp4')

            def validate(row, rank, session, parallel=False):
                self.assertTrue(parallel)
                if row['session_id'] != session:
                    raise ValueError('wrong session')
                return row

            with patch.dict(sys.modules, huggingface_hub=hub), patch.dict(os.environ, HF_TOKEN='test'), \
                    patch.object(worker, 'validate_spec'), \
                    patch.object(worker.source_contract, 'preflight', return_value={'fixed': True}), \
                    patch.object(worker, 'LiveVoiceRuntime', Runtime), \
                    patch.object(worker.baseworker, 'validate_parity', side_effect=validate), \
                    patch.object(worker.subprocess, 'run') as decode:
                try:
                    result = worker.run(root, paths, spec, 'private', time.monotonic()+60)
                except Exception:
                    self.assertEqual(events[-1], ('close',))
                    api.create_commit.assert_not_called()
                    raise
            self.assertEqual(events[-1], ('close',))
            operations = api.create_commit.call_args.kwargs['operations']
            self.assertTrue(all(op['path_in_repo'].startswith('outputs/efficiency/') for op in operations))
            self.assertFalse(any(op['path_in_repo'].endswith(('.wav', '.pt')) for op in operations))
            self.assertEqual(sum(e[0] == 'start' for e in events), 1)
            self.assertFalse(events[0][1]['profile_stages'])
            self.assertTrue(events[0][1]['measure_memory'])
            self.assertTrue(events[0][1]['parallel_decode_acceptance'])
            self.assertEqual(decode.call_count, 15 if parity else 5)
            return result, events

    def test_single_residency_order_and_allowlist(self):
        result, events = self.exercise()
        self.assertTrue(result['parity_gate'])
        self.assertFalse(result['quality_verified'])
        self.assertEqual([e[:2] for e in events[1:]], [
            ('start',), ('render', 'acceptance'), ('render', 'baseline'),
            ('configure', dict(device_audio_validation=True, parallel_decode_waves=True,
                               defer_progress_receipts=True)),
            ('render', 'optimized'), ('close',)])

    def test_numeric_failure_is_completed_finding_no_steady(self):
        result, events = self.exercise(parity=False)
        self.assertTrue(result['completed'])
        self.assertTrue(result['steady_skipped'])
        self.assertEqual([e for e in events if e[0] == 'render'], [('render', 'acceptance')])
        self.assertFalse(any(e[0] == 'configure' for e in events))

    def test_render_failure_closes_without_publication(self):
        with self.assertRaises(RuntimeError):
            self.exercise(fail=True)

    def test_other_window_evidence_remains_strict(self):
        for malformed in ('changed', 'missing', 'extra', 'type'):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(
                    ValueError, 'window receipt differs'):
                self.exercise(window_bad=malformed)

    def test_parallel_acceptance_is_exclusive_to_acceptance_arm(self):
        for arm in ('acceptance', 'baseline', 'optimized'):
            with self.subTest(arm=arm), self.assertRaisesRegex(
                    ValueError, 'profile policy/outcome'):
                self.exercise(acceptance_bad=arm)

    def test_bad_memory_types_rejected(self):
        with self.assertRaises(ValueError):
            self.exercise(memory_bad=True)

    def test_rank_hash_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.exercise(hash_bad=True)

    def test_session_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.exercise(session_bad=True)


if __name__ == '__main__':
    unittest.main()
