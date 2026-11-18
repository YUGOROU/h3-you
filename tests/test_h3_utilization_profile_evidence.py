import copy
import unittest
from research.h3_personal.fast_continuation_render import profile_evidence
from research.h3_personal.fast_ref2va import utilization_name
from research.hf_h3_utilization_comparison import scalar_evidence


class ProfileEvidenceTests(unittest.TestCase):
    def snapshots(self):
        return [dict(session_id='session', utilization_receipts=[dict(
            schema='h3-window-utilization-v1', session_id='session', window_index=i,
            rank=r, trace_policy=None, denoise_policy=None, faulted=False,
            worker_monotonic_start=1., worker_monotonic_end=2., trace_summary=None)
            for r in (0, 1)], continuation_stage_receipts=[dict(committed=True,
            final=i == 2, total_frames=328) for _ in (0, 1)],
            continuation_decode_receipts=[dict(final=i == 2, frame_offset=221,
            frame_count=107) for _ in (0, 1)]) for i in range(3)]

    def test_worker_scalar_contract(self):
        evidence = profile_evidence(self.snapshots(), None, None, 328)
        self.assertTrue(evidence['owner_complete'])
        self.assertEqual(len(evidence['windows']), 3)
        self.assertEqual([r['rank'] for r in scalar_evidence(
            {'utilization': evidence}, 'baseline')], [0, 1])

    def test_fault_missing_rank_stale_session(self):
        for mutate in (
            lambda s: s[1]['utilization_receipts'].pop(),
            lambda s: s[1]['utilization_receipts'][0].update(faulted=True),
            lambda s: s[1]['utilization_receipts'][0].update(session_id='old')):
            snapshots = copy.deepcopy(self.snapshots())
            mutate(snapshots)
            with self.assertRaises(RuntimeError):
                profile_evidence(snapshots, None, None, 328)

    def test_no_false_completion(self):
        snapshots = self.snapshots()
        snapshots[-1]['continuation_stage_receipts'][1]['committed'] = False
        self.assertFalse(profile_evidence(snapshots, None, None, 328)['owner_complete'])

    def test_paths_unique_and_confined(self):
        names = {utilization_name(s, i, r) for s in ('../a', 'b')
                 for i in range(3) for r in (0, 1)}
        self.assertEqual(len(names), 12)
        self.assertTrue(all('/' not in n for n in names))
