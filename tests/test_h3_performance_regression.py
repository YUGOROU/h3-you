import copy
from fractions import Fraction
import unittest

from research.h3_personal.performance_regression import (
    STAGES, SCHEMA, compare_workloads, summarize_stage_profiles,
)


def workload(frames=372, wall=16.202924):
    return dict(frames=frames, fps=24, wall_seconds=wall)


def profile(rank=0, hit=1):
    return dict(schema=SCHEMA, worker_rank=rank, cache_hit=hit,
                synchronization_added=True, benchmark_comparable=False,
                stages=[dict(name=name, host_seconds=rank + 0.1,
                             default_stream_elapsed_ms=rank + 2.5) for name in STAGES])


class WorkloadTests(unittest.TestCase):
    def test_known_v14_vs_optimized(self):
        result = compare_workloads(workload(), workload(328, 20.240361))
        expected = (Fraction(328, 24) / Fraction('20.240361')) / (
            Fraction(372, 24) / Fraction('16.202924'))
        self.assertEqual(result['throughput_ratio'], expected)
        self.assertEqual(result['candidate']['video_seconds_per_wall_second'],
                         Fraction(328, 24) / Fraction('20.240361'))
        self.assertEqual(result['time_increase_ratio'], Fraction('20.240361') / Fraction('16.202924'))
        self.assertEqual(result['useful_output_ratio'], Fraction(328, 372))
        self.assertEqual(result['decomposition']['product'], expected)
        self.assertEqual(result['counterfactual']['candidate_output_at_baseline_time'],
                         Fraction(328, 24) / Fraction('16.202924'))
        self.assertIn('not a performance prediction', result['counterfactual']['label'])
        self.assertTrue(result['regression'])
        self.assertNotIn('frames_per_second', result['candidate'])

    def test_equal_normalized_work(self):
        self.assertEqual(compare_workloads(workload(100, 10), workload(200, 20))[
            'throughput_ratio'], 1)
        self.assertFalse(compare_workloads(workload(100, 10), workload(200, 20))['regression'])

    def test_tolerances(self):
        self.assertTrue(compare_workloads(workload(100, 10), workload(99, 10))['regression'])
        self.assertFalse(compare_workloads(workload(100, 10), workload(99, 10),
                                         relative_tolerance=0.01)['regression'])
        for key in ('relative_tolerance', 'absolute_tolerance'):
            for value in (-0.001, True, '0', float('nan'), float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    compare_workloads(workload(), workload(), **{key: value})

    def test_strict_workload(self):
        for key in ('frames', 'fps', 'wall_seconds'):
            for value in (True, '24', None, 0, -1, float('inf'), float('-inf'), float('nan')):
                bad = workload()
                bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    compare_workloads(bad, workload())
        with self.assertRaises(ValueError):
            compare_workloads(dict(workload(), extra=1), workload())
        with self.assertRaises(ValueError):
            compare_workloads(workload(1.5), workload())


class ProfileTests(unittest.TestCase):
    def test_ordered_per_hit_rank_timings_and_maxima(self):
        records = [profile(1, 2), profile(0), profile(0, 2), profile(1)]
        original = copy.deepcopy(records)
        result = summarize_stage_profiles(records, run_id='v14')
        self.assertEqual(records, original)
        self.assertEqual([h['cache_hit'] for h in result['hits']], [1, 2])
        for hit in result['hits']:
            self.assertEqual([r['worker_rank'] for r in hit['ranks']], [0, 1])
            self.assertEqual(hit['ranks'][0]['stages'][0]['host_seconds'], Fraction('0.1'))
            self.assertEqual(hit['descriptive_stage_maxima'][0]['host_seconds'], Fraction('1.1'))
            self.assertNotIn('wall_seconds', hit)
        self.assertFalse(result['benchmark_comparable'])
        self.assertFalse(result['wall_attribution_proven'])

    def test_pairs_and_provenance(self):
        for records in ([], [profile()], [profile(), profile()],
                        [profile(0, 2), profile(1, 2)],
                        [profile(), profile(1), profile(0, 3), profile(1, 3)]):
            with self.assertRaises(ValueError):
                summarize_stage_profiles(records, run_id='run')
        for run_id in ('', '   ', None, True, 3):
            with self.assertRaises(ValueError):
                summarize_stage_profiles([profile(), profile(1)], run_id=run_id)
        with self.assertRaises(TypeError):
            summarize_stage_profiles([profile(), profile(1)])

    def test_schema_flags_and_indices(self):
        mutations = [('schema', 'other'), ('worker_rank', True), ('worker_rank', 2),
                     ('cache_hit', True), ('cache_hit', 1.0), ('cache_hit', 7),
                     ('synchronization_added', 1), ('benchmark_comparable', 0),
                     ('stages', None), ('extra', 1)]
        for key, value in mutations:
            bad = profile()
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summarize_stage_profiles([bad, profile(1)], run_id='run')

    def test_stage_duplication_order_and_numbers(self):
        for mode in ('duplicate', 'reorder', 'missing', 'extra'):
            bad = profile()
            stages = bad['stages']
            if mode == 'duplicate':
                stages[1] = copy.deepcopy(stages[0])
            elif mode == 'reorder':
                stages.reverse()
            elif mode == 'missing':
                stages.pop()
            else:
                stages[0]['extra'] = 0
            with self.assertRaises(ValueError):
                summarize_stage_profiles([bad, profile(1)], run_id='run')
        for metric in ('host_seconds', 'default_stream_elapsed_ms'):
            for value in (True, '0.1', None, -0.001, float('nan'), float('inf'), float('-inf')):
                bad = profile()
                bad['stages'][0][metric] = value
                with self.subTest(metric=metric, value=value), self.assertRaises(ValueError):
                    summarize_stage_profiles([bad, profile(1)], run_id='run')


if __name__ == '__main__':
    unittest.main()
