from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
from unittest.mock import patch

from research.h3_personal.av_flow_model import FlowCase, FlowModelError, simulate
from research.run_h3_flow_model import main, model_evidence
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal import paired_av_wire as wire


def case(n=3, **kwargs):
    base = FlowCase((1.0,)*n, (1.0,)*n, (1.0,)*n, (.2,)*n)
    return replace(base, **kwargs)


class FlowModelTests(unittest.TestCase):
    def test_analytic_balanced_pipeline(self):
        r = simulate(case())
        self.assertTrue(r['complete'])
        self.assertEqual([x['packet_ready'] for x in r['rows']], [1, 2, 3])
        self.assertEqual([x['accepted'] for x in r['rows']], [2, 3, 4])
        self.assertAlmostEqual(r['virtual_window_playout_idle_seconds'], 0)
        self.assertEqual(r['credit_wait_seconds'], 0)
        self.assertLessEqual(r['slots_highwater'], 2)
        self.assertLessEqual(r['prefetch_highwater'], 1)
        self.assertFalse(r['playback_verified'])

    def test_verified_ack_does_not_release_generation_credit(self):
        r = simulate(case(pulls=(0.0,)*3, forward_delay=1, reverse_delay=1))
        self.assertEqual(r['rows'][0]['verified_ack_received'], 2)
        self.assertEqual(r['rows'][0]['sink_ack_received'], 3)
        self.assertEqual(r['rows'][2]['pull_started'], 3)
        self.assertEqual(r['slots_highwater'], 2)

    def test_final_ack_waits_for_sink_finalization_and_reverse_path(self):
        r = simulate(case(1, forward_delay=1, reverse_delay=1, finalization=3))
        row = r['rows'][0]
        self.assertEqual(row['accepted'], 3)
        self.assertEqual(row['finalized'], 6)
        self.assertEqual(row['sink_ack_received'], 7)
        self.assertEqual(r['virtual_seconds'], 7)
        self.assertTrue(r['final_sink_finalized'])

    def test_withheld_final_ack_hits_deadline_no_success(self):
        r = simulate(case(1, withhold_final_ack=True, deadline=10))
        self.assertFalse(r['complete'])
        self.assertEqual(r['pending'], 1)
        self.assertTrue(r['final_sink_finalized'])
        self.assertEqual(r['virtual_seconds'], 10)
        self.assertIn('no replay', r['terminal'])

    def test_slow_pull_exposes_virtual_starvation(self):
        r = simulate(case(2, pulls=(1, 3), decode=(0, 0)))
        self.assertEqual(r['rows'][1]['packet_ready'], 4)
        self.assertEqual(r['virtual_window_playout_idle_seconds'], 2)
        self.assertEqual(r['max_fixed_media_deadline_deficit_seconds'], 2)

    def test_deadline_during_generation_reports_no_decoded_media(self):
        r = simulate(case(1, pulls=(5,), deadline=2))
        self.assertFalse(r['complete'])
        self.assertTrue(r['generating'])
        self.assertNotIn('packet_ready', r['rows'][0])
        self.assertNotIn('decoded_ready', r['rows'][0])

    def test_deadline_during_decode_does_not_claim_future_readiness(self):
        r = simulate(case(1, sink_service=(5,), decode=(3,), deadline=2))
        self.assertIn('accept_started', r['rows'][0])
        self.assertNotIn('decoded_ready', r['rows'][0])
        self.assertFalse(r['complete'])

    def test_deadline_records_unfinished_credit_wait(self):
        r = simulate(case(3, pulls=(0,)*3, sink_service=(10,)*3, decode=(0,)*3, deadline=5))
        self.assertFalse(r['complete'])
        self.assertEqual(r['completed_credit_wait_seconds'], 0)
        self.assertEqual(r['incomplete_credit_wait_seconds'], 5)
        self.assertEqual(r['credit_wait_seconds'], 5)
        self.assertNotIn('pull_started', r['rows'][2])

    def test_sink_idle_does_not_require_next_decode_completion(self):
        r = simulate(case(2, pulls=(1, 3), sink_service=(1, 3), decode=(0, 2), deadline=5))
        self.assertNotIn('decoded_ready', r['rows'][1])
        self.assertEqual(r['sink_idle_seconds'], 2)
        self.assertEqual(r['virtual_window_playout_idle_seconds'], 3)
        self.assertTrue(r['deadline_deficit_is_lower_bound'])

    def test_deadline_counts_trailing_sink_idle(self):
        r = simulate(case(2, pulls=(1, 5), decode=(0, 0), deadline=4))
        self.assertEqual(r['incomplete_sink_idle_seconds'], 2)
        self.assertEqual(r['sink_idle_seconds'], 2)
        self.assertEqual(r['incomplete_virtual_playout_idle_seconds'], 2)

    def test_zero_time_events_are_finite_and_fifo(self):
        r = simulate(case(144, pulls=(0,)*144, sink_service=(0,)*144, decode=(0,)*144))
        self.assertTrue(r['complete'])
        self.assertEqual(r['pending'], 0)
        self.assertLess(r['events_processed'], 4096)
        self.assertEqual([x['index'] for x in r['rows']], list(range(144)))
        self.assertLessEqual(r['slots_highwater'], 2)

    def test_more_delay_recomputes_credit_wait_not_historical_replay(self):
        baseline = simulate(case(8))
        slower = simulate(case(8, forward_delay=1, reverse_delay=1))
        self.assertGreater(slower['credit_wait_seconds'], baseline['credit_wait_seconds'])
        self.assertGreater(slower['virtual_seconds'], baseline['virtual_seconds'])

    def test_invalid_inputs(self):
        for kwargs in (dict(pulls=()), dict(pulls=[1, 1, 1]), dict(media=(0, 1, 1)),
                       dict(decode=(2, 2, 2)), dict(forward_delay=-1),
                       dict(reverse_delay=float('nan')), dict(deadline=0),
                       dict(withhold_final_ack=1), dict(pulls=(True, 1, 1))):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(FlowModelError): case(**kwargs)

    def test_bad_evidence_error_sanitized(self):
        stream = io.StringIO()
        with patch('research.run_h3_flow_model.model_evidence', side_effect=ValueError('PRIVATE')):
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(['--capture-dir', 'PRIVATE', '--producer-timing', 'PRIVATE',
                                       '--receiver-timing', 'PRIVATE', '--sink-timing', 'PRIVATE']), 1)
        self.assertNotIn('PRIVATE', stream.getvalue())


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.capture = root / 'packets'
        self.capture.mkdir()
        self.producer, self.receiver, self.sink = [root / f'{name}.json' for name in ('p', 'r', 's')]
        binding = LiveAVBinding('PRIVATE_SOURCE', '0'*64, 0)
        p, r, events = [], [], []
        for i in range(2):
            video, pcm = b'video0', b'\0'*4000
            e = encode(binding, i, i, i+1, video, pcm,
                       (AudioRegion('silence', i*2000, (i+1)*2000),), i == 1)
            (self.capture / f'{i:03d}.json').write_text(json.dumps(wire.envelope_record(e)))
            (self.capture / f'{i:03d}.mp4').write_bytes(video)
            (self.capture / f'{i:03d}.pcm').write_bytes(pcm)
            p.append(dict(index=i, credit_wait_started=i*.02, pull_started=i*.02,
                          packet_ready=i*.02+.02, ack_verified=i*.02+.03,
                          ack_sink_accepted=i*.02+.08, video_bytes=6, pcm_bytes=4000))
            r.append(dict(index=i, received=i*.02, verified=i*.02+.001,
                          accept_start=i*.02+.002, accepted=i*.02+.022, sink_acked=i*.02+.023))
            events.append(dict(sequence=i, frame_start=i, frame_stop=i+1,
                               sample_start=i*2000, sample_stop=(i+1)*2000,
                               media_seconds=1/24, decode_seconds=.01))
        self.producer.write_text(json.dumps(dict(type='h3_producer_timing', clock='local_monotonic', windows=p)))
        self.receiver.write_text(json.dumps(dict(clock='local_monotonic', receive=r)))
        self.sink.write_text(json.dumps(dict(events=events)))

    def run_model(self):
        return model_evidence(self.capture, self.producer, self.receiver, self.sink)

    def test_real_parser_no_media_decode_or_network(self):
        result = self.run_model()
        self.assertEqual(len(result['cases']), 3)
        self.assertFalse(result['network_executed'])
        self.assertNotIn('PRIVATE_SOURCE', json.dumps(result))
        self.assertNotIn(str(self.capture), json.dumps(result))
        self.assertTrue(all(r['complete'] for r in result['cases'].values()))

    def test_historical_wait_and_absolute_readiness_not_replayed(self):
        value = json.loads(self.producer.read_text())
        row = value['windows'][1]
        for key in ('pull_started', 'packet_ready', 'ack_verified', 'ack_sink_accepted'):
            row[key] += 1
        self.producer.write_text(json.dumps(value))
        result = self.run_model()
        self.assertAlmostEqual(result['historical_credit_wait_seconds'][1], 1)
        modeled = result['cases']['zero-carrier-delay-bound']['rows'][1]
        self.assertAlmostEqual(modeled['packet_ready'], .04)
        self.assertAlmostEqual(modeled['credit_wait_seconds'], 0)

    def test_final_preparation_proxy_is_explicit(self):
        result = self.run_model()
        self.assertAlmostEqual(result['observed_final_accept_to_ack_enqueued_seconds'], .001)
        row = result['cases']['zero-carrier-delay-bound']['rows'][-1]
        self.assertAlmostEqual(row['finalized'] - row['accepted'], .001)
        self.assertIn('proxy', result['parameter_caveat'])

    def test_mismatched_sink_range_rejected(self):
        value = json.loads(self.sink.read_text())
        value['events'][0]['frame_stop'] += 1
        self.sink.write_text(json.dumps(value))
        with self.assertRaises(FlowModelError): self.run_model()

    def test_invalid_local_timestamp_order_rejected(self):
        value = json.loads(self.receiver.read_text())
        value['receive'][0]['accepted'] = 0
        self.receiver.write_text(json.dumps(value))
        with self.assertRaises(FlowModelError): self.run_model()

    def test_decode_outside_sink_service_rejected(self):
        value = json.loads(self.sink.read_text())
        value['events'][0]['decode_seconds'] = 1
        self.sink.write_text(json.dumps(value))
        with self.assertRaises(FlowModelError): self.run_model()


if __name__ == '__main__':
    unittest.main()
