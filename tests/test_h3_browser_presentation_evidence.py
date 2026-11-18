"""Per-turn browser presentation evidence: bounded, fake-only, no browser launch."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research.h3_personal.browser_av import BrowserAVServer, BrowserAVSink
from research.h3_personal.browser_presentation import BrowserPresentationEvidence
from research.h3_personal.half_duplex_gate import EVIDENCE_SCHEMA


class PresentationModuleTests(unittest.TestCase):
    def setUp(self):
        self.session = 'a' * 32
        self.evidence = BrowserPresentationEvidence(session=self.session,
                                                    deadline=time.monotonic() + 30)

    def test_submission_alone_never_completes(self):
        self.evidence.note_submitted(1, 2.0)
        self.assertEqual(self.evidence.records(), ())
        self.assertIsNone(self.evidence.record(1))

    def test_playhead_before_turn_end_does_not_complete(self):
        self.evidence.note_submitted(1, 2.0)
        self.assertEqual(self.evidence.note_playhead(1.999), ())
        self.assertEqual(self.evidence.records(), ())

    def test_playhead_reaching_end_emits_one_record(self):
        self.evidence.note_submitted(1, 2.0)
        records = self.evidence.note_playhead(2.0)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record), {'schema', 'session', 'epoch',
                                       'completed_at_monotonic', 'output_seconds'})
        self.assertEqual(record['schema'], EVIDENCE_SCHEMA)
        self.assertEqual(record['session'], self.session)
        self.assertEqual(record['epoch'], 1)
        self.assertAlmostEqual(record['output_seconds'], 2.0)
        self.assertLessEqual(record['completed_at_monotonic'], time.monotonic())
        # Exactly one record per epoch, repeated playheads add nothing.
        self.assertEqual(self.evidence.note_playhead(2.5), ())
        self.assertEqual(len(self.evidence.records()), 1)

    def test_multiple_epochs_complete_in_order_on_one_jump(self):
        self.evidence.note_submitted(1, 2.0)
        self.evidence.note_submitted(2, 3.5)
        records = self.evidence.note_playhead(4.0)
        self.assertEqual([r['epoch'] for r in records], [1, 2])
        self.assertAlmostEqual(records[0]['output_seconds'], 2.0)
        self.assertAlmostEqual(records[1]['output_seconds'], 1.5)

    def test_duplicate_or_out_of_order_submission_rejected(self):
        self.evidence.note_submitted(1, 2.0)
        for epoch, end in ((1, 2.0), (3, 3.0), (2, 2.0), (2, 1.0)):
            with self.assertRaises(ValueError):
                self.evidence.note_submitted(epoch, end)
        for epoch, end in ((0, 1.0), (True, 1.0), (2, 0.0), (2, float('nan')), (2, '3')):
            with self.assertRaises(ValueError):
                self.evidence.note_submitted(epoch, end)
        self.assertEqual(self.evidence.snapshot()['submitted_ends'], (2.0,))

    def test_playhead_validation(self):
        self.evidence.note_submitted(1, 2.0)
        for value in (float('nan'), float('inf'), -1, '2', None):
            with self.assertRaises(ValueError):
                self.evidence.note_playhead(value)
        with self.assertRaises(ValueError):
            self.evidence.note_playhead(9.0)  # beyond submitted media
        self.assertEqual(self.evidence.records(), ())

    def test_playhead_regression_blocks_without_replay(self):
        self.evidence.note_submitted(1, 2.0)
        self.evidence.note_playhead(2.0)
        with self.assertRaises(ValueError):
            self.evidence.note_playhead(1.0)
        self.assertTrue(self.evidence.snapshot()['stopped'])
        self.assertEqual(self.evidence.snapshot()['reason'], 'playhead_regression')
        with self.assertRaises(ValueError):
            self.evidence.note_playhead(5.0)

    def test_bounded_wait_and_deadline(self):
        self.evidence.note_submitted(1, 2.0)
        with self.assertRaises(TimeoutError):
            self.evidence.wait_for_presentation_complete(1, time.monotonic() + .05)
        self.evidence.note_playhead(2.0)
        record = self.evidence.wait_for_presentation_complete(1, time.monotonic() + 1)
        self.assertEqual(record['epoch'], 1)

    def test_expiry_fails_closed(self):
        evidence = BrowserPresentationEvidence(session=self.session,
                                               deadline=time.monotonic() + .05)
        evidence.note_submitted(1, 2.0)
        time.sleep(.08)
        with self.assertRaises(ValueError):
            evidence.note_playhead(2.0)
        with self.assertRaises(RuntimeError):
            evidence.wait_for_presentation_complete(1, time.monotonic() + 1)
        self.assertFalse(evidence.snapshot()['completed_epochs'])

    def test_stop_releases_waiters_and_refuses_evidence(self):
        self.evidence.note_submitted(1, 2.0)
        self.assertTrue(self.evidence.stop('cancelled'))
        self.assertFalse(self.evidence.stop('again'))
        with self.assertRaises(RuntimeError):
            self.evidence.wait_for_presentation_complete(1, time.monotonic() + 1)
        with self.assertRaises(ValueError):
            self.evidence.note_playhead(2.0)

    def test_record_matches_capture_gate_contract(self):
        from research.h3_personal.half_duplex_gate import HalfDuplexCaptureGate
        gate = HalfDuplexCaptureGate(session=self.session, deadline=time.monotonic() + 30)
        self.assertTrue(gate.request_capture(1))
        gate.note_publication(1)
        self.evidence.note_submitted(1, 2.0)
        record = self.evidence.note_playhead(2.0)[0]
        self.assertTrue(gate.note_presentation_complete(record))
        self.assertTrue(gate.request_capture(2))


class ServerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.output = {'default_is_builtin_speaker': True, 'stable': True}
        self.evidence = BrowserPresentationEvidence(session='b' * 32,
                                                    deadline=time.monotonic() + 30)

    def tearDown(self):
        self.temp.cleanup()

    def server(self, **kwargs):
        return BrowserAVServer(self.root, lambda: self.output, 2,
                               presentation=self.evidence, **kwargs)

    def test_playhead_reaching_declared_end_completes(self):
        server = self.server()
        server._event({'event': 'client_ready'})
        server.note_turn_submitted(1, 2.0)
        server._event({'event': 'playing', 'media_seconds': 0.5})
        self.assertIsNone(server.presentation.record(1))
        server._event({'event': 'waiting', 'media_seconds': 2.0})
        self.assertEqual(server.presentation.record(1)['epoch'], 1)
        server.close()

    def test_ack_like_events_never_complete(self):
        server = self.server()
        server._event({'event': 'client_ready'})
        server._event({'event': 'first_append'})
        server._event({'event': 'playing'})
        server.note_turn_submitted(1, 2.0)
        server._event({'event': 'first_video'})
        self.assertEqual(server.presentation.records(), ())
        server.close()

    def test_duplicate_turn_submission_fails_closed(self):
        server = self.server()
        server._event({'event': 'client_ready'})
        server.note_turn_submitted(1, 2.0)
        with self.assertRaises(ValueError):
            server.note_turn_submitted(1, 2.0)
        self.assertEqual(server.stream_state, 'failed')
        server.close()

    def test_playhead_violation_fails_the_stream(self):
        server = self.server()
        server._event({'event': 'client_ready'})
        server.note_turn_submitted(1, 2.0)
        server._event({'event': 'playing', 'media_seconds': 2.0})
        with self.assertRaises(ValueError):
            server._event({'event': 'stalled', 'media_seconds': 1.0})
        self.assertEqual(server.stream_state, 'failed')
        self.assertEqual(server.presentation.snapshot()['reason'], 'playhead_regression')
        server.close()

    def test_terminal_client_event_stops_evidence(self):
        server = self.server()
        server._event({'event': 'client_ready'})
        server.note_turn_submitted(1, 2.0)
        server._event({'event': 'cancelled', 'elapsed_seconds': 1.0, 'media_seconds': 0.4})
        self.assertEqual(server.stream_state, 'failed')
        self.assertTrue(server.presentation.snapshot()['stopped'])
        server.close()

    def test_evidence_receipt_records_only_bounded_counts(self):
        server = self.server()
        server._event({'event': 'client_ready'})
        server.note_turn_submitted(1, 2.0)
        server._event({'event': 'playing', 'media_seconds': 2.0})
        server.close()
        receipt = json.loads((self.root / 'browser-av-evidence.json').read_text())
        presentation = receipt['presentation']
        self.assertEqual(presentation['completed_epochs'], [1])
        self.assertEqual(presentation['submitted_ends'], [2.0])
        self.assertNotIn('http:', json.dumps(receipt))
        self.assertNotIn('browser-av-evidence', json.dumps(presentation))

    def test_presentation_deadline_beyond_session_rejected_before_socket(self):
        evidence = BrowserPresentationEvidence(session='c' * 32,
                                               deadline=time.monotonic() + 60)
        with patch('research.h3_personal.browser_av._JoinedServer') as joined:
            with self.assertRaises(ValueError):
                with BrowserAVServer(self.root, lambda: self.output, 2,
                                     presentation=evidence):
                    pass
            joined.assert_not_called()

    def test_explicit_evidence_object_required(self):
        with self.assertRaises(ValueError):
            BrowserAVServer(self.root, lambda: self.output, 2, presentation={})

    def test_default_route_keeps_no_presentation_receipt(self):
        with patch.object(BrowserAVServer, '_read', staticmethod(
                lambda directory, name: b'<script>void 0;</script>')):
            server = BrowserAVServer(self.root, lambda: self.output, 1)
            self.assertIsNone(server.presentation)
            server._event({'event': 'client_ready'})
            server._event({'event': 'playing', 'media_seconds': 9.0})
            with self.assertRaises(RuntimeError):
                server.note_turn_submitted(1, 2.0)
            server.close()
        receipt = json.loads((self.root / 'browser-av-evidence.json').read_text())
        self.assertNotIn('presentation', receipt)


class SinkSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.output = {'default_is_builtin_speaker': True, 'stable': True}
        self.evidence = BrowserPresentationEvidence(session='d' * 32,
                                                    deadline=time.monotonic() + 30)
        self.server = BrowserAVServer(self.root, lambda: self.output, 2,
                                      presentation=self.evidence)
        self.server._event({'event': 'client_ready'})

    def tearDown(self):
        self.server.close()
        self.temp.cleanup()

    def test_mark_turn_submitted_declares_current_sink_media(self):
        sink = BrowserAVSink.__new__(BrowserAVSink)
        sink.server = self.server
        sink.media_seconds_end = 3.0
        sink.mark_turn_submitted(1)
        self.assertEqual(self.evidence.snapshot()['submitted_ends'], (3.0,))

    def test_mark_turn_submitted_requires_submitted_media(self):
        sink = BrowserAVSink.__new__(BrowserAVSink)
        sink.server = self.server
        sink.media_seconds_end = None
        with self.assertRaises(RuntimeError):
            sink.mark_turn_submitted(1)

    def test_mark_turn_submitted_requires_enabled_evidence(self):
        sink = BrowserAVSink.__new__(BrowserAVSink)
        with patch('research.h3_personal.browser_av.BrowserAVServer.presentation',
                   new_callable=lambda: property(lambda self: None)):
            sink.server = self.server
            sink.media_seconds_end = 3.0
            with self.assertRaises(RuntimeError):
                sink.mark_turn_submitted(1)

    def test_discontinuous_origin_refused_before_encoder_spawn(self):
        from research.h3_personal.paired_av import LiveAVBinding
        binding = LiveAVBinding('browser-test', 'a' * 64, 1, max_segments=8)
        with patch('research.h3_personal.headless_av_sink.HeadlessAVSink.__init__') as factory:
            with self.assertRaises(ValueError):
                BrowserAVSink(self.root, binding, server=self.server,
                              playback_approved=True, discontinuity_origin=(2, 5))
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
