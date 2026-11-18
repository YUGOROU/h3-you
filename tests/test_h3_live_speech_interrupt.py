"""CPU native-boundary scheduling; does not test presentation invalidation."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import test_h3_live_conversation as fixtures
from research.h3_personal.live_conversation import LiveConversation
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import (
    ResamplingContext, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256)


class InterruptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.driver = LiveConversation(Path(self.temp.name)/'live',
            binding=LiveAVBinding('conversation', fixtures.SHA, 0, max_segments=24),
            context=ResamplingContext('native-48k32k-prefix128',
                NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256))
        self.addCleanup(self.driver.close)
        self.fixture = fixtures.ConversationTests()
        self.backend = self.fixture.backend
        self.driver.step(self.backend)

    def speech(self, epoch=1, samples=204128, seal=False):
        stream, pcm = self.fixture.source(epoch, samples)
        if seal:
            stream.seal(total_samples=samples, sha256=hashlib.sha256(pcm).hexdigest())
        self.driver.attach(stream)
        return stream, pcm

    def test_interrupt_in_forward_completes_once_before_retirement(self):
        d = self.driver; old, pcm = self.speech()
        calls = []
        def forward(manifest):
            calls.append(manifest)
            self.assertTrue(d.request_speech_interrupt(1))
            self.assertFalse(d.request_speech_interrupt(1))
            self.assertEqual(d.inspect().status, 'busy')
            with self.assertRaises(RuntimeError):
                d.retire_interrupted_speech()
            return self.backend(manifest)
        first = d.step(forward)
        self.assertEqual(len(calls), 1)
        self.assertEqual(d.inspect().status, 'interrupt')
        before = d.control.snapshot()
        marker = d.retire_interrupted_speech()
        self.assertEqual(marker, dict(schema='h3-speech-interruption-v1',
                         source_id='speech-1', committed_frame=226))
        self.assertEqual(d.control.snapshot()['committed_frame'], before['committed_frame'])
        self.assertEqual(d.control.snapshot()['emitted_frame'], 221)
        self.assertEqual(d.control.snapshot()['completed_turns'], 0)
        self.assertEqual(d.control.snapshot()['interrupted_turns'], 1)
        # Old ingress must survive for decoder lookahead, not be cancelled early.
        self.assertEqual(old.read(0, 1).pcm, pcm[:2])
        p = d.prepare()
        self.assertEqual(p.manifest['streaming_audio']['previous_speech_interrupted'], marker)
        tail = d.complete_native(p, self.backend(p.manifest))
        self.assertEqual(tail.envelope.frame_start, first.envelope.frame_stop)
        self.assertEqual(sum(r.sample_stop-r.sample_start for r in tail.envelope.regions
                             if r.source_epoch == 1), 10000)
        self.assertNotIn('previous_speech_interrupted', d.prepare().manifest['streaming_audio'])

    def test_next_speech_reuses_driver_and_monotonic_clock(self):
        d = self.driver; self.speech(); d.step(self.backend)
        d.request_speech_interrupt(1); d.retire_interrupted_speech()
        self.speech(2, 40000, seal=True)
        p = d.prepare()
        self.assertEqual(p.manifest['streaming_audio']['role'], 'speech')
        self.assertEqual(p.manifest['streaming_audio']['logical_index'], 0)
        self.assertEqual(p.manifest['streaming_audio']['committed_before'], 226)
        packet = d.complete_native(p, self.backend(p.manifest))
        self.assertEqual(packet.envelope.frame_start, 221)
        self.assertTrue(any(r.source_epoch == 2 for r in packet.envelope.regions))
        self.assertEqual(d.inspect().status, 'end')
        d.retire_speech(); d.request_stop(); d.step(self.backend)
        self.assertEqual(d.control.snapshot()['phase'], 'closed')

    def test_pending_source_needs_no_native_interruption_marker(self):
        d = self.driver; old, _ = self.speech(samples=0)
        d.request_speech_interrupt(1)
        with self.assertRaises(RuntimeError):
            d.cancel_pending_speech()
        self.assertIsNone(d.retire_interrupted_speech())
        self.assertNotIn('previous_speech_interrupted', d.prepare().manifest['streaming_audio'])
        self.assertEqual(old.snapshot()['received_samples'], 0)

    def test_finished_source_not_falsely_marked_as_unfinished(self):
        d = self.driver; self.speech(samples=40000, seal=True); d.step(self.backend)
        d.request_speech_interrupt(1)
        self.assertIsNone(d.retire_interrupted_speech())
        self.assertNotIn('previous_speech_interrupted', d.prepare().manifest['streaming_audio'])

    def test_foreign_epoch_and_duplicate_retirement_refused(self):
        d = self.driver; self.speech()
        with self.assertRaises(ValueError):
            d.request_speech_interrupt(2)
        d.request_speech_interrupt(1); d.retire_interrupted_speech()
        with self.assertRaises(RuntimeError):
            d.retire_interrupted_speech()
        replay, _ = self.fixture.source(1, 100)
        try:
            with self.assertRaises(ValueError):
                d.attach(replay)
        finally:
            replay.cancel()


if __name__ == '__main__':
    unittest.main()
