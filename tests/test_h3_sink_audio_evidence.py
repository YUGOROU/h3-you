"""Device-free encoder-input evidence; no remote/playback assertion."""
import hashlib
import json
import struct
import unittest
from unittest.mock import patch

import test_h3_headless_av_sink as fixtures
from research.h3_personal.av_delivery import Binding, Delivery
from research.h3_personal.conversation_state import SpeechSource
from research.h3_personal.headless_av_sink import _pcm16_evidence


class MeterTests(unittest.TestCase):
    def test_exact_signed_little_endian(self):
        self.assertEqual(_pcm16_evidence(struct.pack('<5h', -32768, 32767, -1, 0, 256)),
                         (4, 32768, 32768**2 + 32767**2 + 1 + 256**2))
        self.assertEqual(_pcm16_evidence(b''), (0, 0, 0))
        with self.assertRaises(ValueError):
            _pcm16_evidence(b'\x00')


@unittest.skipUnless(fixtures.shutil.which('ffmpeg') and fixtures.shutil.which('ffprobe'),
                     'FFmpeg required')
class AudioEvidenceTests(unittest.TestCase):
    # Reuse media fixtures without inheriting the unrelated sink test suite.
    setUp = fixtures.SinkTests.setUp
    tearDown = fixtures.SinkTests.tearDown
    lease = fixtures.SinkTests.lease
    sink = fixtures.SinkTests.sink

    def test_source_and_silence_with_explicit_padding(self):
        for silent in (False, True):
            with self.subTest(silent=silent):
                if silent:
                    self.pcm = bytes(len(self.pcm))
                    self.binding = Binding('test', 'source', SpeechSource(
                        hashlib.sha256(self.pcm).hexdigest(), 8000, 1200), 'a'*64)
                    self.delivery = Delivery(self.binding, self.pcm)
                with self.sink() as sink:
                    initial = sink.audio_evidence
                    self.assertEqual(initial['ack_count'], 0)
                    first = self.lease(0)
                    sink.accept('0.mkv', first)
                    self.delivery.ack(first)
                    snapshot = sink.audio_evidence
                    second = self.lease(1)
                    sink.accept('1.mkv', second)
                    sink.finish()
                    evidence = sink.audio_evidence
                    nonzero, peak, squares = _pcm16_evidence(self.pcm)
                    self.assertEqual(evidence['source_pcm_samples'], 1200)
                    self.assertEqual(evidence['source_pcm16_samples'], 1200)
                    self.assertEqual(evidence['padding_pcm_samples'], second.segment.padding_samples)
                    self.assertGreater(evidence['padding_pcm_samples'], 0)
                    self.assertEqual(evidence['nonzero_pcm16_samples'], nonzero)
                    self.assertEqual(evidence['peak_pcm16'], peak)
                    self.assertEqual(evidence['sum_squares_pcm16'], squares)
                    self.assertAlmostEqual(evidence['rms_pcm16'], (squares/1200)**.5)
                    self.assertEqual(evidence['ack_count'], 2)
                    self.assertEqual(snapshot['ack_count'], 1)
                    evidence['ack_count'] = -1
                    self.assertEqual(sink.audio_evidence['ack_count'], 2)
                    self.assertTrue(evidence['encoder_input_only'])
                    self.assertFalse(evidence['remote_verified'])
                    json.dumps(evidence, allow_nan=False)

    def test_failed_writer_does_not_commit(self):
        with self.sink() as sink:
            first = self.lease(0)
            sink.accept('0.mkv', first)
            self.delivery.ack(first)
            before = sink.audio_evidence
            with patch('research.h3_personal.headless_av_sink.os.write',
                       side_effect=BrokenPipeError('injected')):
                with self.assertRaises(BrokenPipeError):
                    sink.accept('1.mkv', self.lease(1))
            self.assertEqual(sink.audio_evidence, before)

    def test_encoder_dead_after_writes_does_not_commit(self):
        with self.sink() as sink:
            before = sink.audio_evidence
            real_poll = sink._encoder.poll
            calls = []
            def exited_once():
                calls.append(True)
                return 1 if len(calls) == 1 else real_poll()
            with patch.object(sink._encoder, 'poll', side_effect=exited_once):
                with self.assertRaisesRegex(RuntimeError, 'encoder exited'):
                    sink.accept('0.mkv', self.lease(0))
            self.assertEqual(sink.audio_evidence, before)

    def test_invalid_lease_does_not_commit(self):
        with self.sink() as sink:
            before = sink.audio_evidence
            with self.assertRaises(ValueError):
                sink.accept('0.mkv', object())
            self.assertEqual(sink.audio_evidence, before)
