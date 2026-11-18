"""Synthetic PCM/score fixtures; not human speech or echo acceptance."""
from array import array
from dataclasses import replace
from io import BytesIO
import math
import random
import unittest

from research.h3_personal.local_vad import VADBoundary, VADConfig, speech_score


def frame(value):
    return int(value).to_bytes(2, 'little', signed=True)*320


def read_all(reader, size=5120):
    output = bytearray()
    while True:
        part = reader.read(size)
        if not part:
            return bytes(output)
        output.extend(part)


class VADTests(unittest.TestCase):
    def boundary(self, scores, *, tail=0, config=None):
        # Unique per-frame PCM proves exact retention, ordering and no replay.
        pcm = b''.join(frame(i+1) for i in range(len(scores)))
        it = iter(scores)
        return VADBoundary(BytesIO(pcm + bytes(tail)), config=config,
                           scorer=lambda _: next(it)), pcm

    def test_short_question_ends_500ms_after_last_speech(self):
        reader, pcm = self.boundary([0.]*25 + [1.]*20 + [0.]*100)
        actual = read_all(reader)
        # Six confirmation frames plus ten frames of true pre-onset history.
        self.assertEqual(actual, pcm[15*640:70*640])
        self.assertEqual(reader.reason, 'application_vad_quiet')
        self.assertEqual(reader.last_speech, .9)
        self.assertEqual(reader.vad_end, 1.4)
        self.assertEqual(reader.sent_samples*2, len(actual))
        self.assertFalse(reader.evidence()['speaker_identity_verified'])

    def test_long_question_brief_pause_and_quiet_ending(self):
        scores = [1.]*100 + [0.]*15 + [1.]*100 + [.4]*10 + [0.]*25
        reader, pcm = self.boundary(scores)
        self.assertEqual(read_all(reader), pcm)
        self.assertEqual(reader.last_speech, 4.5)
        self.assertEqual(reader.reason, 'application_vad_quiet')

    def test_short_noise_then_silence_is_no_speech(self):
        reader, _ = self.boundary([1.]*4 + [0.]*396)
        self.assertEqual(read_all(reader), b'')
        self.assertEqual(reader.reason, 'no_speech_window')
        self.assertIsNone(reader.speech_start)
        self.assertEqual(len(reader._pre), 0)

    def test_hysteresis_onset_and_minimum_are_distinct(self):
        reader, pcm = self.boundary([.8]*3 + [.4]*3 + [.4]*5 + [0.]*25)
        self.assertEqual(read_all(reader), pcm)
        self.assertEqual(reader.speech_start, 0.)
        reader, _ = self.boundary([.8, .4]*200)
        self.assertEqual(read_all(reader), b'')
        self.assertEqual(reader.reason, 'no_speech_window')

    def test_maximum_utterance_and_multiple_read_sizes(self):
        config = replace(VADConfig(), no_speech_ms=1000, maximum_ms=2000)
        for size in (2, 640, 5120, 65536):
            reader, pcm = self.boundary([1.]*120, config=config)
            self.assertEqual(read_all(reader, size), pcm[:100*640])
            self.assertEqual(reader.reason, 'maximum_utterance')
            self.assertEqual(read_all(reader, size), b'')

    def test_speech_at_window_start_and_eof_unpadded_tail(self):
        data = frame(1000)*6 + b'\x01\x00'*35
        reader = VADBoundary(BytesIO(data), scorer=lambda _: 1.)
        self.assertEqual(read_all(reader), data)
        self.assertEqual(reader.reason, 'source_eof')

    def test_fragmented_capture_preserves_samples(self):
        data = frame(1000)*15
        class Fragmented(BytesIO):
            def read(self, size):
                return super().read(min(size, 73))
        reader = VADBoundary(Fragmented(data), scorer=lambda _: 1.)
        self.assertEqual(read_all(reader), data)
        with self.assertRaises(ValueError):
            read_all(VADBoundary(BytesIO(data+b'x'), scorer=lambda _: 1.))

    def test_score_validation_and_bounds(self):
        for score in (float('nan'), float('inf'), -.1, 1.1, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                VADBoundary(BytesIO(frame(1)), scorer=lambda _: score).read(640)
        for kwargs in ({'quiet_ms': 1000}, {'pre_roll_ms': 100},
                       {'onset_score': float('nan')}, {'maximum_ms': 60000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VADConfig(**kwargs)

    def test_acoustic_features_not_only_energy(self):
        rng = random.Random(7)
        voice_like = b''.join(int(2000*math.sin(2*math.pi*200*i/16000)).to_bytes(
            2, 'little', signed=True) for i in range(320))
        white = b''.join(rng.randint(-3500, 3500).to_bytes(2, 'little', signed=True)
                         for _ in range(320))
        self.assertGreater(speech_score(voice_like), .65)
        self.assertLess(speech_score(white), .35)
        self.assertEqual(speech_score(frame(2000)), 0.)
        self.assertEqual(speech_score(bytes(640)), 0.)

    def test_ai_voice_not_misrepresented_as_human(self):
        # A speech-like AI score is intentionally indistinguishable here.
        reader, _ = self.boundary([1.]*20 + [0.]*25)
        read_all(reader)
        self.assertIsNotNone(reader.speech_start)
        self.assertFalse(reader.evidence()['speaker_identity_verified'])


if __name__ == '__main__':
    unittest.main()
