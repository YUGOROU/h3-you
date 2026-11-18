import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from research.h3_personal import tts_streaming_measure as m


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class StreamingMeasureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.clock = Clock()
        patcher = patch.object(m.time, 'monotonic', self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def candidate(self, chunks, delays=None, rate=10):
        clock = self.clock

        class Candidate:
            sample_rate = rate
            closed = False

            def stream(inner, text):
                # Setup cost must be included in TTFA.
                clock.now += 2

                def generate():
                    try:
                        for index, chunk in enumerate(chunks):
                            clock.now += delays[index] if delays else 0
                            yield chunk
                        clock.now += 0.25
                    finally:
                        inner.closed = True
                return generate()
        return Candidate()

    def test_timing_totals_exact_wav_and_privacy(self):
        chunks = [b'', b'\x00\x80' * 10, b'\xff\x7f' * 5]
        candidate = self.candidate(chunks, [0.5, 0.5, 2])
        result = m.measure(candidate, 'private utterance', self.directory, 'pcm16')
        self.assertEqual(result['ttfa_seconds'], 3)
        self.assertEqual(result['iterator_completion_seconds'], 5.25)
        self.assertEqual(result['max_late_supply_seconds'], 1)
        self.assertEqual(result['total_samples'], 15)
        self.assertEqual(result['duration_seconds'], 1.5)
        self.assertEqual(result['chunk_count'], 3)
        self.assertEqual(result['pcm_sha256'], hashlib.sha256(b''.join(chunks)).hexdigest())
        with wave.open(str(self.directory / result['wav_file']), 'rb') as wav:
            self.assertEqual(wav.getframerate(), 10)
            self.assertEqual(wav.getnframes(), 15)
            self.assertEqual(wav.readframes(15), b''.join(chunks))
        saved = (self.directory / 'timing.json').read_text()
        self.assertNotIn('private utterance', saved)
        self.assertEqual(json.loads(saved), result)
        self.assertTrue(candidate.closed)

    def test_optional_timing_attribution(self):
        candidate = self.candidate([b'', b'\0\0', b'\1\0'], [0.5, 0.5, 2])
        # Independent deterministic clock: must not change monotonic semantics.
        ticks = iter(range(0, 100_000_000, 1_000_000))
        with patch.object(m.time, 'perf_counter_ns', side_effect=lambda: next(ticks)):
            result = m.measure(candidate, 'x', self.directory, 'pcm16',
                               timing_attribution=True)
        self.assertEqual(result['ttfa_seconds'], 3)
        self.assertEqual(result['iterator_completion_seconds'], 5.25)
        self.assertEqual(result['final_next_call_seconds'], 0.001)
        self.assertEqual(result['attribution_clock'],
                         'perf_counter_ns-relative-to-measure-start')
        previous_ready = 0
        for row in result['chunks']:
            self.assertEqual(row['next_call_seconds'], 0.001)
            self.assertEqual(row['consumer_seconds'], 0.002)
            self.assertGreater(row['validated_ready_seconds'], previous_ready)
            previous_ready = row['validated_ready_seconds']
        self.assertEqual(json.loads((self.directory / 'timing.json').read_text()), result)
        self.assertTrue(candidate.closed)

    def test_default_does_not_use_attribution_clock(self):
        with patch.object(m.time, 'perf_counter_ns', side_effect=AssertionError('unexpected clock')):
            result = m.measure(self.candidate([b'\0\0']), 'x', self.directory, 'pcm16')
        self.assertNotIn('final_next_call_seconds', result)
        self.assertNotIn('attribution_clock', result)
        for field in ('next_call_seconds', 'consumer_seconds', 'validated_ready_seconds'):
            self.assertNotIn(field, result['chunks'][0])

    def assert_rejected(self, candidate, message):
        with self.assertRaisesRegex(ValueError, message):
            m.measure(candidate, 'x', self.directory, 'pcm16')
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_native_completion_is_separate_and_unknown_keeps_artifacts(self):
        for index, evidence in enumerate((None, {'reason':'max_len','natural_stop':False},
                                           {'reason':'natural_stop','natural_stop':True,'stream_exhausted':True,'decision_monotonic_ns':123},
                                           {'reason':'natural_stop','natural_stop':True,'stream_exhausted':False,'decision_monotonic_ns':123})):
            candidate = self.candidate([b'\0\0'])
            original = candidate.stream
            def stream(text):
                yield from original(text)
                candidate.completion = evidence
                return {'iterator':'returned'}
            candidate.stream = stream
            directory = self.directory / str(index)
            result = m.measure(candidate, 'x', directory, 'pcm16')
            self.assertEqual(result['native_completion'], evidence)
            self.assertEqual(result['iterator_return'], {'iterator':'returned'})
            self.assertEqual(result['formal_completion_accepted'], index == 2)
            self.assertLess(result['last_pcm_arrival_seconds'], result['iterator_completion_seconds'])
            self.assertTrue((directory / 'native.wav').exists())

    def test_audio_bound_removes_partial(self):
        candidate = self.candidate([b'\0\0', b'\0\0' * 300])
        self.assert_rejected(candidate, 'duration')
        self.assertTrue(candidate.closed)

    def test_wall_bound(self):
        self.assert_rejected(self.candidate([b'\0\0', b'\0\0'], [0, 121]), 'wall')

    def test_chunk_bound_including_empty_chunks(self):
        self.assert_rejected(self.candidate([b'\0\0'] + [b''] * 1024), 'chunk')

    def test_empty_and_unaligned(self):
        self.assert_rejected(self.candidate([]), 'no playable')
        self.assert_rejected(self.candidate([b'x']), 'aligned')

    def test_primary_error_survives_close_failure(self):
        class Broken:
            sample_rate = 10
            closed = False

            def stream(inner, text):
                return inner

            def __iter__(inner):
                return inner

            def __next__(inner):
                raise RuntimeError('primary')

            def close(inner):
                inner.closed = True
                raise RuntimeError('secondary')
        candidate = Broken()
        with self.assertRaisesRegex(RuntimeError, '^primary$'):
            m.measure(candidate, 'x', self.directory, 'pcm16')
        self.assertTrue(candidate.closed)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_rate_validation(self):
        for rate in (True, 0, 48001, 1.5):
            self.assert_rejected(self.candidate([], rate=rate), 'sample_rate')


if __name__ == '__main__':
    unittest.main()
