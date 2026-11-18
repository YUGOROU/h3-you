"""Synthetic contracts only: the pinned private WAV is not present or verified."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from research.h3_personal import continuation_diagnostic as diagnostic
from research.h3_personal.continuation_plan import plan
from research.h3_personal.fast_continuation_render import plan as wav_plan


def synthetic_plan():
    return dict(schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
                sha256=diagnostic.SOURCE_SHA256, source_samples=577920,
                source_sample_rate=48000, source_channels=1,
                total_frames=328, window_frames=124, overlap_frames=22,
                expected_windows=3, sampler='SMD4', fps=24, seed=314159,
                windows=[w.record() for w in plan(328, 124, 22)])


class DiagnosticTests(unittest.TestCase):
    def test_synthetic_scope_is_not_verification(self):
        result = diagnostic.validate_metadata(diagnostic.source_contract(),
                                              synthetic_plan(), timeout_seconds=1800)
        self.assertEqual(result['suffix_frames'], [119, 102, 107])
        self.assertEqual(sum(result['suffix_frames']), result['total_frames'])
        self.assertEqual((result['job_count'], result['gpu'], result['gpu_count']),
                         (1, 'H200', 2))
        self.assertEqual((result['sampler'], result['precision'], result['policy']),
                         ('SMD4', 'BF16', 'resident-visual-cached'))
        for key in ('local_source_verified', 'execution_authorized', 'gpu_ready',
                    'quality_verified', 'realtime_verified', 'fallback_allowed'):
            self.assertIs(result[key], False)
        self.assertIs(result['source']['new_speech'], False)

    def test_source_mismatches(self):
        source = diagnostic.source_contract()
        for key in source:
            bad = dict(source, **{key: 'wrong'})
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnostic.validate_metadata(bad, synthetic_plan(), timeout_seconds=60)
        source['source_channels'] = True
        with self.assertRaises(ValueError):
            diagnostic.validate_metadata(source, synthetic_plan(), timeout_seconds=60)

    def test_plan_mismatches(self):
        for key in synthetic_plan():
            bad = copy.deepcopy(synthetic_plan())
            bad[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnostic.validate_metadata(diagnostic.source_contract(), bad,
                                             timeout_seconds=60)

    def test_timeout_before_io(self):
        for timeout in (None, True, 0, -1, 3601, 1.0, float('inf'), float('nan')):
            with self.subTest(timeout=timeout), patch.object(diagnostic, 'wav_plan') as reader:
                with self.assertRaises(ValueError):
                    diagnostic.preflight('not-read.wav', timeout_seconds=timeout)
                reader.assert_not_called()
        for timeout in (1, 3600):
            diagnostic.validate_metadata(diagnostic.source_contract(), synthetic_plan(),
                                         timeout_seconds=timeout)

    def test_actual_synthetic_wav_is_not_fixed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
                stream.writeframes(bytes(577920 * 2))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            actual = wav_plan(path, digest)
            self.assertEqual(actual['expected_windows'], 3)
            self.assertEqual(actual['source_samples'], 577920)
            with self.assertRaises(RuntimeError):
                diagnostic.preflight(path, timeout_seconds=60)
            with self.assertRaises(ValueError):
                diagnostic.validate_metadata(diagnostic.source_contract(), actual,
                                             timeout_seconds=60)
            # Existing planner validates the real header even with a matching hash.
            path.write_bytes(b'not a WAV')
            with self.assertRaises((wave.Error, EOFError)):
                wav_plan(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_preflight_delegates_fixed_digest_and_rejects_header_drift(self):
        bad = synthetic_plan()
        bad['source_sample_rate'] = 44100
        with patch.object(diagnostic, 'wav_plan', return_value=bad) as reader:
            with self.assertRaises(ValueError):
                diagnostic.preflight('caller-local.wav', timeout_seconds=60)
            reader.assert_called_once_with('caller-local.wav', diagnostic.SOURCE_SHA256)


if __name__ == '__main__':
    unittest.main()
