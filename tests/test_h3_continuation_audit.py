import json
from pathlib import Path
import tempfile
import unittest

from research.h3_personal.continuation_audit import compare


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.roots = [Path(self.tmp.name) / n for n in ('baseline', 'continuation')]
        self.specs = [dict(fast_config=dict(fps=24, prompt='PRIVATE SENTINEL'),
                           execution=dict(timeout=100 + i)) for i in range(2)]
        self.profiles = [dict(independent_ref2va_clips=True, external_audio_conditioning=False,
                             unique_frames=496,
                             generation_intervals=[[0, 90], [90, 95], [95, 100], [100, 105]],
                             derived_adapter_receipt=dict(derived_sha256='a')),
                         dict(mode='latent-tail-v1', speech_pcm_exact=True, unique_frames=328,
                              generation_intervals=[[0, 7], [8, 15], [16, 23]],
                              derived_adapter_receipt=dict(derived_sha256='b'))]
        self.stage()

    def stage(self):
        for root, spec, profile in zip(self.roots, self.specs, self.profiles):
            for name, document in [('deployment/spec.json', spec),
                                   ('deployment/model-manifest.json', {}),
                                   ('deployment/asset-manifest.json', {}),
                                   ('verified-results/profile.json', profile)]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(document))

    def test_scope_denominators_and_privacy(self):
        report = compare(*self.roots)
        self.assertEqual(report['baseline_warm_generation']['generation_seconds'], 15)
        self.assertEqual(report['baseline_warm_generation']['video_seconds'], 15.5)
        self.assertAlmostEqual(report['continuation_generation']['throughput'], 328 / 24 / 21)
        self.assertEqual(report['spec_difference_fields'], ['execution.timeout'])
        self.assertFalse(report['controlled_comparison'])
        self.assertEqual(report['derived_adapter_tensor_equality'], 'unverified')
        self.assertNotIn('PRIVATE SENTINEL', json.dumps(report))

    def test_actual_byte_hash_changes_even_for_equal_json(self):
        before = compare(*self.roots)
        path = self.roots[0] / 'deployment/spec.json'
        path.write_text(path.read_text() + '\n')
        after = compare(*self.roots)
        self.assertNotEqual(before['evidence_sha256'], after['evidence_sha256'])
        self.assertEqual(before['spec_difference_fields'], after['spec_difference_fields'])

    def test_invalid_evidence_rejected(self):
        for bad in ([[1, 0]], [[0, float('nan')]], [[0, float('inf')]], [[0, 0]]):
            with self.subTest(intervals=bad):
                self.profiles[1]['generation_intervals'] = bad
                self.stage()
                with self.assertRaises(ValueError):
                    compare(*self.roots)

    def test_wrong_clock_and_audio_contract_rejected(self):
        self.specs[1]['fast_config']['fps'] = 30
        self.stage()
        with self.assertRaisesRegex(ValueError, '24fps'):
            compare(*self.roots)
        self.specs[1]['fast_config']['fps'] = 24
        self.profiles[0]['external_audio_conditioning'] = True
        self.stage()
        with self.assertRaisesRegex(ValueError, 'audio evidence'):
            compare(*self.roots)


if __name__ == '__main__':
    unittest.main()
