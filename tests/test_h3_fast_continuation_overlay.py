"""Real pinned-source installer composition, offline and without native imports.

Run with unittest discovery and H3_FASTVIDEO_ARCHIVE pointing to an existing
pinned archive. Missing automatic fixtures skip; incompatible fixtures fail.
These tests do not establish GPU/numerical/runtime acceptance.
"""
import hashlib
import types
import unittest

from test_h3_fast_continuation_stages import pinned_source
from research.h3_personal.fast_ref2va import install as install_smd
from research.h3_personal.external_audio import install as install_audio
from research.h3_personal.fast_continuation_overlay import install as install_continuation


class ComposedOverlayTests(unittest.TestCase):
    def test_actual_runtime_install_order_compiles(self):
        with pinned_source() as source:
            # Same order as LiveVoiceRuntime._start; no mocked installers or
            # fabricated anchor snippets. Only the private source copy changes.
            install_smd(source)
            install_audio(source)
            before = {p: p.read_bytes() for p in source.rglob('*.py')}
            try:
                report = install_continuation(source)
            except Exception:
                # The continuation installer promises transactional validation.
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                raise  # Preserve actual compiler/anchor diagnostic, not xfail.
            from research.h3_personal.fast_continuation import verify_sequence_transport
            packing = types.SimpleNamespace(__file__=str(source/'fastvideo/pipelines/basic/minimax_h3/packing.py'))
            verify_sequence_transport(packing, 55551, 2)
            self.assertEqual(report['mode'], 'latent-tail-v1-opt-in')
            self.assertFalse(report['gpu_verified'])
            self.assertEqual(len(report['stages']), 3)
            for path in sorted(source.rglob('*.py')):
                compile(path.read_text(), str(path), 'exec')
                if path.name in report['stages']:
                    row = report['stages'][path.name]
                    self.assertEqual(row['original_sha256'], hashlib.sha256(before[path]).hexdigest())
                    self.assertEqual(row['patched_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertIn('fast_continuation_', path.read_text())
            denoising = next(p for p in before if p.name == 'minimax_h3_denoising.py').read_text()
            self.assertEqual(denoising.count('with forward_profile(batch, index):'), 1)
            self.assertIn('with forward_profile(batch, index):\n                            video_velocity, audio_velocity = self.transformer(', denoising)
            after = {p: p.read_bytes() for p in before}
            with self.assertRaisesRegex(ValueError, 'single-use'):
                install_continuation(source)
            self.assertEqual(after, {p: p.read_bytes() for p in before})


if __name__ == '__main__':
    unittest.main()
