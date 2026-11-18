"""Execute the installed warm hook on CPU stubs; no FastVideo/Torch required."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.h3_personal import fast_ref2va


PIPELINE = '''
def h3_encoder_split_enabled(args): return False
class Pipeline:
    def forward(self, batch, fastvideo_args):
        if h3_encoder_split_enabled(fastvideo_args):
            try:
                return batch
            finally:
                pass
        self.cold_calls += 1
        self._release_text_encoder()
        self._load_denoise_modules(fastvideo_args)
        return batch
    def _release_text_encoder(self):
        self.releases += 1
    def _load_denoise_modules(self, fastvideo_args):
        pass
    def load(self, loaded):
        if loaded:
            self._apply_inference_compile(tuple(name for name in loaded if name in _DENOISE_MODULE_NAMES))
    def _release_all_lazy_modules(self):
        self.releases += 1
'''


class WarmFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        base = root / 'fastvideo/pipelines/basic/minimax_h3'
        files = {
            root / 'fastvideo/pipelines/lora_pipeline.py':
                '        elif not self.training_mode and self.lora_path is not None:\n',
            base / 'minimax_h3_pipeline.py': PIPELINE,
            base / 'stages/minimax_h3_denoising.py': '        elif dmd_steps is None:\n',
            base / 'stages/minimax_h3_conditioning.py':
                '        batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = text_token_tags\n'
                '            return self._ref2va_memo.get_or_compute(key, lambda: self._present_ref2va(batch, references, device))\n'
                '            video_block_token_counts,\n        )\n        return self._encode_tokens(\n',
            base / 'stages/minimax_h3_input_preparation.py':
                '        batch.references = [prepare_reference(reference, num_frames, target_sample_rate) for reference in references]\n',
        }
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        with patch('research.h3_personal.ref2va_runtime.install_ref_guard', return_value={}):
            fast_ref2va.install(root)
        namespace = {}
        exec(compile((base / 'minimax_h3_pipeline.py').read_text(), 'installed-pipeline', 'exec'), namespace)
        self.pipeline = namespace['Pipeline']()
        self.pipeline._denoise_stages_ready = True
        self.pipeline._stage_name_mapping = {
            'conditioning_stage': SimpleNamespace(_smd_condition_cache=('key', [], None))}
        self.pipeline.releases = self.pipeline.cold_calls = 0
        self.batch = SimpleNamespace(extra={})
        self.manifest = root / 'window.json'
        env = patch.dict(os.environ, {'H3_SMD_REF2VA': '1'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        key = patch.object(fast_ref2va, 'condition_key', return_value='key')
        key.start()
        self.addCleanup(key.stop)

    def activate(self, mode='latent-tail-v1'):
        self.manifest.write_text(json.dumps(dict(
            mode=mode, index=0, total_frames=124, window_frames=124,
            overlap_frames=22, seed=314159, expected_windows=1, final=True,
            session_id='test', wav='/unused.wav', sha256='a' * 64)))
        os.environ['H3_SMD_CONTINUATION_WINDOW_FILE'] = str(self.manifest)

    def call(self):
        return self.pipeline.forward(self.batch, None)

    def test_independent_error_releases_even_with_manifest_environment(self):
        self.activate('independent')
        error = RuntimeError('independent failure')
        with patch.object(fast_ref2va, 'warm_forward', side_effect=error) as forward:
            with self.assertRaises(RuntimeError) as caught:
                self.call()
        self.assertIs(caught.exception, error)
        forward.assert_called_once()
        self.assertEqual(self.pipeline.releases, 1)
        self.assertFalse(getattr(self.pipeline, '_smd_continuation_warm_faulted', False))

    def test_continuation_error_retains_and_refuses_all_subsequent_use(self):
        self.activate()
        error = KeyboardInterrupt('injected failure before batch context exists')

        def fail(*args):
            self.manifest.unlink()  # Teardown must not re-read this manifest.
            raise error

        with patch.object(fast_ref2va, 'warm_forward', side_effect=fail) as forward:
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.call()
            self.assertIs(caught.exception, error)
            self.assertEqual(self.pipeline.releases, 0)
            self.assertTrue(self.pipeline._smd_continuation_warm_faulted)
            # Refuse even a cold miss or disabled SMD, before key/manifest work.
            self.pipeline._denoise_stages_ready = False
            os.environ.clear()
            with self.assertRaisesRegex(RuntimeError, 'reuse forbidden'):
                self.call()
            forward.assert_called_once()
        self.assertEqual(self.pipeline.cold_calls, 0)

    def test_batch_activation_marker_needs_no_manifest_read(self):
        self.batch.extra['_h3_continuation'] = {'owner': object()}
        os.environ['H3_SMD_CONTINUATION_WINDOW_FILE'] = '/nonexistent'
        error = ValueError('stage failed')
        with patch.object(fast_ref2va, 'warm_forward', side_effect=error):
            with self.assertRaises(ValueError) as caught:
                self.call()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.pipeline.releases, 0)
        self.assertTrue(self.pipeline._smd_continuation_warm_faulted)

    def test_invalid_manifest_fails_before_warm_execution(self):
        self.activate()
        self.manifest.write_text('{')
        with patch.object(fast_ref2va, 'warm_forward') as forward:
            with self.assertRaises(ValueError):
                self.call()
            forward.assert_not_called()
        self.assertEqual(self.pipeline.releases, 0)

    def test_success_remains_reusable(self):
        self.activate()
        with patch.object(fast_ref2va, 'warm_forward', return_value=self.batch) as forward:
            self.assertIs(self.call(), self.batch)
            self.assertIs(self.call(), self.batch)
            self.assertEqual(forward.call_count, 2)
        self.assertEqual(self.pipeline.releases, 0)


if __name__ == '__main__':
    unittest.main()
