"""Device-free tests; optional installed CPU torch, never installs packages."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

from research.h3_personal.voxcpm_supply_diagnostic import run, _decode, _targets


class Guards(unittest.TestCase):
    def test_text_bound(self):
        for text in ('', 'x'*4097, None):
            with self.assertRaises(ValueError):
                run(None, text, None, None, None)

    def test_trace_must_be_off(self):
        candidate = types.SimpleNamespace(observe_stop=True, _active=False)
        with self.assertRaises(RuntimeError):
            run(candidate, 'hello', None, None, None)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'optional local torch/numpy absent')
class CPUIntegration(unittest.TestCase):
    def test_exact_fold_and_sequence(self):
        import torch
        import numpy as np

        class VAE(torch.nn.Module):
            latent_dim = 3

            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 1, 1))

            def streaming_decode(self):
                vae = self

                class Context:
                    def __enter__(self):
                        self.count = 0
                        return self

                    def decode_chunk(self, z):
                        self.count += 1
                        return vae.decoder(z) + self.count

                    def __exit__(self, *args):
                        pass
                return Context()

        model = torch.nn.Module()
        model.audio_vae = VAE()
        model.patch_size = 2
        model.eval()
        candidate = types.SimpleNamespace(observe_stop=False, _active=False,
                                           _model=types.SimpleNamespace(tts_model=model))
        calls = []

        def measure(c, text, out, kind, **kwargs):
            calls.append((len(_targets(model.audio_vae.decoder, torch)),
                          float(np.random.rand()), float(torch.rand(()))))
            Path(out).mkdir(parents=True)
            (Path(out)/'native.wav').write_bytes(b'fake')
            return {'pcm_sha256': 'test-only', 'iterator_completion_seconds': 0.1}

        with tempfile.TemporaryDirectory() as root:
            report = run(candidate, 'same text', Path(root)/'experiment', torch, measure)
            self.assertTrue(report['exact_pcm_equal'])
            self.assertTrue(report['parameter_dtypes_unchanged'])
            self.assertEqual(report['remaining_hooks'], 0)
            self.assertEqual(len(report['cases']), 4)
            self.assertEqual([n for n, _, _ in calls], [1, 1, 1, 0, 0, 0])
            self.assertEqual(len(set((a, b) for _, a, b in calls)), 1)
            self.assertTrue((Path(root)/'experiment'/'report.json').is_file())
            self.assertEqual(len(list((Path(root)/'experiment').rglob('native.wav'))), 4)

    def test_decode_clones_and_preserves_context(self):
        import torch
        entries = []

        class VAE:
            def streaming_decode(self):
                class Context:
                    def __enter__(self):
                        entries.append(1)
                        self.i = 0
                        return self

                    def decode_chunk(self, z):
                        self.i += 1
                        z.add_(self.i)
                        return z

                    def __exit__(self, *args):
                        pass
                return Context()

        patches = [torch.zeros(1), torch.zeros(1)]
        output = _decode(VAE(), patches, torch)
        self.assertEqual(entries, [1])
        self.assertEqual([v.item() for v in output], [1, 2])
        self.assertEqual([v.item() for v in patches], [0, 0])


if __name__ == '__main__':
    unittest.main()
