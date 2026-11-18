"""No weights: fake encoders, real CPU tensors, pinned decode and presentation."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from research.h3_personal import component_cache as cache
from research.h3_personal import component_encode as encode

PIN = Path('/tmp/h3-identity-audit/aviad')
TENSORS = all(importlib.util.find_spec(n) is not None for n in ('torch', 'safetensors', 'PIL'))
HASHES = {k: encode.digest({'SYNTHETIC': k}) for k in cache.KINDS}


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('video', 'audio', 'ref1', 'ref2'):
            (self.root / name).write_bytes(name.encode())
        self.row = dict(id='a', video=str(self.root / 'video'), audio=str(self.root / 'audio'),
                        reference_image=str(self.root / 'ref1'), caption='caption', width=32, height=32, num_frames=22)

    def plan(self, rows=None, **kwargs):
        return encode.make_plan(rows or [self.row], HASHES, attested_by='SYNTHETIC TEST', **kwargs)

    def test_unique_assignments_and_reference_change(self):
        rows = [self.row, dict(self.row, id='b'), dict(self.row, id='c', reference_image=str(self.root / 'ref2'))]
        plan = self.plan(rows)
        self.assertEqual(len(plan['units']), 8)  # 2 targets + 2*(reference/text/probe)
        flat = sum(plan['media_shards'], [])
        self.assertEqual(len(flat), 2)
        self.assertEqual(len(flat), len(set(flat)))
        first, second, third = [r['units'] for r in plan['samples']]
        self.assertEqual(first, second)
        for kind in ('video', 'audio'):
            self.assertEqual(first[kind], third[kind])
        for kind in ('reference', 'text', 'identity_uncond'):
            self.assertNotEqual(first[kind], third[kind])
        self.assertFalse(plan['c_gate_enabled'])
        relocated = self.root / 'copied'
        shutil.copyfile(self.root / 'video', relocated)
        other = self.plan([dict(self.row, video=str(relocated))])
        self.assertEqual(first['video'], other['samples'][0]['units']['video'])
        (self.root / 'video').write_bytes(b'changed bytes')
        self.assertNotEqual(first['video'], self.plan()['samples'][0]['units']['video'])

    def test_fail_closed_inputs(self):
        for rows in ([self.row, self.row], [dict(self.row, reference_image=['ref1', 'ref2'])],
                     [dict(self.row, start_frame=0)], [dict(self.row, caption=None)]):
            with self.assertRaises(ValueError):
                self.plan(rows)
        for count in (0, 3, True):
            with self.assertRaises(ValueError):
                self.plan(media_shards=count)
        bad = self.plan()
        bad['media_shards'][1] += bad['media_shards'][0]
        with self.assertRaises(ValueError):
            encode.Engine(self.root / 'out', bad, None)


class FakeBackend:
    def __init__(self):
        self.calls = []

    def video(self, unit):
        import torch
        self.calls.append('video')
        g = unit['settings']
        vr, _ = cache.geometry_rows(g['width'], g['height'], g['num_frames'])
        return {'latents': torch.zeros(vr, 96)}

    def audio(self, unit):
        import torch
        self.calls.append('audio')
        _, ar = cache.geometry_rows(32, 32, unit['settings']['num_frames'])
        return {'latents': torch.zeros(ar, 32)}

    def image(self, unit):
        return Path(unit['sources']['image']['path']).read_bytes()

    def reference(self, unit, image):
        import torch
        self.calls.append('reference')
        return {'latents': torch.full((1, 96), float(sum(image))), 'geometry': torch.tensor([[0, 1, 2, 2, 0]])}

    def presentation(self, image, geometry):
        return [SimpleNamespace(kind='image', image=image)]

    def text(self, prompt, references):
        import torch
        self.calls.append(('text', prompt, id(references[0].image), references[0].image))
        n = 2 + len(prompt)
        return {'prompt_embeds': torch.zeros(n, 5120, dtype=torch.bfloat16),
                'text_token_tags': torch.tensor([1, 0] + [1] * len(prompt))}


@unittest.skipUnless(TENSORS, 'use /tmp/h3-media-env for tensor tests')
class ExecutionTests(unittest.TestCase):
    setUp = PlanTests.setUp
    plan = PlanTests.plan

    def execute(self, plan, root, backend=None, reuse=None):
        backend = backend or FakeBackend()
        engine = encode.Engine(root, plan, backend, reuse_catalog=reuse)
        for rank in range(len(plan['media_shards'])):
            engine.run_media(rank)
        engine.run_references()
        engine.run_text()
        output = engine.finish()
        return json.loads(output.read_text()), backend

    def test_execute_reuse_and_partial_missing_components(self):
        rows = [self.row, dict(self.row, id='b'), dict(self.row, id='c', reference_image=str(self.root / 'ref2'))]
        old, backend = self.execute(self.plan(rows), self.root / 'old')
        self.assertEqual(len(old['samples']), 3)
        self.assertEqual(backend.calls.count('video'), 1)
        self.assertEqual(backend.calls.count('audio'), 1)
        self.assertEqual(backend.calls.count('reference'), 2)
        calls = [c for c in backend.calls if isinstance(c, tuple)]
        self.assertEqual(len(calls), 4)
        for a, b in zip(calls[::2], calls[1::2]):
            self.assertEqual(a[2:], b[2:])  # exact same prepared pixels object
            self.assertNotEqual(len(a[1]), len(b[1]))
        newroot = self.root / 'new'
        shutil.copytree(self.root / 'old' / 'objects', newroot / 'objects')
        changed = [dict(self.row, caption='changed text', reference_image=str(self.root / 'ref2'))]
        new, backend = self.execute(self.plan(changed), newroot, reuse=old['components'])
        self.assertNotIn('video', backend.calls)
        self.assertNotIn('audio', backend.calls)
        self.assertNotIn('reference', backend.calls)  # common ref2 already encoded
        for kind in ('video', 'audio'):
            self.assertEqual(old['samples'][0]['components'][kind], new['samples'][0]['components'][kind])
        self.assertEqual(len(new['components']), 5)  # no stale/unselected catalog entries
        partialroot = self.root / 'partial'
        shutil.copytree(newroot / 'objects', partialroot / 'objects')
        partial = {k: v for k, v in new['components'].items() if v['kind'] != 'audio'}
        # Remove ONLY missing audio output, not its good video sibling.
        audio = new['components'][new['samples'][0]['components']['audio']]
        (partialroot / audio['path']).unlink()
        result, backend = self.execute(self.plan(changed), partialroot, reuse=partial)
        self.assertEqual(backend.calls, ['audio'])
        self.assertEqual(result['samples'][0]['components'], new['samples'][0]['components'])

    def test_common_probe_computation_for_different_captions(self):
        rows = [self.row, dict(self.row, id='b', caption='a longer caption')]
        payload, backend = self.execute(self.plan(rows), self.root / 'out')
        prompts = [c[1] for c in backend.calls if isinstance(c, tuple)]
        self.assertEqual(prompts, ['caption', 'a longer caption', ' '])
        a, b = payload['samples']
        self.assertNotEqual(a['components']['text'], b['components']['text'])
        self.assertNotEqual(a['components']['identity_uncond'], b['components']['identity_uncond'])
        self.assertEqual(a['components']['reference'], b['components']['reference'])
        cache.validate_index(self.root / 'out', payload)  # separate correct paired metadata

    def test_independent_missing_reference_text_and_probe(self):
        plan = self.plan()
        old, _ = self.execute(plan, self.root / 'old')
        for missing in ('reference', 'text', 'identity_uncond', 'video'):
            with self.subTest(missing=missing):
                root = self.root / ('missing-' + missing)
                shutil.copytree(self.root / 'old' / 'objects', root / 'objects')
                removed = {missing}
                if missing in ('reference', 'text'):
                    removed.add('identity_uncond')  # a reusable probe must retain valid dependencies
                reuse = {k: v for k, v in old['components'].items() if v['kind'] not in removed}
                for entry in old['components'].values():
                    if entry['kind'] in removed:
                        (root / entry['path']).unlink()
                result, backend = self.execute(plan, root, reuse=reuse)
                numerical = [c[0] if isinstance(c, tuple) else c for c in backend.calls]
                expected = [('text' if k == 'identity_uncond' else k) for k in removed]
                self.assertCountEqual(numerical, expected)
                for kind in cache.KINDS - removed:
                    self.assertEqual(result['samples'][0]['components'][kind], old['samples'][0]['components'][kind])
                from safetensors.torch import load_file
                for kind in removed:
                    before = old['components'][old['samples'][0]['components'][kind]]
                    after = result['components'][result['samples'][0]['components'][kind]]
                    a = load_file(str(self.root / 'old' / before['path']))
                    b = load_file(str(root / after['path']))
                    self.assertEqual(set(a), set(b))
                    for key in a:
                        self.assertTrue(a[key].equal(b[key]))

    def test_no_silent_skip_failure_and_mutation(self):
        plan = self.plan()
        engine = encode.Engine(self.root / 'out', plan, FakeBackend())
        with self.assertRaises(FileNotFoundError):
            engine.finish()
        (self.root / 'video').write_bytes(b'changed')
        rank = next(i for i, shard in enumerate(plan['media_shards']) if plan['samples'][0]['units']['video'] in shard)
        with self.assertRaisesRegex(ValueError, 'source bytes'):
            engine.run_media(rank)
        self.assertFalse((self.root / 'out/index.json').exists())
        self.setUp()
        class Broken(FakeBackend):
            def text(self, prompt, references):
                raise RuntimeError('conditioner failed')
        engine = encode.Engine(self.root / 'out', self.plan(), Broken())
        for rank in (0, 1):
            engine.run_media(rank)
        engine.run_references()
        with self.assertRaisesRegex(RuntimeError, 'conditioner failed'):
            engine.run_text()
        self.assertFalse((engine.root / 'text.json').exists())
        self.assertFalse((engine.root / 'index.json').exists())

    def test_exclusive_publication_and_wrong_receipt(self):
        plan = self.plan()
        payload, _ = self.execute(plan, self.root / 'out')
        cache.validate_index(self.root / 'out', payload)
        engine = encode.Engine(self.root / 'out', plan, FakeBackend())
        with self.assertRaises(FileExistsError):
            engine.finish()
        receipt = engine.root / 'text.json'
        value = json.loads(receipt.read_text())
        value['bindings'] = {}
        receipt.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            engine.finish()


@unittest.skipUnless(TENSORS and PIN.exists() and shutil.which('ffmpeg'), 'pinned media/tensor fixture required')
class RealPinnedTests(unittest.TestCase):
    setUp = PlanTests.setUp
    plan = PlanTests.plan
    execute = ExecutionTests.execute

    def test_actual_encoder_methods_with_fake_posteriors(self):
        import numpy as np
        import torch
        from PIL import Image
        sys.path.insert(0, str(PIN / 'src'))
        self.addCleanup(lambda: sys.path.remove(str(PIN / 'src')))
        from h3_trainer.preprocessing.encoders import H3Encoders
        h3 = H3Encoders.__new__(H3Encoders)
        h3.device = torch.device('cpu')
        h3._pixel_mean = torch.zeros(1, 3, 1, 1, 1)
        h3._pixel_std = torch.ones(1, 3, 1, 1, 1)
        h3._latents_mean = torch.zeros(1, 24, 1, 1, 1)
        h3._latents_std = torch.ones(1, 24, 1, 1, 1)
        h3._audio_mean = torch.zeros(1, 1, 32)
        h3._audio_std = torch.ones(1, 1, 32)
        h3.audio_latent_channels = 32
        seen = []
        class Mode:
            def __init__(self, tensor):
                self.tensor = tensor
            def mode(self):
                seen.append('mode')
                return self.tensor
            def sample(self, *args, **kwargs):
                raise AssertionError('targets must not sample')
        class VideoVAE:
            def encode(self, pixels):
                assert pixels.shape == (1, 3, 22, 32, 32)
                assert pixels.dtype == torch.float32
                assert float(pixels.max()) == 1.0
                return SimpleNamespace(latent_dist=Mode(torch.ones(1, 24, 7, 2, 2)))
            def _encode_clip(self, pixels):
                assert pixels.shape == (1, 3, 1, 32, 32)
                return torch.zeros(1, 48, 1, 2, 2)
        class AudioVAE:
            def encode(self, waveform, return_dict):
                assert waveform.shape == (2, 1, 29600)
                assert not return_dict
                # Different stereo channels prove channel-major layout.
                return (Mode(torch.stack([torch.ones(32, 37), torch.full((32, 37), 2.0)])),)
        h3.vae, h3.audio_vae = VideoVAE(), AudioVAE()
        video = h3.encode_video_target(np.full((22, 32, 32, 3), 255, dtype=np.uint8))
        self.assertEqual(tuple(video.shape), (7, 96))
        self.assertTrue(bool((video == 1).all()))
        audio = h3.encode_audio(torch.zeros(2, 29600))
        self.assertEqual(tuple(audio.shape), (74, 32))
        self.assertTrue(bool((audio[:37] == 1).all()))
        self.assertTrue(bool((audio[37:] == 2).all()))
        image = Image.new('RGB', (32, 32), 'red')
        ref = h3.encode_reference('image', image=image)
        other = h3.encode_reference('image', image=image)
        self.assertEqual(ref.geometry.tolist(), [0, 1, 2, 2, 0])
        self.assertEqual(tuple(ref.video_rows.shape), (1, 96))
        self.assertTrue(torch.equal(ref.video_rows, other.video_rows))
        self.assertTrue(torch.equal(ref.video_rows, ref.video_rows.half().float()))
        self.assertIsNone(ref.audio_rows)
        self.assertEqual(seen, ['mode', 'mode'])

    def test_actual_decode_silence_reference_presentation_and_loader(self):
        import numpy as np
        import torch
        from PIL import Image
        sys.path.insert(0, str(PIN / 'src'))
        self.addCleanup(lambda: sys.path.remove(str(PIN / 'src')))
        from h3_trainer.preprocessing.encoders import H3Encoders, EncodedReference
        from h3_trainer.preprocessing.media import extract_audio
        media = self.root / 'silent.mp4'
        result = subprocess.run(['ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                                 '-f', 'lavfi', '-i', 'color=c=red:s=32x32:r=24:d=1',
                                 '-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=stereo',
                                 '-t', '1', '-c:v', 'libx264', '-c:a', 'aac', str(media)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        Image.new('RGB', (40, 50), 'red').save(self.root / 'one.png')
        Image.new('RGB', (50, 40), 'blue').save(self.root / 'two.png')
        self.assertIsNone(extract_audio(media, 22))  # upstream really discards silence
        recovered = encode.recover_audio(media, 22)
        self.assertEqual(tuple(recovered.shape), (2, 29600))
        self.assertEqual(float(recovered.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, 'real audio'):
            encode.recover_audio(self.root / 'one.png', 22)
        h3 = H3Encoders.__new__(H3Encoders)  # NEVER __init__: no weight loading
        seen = []
        def video(frames):
            seen.append(('video', frames.shape, frames.dtype))
            return torch.zeros(7, 96)
        def audio(waveform):
            seen.append(('audio', waveform.clone()))
            return torch.zeros(74, 32)
        def reference(kind, image):
            seen.append(('reference', kind, image.size))
            return EncodedReference(kind, torch.zeros(1, 96), None, torch.tensor([0, 1, 2, 2, 0]), [])
        class Tokenizer:
            def __call__(self, text, add_special_tokens):
                assert add_special_tokens is False
                return {'input_ids': [ord(c) for c in text]}
            def convert_tokens_to_ids(self, token):
                return {'<|vision_start|>': 1000, '<|image_pad|>': 1001, '<|vision_end|>': 1002}[token]
        class Processor:
            merge_size = 2
            def __call__(self, images, return_tensors):
                self_image = images[0]
                seen.append(('pixels', id(self_image), np.array(self_image).copy()))
                return {'pixel_values': torch.tensor(np.array(self_image).mean(axis=(0, 1))),
                        'image_grid_thw': torch.tensor([[1, 4, 4]])}
        def conditioner(ids, tags, pixels, grid, *args):
            seen.append(('presentation', ids, tags, pixels.clone()))
            return torch.zeros(len(ids), 5120, dtype=torch.bfloat16), torch.tensor(tags)
        h3.encode_video_target, h3.encode_audio, h3.encode_reference = video, audio, reference
        h3.text = SimpleNamespace(tokenizer=Tokenizer(), processor=SimpleNamespace(image_processor=Processor()))
        h3._run_conditioner = conditioner
        backend = encode.PinnedBackend(h3, PIN)
        rows = [dict(self.row, video=str(media), audio=str(media), reference_image=str(self.root / 'one.png')),
                dict(self.row, id='b', video=str(media), audio=str(media), reference_image=str(self.root / 'two.png'))]
        payload, _ = self.execute(self.plan(rows), self.root / 'real', backend)
        self.assertEqual(len(payload['samples']), 2)
        self.assertEqual([x for x in seen if x[0] == 'video'], [('video', (22, 32, 32, 3), np.dtype('uint8'))])
        audio_seen = [x for x in seen if x[0] == 'audio']
        self.assertEqual(len(audio_seen), 1)
        self.assertTrue(torch.equal(audio_seen[0][1], recovered))
        pixels = [x for x in seen if x[0] == 'pixels']
        presentations = [x for x in seen if x[0] == 'presentation']
        for i in (0, 2):
            self.assertEqual(pixels[i][1], pixels[i + 1][1])
            np.testing.assert_array_equal(pixels[i][2], pixels[i + 1][2])
            self.assertNotEqual(len(presentations[i][1]), len(presentations[i + 1][1]))
            self.assertEqual(presentations[i][2].count(0), 6)
            self.assertEqual(presentations[i + 1][2].count(0), 6)
            self.assertEqual(presentations[i][1][-len('caption'):], [ord(c) for c in 'caption'])
            self.assertEqual(presentations[i + 1][1][-1], ord(' '))
        self.assertFalse(np.array_equal(pixels[0][2], pixels[2][2]))
        # Actual composed v2 loader, in a fresh interpreter to avoid source module collisions.
        from research import install_identity_component_cache as installer
        tree = installer.install(PIN, self.root / 'trainer')
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
from h3_trainer.datasets import PrecomputedDataset
sources = {k: 'ignored' for k in ('video','audio','reference','text','identity_uncond')}
d = PrecomputedDataset(sys.argv[2], sources, val_split_every=0)
assert len(d) == 2
for i in range(2):
    s = d[i]
    assert s['video'].shape == (7,96)
    assert s['audio'].shape == (74,32)
    assert s['text'].shape[0] != s['identity_uncond_text'].shape[0]
'''
        result = subprocess.run([sys.executable, '-c', script, str(tree / 'src'), str(self.root / 'real')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
