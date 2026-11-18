"""Locked external-speech contract tests.

Always-on tests are pure Python. The pinned-source tests run only when
H3_FASTVIDEO_SOURCE_ARCHIVE points at the verified code-only FastVideo archive;
they never modify that archive and never load weights or a GPU.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal import external_audio as e


class LockedAudioPlanTests(unittest.TestCase):
    def test_window_arithmetic_matches_pinned_constants(self):
        self.assertEqual(e.audio_latent_frames(124), 207)
        self.assertEqual(e.expected_audio_rows(124), 414)
        self.assertAlmostEqual(e.window_seconds(124), 5.175)
        with self.assertRaises(ValueError):
            e.audio_latent_frames(123)

    def test_plan_covers_speech_without_truncation(self):
        plan = e.plan_windows(11.64, wav='/private/reply.wav', sha256_value='a' * 64)
        self.assertEqual(plan['windows_required'], 3)
        self.assertFalse(plan['truncation'])
        self.assertFalse(plan['time_stretch'])
        self.assertGreaterEqual(plan['windows_required'] * plan['window_seconds'], 11.64)
        self.assertGreaterEqual(plan['trailing_silence_seconds'], 0)
        self.assertEqual([row['index'] for row in plan['windows']], [0, 1, 2])
        self.assertEqual(plan['windows'][1]['latent_start'], plan['window_latents'])
        for row in plan['windows']:
            self.assertEqual(row['expected_rows'], row['latent_count'] * e.AUDIO_CHANNELS)

    def test_plan_rejects_overlong_speech_and_bad_bounds(self):
        # Exactly four windows is allowed; anything that would need a fifth is not.
        self.assertEqual(e.plan_windows(4 * e.window_seconds(), wav='/private/reply.wav',
                                        sha256_value='a' * 64)['windows_required'], 4)
        with self.assertRaises(ValueError):
            e.plan_windows(4 * e.window_seconds() + 0.1, wav='/private/reply.wav', sha256_value='a' * 64)
        for value in (0, -1, float('nan'), 'x'):
            with self.assertRaises(ValueError):
                e.plan_windows(value, wav='/private/reply.wav', sha256_value='a' * 64)
        with self.assertRaises(ValueError):
            e.plan_windows(5.0, wav='/private/reply.wav', sha256_value='short')

    def test_window_bounds_are_exact_and_never_clip(self):
        # 11.64s at 48 kHz, resampled to the pinned 32 kHz audio-VAE rate.
        speech = round(11.64 * e.AUDIO_SAMPLE_RATE)
        self.assertEqual(speech, 372480)
        total = 3 * e.window_latents()
        first = e.window_bounds(speech, 0, e.window_latents(), total)
        self.assertEqual((first['start'], first['available'], first['padded']), (0, 165600, 0))
        second = e.window_bounds(speech, 1 * e.window_latents(), e.window_latents(), total)
        self.assertEqual((second['start'], second['available']), (165600, 165600))
        last = e.window_bounds(speech, 2 * e.window_latents(), e.window_latents(), total)
        self.assertEqual((last['start'], last['available'], last['padded']), (331200, 41280, 124320))
        for bad in (dict(speech_samples=2 * e.window_latents() * e.HOP_LENGTH + 1, planned_latent_total=2 * e.window_latents()),
                    dict(latent_count=0), dict(latent_start=-1), dict(speech_samples=-1),
                    dict(planned_latent_total=e.window_latents())):
            values = dict(speech_samples=speech, latent_start=0, latent_count=e.window_latents(), planned_latent_total=total)
            values.update(bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                e.window_bounds(**values)

    def test_manifest_binds_one_window_and_default_off(self):
        plan = e.plan_windows(12.0, wav='/private/reply.wav', sha256_value='b' * 64)
        manifest = e.window_manifest(plan, 1)
        self.assertEqual(manifest['schema'], e.SCHEMA)
        self.assertEqual(manifest['index'], 1)
        self.assertEqual(manifest['expected_rows'], e.expected_audio_rows())
        self.assertEqual(manifest['planned_latent_total'], plan['windows_required'] * plan['window_latents'])
        with self.assertRaises(ValueError):
            e.window_manifest(plan, 3)
        with self.assertRaises(ValueError):
            e.window_manifest(dict(plan, schema='other'), 0)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                e.load_manifest()

    def test_module_never_plays_publishes_or_truncates(self):
        source = Path(e.__file__).read_text()
        ast.parse(source)
        for forbidden in ('ffmpeg', 'AVFoundation', 'AudioObject', 'urllib', 'requests', 'huggingface_hub',
                          'import subprocess', 'os.system', 'upload_folder', 'create_commit'):
            self.assertNotIn(forbidden, source)
        self.assertIn('instead of truncating', source)
        self.assertIn('supplied speech would be truncated by the planned windows', source)


class RunnerLockedAudioTests(unittest.TestCase):
    def speech(self, folder, seconds=11.64, rate=48000):
        import wave
        path = Path(folder) / 'reply.wav'
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
            handle.writeframes(b'\x00\x00' * int(seconds * rate))
        return path

    def args(self, path, sha, **overrides):
        from types import SimpleNamespace
        values = dict(locked_speech=path, locked_speech_sha256=sha, fast_ref_smd=True, runtime_policy='resident-visual')
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_runner_plan_binds_sha_duration_and_rejects_drift(self):
        from research.run_h3_ref2va import locked_audio_plan, speech_duration
        from research.h3_personal.external_audio import sha256
        with tempfile.TemporaryDirectory() as folder:
            path = self.speech(folder)
            self.assertAlmostEqual(speech_duration(path), 11.64, places=6)
            plan = locked_audio_plan(self.args(path, sha256(path)))
            self.assertEqual(plan['windows_required'], 3)
            self.assertFalse(plan['truncation'])
            self.assertIsNone(locked_audio_plan(self.args(None, None)))
            for overrides in (dict(locked_speech_sha256='0' * 64), dict(runtime_policy='baseline'),
                              dict(fast_ref_smd=False)):
                with self.assertRaises(ValueError):
                    locked_audio_plan(self.args(path, sha256(path), **overrides))
            with self.assertRaises(ValueError):
                locked_audio_plan(self.args(path, None))

    def test_runner_defaults_remain_offline_and_visual_only(self):
        source = Path(__import__('research.run_h3_ref2va', fromlist=['x']).__file__).read_text()
        self.assertIn("external_audio_conditioning=locked is not None", source)
        self.assertIn("visual_only=True", source)
        self.assertIn("if not a.run:", source)
        self.assertIn("H3_SMD_LOCKED_AUDIO='1' if locked is not None else '0'", source)
        self.assertIn("from research.h3_personal.external_audio import verify_receipts,window_manifest", source)


class LockedAudioReceiptTests(unittest.TestCase):
    def test_both_rank_receipts_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            base = dict(schema=e.SCHEMA, audio_timestep_policy='fixed_clean_1.0',
                        audio_scheduler_step_skipped=True, steps_held=4, rows_unchanged=True,
                        truncation=False, time_stretch=False, expected_rows=414)
            for index in range(3):
                for rank in (0, 1):
                    row = dict(base, window_index=index, encoded_rows=414, worker_rank=rank)
                    (Path(folder) / f'locked-audio-rank-{rank}-window-{index}.json').write_text(json.dumps(row))
            self.assertEqual(len(e.verify_receipts(folder, 3)), 6)
            path = Path(folder) / 'locked-audio-rank-1-window-2.json'
            path.write_text(json.dumps(dict(base, window_index=2, encoded_rows=414, worker_rank=1, steps_held=0)))
            with self.assertRaises(RuntimeError):
                e.verify_receipts(folder, 3)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch required')
    def test_lock_and_hold_reject_row_change(self):
        import torch
        from types import SimpleNamespace
        plan = e.plan_windows(5.0, wav='/private/reply.wav', sha256_value='c' * 64)
        state = SimpleNamespace(hop_length=e.HOP_LENGTH, sampling_rate=e.AUDIO_SAMPLE_RATE,
                                latent_channels=e.AUDIO_LATENT_CHANNELS)
        layout = SimpleNamespace(num_audio_latents=e.window_latents(), num_condition_audio_rows=0)
        batch = SimpleNamespace(audio_latents=torch.zeros(e.expected_audio_rows(), e.AUDIO_LATENT_CHANNELS),
                                extra={e.LAYOUT_KEY: layout})
        stage = SimpleNamespace(audio_vae=state)
        self.assertEqual(e.locked_audio_timestep(0.5), 1.0)
        with self.assertRaises(ValueError):
            e.hold_locked_audio_rows(stage, batch, torch.zeros(1))


@unittest.skipUnless(os.environ.get('H3_FASTVIDEO_SOURCE_ARCHIVE'), 'pinned archive required')
class PinnedSourceTests(unittest.TestCase):
    def extract(self, folder):
        archive = Path(os.environ['H3_FASTVIDEO_SOURCE_ARCHIVE'])
        before = e.sha256(archive)
        with tarfile.open(archive) as handle:
            handle.extractall(folder, filter='data')
        source = next(Path(folder).iterdir())
        return archive, before, source

    def test_pinned_constants_and_install_are_auditable(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, before, source = self.extract(folder)
            packing = (source / 'fastvideo/pipelines/basic/minimax_h3/packing.py').read_text()
            tree = ast.parse(packing)
            values = {node.targets[0].id: node.value.value for node in ast.walk(tree)
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and isinstance(node.value, ast.Constant)}
            self.assertEqual(values['MINIMAX_H3_FPS'], e.FPS)
            self.assertEqual(values['MINIMAX_H3_AUDIO_LATENTS_PER_SECOND'], e.AUDIO_LATENTS_PER_SECOND)
            self.assertEqual(values['MINIMAX_H3_AUDIO_CHANNELS'], e.AUDIO_CHANNELS)
            self.assertEqual(values['MINIMAX_H3_FRAMES_PER_CHUNK'], e.FRAMES_PER_CHUNK)
            self.assertEqual(values['MINIMAX_H3_LATENTS_PER_CHUNK'], e.LATENTS_PER_CHUNK)
            audio_config = (source / 'fastvideo/configs/models/vaes/minimax_h3_audio.py').read_text()
            arch = ast.parse(audio_config)
            defaults = {node.target.id: node.value.value for node in ast.walk(arch)
                        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                        and isinstance(node.value, ast.Constant)}
            self.assertEqual(defaults['sampling_rate'], e.AUDIO_SAMPLE_RATE)
            self.assertEqual(defaults['latent_channels'], e.AUDIO_LATENT_CHANNELS)
            rates = next(node.value for node in ast.walk(arch) if isinstance(node, ast.AnnAssign)
                         and getattr(node.target, 'id', None) == 'encoder_rates')
            self.assertEqual(math_prod(ast.literal_eval(rates)), e.HOP_LENGTH)
            latent_source = (source / 'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py').read_text()
            keys = {node.targets[0].id: node.value.value for node in ast.walk(ast.parse(latent_source))
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)}
            self.assertEqual(keys['MINIMAX_H3_LAYOUT_KEY'], e.LAYOUT_KEY)

            from research.h3_personal.fast_ref2va import install as smd_install
            smd_install(source)
            report = e.install(source)
            for name in ('minimax_h3_latent_preparation.py', 'minimax_h3_denoising.py'):
                path = source / 'fastvideo/pipelines/basic/minimax_h3/stages' / name
                ast.parse(path.read_text())
            denoising = (source / 'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py').read_text()
            self.assertEqual(denoising.count('H3_SMD_LOCKED_AUDIO'), 2)
            self.assertIn('hold_locked_audio_rows(self, batch, audio_velocity)', denoising)
            self.assertIn('audio_value = locked_audio_timestep(audio_value)', denoising)
            latent = (source / 'fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py').read_text()
            self.assertIn('batch = lock_target_audio_rows(self, batch, fastvideo_args)', latent)
            # The lock must run only after the packed layout is attached to the batch.
            self.assertLess(latent.index('batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout'),
                            latent.index('batch = lock_target_audio_rows('))
            self.assertNotEqual(report['latent_preparation_original_sha256'],
                                report['latent_preparation_patched_sha256'])
            with self.assertRaises(ValueError):
                e.install(source)
        self.assertEqual(e.sha256(archive), before)


def math_prod(values):
    result = 1
    for value in values:
        result *= value
    return result


if __name__ == '__main__':
    unittest.main()
