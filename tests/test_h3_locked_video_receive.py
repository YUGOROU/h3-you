"""Offline CPU contract tests: no Hub, private files, GPU or media fixtures."""
import copy
import hashlib
from types import SimpleNamespace
import unittest
from pathlib import Path

from research.bootstrap_h3_fast_ref2va import locked_speech_plan
from research.receive_h3_locked_video import validate_profile, validate_metadata, verify_remote_bytes, concat_manifest

class ConcatDurationTests(unittest.TestCase):
    def test_absolute_pts_container_durations_are_not_used(self):
        text=concat_manifest([Path(f'segment-{i:03d}.mp4') for i in range(3)])
        self.assertEqual(text.count('duration 5.166666666667\n'),3)
        self.assertEqual(text.count('file '),3)

from research.h3_personal.external_audio import SCHEMA


def fixture(seconds=11.64):
    spec = dict(layout='portrait', runtime_policy='resident-visual-cached',
                locked_speech=dict(repo='fixture/speech', revision='a'*40,
                                   path='generated/reply.wav', sha256='b'*64,
                                   sample_rate=48000, samples=round(seconds*48000)))
    plan = locked_speech_plan(spec)
    receipts = []
    for window in plan['windows']:
        for rank in (0, 1):
            receipts.append(dict(schema=SCHEMA, window_index=window['index'], worker_rank=rank,
                wav_sha256='b'*64, latent_start=window['latent_start'], latent_count=window['latent_count'],
                expected_rows=window['expected_rows'], encoded_rows=window['expected_rows'],
                audio_timestep_policy='fixed_clean_1.0', steps_held=4,
                audio_scheduler_step_skipped=True, audio_velocity_discarded=True,
                rows_unchanged=True, truncation=False, time_stretch=False,
                supplied_speech_not_generated=True, generated_audio_decode_skipped=True))
    n = plan['windows_required']
    profile = dict(external_audio_conditioning=True, originals_modified=False,
                   visual_only=True, generated_audio_discarded=True,
                   independent_ref2va_clips=True, shared_head_trim=False,
                   expected_forwards_per_segment=4, unique_frames=124*n,
                   locked_audio_plan=plan, locked_audio_receipts=receipts,
                   loaded_coverage=[dict(worker_rank=r, identity_modules=208, total_modules=312,
                                        unmatched=0, rank='mixed128-144', partition='transformer_ref')
                                    for r in (0, 1)], segments=[{} for _ in range(n)])
    return spec, profile


class LockedVideoTests(unittest.TestCase):
    def test_planned_one_to_four(self):
        for seconds, count in ((1, 1), (7, 2), (11.64, 3), (17, 4)):
            spec, profile = fixture(seconds)
            self.assertEqual(validate_profile(profile, spec), count)

    def test_profile_refusals(self):
        spec, original = fixture()
        for key, value in [('external_audio_conditioning', False), ('originals_modified', True),
                           ('shared_head_trim', True), ('unique_frames', 369)]:
            profile = copy.deepcopy(original)
            profile[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile(profile, spec)

    def test_receipt_refusals(self):
        spec, original = fixture()
        for key, value in [('steps_held', 3), ('rows_unchanged', False),
                           ('wav_sha256', 'c'*64), ('encoded_rows', 0),
                           ('time_stretch', True), ('audio_velocity_discarded', False)]:
            profile = copy.deepcopy(original)
            profile['locked_audio_receipts'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile(profile, spec)
        original['locked_audio_receipts'][1] = original['locked_audio_receipts'][0]
        with self.assertRaises(ValueError):
            validate_profile(original, spec)

    def test_original64_coverage(self):
        spec, profile = fixture()
        profile['loaded_coverage'][1]['identity_modules'] = 207
        with self.assertRaises(ValueError):
            validate_profile(profile, spec)

    def test_metadata(self):
        video = dict(codec_type='video', width=252, height=448, avg_frame_rate='24/1',
                     r_frame_rate='24/1', nb_read_frames='124')
        validate_metadata({'streams': [video]}, 124)
        for key, value in [('width', 256), ('nb_read_frames', '123'), ('avg_frame_rate', '30/1')]:
            bad = dict(video, **{key: value})
            with self.assertRaises(ValueError):
                validate_metadata({'streams': [bad]}, 124)
        with self.assertRaises(ValueError):
            validate_metadata({'streams': [video, {'codec_type': 'audio'}]}, 124)

    def test_remote_git_and_lfs(self):
        data = b'fixture'
        git = hashlib.sha1(b'blob 7\0'+data).hexdigest()
        sha = hashlib.sha256(data).hexdigest()
        for row in (SimpleNamespace(size=7, lfs=None, blob_id=git),
                    SimpleNamespace(size=7, lfs=SimpleNamespace(sha256=sha))):
            self.assertEqual(verify_remote_bytes(data, row), sha)
            with self.assertRaises(ValueError):
                verify_remote_bytes(b'changed', row)


if __name__ == '__main__':
    unittest.main()
