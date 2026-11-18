"""No original recordings, cloud access, credentials or paid calls."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / 'research'
sys.path.insert(0, str(RESEARCH))
import hf_prepare_identity_media as worker
import launch_hf_prepare_identity_media as launcher


class Sink:
    def __init__(self, restored=None):
        self.calls = []
        self.restored = restored

    def upload(self, records, dirty):
        self.calls.append((len(records), len(dirty)))

    def restore(self, rows):
        return self.restored


class FakeAPI:
    def __init__(self):
        self.uploads = []
        self.runs = []
        self.jobs = []
        self.private = True
        self.rate = .03
        self.created = []

    def whoami(self):
        return {'name': 'tester', 'isPro': True, 'auth': {'accessToken': {'role': 'write'}}}

    def list_jobs(self, **kwargs):
        return self.jobs

    def list_jobs_hardware(self):
        return [NS(name='cpu-upgrade', accelerator=None, unit_cost_usd=self.rate, unit_label='hour')]

    def create_repo(self, repo, **kwargs):
        self.created.append((repo, kwargs))

    def repo_info(self, *args, **kwargs):
        return NS(private=self.private, sha='a' * 40)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        return NS(oid='a' * 40)

    def run_job(self, **kwargs):
        self.runs.append(kwargs)
        return NS(id='fake-job')


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'originals').mkdir()
        self.freeze, self.preprocess = worker.helpers(RESEARCH / 'h3_personal')
        self.sources = []
        for i in range(6):
            path = self.root / 'originals' / ('source%d.mov' % i)
            path.write_bytes(('synthetic-original-%d' % i).encode())
            self.sources.append(path)
        selected = []
        for i in range(715):
            source = self.sources[i % 6]
            selected.append({'id': 'clip_%04d' % i, 'source': str(source),
                'source_sha256': worker.sha256(source), 'session_id': 'session%d' % (i % 6),
                'start_seconds': i * 2.0, 'end_seconds': i * 2.0 + 124 / 24,
                'pool': 'identity' if i < 590 else 'motion', 'human_review': 'approved_sampled_images'})
        self.selection = self.freeze.build(selected)
        self.frozen = self.root / 'freeze.json'
        worker.dump(self.frozen, self.selection)
        self.target = self.root / 'stage'
        self.output = self.root / 'separate' / 'raw'
        self.ledger = self.root / 'budget.json'
        worker.dump(self.ledger, {'spent_usd': 29.83, 'reservations': {}})
        self.image = 'python@sha256:' + 'b' * 64

    def stage(self):
        launcher.stage(self.frozen, self.target, 'tester/private-raw')
        request = json.loads((self.target / 'request.json').read_text())
        request['source_map'] = json.loads((self.target / 'local_sources.json').read_text())
        request['input_revision'] = 'a' * 40
        return request

    @staticmethod
    def fake_encode(row, source, output, preprocess, helper_dir):
        video = 'clips/' + row['id'] + '.mp4'
        audio = 'audio/' + row['id'] + '.wav'
        (output / video).write_bytes(b'fake-video')
        (output / audio).write_bytes(b'fake-audio')
        return dict(row, video=video, audio=audio, video_sha256=worker.sha256(output / video),
                    audio_sha256=worker.sha256(output / audio), preparation_seconds=.001,
                    input_provenance={k: row[k] for k in ('source_sha256', 'start_seconds', 'end_seconds')})

    def test_stage_small_allowlist_no_original_copies(self):
        before = [p.read_bytes() for p in self.sources]
        self.stage()
        self.assertEqual({p.name for p in self.target.iterdir()}, {'frozen_selection.json',
            'hf_prepare_identity_media.py', 'freeze_dataset.py', 'preprocess.py',
            'request.json', 'local_sources.json', 'plan.json'})
        self.assertEqual(before, [p.read_bytes() for p in self.sources])
        self.assertFalse(any(p.is_symlink() for p in self.target.iterdir()))
        self.assertFalse(any(p.suffix == '.mov' for p in self.target.iterdir()))

    def test_semantic_and_geometry_and_verified_gates(self):
        for change in ({'source_bytes_verified': False}, {'dataset_sha256': '0' * 64},
                       {'sample_count': 714}, {'geometry': {}}, {'clip_seconds_with_tail_overlap': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.check_selection(dict(self.selection, **change), self.freeze)
        altered = copy.deepcopy(self.selection)
        altered['samples'][0]['start_seconds'] += 1
        with self.assertRaises(ValueError):
            worker.check_selection(altered, self.freeze)
        altered = copy.deepcopy(self.selection)
        altered['samples'][0]['quality_pool'] = 'motion'
        with self.assertRaises(ValueError):
            worker.check_selection(altered, self.freeze)

    def test_source_change_fails_before_encoding(self):
        request = self.stage()
        self.sources[0].write_bytes(b'changed')
        with self.assertRaises(ValueError):
            worker.prepare(request, self.target, self.output, Sink(), encoder=lambda *args: self.fail('encoded'))
        self.assertFalse(self.output.exists())

    def test_unknown_source_and_symlink_rejected(self):
        request = self.stage()
        extra = dict(request['source_map'], extra=str(self.sources[0]))
        with self.assertRaises(ValueError):
            worker.verify_sources(self.selection['samples'], extra)
        link = self.root / 'link.mov'
        link.symlink_to(self.sources[0])
        source_map = dict(request['source_map'])
        source_map[worker.sha256(self.sources[0])] = str(link)
        with self.assertRaises(ValueError):
            worker.verify_sources(self.selection['samples'], source_map)

    def test_complete_ordered_715_snapshots_and_hashes(self):
        request = self.stage()
        sink = Sink()
        summary = worker.prepare(request, self.target, self.output, sink, encoder=self.fake_encode)
        self.assertTrue(summary['complete_raw_preprocessing'])
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual([r['id'] for r in manifest['samples']], [r['id'] for r in self.selection['samples']])
        self.assertEqual(sink.calls[0], (0, 0))
        self.assertEqual(sink.calls[-1], (715, 5))
        self.assertLessEqual(max(d for _, d in sink.calls), 10)
        for row, frozen in zip(manifest['samples'], self.selection['samples']):
            self.assertEqual({k: row[k] for k in frozen}, frozen)
            self.assertEqual(row['audio_sha256'], worker.sha256(self.output / row['audio']))
        self.assertFalse(summary['technical_training_ready'])

    def test_exception_no_skip_or_retry_and_remote_stays_incomplete(self):
        request = self.stage()
        calls = []
        def encoder(row, *args):
            calls.append(row['id'])
            if len(calls) == 12:
                raise subprocess.TimeoutExpired('fake-ffmpeg', 180)
            return self.fake_encode(row, *args)
        sink = Sink()
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.prepare(request, self.target, self.output, sink, encoder=encoder)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertFalse(summary['complete_raw_preprocessing'])
        self.assertEqual(len(calls), 12)
        self.assertEqual(sink.calls[-1], (10, 10))

    def test_explicit_resume_revalidates_and_no_reencode(self):
        request = self.stage()
        worker.prepare(request, self.target, self.output, Sink(), encoder=self.fake_encode)
        manifest = json.loads((self.output / 'manifest.json').read_text())
        with patch.object(worker, 'validate_files') as validation:
            result = worker.prepare(request, self.target, self.output, Sink(manifest), resume=True,
                                    encoder=lambda *args: self.fail('unexpected reencode'))
        self.assertEqual(validation.call_count, 715)
        self.assertTrue(result['complete_raw_preprocessing'])
        (self.output / manifest['samples'][0]['video']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            worker.prepare(request, self.target, self.output, Sink(manifest), resume=True)

    def test_pinned_helpers_reject_changed_code(self):
        self.stage()
        (self.target / 'preprocess.py').write_text('raise Exception("must not execute")')
        with self.assertRaises(ValueError):
            worker.helpers(self.target)

    def test_fake_submit_bounded_private_exact_whitelist_direct_sources(self):
        self.stage()
        api = FakeAPI()
        with patch.dict(os.environ, {'HF_TOKEN': 'fake-secret'}):
            report = launcher.submit(self.target, self.ledger, self.image, api, lambda **kw: kw)
        self.assertEqual(report['reserved_upper_bound_usd'], .09)
        self.assertEqual(len(api.uploads), 11)
        self.assertEqual(len(api.runs), 1)
        self.assertEqual(api.runs[0]['timeout'], '3h')
        self.assertEqual(api.runs[0]['flavor'], 'cpu-upgrade')
        self.assertEqual({u['path_or_fileobj'] for u in api.uploads[:6]}, {str(p) for p in self.sources})
        self.assertNotIn('fake-secret', (self.target / 'submission_guard.json').read_text())
        self.assertTrue(api.created[0][1]['private'])
        with self.assertRaises(ValueError):
            launcher.submit(self.target, self.ledger, self.image, api, lambda **kw: kw)
        self.assertEqual(len(api.runs), 1)

    def test_active_jobs_price_permission_budget_gates(self):
        self.stage()
        for kind in ('active', 'rate', 'permission', 'budget'):
            api = FakeAPI()
            ledger = self.root / (kind + '.json')
            worker.dump(ledger, {'spent_usd': 149.99 if kind == 'budget' else 29.83, 'reservations': {}})
            if kind == 'active':
                api.jobs = [NS(status=NS(stage='RUNNING'), labels={})]
            if kind == 'rate':
                api.rate = .051
            if kind == 'permission':
                api.whoami = lambda: {'name': 'tester', 'isPro': False}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                launcher.submit(self.target, ledger, self.image, api, lambda **kw: kw)
            self.assertFalse(api.created)
            self.assertFalse(api.runs)

    def test_ambiguous_submission_error_reservation_survives(self):
        self.stage()
        api = FakeAPI()
        def fail(**kwargs):
            raise RuntimeError('ambiguous failure')
        api.run_job = fail
        with patch.dict(os.environ, {'HF_TOKEN': 'fake'}), self.assertRaises(RuntimeError):
            launcher.submit(self.target, self.ledger, self.image, api, lambda **kw: kw)
        ledger = json.loads(self.ledger.read_text())
        self.assertEqual(len(ledger['reservations']), 1)
        with self.assertRaises(ValueError):
            launcher.submit(self.target, self.ledger, self.image, api, lambda **kw: kw)

    def test_nonempty_output_requires_explicit_resume(self):
        request = self.stage()
        self.output.mkdir(parents=True)
        (self.output / 'unrelated.txt').write_text('keep')
        with self.assertRaises(ValueError):
            worker.prepare(request, self.target, self.output, Sink(), encoder=self.fake_encode)
        self.assertEqual((self.output / 'unrelated.txt').read_text(), 'keep')

    def test_private_snapshot_upload_exact_paths_no_folder_walk(self):
        api = FakeAPI()
        api.commits = []
        api.create_commit = lambda **kw: api.commits.append(kw)
        sink = worker.PrivateSnapshots(api, 'tester/private', self.root, 'c' * 64, 'a' * 40)
        fake_hub = NS(CommitOperationAdd=lambda **kw: NS(**kw))
        with patch.dict(sys.modules, {'huggingface_hub': fake_hub}):
            sink.upload([], [{'video': 'clips/clip_0000.mp4', 'audio': 'audio/clip_0000.wav'}])
            api.private = False
            with self.assertRaises(ValueError):
                sink.upload([], [])
        self.assertEqual(len(api.commits), 1)
        names = {op.path_in_repo for op in api.commits[0]['operations']}
        prefix = 'raw/' + 'c' * 64 + '/'
        self.assertEqual(names, {prefix + n for n in ('manifest.json', 'summary.json',
                                                     'clips/clip_0000.mp4', 'audio/clip_0000.wav')})

    def test_time_interval_snapshot_even_below_ten_clips(self):
        request = self.stage()
        sink = Sink()
        clock = [0.0]
        def monotonic():
            clock[0] += 301
            return clock[0]
        with patch.object(worker.time, 'monotonic', monotonic):
            worker.prepare(request, self.target, self.output, sink, encoder=self.fake_encode)
        self.assertTrue(all(dirty <= 1 for _, dirty in sink.calls))
        self.assertEqual(sink.calls[-1][0], 715)

    def test_process_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'], .05)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg not installed')
    def test_synthetic_ffmpeg_exact_grid_fitpad_and_validation(self):
        source = self.root / 'fixture.mov'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'testsrc2=size=96x64:rate=30:duration=6', '-f', 'lavfi', '-i',
            'sine=frequency=440:sample_rate=48000:duration=6', '-c:v', 'libx264', '-pix_fmt',
            'yuv420p', '-c:a', 'pcm_s16le', str(source)], check=True, capture_output=True, timeout=30)
        self.output.mkdir(parents=True)
        (self.output / 'clips').mkdir()
        (self.output / 'audio').mkdir()
        row = dict(self.selection['samples'][0], source=str(source), source_sha256=worker.sha256(source),
                   start_seconds=0, end_seconds=124 / 24)
        before = worker.sha256(source)
        record = worker.encode(row, source, self.output, self.preprocess, RESEARCH / 'h3_personal')
        self.preprocess.validate(self.output / record['video'], self.output / record['audio'], 768, 1344)
        self.assertEqual(worker.sha256(source), before)
        self.assertEqual(self.preprocess.AUDIO_SAMPLES, 165600)
        self.assertGreater(record['preparation_seconds'], 0)
        # Black fit-pad at top-left; content is not cropped to fill the portrait canvas.
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(self.output / record['video']),
            '-vf', 'crop=2:2:0:0', '-frames:v', '1', '-pix_fmt', 'gray', '-f', 'rawvideo', '-'], timeout=30)
        self.assertTrue(all(pixel < 5 for pixel in raw))
        (self.output / record['audio']).write_bytes(b'not WAV')
        with self.assertRaises(subprocess.CalledProcessError):
            worker.validate_files(self.output / record['video'], self.output / record['audio'], RESEARCH / 'h3_personal')


if __name__ == '__main__':
    unittest.main()
