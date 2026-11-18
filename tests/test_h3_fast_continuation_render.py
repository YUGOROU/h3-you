"""CPU protocol tests only: no real tensors, GPU, FFmpeg or media playback."""
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from research.h3_personal import fast_continuation_render as renderer
from research.h3_personal.continuation_plan import audio_boundary, plan as grid, video_t
from research.h3_personal.continuation_candidate import make_options
from research.h3_personal.continuation_efficiency import make_options as efficiency_options


def wav_at(path, samples, rate=32000, width=2, channels=1):
    with wave.open(str(path), 'wb') as handle:
        handle.setparams((channels, width, rate, 0, 'NONE', 'not compressed'))
        handle.writeframes(b'\x01' * samples * width * channels)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Frames:
    def __init__(self, shape, dtype='uint8'):
        self.shape, self.dtype = shape, dtype

    def copy(self):
        return Frames(self.shape, self.dtype)

    def __iter__(self):
        for _ in range(self.shape[0]):
            yield Frames(self.shape[1:], self.dtype)

    def __getitem__(self, key):
        # Exercise the actual fixed two-pixel side crop, not an arbitrary reshape.
        assert key == (slice(None), slice(None), slice(2, -2), slice(None))
        return Frames((self.shape[0], 448, 252, 3), self.dtype)

    def tobytes(self):
        return b'fake-rgb'


NP = SimpleNamespace(uint8='uint8', asarray=lambda f: f,
                     stack=lambda frames: Frames((len(frames), *frames[0].shape), frames[0].dtype))


class Runtime:
    """Fake worker writes the real scalar receipt protocol at generation time."""
    def __init__(self, root):
        self.coverage = root / 'coverage'
        self.coverage.mkdir()
        self.continuation_file = root / 'mode.json'
        renderer._json(self.continuation_file, {'mode': 'independent', 'preserve': 'test'})
        self._requests = 1  # warmed owner
        self._last_seed = 314159
        self.proof = {'reference_sha256': 'reference-proof'}
        self.derived = {'adapter': 'unchanged'}
        self.seen, self.events = [], []
        self.mutate = lambda stage, decode: None
        self.dtype, self.wrong_frames = 'uint8', False
        self.bad_check = None

    def _generate(self, *, continuation=False):
        assert continuation is True
        m = json.loads(self.continuation_file.read_text())
        assert m['mode'] == 'latent-tail-v1'
        assert renderer._sha(m['wav']) == m['sha256']
        assert Path(m['wav']).stat().st_mode & 0o222 == 0
        self.seen.append(m)
        index = m['index']
        for rank in (0, 1):
            for stem in ('h3-continuation', 'h3-continuation-decode'):
                assert not (self.coverage / f'{stem}-rank-{rank}-window-{index}.json').exists()
        self.events.append(('generate', index))
        self._requests += 1
        w = grid(m['total_frames'], 124, 22)[index]
        offset = 0 if index == 0 else w.start+w.overlap-5
        count = (w.end if m['final'] else w.end-5)-offset
        per_frame = 448
        cumulative = video_t(w.end)*per_frame
        previous = 0 if index == 0 else video_t(w.start+w.overlap)*per_frame
        for rank in (0, 1):
            stage = dict(schema='h3-smd-continuation-stage-v1', session_id=m['session_id'],
                window_index=index, worker_rank=rank, mode=m['mode'], wav_sha256=m['sha256'],
                total_frames=m['total_frames'], window_frames=124, overlap_frames=22,
                seed=314159, expected_windows=m['expected_windows'], final=m['final'],
                audio_start=w.audio_start, audio_end=w.audio_end,
                guide_video_t=w.guide_video_t, guide_audio_t=w.guide_audio_t,
                target_audio_rows=2*(w.audio_end-w.audio_start), guide_audio_rows=2*w.guide_audio_t,
                forwards=4, steps_held=4, sampler='SMD4', audio_timestep_policy='fixed_clean_1.0',
                audio_velocity_discarded=True, audio_scheduler_step_skipped=True,
                rows_unchanged=True, committed=True, truncation=False, gpu_validated=False,
                source_samples=m['source_samples'], source_sample_rate=m['source_sample_rate'],
                full_source_encodes=1, global_audio_t=audio_boundary(m['total_frames']),
                padded_samples=audio_boundary(m['total_frames'])*800-
                               m['source_samples']*32000//m['source_sample_rate'],
                suffix_video_rows=cumulative-previous, video_row_offset=previous,
                cumulative_video_rows=cumulative,
                suffix_audio_t=w.audio_end-audio_boundary(w.start+w.overlap),
                audio_time_offset=audio_boundary(w.start+w.overlap), cumulative_audio_t=w.audio_end)
            if 'candidate' in m:
                candidate = m['candidate']
                window_context = candidate['audio_context'] == 'window-context-canonical-v1'
                stage.update(candidate_audio_context=candidate['audio_context'],
                    candidate_progress_receipts=candidate['progress_receipts'],
                    source_window_encodes=m['expected_windows'] if window_context else 0,
                    full_source_encodes=0 if window_context else 1)
                if window_context:
                    stage['audio_overlap_policy'] = 'prior-canonical-wins'
            decode = dict(session_id=m['session_id'], window_index=index, worker_rank=rank,
                total_frames=m['total_frames'], frame_offset=offset, frame_count=count,
                latent_received=video_t(w.end), retained_latents=0 if m['final'] else 2,
                retained_overlap_frames=0 if m['final'] else 5,
                max_clip_latents=7, two_latent_lookahead=2,
                bounded_buffers=True, final=m['final'], causal=False,
                execution_policy='two-rank-alternating-native-clips', parallel_preserved=True,
                participation=True, output_worker=rank == 0)
            if 'efficiency' in m:
                efficiency = m['efficiency']
                stage.update(efficiency_audio_validation=efficiency['audio_validation'],
                             efficiency_decode_schedule=efficiency['decode_schedule'])
                if efficiency['decode_schedule'] == 'two-clip-waves-v1':
                    decode.update(execution_policy='two-rank-two-clip-waves',
                                  decode_schedule='two-clip-waves-v1',
                                  max_buffer_latents=12, max_queued_clips=1)
            self.mutate(stage, decode)
            renderer._json(self.coverage / f'h3-continuation-rank-{rank}-window-{index}.json', stage)
            renderer._json(self.coverage / f'h3-continuation-decode-rank-{rank}-window-{index}.json', decode)
        return SimpleNamespace(frames=Frames((count+int(self.wrong_frames), 448, 256, 3), self.dtype)), (1., 2.)

    def _loaded(self):
        self.events.append(('loaded', len(self.seen)-1))
        if self.bad_check == 'loaded':
            raise RuntimeError('both-rank coverage rejected')
        return {'loaded_coverage': ['rank0', 'rank1'], 'static_merge_receipts': ['merged', 'merged']}

    def _read(self, stem):
        if self.bad_check == stem:
            return [{}, {}]
        return {'cache': [{'hits': self._requests-1}]*2,
                'packed': [{'warm_counter_removed': True, 'packed_route_executed': True}]*2,
                'residency': [{'policy': renderer.POLICY, 'audio_decode_skipped': True}]*2}[stem]


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_import_and_planning_are_stdlib_only(self):
        # A clean interpreter, so accidental eager imports cannot hide in sys.modules.
        script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'torch', 'PIL', 'fastvideo'):
        raise AssertionError('forbidden planning import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from research.h3_personal.fast_continuation_render import plan
assert plan(sys.argv[1], sys.argv[2])['total_frames'] == 124
'''
        wav = self.root / 'speech.wav'
        digest = wav_at(wav, 32000)
        subprocess.run([sys.executable, '-c', script, str(wav), digest], check=True,
                       capture_output=True, timeout=15)

    def test_minimal_grid_and_exact_metadata(self):
        for n, samples in ((1, 1), (1, 165333), (2, 165334), (3, 301334), (4, 437334)):
            with self.subTest(n=n, samples=samples):
                wav = self.root / f'{samples}.wav'
                digest = wav_at(wav, samples)
                before = wav.read_bytes()
                p = renderer.plan(wav, digest)
                self.assertEqual((p['expected_windows'], p['total_frames']), (n, 124+(n-1)*102))
                self.assertEqual((p['source_samples'], p['source_sample_rate']), (samples, 32000))
                self.assertEqual(p['seed'], 314159)
                self.assertEqual(len(p['windows']), n)
                self.assertNotIn('session_id', p)
                self.assertEqual(wav.read_bytes(), before)

    def test_both_global_clocks_must_cover_source(self):
        # All bounded 124+102k grids round audio UP: video is the tighter
        # clock. Audio capacity alone would incorrectly admit 573600 samples.
        wav = self.root / 'capacity.wav'
        digest = wav_at(wav, 573334)
        self.assertLessEqual(573334, audio_boundary(430)*800)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            renderer.plan(wav, digest)
        digest = wav_at(wav, 573333)
        self.assertEqual(renderer.plan(wav, digest)['total_frames'], 430)
        digest = wav_at(wav, 573601)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            renderer.plan(wav, digest)

    def test_arbitrary_rate_pcm_and_invalid_input(self):
        wav = self.root / 'rate.wav'
        digest = wav_at(wav, 220501, rate=44100, width=3, channels=2)
        p = renderer.plan(wav, digest)
        self.assertEqual((p['source_samples'], p['source_sample_rate'], p['source_sample_width']),
                         (220501, 44100, 3))
        with self.assertRaisesRegex(RuntimeError, 'SHA'):
            renderer.plan(wav, '0'*64)
        digest = wav_at(wav, 0)
        with self.assertRaisesRegex(RuntimeError, 'nonempty'):
            renderer.plan(wav, digest)
        digest = wav_at(wav, 10)
        wav.write_bytes(wav.read_bytes()[:-2])
        with self.assertRaisesRegex(RuntimeError, 'truncated'):
            renderer.plan(wav, renderer._sha(wav))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = Runtime(self.root)
        self.wav = self.root / 'original.wav'
        self.digest = wav_at(self.wav, 350000)
        self.original = self.wav.read_bytes()
        self.timings, self.commands, self.callbacks = [], [], []
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(sys.modules, {'numpy': NP}))
        # Import the production encoder module without invoking it.
        encoder = importlib.import_module('research.run_h3_realtime_first')
        stack.enter_context(patch.object(encoder, 'encode_rgb_segment', self.encode))
        stack.enter_context(patch.object(renderer, '_metadata', self.metadata))
        stack.enter_context(patch.object(renderer, '_command', self.command))
        stack.enter_context(patch.object(renderer, '_pcm', return_value=b'original-pcm'))

    def encode(self, raw, path, timing, *, width, height):
        self.assertEqual((raw, width, height), (b'fake-rgb', 252, 448))
        self.runtime.events.append(('encode', timing.segment_id))
        self.timings.append(timing)
        path.write_bytes(b'encoded')

    def metadata(self, path, frames, *, audio=False):
        self.runtime.events.append(('metadata', path.name))
        return {'frames': frames, 'audio': audio}

    def command(self, command):
        self.commands.append(command)
        Path(command[-1]).write_bytes(b'joined-or-muxed')
        return b''

    def callback(self, segment):
        self.assertTrue(Path(segment['path']).is_file())
        self.assertEqual(self.runtime.events[-1], ('metadata', Path(segment['path']).name))
        self.assertTrue(segment['video_only'])
        self.assertFalse(segment['remote_acceptance'])
        self.callbacks.append(segment)

    def run_render(self, name='turn', callback=None):
        return renderer.render(self.runtime, self.wav, self.digest, self.root / name,
                               on_segment=self.callback if callback is None else callback)

    def assert_restored(self):
        self.assertEqual(json.loads(self.runtime.continuation_file.read_text()),
                         {'mode': 'independent', 'preserve': 'test'})
        self.assertEqual(self.wav.read_bytes(), self.original)

    def test_all_grids_suffix_counts_pts_and_mux(self):
        for n, samples, counts in ((1, 32000, [124]), (2, 200000, [119, 107]),
                                  (3, 350000, [119, 102, 107]),
                                  (4, 500000, [119, 102, 102, 107])):
            with self.subTest(n=n):
                self.digest = wav_at(self.wav, samples)
                self.original = self.wav.read_bytes()
                self.timings.clear()
                self.callbacks.clear()
                self.runtime.seen.clear()
                result = self.run_render(f'turn-{n}')
                profile = json.loads(result.profile.read_text())
                self.assertEqual(profile['unique_frames'], sum(counts))
                self.assertEqual([t.newly_emitted_frames for t in self.timings], counts)
                offset = 0
                for i, (t, count, m, cb) in enumerate(zip(self.timings, counts, self.runtime.seen, self.callbacks)):
                    self.assertEqual((t.segment_id, t.head_trim_frames, t.pts_start, t.pts_end),
                                     (i, 0, offset, offset+count))
                    self.assertEqual(cb['frame_offset'], offset)
                    self.assertEqual((m['index'], m['final'], m['seed']), (i, i == n-1, 314159))
                    self.assertNotEqual(m['wav'], str(self.wav))
                    self.assertEqual(m['session_id'], profile['session_id'])
                    offset += count
                self.assertEqual(len(self.commands), 2*n)
                mux = self.commands[-1]
                self.assertEqual(mux.count('-c:a'), 1)
                self.assertIn('alac', mux)
                self.assertNotIn('-shortest', mux)
                self.assertNotIn('-t', mux)
                self.assertEqual(profile['review_metadata'], {'frames': sum(counts), 'audio': True})
                self.assertFalse(profile['causal'])
                self.assertFalse(profile['remote_acceptance'])
                self.assertEqual(profile['two_latent_vae_lookahead'], 2)
                self.assert_restored()

    def test_candidate_snapshot_and_independent_policies(self):
        for window, progress in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(window=window, progress=progress):
                options = make_options(window_audio_context=window,
                                       defer_progress_receipts=progress)
                if options is None:
                    self.runtime.__dict__.pop('_candidate', None)
                else:
                    self.runtime._candidate = options
                self.runtime.seen.clear()
                result = self.run_render(f'candidate-{window}-{progress}')
                profile = json.loads(result.profile.read_text())
                speech_plan = json.loads((result.profile.parent / 'speech-plan.json').read_text())
                self.assertEqual(profile['continuation_plan'], speech_plan)
                self.assertEqual(len(self.runtime.seen), 3)
                for manifest in [speech_plan, *self.runtime.seen]:
                    if options is None:
                        self.assertNotIn('candidate', manifest)
                    else:
                        self.assertEqual(manifest['candidate'], options)
                for index in range(3):
                    for rank in (0, 1):
                        stage = json.loads((self.runtime.coverage /
                            f'h3-continuation-rank-{rank}-window-{index}.json').read_text())
                        self.assertEqual(stage['full_source_encodes'], 0 if window else 1)
                        if options is None:
                            for key in ('candidate_audio_context', 'candidate_progress_receipts',
                                        'source_window_encodes', 'audio_overlap_policy'):
                                self.assertNotIn(key, stage)
                        else:
                            self.assertEqual(stage['candidate_audio_context'], options['audio_context'])
                            self.assertEqual(stage['candidate_progress_receipts'], options['progress_receipts'])
                            self.assertEqual(stage['source_window_encodes'], 3 if window else 0)
                            if window:
                                self.assertEqual(stage['audio_overlap_policy'], 'prior-canonical-wins')
                            else:
                                self.assertNotIn('audio_overlap_policy', stage)
                self.assertTrue(profile['speech_pcm_exact'])
                self.assertEqual((result.profile.parent / 'speech.wav').read_bytes(), self.original)
                self.assert_restored()

    def test_efficiency_snapshot_and_both_rank_evidence(self):
        for waves in (False, True):
            with self.subTest(waves=waves):
                self.runtime._candidate = make_options(window_audio_context=True,
                                                       defer_progress_receipts=True)
                options = efficiency_options(device_audio_validation=True,
                                             parallel_decode_waves=waves)
                self.runtime._efficiency = dict(options)
                self.runtime.seen.clear()

                def mutate_runtime(stage, decode):
                    # Runtime configuration changes cannot rewrite the admitted policy.
                    self.runtime._efficiency['decode_schedule'] = 'unknown'
                    self.runtime._efficiency['audio_validation'] = 'unknown'

                self.runtime.mutate = mutate_runtime
                result = self.run_render(f'efficiency-{waves}')
                profile = json.loads(result.profile.read_text())
                speech_plan = json.loads((result.profile.parent / 'speech-plan.json').read_text())
                self.assertEqual(profile['continuation_plan'], speech_plan)
                self.assertEqual(len(self.runtime.seen), 3)
                for manifest in [speech_plan, *self.runtime.seen]:
                    self.assertEqual(manifest['efficiency'], options)
                for index in range(3):
                    for rank in (0, 1):
                        stage = json.loads((self.runtime.coverage /
                            f'h3-continuation-rank-{rank}-window-{index}.json').read_text())
                        decode = json.loads((self.runtime.coverage /
                            f'h3-continuation-decode-rank-{rank}-window-{index}.json').read_text())
                        self.assertEqual(stage['efficiency_audio_validation'], options['audio_validation'])
                        self.assertEqual(stage['efficiency_decode_schedule'], options['decode_schedule'])
                        self.assertTrue(decode['participation'])
                        self.assertEqual(decode['execution_policy'], 'two-rank-two-clip-waves'
                                         if waves else 'two-rank-alternating-native-clips')
                        if waves:
                            self.assertEqual(decode['decode_schedule'], 'two-clip-waves-v1')
                            self.assertEqual(decode['max_buffer_latents'], 12)
                            self.assertEqual(decode['max_queued_clips'], 1)
                        else:
                            for key in ('decode_schedule', 'max_buffer_latents', 'max_queued_clips'):
                                self.assertNotIn(key, decode)
                self.assertTrue(profile['speech_pcm_exact'])
                self.assert_restored()

    def test_efficiency_malformed_per_rank_evidence_rejected_before_encode(self):
        self.runtime._candidate = make_options(window_audio_context=True,
                                               defer_progress_receipts=True)
        for waves in (False, True):
            self.runtime._efficiency = efficiency_options(device_audio_validation=True,
                                                          parallel_decode_waves=waves)
            cases = [('stage', 'efficiency_audio_validation', False),
                     ('stage', 'efficiency_decode_schedule', 'unknown'),
                     ('decode', 'execution_policy', 'unknown'),
                     ('decode', 'participation', False),
                     ('decode', 'decode_schedule', 'unknown'),
                     ('decode', 'max_buffer_latents', 13),
                     ('decode', 'max_queued_clips', 2),
                     ('decode', 'max_queued_clips', True)]
            for rank in (0, 1):
                for kind, key, value in cases:
                    with self.subTest(waves=waves, rank=rank, key=key, value=value):
                        def mutate(stage, decode):
                            if stage['worker_rank'] == rank:
                                (stage if kind == 'stage' else decode)[key] = value
                        self.runtime.mutate = mutate
                        with self.assertRaises(RuntimeError):
                            self.run_render(f'efficiency-bad-{waves}-{rank}-{key}-{value}')
                        self.assertEqual(self.timings, [])
                        self.assertEqual(self.callbacks, [])
                        self.assert_restored()

    def test_efficiency_requires_v6_before_generation(self):
        for waves in (False, True):
            for candidate in (None, make_options(defer_progress_receipts=True)):
                with self.subTest(waves=waves, candidate=candidate):
                    self.runtime._candidate = candidate
                    self.runtime._efficiency = efficiency_options(device_audio_validation=True,
                                                                  parallel_decode_waves=waves)
                    with self.assertRaisesRegex(RuntimeError, 'efficiency requires v6'):
                        self.run_render(f'no-v6-{waves}-{candidate is None}')
                    self.assertEqual(self.runtime.seen, [])
                    self.assertEqual(self.timings, [])
                    self.assert_restored()

    def test_no_efficiency_rejects_unexpected_fields(self):
        for v6 in (False, True):
            self.runtime._candidate = make_options(window_audio_context=True,
                                                   defer_progress_receipts=True) if v6 else None
            for rank in (0, 1):
                for kind, key, value in (
                        ('stage', 'efficiency_audio_validation', 'per-step-canonical-v1'),
                        ('stage', 'efficiency_decode_schedule', 'alternating-v1'),
                        ('decode', 'decode_schedule', 'two-clip-waves-v1'),
                        ('decode', 'max_buffer_latents', 12),
                        ('decode', 'max_queued_clips', 1)):
                    with self.subTest(v6=v6, rank=rank, key=key):
                        def mutate(stage, decode):
                            if stage['worker_rank'] == rank:
                                (stage if kind == 'stage' else decode)[key] = value
                        self.runtime.mutate = mutate
                        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                            self.run_render(f'no-efficiency-{v6}-{rank}-{key}')
                        self.assertEqual(self.timings, [])
                        self.assert_restored()

    def test_candidate_tampering_rejected_per_rank_before_encode(self):
        self.runtime._candidate = make_options(window_audio_context=True,
                                               defer_progress_receipts=True)
        for rank in (0, 1):
            for key, value in (('candidate_audio_context', 'full-source-v1'),
                               ('candidate_progress_receipts', 'per-step-v1'),
                               ('source_window_encodes', 0), ('full_source_encodes', 1),
                               ('audio_overlap_policy', 'last-wins')):
                with self.subTest(rank=rank, key=key):
                    def mutate(stage, decode):
                        if stage['worker_rank'] == rank:
                            stage[key] = value
                    self.runtime.mutate = mutate
                    with self.assertRaises(RuntimeError):
                        self.run_render(f'candidate-tamper-{rank}-{key}')
                    self.assertEqual(self.timings, [])
                    self.assert_restored()

    def test_unexpected_candidate_evidence_in_legacy_rejected(self):
        for rank in (0, 1):
            for key, value in (('candidate_audio_context', 'full-source-v1'),
                               ('candidate_progress_receipts', 'per-step-v1'),
                               ('source_window_encodes', 0),
                               ('audio_overlap_policy', 'prior-canonical-wins')):
                with self.subTest(rank=rank, key=key):
                    def mutate(stage, decode):
                        if stage['worker_rank'] == rank:
                            stage[key] = value
                    self.runtime.mutate = mutate
                    with self.assertRaisesRegex(RuntimeError, 'unexpected candidate evidence'):
                        self.run_render(f'legacy-tamper-{rank}-{key}')
                    self.assertEqual(self.timings, [])
                    self.assert_restored()

    def test_monotonic_timing_boundaries_and_unique_throughput(self):
        # Three suffixes: 119+102+107 unique frames, not three full 124-frame windows.
        ticks = iter([100., *([105.]*6), 110., *([120.]*6), 125.,
                      *([135.]*6), 140., 145., 160.])
        boundaries = []
        hashed = []
        original_sha = renderer._sha

        def sha(path):
            hashed.append(Path(path).name)
            return original_sha(path)

        def clock():
            boundaries.append((list(self.runtime.events), len(self.callbacks), list(hashed)))
            return next(ticks)

        with patch.object(renderer.time, 'monotonic', side_effect=clock), \
                patch.object(renderer, '_sha', side_effect=sha):
            result = self.run_render()
        profile = json.loads(result.profile.read_text())
        timing = profile['timing_metrics']
        self.assertEqual(timing['total_wall_seconds'], 60.)
        self.assertEqual(timing['first_segment_ready_seconds'], 10.)
        self.assertEqual(timing['segment_ready_elapsed_seconds'], [10., 25., 40.])
        self.assertEqual(timing['mux_export_seconds'], 15.)
        self.assertEqual(len(profile['pipeline_timing']), 3)
        for window in profile['pipeline_timing']:
            self.assertEqual(window['clock'], 'parent_monotonic')
            self.assertLessEqual(window['request_end'], window['encoding_start'])
            self.assertLessEqual(window['encoding_end'], window['validation_start'])
        self.assertEqual(timing['generation_total_seconds'], 3.)
        self.assertEqual(profile['generation_intervals'], [[1., 2.]]*3)
        self.assertEqual(profile['generation_seconds'], [1.]*3)
        self.assertEqual(timing['unique_video_seconds'], 328/24)
        self.assertAlmostEqual(timing['unique_throughput_total'], (328/24)/60)
        self.assertAlmostEqual(timing['unique_throughput_generation'], (328/24)/3)
        self.assertEqual(boundaries[0], ([], 0, []))
        for i in range(3):
            events, callbacks, _ = boundaries[(i+1)*7]
            self.assertEqual(events[-1], ('metadata', f'segment-{i:03d}.mp4'))
            self.assertEqual(callbacks, i)  # Ready time excludes this callback.
        self.assertEqual(boundaries[-2][1], 3)
        self.assertEqual(boundaries[-1][0][-1], ('metadata', 'review-with-speech.mp4'))
        self.assertEqual(boundaries[-1][2][-6:], ['speech.wav', 'segment-000.mp4',
            'segment-001.mp4', 'segment-002.mp4', 'joined.mp4', 'review-with-speech.mp4'])
        self.assertIn('excludes per-segment encoding', timing['mux_export_scope'])
        self.assertIn('not isolated denoise/decode', timing['generation_scope'])
        self.assertFalse(profile['realtime_complete'])
        self.assertEqual(len(self.runtime.seen), 3)
        self.assert_restored()

    def test_zero_duration_timing_has_no_infinite_throughput(self):
        generate = self.runtime._generate

        def instantaneous(**kwargs):
            result, _ = generate(**kwargs)
            return result, (2., 2.)

        with patch.object(renderer.time, 'monotonic', return_value=100.), \
                patch.object(self.runtime, '_generate', side_effect=instantaneous):
            result = self.run_render()
        timing = json.loads(result.profile.read_text())['timing_metrics']
        self.assertEqual(timing['total_wall_seconds'], 0.)
        self.assertEqual(timing['generation_total_seconds'], 0.)
        self.assertEqual(timing['segment_ready_elapsed_seconds'], [0.]*3)
        self.assertIsNone(timing['unique_throughput_total'])
        self.assertIsNone(timing['unique_throughput_generation'])
        self.assert_restored()

    def test_stale_receipts_removed_before_generation(self):
        for rank in (0, 1):
            for stem in ('h3-continuation', 'h3-continuation-decode'):
                renderer._json(self.runtime.coverage / f'{stem}-rank-{rank}-window-0.json',
                               {'session_id': 'stale', 'committed': True})
        self.run_render()
        self.assert_restored()

    def test_metadata_failure_restores_independent_before_any_callback(self):
        with patch.object(renderer, '_metadata', side_effect=RuntimeError('metadata failed')):
            with self.assertRaisesRegex(RuntimeError, 'metadata failed'):
                self.run_render()
        self.assertEqual(self.callbacks, [])
        self.assert_restored()

    def test_unique_session_for_identical_source(self):
        first = self.run_render('first')
        second = self.run_render('second')
        a, b = [json.loads(r.profile.read_text()) for r in (first, second)]
        self.assertNotEqual(a['session_id'], b['session_id'])
        self.assert_restored()

    def test_receipt_tampering_rejected_before_encode(self):
        cases = [('stage', 'session_id', 'stale'), ('stage', 'wav_sha256', '0'*64),
                 ('stage', 'forwards', 3), ('stage', 'steps_held', 3),
                 ('stage', 'committed', False), ('stage', 'full_source_encodes', 2),
                 ('stage', 'source_samples', 1), ('stage', 'target_audio_rows', 1),
                 ('stage', 'audio_time_offset', 1), ('stage', 'video_row_offset', 1),
                 ('stage', 'suffix_video_rows', 1), ('stage', 'padded_samples', -1),
                 ('decode', 'frame_offset', 1), ('decode', 'frame_count', 124),
                 ('decode', 'two_latent_lookahead', 0), ('decode', 'participation', False),
                 ('decode', 'retained_overlap_frames', 0),
                 ('decode', 'retained_overlap_frames', True)]
        for i, (kind, key, value) in enumerate(cases):
            with self.subTest(kind=kind, key=key):
                def mutate(stage, decode):
                    if stage['worker_rank'] == 1:
                        (stage if kind == 'stage' else decode)[key] = value
                self.runtime.mutate = mutate
                with self.assertRaises(RuntimeError):
                    self.run_render(f'tamper-{i}')
                self.assertEqual(self.timings, [])
                self.assert_restored()

    def test_final_overlap_receipt_rejected_before_final_export(self):
        def mutate(stage, decode):
            if decode['final'] and decode['worker_rank'] == 1:
                decode['retained_overlap_frames'] = 5
        self.runtime.mutate = mutate
        with self.assertRaisesRegex(RuntimeError, 'suffix decode/PTS differs'):
            self.run_render('bad-final-overlap')
        self.assertEqual(len(self.callbacks), 2)  # only prior verified suffixes
        self.assertEqual(len(self.timings), 2)
        self.assert_restored()

    def test_parent_quality_checks_not_bypassed(self):
        for check in ('loaded', 'cache', 'packed', 'residency'):
            with self.subTest(check=check):
                self.runtime.bad_check = check
                with self.assertRaises(RuntimeError):
                    self.run_render(check)
                self.assertEqual(self.timings, [])
                self.assert_restored()

    def test_frame_shape_dtype_and_actual_seed_rejected(self):
        for name in ('shape', 'dtype', 'seed'):
            with self.subTest(name=name):
                self.runtime.wrong_frames = name == 'shape'
                self.runtime.dtype = 'float32' if name == 'dtype' else 'uint8'
                self.runtime._last_seed = 42 if name == 'seed' else 314159
                with self.assertRaises(RuntimeError):
                    self.run_render(name)
                self.assertEqual(self.timings, [])
                self.assert_restored()

    def test_restore_on_encode_callback_and_generation_failure(self):
        for name in ('encode', 'callback', 'generate'):
            with self.subTest(name=name):
                def fail(*args, **kwargs):
                    raise RuntimeError('injected failure')
                encoder = importlib.import_module('research.run_h3_realtime_first')
                target, attribute = ((encoder, 'encode_rgb_segment') if name == 'encode'
                                     else (self.runtime, '_generate'))
                with ExitStack() as stack:
                    if name != 'callback':
                        stack.enter_context(patch.object(target, attribute, fail))
                    with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                        self.run_render(name, fail if name == 'callback' else self.callback)
                self.assert_restored()

    def test_original_mutation_and_pcm_mismatch_rejected(self):
        with patch.object(renderer, '_pcm', side_effect=[b'original', b'wrong']):
            with self.assertRaisesRegex(RuntimeError, 'PCM'):
                self.run_render('bad-pcm')
        self.assert_restored()
        def mutation(segment):
            self.wav.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'SHA'):
            self.run_render('mutated', mutation)
        self.assertEqual(json.loads(self.runtime.continuation_file.read_text())['mode'], 'independent')

    def test_requires_preexisting_independent_manifest_and_new_turn(self):
        self.runtime.continuation_file.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_render()
        self.assertEqual(self.runtime.seen, [])
        renderer._json(self.runtime.continuation_file, {'mode': 'latent-tail-v1'})
        with self.assertRaisesRegex(RuntimeError, 'independent'):
            self.run_render()
        renderer._json(self.runtime.continuation_file, {'mode': 'independent', 'preserve': 'test'})
        self.run_render()
        before = len(self.runtime.seen)
        with self.assertRaises(FileExistsError):
            self.run_render()
        self.assertEqual(len(self.runtime.seen), before)
        self.assert_restored()


if __name__ == '__main__':
    unittest.main()
