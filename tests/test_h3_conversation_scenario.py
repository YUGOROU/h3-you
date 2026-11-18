import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from research.h3_personal.conversation_av import ConversationAVStream
from research.h3_personal.conversation_audio import audio_origin
from research.h3_personal.conversation_scenario import (
    TOTAL_FRAMES, ConversationRenderer, ScenarioPlan, plan, segment_timing,
)
from research.h3_personal.conversation_stages import validate_manifest
from research.h3_personal.fast_continuation_stages import ENV as MANIFEST_ENV


RATE = 48000
PROFILE = 'c' * 64


def write_wav(path, pcm, rate=RATE, channels=1, width=2):
    with wave.open(str(path), 'wb') as handle:
        handle.setparams((channels, width, rate, 0, 'NONE', 'nonce'))
        handle.writeframes(pcm)
    return path


def silence(samples, channels=1):
    return bytes(samples * channels * 2)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.listening = (write_wav(self.root / 'listen-0.wav', silence(4800)),
                          write_wav(self.root / 'listen-2.wav', silence(4800)))
        self.speech_pcm = bytes((i * 11 + 5) % 241 for i in range(4800 * 2))
        self.speech = write_wav(self.root / 'speech.wav', self.speech_pcm)

    def tearDown(self):
        self.tmp.cleanup()

    def scenario(self, **kwargs):
        options = dict(conversation_id='conversation', profile_sha256=PROFILE,
                       listening_wavs=self.listening, speech_wav=self.speech)
        options.update(kwargs)
        return plan(**options)

    def test_fixed_endpoints_and_segments(self):
        p = self.scenario()
        self.assertEqual(p.frames, TOTAL_FRAMES)
        self.assertEqual(p.committed_endpoints, (124, 226, 328, 430, 532))
        self.assertEqual(p.emitted_endpoints, (119, 221, 323, 425, 532))
        self.assertEqual(p.segments, ((0, 119), (119, 221), (221, 323), (323, 425), (425, 532)))
        emissions = p.window_emissions()
        self.assertEqual([e[2] for e in emissions], [119, 102, 102, 102, 107])
        self.assertEqual([e[0] for e in emissions], [0, 1, 1, 1, 2])
        self.assertEqual([e[3] for e in emissions], [False, False, False, False, True])
        self.assertEqual([s.role for s in p.sources], ['listening', 'speech', 'listening'])
        self.assertEqual([s.onset_frame for s in p.sources], [0, 124, 430])
        self.assertEqual([s.guide_start for s in p.sources], [None, 102, 408])
        self.assertEqual(p.sources[1].committed_endpoints, (226, 328, 430))
        self.assertEqual(p.sources[2].committed_endpoints, (532,))

    def test_manifests_validate_against_native_contracts(self):
        p = self.scenario()
        manifests = p.manifests()
        self.assertEqual(len(manifests), 5)
        self.assertEqual([(e, i) for e, i, _ in manifests],
                         [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)])
        previous = os.environ.get(MANIFEST_ENV)
        path = self.root / 'window.json'
        try:
            os.environ[MANIFEST_ENV] = str(path)
            from research.h3_personal.fast_continuation_stages import _manifest
            for epoch, index, manifest in manifests:
                # Round-trip through the real worker-facing file contract.
                path.write_text(json.dumps(manifest))
                loaded = _manifest()
                c = validate_manifest(loaded)
                self.assertEqual(c['source_epoch'], epoch)
                self.assertEqual(loaded['index'], index)
                self.assertEqual(loaded['session_id'], f'conversation-{epoch}')
                self.assertEqual(c['role'], ('listening', 'speech', 'listening')[epoch])
                self.assertEqual(loaded['total_frames'], (124, 328, 124)[epoch])
                self.assertEqual(c['total_frames'], TOTAL_FRAMES)
                self.assertEqual(c['source_onset_frame'], (0, 124, 430)[epoch])
                self.assertEqual(manifest['final'], index == len(p.sources[epoch].windows) - 1)
                self.assertEqual(c['conversation_final'],
                                 epoch == 2 and manifest['final'])
            final = manifests[-1][2]['conversation']
            self.assertTrue(final['conversation_final'])
            for epoch, index, manifest in manifests[:-1]:
                self.assertFalse(manifest['conversation']['conversation_final'])
            # The speech source must exist for both listening epochs too.
            self.assertEqual([m['wav'] for _, _, m in manifests],
                             [str(Path(self.listening[0]).resolve()), str(self.speech.resolve()),
                              str(self.speech.resolve()), str(self.speech.resolve()),
                              str(Path(self.listening[1]).resolve())])
        finally:
            if previous is None:
                os.environ.pop(MANIFEST_ENV, None)
            else:
                os.environ[MANIFEST_ENV] = previous

    def test_binding_preserves_original_speech_pcm(self):
        p = self.scenario()
        binding = p.binding()
        self.assertEqual(binding.conversation_id, 'conversation')
        self.assertEqual([(s.frame_start, s.frame_stop, s.kind) for s in binding.spans],
                         [(0, 124, 'listening'), (124, 430, 'speech'), (430, 532, 'listening')])
        self.assertEqual(binding.sources[0].pcm, self.speech_pcm)
        self.assertEqual(binding.sources[0].sample_rate, RATE)
        self.assertEqual(binding.sources[0].sha256, hashlib.sha256(self.speech_pcm).hexdigest())
        producer = binding.producer()
        from research.h3_personal.conversation_timeline import ConversationSegment
        slices = producer.emit(ConversationSegment(binding, 0, 0, 119))
        self.assertEqual([s.kind for s in slices], ['listening'])
        slices = producer.emit(ConversationSegment(binding, 1, 119, 221))
        # The staged speech source is short, so its span ends in explicit silence.
        self.assertEqual([s.kind for s in slices], ['listening', 'speech', 'tailpadding'])
        self.assertEqual(slices[0].sample_stop, binding.boundary(124))
        self.assertEqual(slices[1].pcm, self.speech_pcm)
        self.assertEqual(slices[1].source_start, 0)
        self.assertEqual(slices[2].source_id, None)

    def test_staged_source_rejections(self):
        noisy = write_wav(self.root / 'noisy.wav', b'\x01\x00' * 4800)
        with self.assertRaises(ValueError):
            self.scenario(listening_wavs=(noisy, self.listening[1]))
        with self.assertRaises(ValueError):
            self.scenario(listening_wavs=(self.listening[0],))
        long_pcm = bytes(612002 * 2)
        too_long = write_wav(self.root / 'long.wav', long_pcm)
        with self.assertRaises(ValueError):
            self.scenario(speech_wav=too_long)
        with self.assertRaises(ValueError):
            ScenarioPlan('', PROFILE, self.scenario().sources)
        with self.assertRaises(ValueError):
            ScenarioPlan('conversation', PROFILE, self.scenario().sources[:2])

    def test_duplicate_and_producer_bounds(self):
        p = self.scenario()
        binding = p.binding()
        from research.h3_personal.conversation_timeline import ConversationSegment
        producer = binding.producer()
        with self.assertRaises(RuntimeError):
            binding.producer()
        with self.assertRaises(ValueError):
            producer.emit(ConversationSegment(binding, 1, 0, 1))
        with self.assertRaises(ValueError):
            producer.emit(ConversationSegment(binding, 0, 0, 120))


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class RendererTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.turn = self.root / 'turn'
        self.turn.mkdir(mode=0o700)
        self.listening = (write_wav(self.root / 'listen-0.wav', silence(4800)),
                          write_wav(self.root / 'listen-2.wav', silence(4800)))
        self.speech_pcm = bytes((i * 11 + 5) % 241 for i in range(4800 * 2))
        self.speech = write_wav(self.root / 'speech.wav', self.speech_pcm)
        self.scenario = plan(conversation_id='conversation', profile_sha256=PROFILE,
                             listening_wavs=self.listening, speech_wav=self.speech)

    def tearDown(self):
        self.tmp.cleanup()

    def backend(self, counts, bad=False):
        calls = []
        class Backend:
            def render_window(self, epoch, index, final):
                calls.append((epoch, index, final))
                count = counts[len(calls) - 1]
                if bad:
                    count += 1
                return dict(raw=bytes(count * 252 * 448 * 3), count=count,
                            interval=(0.1 * len(calls), 0.2 * len(calls)))
        self.calls = calls
        return Backend()

    def stream(self):
        return ConversationAVStream(self.root, self.scenario.binding(),
                                    width=252, height=448, timeout=180)

    def test_end_to_end_frames_and_audio(self):
        counts = [e[2] for e in self.scenario.window_emissions()]
        stream = self.stream()
        renderer = ConversationRenderer(self.scenario, stream.binding, stream, self.turn,
                                        self.backend(counts))
        receipt = renderer.run()
        self.assertEqual(self.calls, [(0, 0, False), (1, 0, False), (1, 1, False),
                                      (1, 2, False), (2, 0, True)])
        self.assertEqual(receipt['frames'], TOTAL_FRAMES)
        self.assertEqual([s['frame_start'] for s in receipt['segments']], [0, 119, 221, 323, 425])
        self.assertEqual([s['frame_stop'] for s in receipt['segments']], [119, 221, 323, 425, 532])
        self.assertFalse(receipt['listening_quality_verified'])
        self.assertFalse(receipt['device_playback_verified'])
        av = receipt['av']
        self.assertEqual(av['audio_producers'], 1)
        self.assertLessEqual(av['bounded_queue_highwater'], 2)
        audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', receipt['output'],
            '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=60)
        binding = self.scenario.binding()
        expected = (bytes(binding.boundary(124) * 2) + self.speech_pcm
                    + bytes((binding.boundary(532) - binding.boundary(124)
                             - len(self.speech_pcm) // 2) * 2))
        self.assertEqual(audio, expected)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', receipt['output']], timeout=60)
        times = [float(x) for x in probe.split()]
        self.assertEqual(len(times), TOTAL_FRAMES)
        for index in (0, 119, 221, 323, 425, 531):
            self.assertAlmostEqual(times[index], index / 24, places=6)
        kinds = [s['kind'] for r in av['ranges'] for s in r['slices']]
        self.assertIn('speech', kinds)
        stream.cancel()

    def test_actual_count_mismatch_is_terminal(self):
        counts = [e[2] for e in self.scenario.window_emissions()]
        stream = self.stream()
        renderer = ConversationRenderer(self.scenario, stream.binding, stream, self.turn,
                                        self.backend(counts, bad=True))
        with self.assertRaises(ValueError):
            renderer.run()
        self.assertEqual(len(self.calls), 1)
        stream.cancel()

    def test_manifest_order_enforced(self):
        counts = [e[2] for e in self.scenario.window_emissions()]
        stream = self.stream()
        path = self.root / 'window.json'
        renderer = ConversationRenderer(self.scenario, stream.binding, stream, self.turn,
                                        self.backend(counts), manifest_path=path)
        receipt = renderer.run()
        self.assertEqual(json.loads(path.read_text())['conversation']['source_epoch'], 2)
        self.assertEqual(len(receipt['segments']), 5)
        stream.cancel()


class TimingTests(unittest.TestCase):
    def test_segment_timing_is_global_half_open(self):
        timing = segment_timing(3, 323, 425)
        self.assertEqual((timing.pts_start, timing.pts_end, timing.newly_emitted_frames),
                         (323, 425, 102))
        for args in ((0, 0, 0), (-1, 0, 1), (0, -1, 1)):
            with self.assertRaises(ValueError):
                segment_timing(*args)


if __name__ == '__main__':
    unittest.main()


class SequenceTests(unittest.TestCase):
    def sequence(self):
        from research.h3_personal.conversation_sequence import ConversationSequence
        return ConversationSequence(speech_digest=hashlib.sha256(b'speech').hexdigest(),
                                    sample_rate=RATE, samples=577920)

    def test_ordered_transitions_and_exactly_once(self):
        sequence = self.sequence()
        sequence.listen(handoff_frame=0, visual_receipt=None, audio_guide_reset_receipt=None)
        self.assertEqual(sequence.phase.value, 'listening')
        output = sequence.emit(index=0, frames=119, emitted_end=119)
        self.assertEqual((output.suffix.start, output.suffix.frames), (0, 119))
        with self.assertRaises(ValueError):
            sequence.speech_prepared(handoff_frame=119, visual_receipt='a' * 64,
                                     audio_guide_reset_receipt='b' * 64)
        sequence.request_speech()
        self.assertEqual(sequence.phase.value, 'preparing_speech')
        sequence.speech_prepared(handoff_frame=119, visual_receipt='a' * 64,
                                 audio_guide_reset_receipt='b' * 64)
        self.assertEqual(sequence.phase.value, 'speaking')
        for index, frames in ((1, 102), (2, 102), (3, 102)):
            sequence.emit(index=index, frames=frames, emitted_end=119 + 102 * index)
        self.assertEqual(sequence.state.sample_cursor, 577920)
        sequence.listen(handoff_frame=425, visual_receipt='c' * 64,
                        audio_guide_reset_receipt='d' * 64)
        self.assertEqual(sequence.phase.value, 'listening')
        sequence.emit(index=4, frames=107, emitted_end=532)
        sequence.close()
        receipt = sequence.receipt()
        self.assertEqual([t['action'] for t in receipt['transitions']],
                         ['listen', 'reserve', 'commit', 'prepare', 'prepared', 'reserve',
                          'commit', 'reserve', 'commit', 'reserve', 'commit', 'listen',
                          'reserve', 'commit', 'close'])
        self.assertEqual(receipt['final_phase'], 'closed')
        self.assertEqual(len(receipt['retired_sources']), 1)
        self.assertEqual(receipt['policy'], 'encoded_silence')

    def test_rejections(self):
        sequence = self.sequence()
        with self.assertRaises(ValueError):
            sequence.close()
        sequence.listen(handoff_frame=0, visual_receipt=None, audio_guide_reset_receipt=None)
        with self.assertRaises(ValueError):
            sequence.emit(index=1, frames=102, emitted_end=102)
        with self.assertRaises(ValueError):
            sequence.speech_prepared(handoff_frame=0, visual_receipt='a' * 64,
                                     audio_guide_reset_receipt='short')
        sequence.emit(index=0, frames=119, emitted_end=119)
        sequence.request_speech()
        with self.assertRaises(ValueError):
            sequence.request_speech()


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class RendererSequenceTests(RendererTests):
    def receipts(self, coverage, scenario, bad=False):
        for source in scenario.sources:
            for index, committed in enumerate(source.committed_endpoints):
                for rank in (0, 1):
                    payload = dict(schema='h3-model-conversation-v1',
                        conversation=dict(schema='h3-model-conversation-v1',
                            id=scenario.conversation_id, profile_sha256=scenario.profile_sha256,
                            source_epoch=source.epoch, source_id=source.source_id,
                            role=source.role, source_onset_frame=source.onset_frame,
                            conversation_final=source.conversation_final
                                and index == len(source.committed_endpoints) - 1,
                            total_frames=TOTAL_FRAMES),
                        committed_frames=committed, source_id=source.source_id,
                        audio_origin=audio_origin(source.onset_frame, source.total_frames),
                        canonical_audio_sha256=(('e' if (bad and rank == 1) else 'f') * 64),
                        guide_digests=[None, None, None] if source.epoch == 0 else ['a' * 64, 'b' * 64, 'c' * 64])
                    (coverage / f'h3-conversation-rank-{rank}-epoch-{source.epoch}-window-{index}.json'
                     ).write_text(json.dumps(payload))
        from research.h3_personal.conversation_scenario import read_window_receipts
        return lambda epoch, index: read_window_receipts(coverage, scenario, epoch, index)

    def test_sequence_and_receipts_end_to_end(self):
        from research.h3_personal.conversation_sequence import ConversationSequence
        coverage = self.root / 'coverage'
        coverage.mkdir(mode=0o700)
        counts = [e[2] for e in self.scenario.window_emissions()]
        stream = self.stream()
        sequence = ConversationSequence(
            speech_digest=hashlib.sha256(self.speech_pcm).hexdigest(),
            sample_rate=RATE, samples=4800)
        renderer = ConversationRenderer(self.scenario, stream.binding, stream, self.turn,
                                        self.backend(counts), receipts=self.receipts(coverage, self.scenario),
                                        sequence=sequence)
        receipt = renderer.run()
        phases = [t['phase'] for t in receipt['sequence']['transitions']]
        self.assertEqual(phases[0], 'listening')
        self.assertIn('preparing_speech', phases)
        self.assertEqual(phases[phases.index('preparing_speech') + 1], 'speaking')
        self.assertEqual(phases[-2:], ['listening', 'closed'])
        self.assertEqual([s['committed_frames'] for s in receipt['segments']],
                         [124, 226, 328, 430, 532])
        self.assertEqual(receipt['segments'][0]['canonical_audio_sha256'], 'f' * 64)
        self.assertTrue(all(s['receipt_digest'] for s in receipt['segments']))
        self.assertFalse(receipt['listening_quality_verified'])
        stream.cancel()

    def test_receipt_disagreement_is_terminal(self):
        coverage = self.root / 'coverage'
        coverage.mkdir(mode=0o700)
        counts = [e[2] for e in self.scenario.window_emissions()]
        stream = self.stream()
        renderer = ConversationRenderer(self.scenario, stream.binding, stream, self.turn,
                                        self.backend(counts),
                                        receipts=self.receipts(coverage, self.scenario, bad=True))
        with self.assertRaises(ValueError):
            renderer.run()
        stream.cancel()
