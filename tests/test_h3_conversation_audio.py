"""Source origin crosses waveform planning, marker encode, owner trim and PCM clock."""
import unittest
from test_h3_fast_continuation import Tensor
import test_h3_conversation_continuation as fixtures
from research.h3_personal.conversation_audio import audio_origin, prepare_waveform, WaveHistory
from research.h3_personal.continuation_plan import audio_boundary


class WaveOps:
    def shape(self, w): return (len(w), len(w[0]))
    def copy(self, w): return [row[:] for row in w]
    def slice(self, w, a, b): return [row[a:b] for row in w]
    def zeros(self, like, count): return [[0]*count for _ in range(2)]
    def cat(self, pieces): return [sum((p[c] for p in pieces), []) for c in range(2)]


def marker_encode(wave):
    # Deterministic stand-in ONLY for checking sample/latent index mapping.
    # Native audio VAE numerical behavior is not established by this test.
    n = len(wave[0])//800
    return Tensor((2, n, 32),
                  [wave[c][i*800] for c in range(2) for i in range(n) for _ in range(32)])


class AudioOriginTests(unittest.TestCase):
    def test_native_prefix_capacity_and_phase(self):
        first = audio_origin(0, 124)
        speech = audio_origin(124, 328)
        last = audio_origin(430, 124)
        self.assertEqual(first['preroll_latents'], 0)
        self.assertEqual(speech['model_start_frame'], 102)
        self.assertEqual(speech['preroll_samples_32000'], 29600)
        self.assertEqual(last['preroll_latents'], 37)
        for bad in (-1, 1, True, 22):
            with self.assertRaises(ValueError): audio_origin(bad, 124)

    def test_history_and_capacity_fail_closed(self):
        ops = WaveOps(); source = [[1]*800 for _ in range(2)]
        p = audio_origin(124, 124)
        for history in (None, WaveHistory(123, [[0]*29600]*2), WaveHistory(124, [[0]*8]*2)):
            with self.assertRaises(ValueError): prepare_waveform(source, p, history, ops)
        history = WaveHistory(124, [[0]*29600]*2)
        with self.assertRaisesRegex(ValueError, 'never truncate'):
            prepare_waveform([[1]*p['capacity_samples_32000']]*2, p, history, ops)
        with self.assertRaises(ValueError):
            prepare_waveform(source, audio_origin(0, 124), history, ops)

    def test_marker_source_onset_and_reverse_history_through_actual_owner(self):
        case = fixtures.ConversationTests(); owner = case.owner(); ops = WaveOps(); history = None
        for epoch, (onset, total, samples) in enumerate(((0,124,32000), (124,328,385280), (430,124,32000))):
            origin = audio_origin(onset, total)
            speech = [[i+1 for i in range(samples)] for _ in range(2)] if epoch == 1 else [[0]*samples for _ in range(2)]
            original_first = speech[0][0]
            wave, next_history, padding = prepare_waveform(speech, origin, history, ops)
            prefix = origin['preroll_samples_32000']
            self.assertEqual(wave[0][prefix:prefix+samples], speech[0])
            self.assertEqual(speech[0][0], original_first)  # no source mutation
            if prefix:
                self.assertEqual(wave[0][:prefix], history.tail[0])
            if epoch == 1:
                self.assertFalse(any(wave[0][:prefix]))
                self.assertEqual(wave[0][prefix], 1)
            if epoch == 2:
                # Reverse guide must include the real preceding speech tail,
                # not substitute all-silent audio for a still-speaking guide.
                self.assertTrue(any(wave[0][:prefix]))
                self.assertEqual(wave[0][prefix-1], 0)
            canonical = marker_encode(wave)
            owner.start_source(str(epoch), canonical_audio=canonical, total_frames=total,
                profile='reference1024-bound', height=2, width=2, audio_origin=origin)
            windows = len(owner.source_session.windows)
            for index in range(windows):
                token = owner.begin_window()
                result = case.commit(owner, token, final=epoch == 2 and index == windows-1)
                # First newly retained source video corresponds to source0,
                # because its preceding latent interval is now actual history.
                if epoch == 1 and index == 0:
                    self.assertEqual(result.source_audio_start, 37)
                    self.assertEqual(result.suffix.audio.values[0], speech[0][0])
                    self.assertEqual(audio_boundary(onset)-token.window.audio_start, 37)
                    # Original PCM clock still starts at onset, not model base.
                    self.assertEqual(onset*48000//24, 248000)
                owner.ack_decoded_count(owner.committedF if epoch == 2 and index == windows-1
                                        else owner.committedF-5,
                                        final_flush=epoch == 2 and index == windows-1)
            history = next_history
        self.assertEqual(owner.committedF, 532)
        self.assertEqual(owner.emittedF, 532)

    def test_owner_rejects_missing_or_wrong_origin(self):
        case = fixtures.ConversationTests(); owner = case.owner()
        canonical = Tensor((2, audio_boundary(124), 32))
        for origin in (None, audio_origin(124,124)):
            with self.assertRaisesRegex(ValueError, 'audio origin'):
                owner.start_source('a',canonical_audio=canonical,total_frames=124,
                    profile='reference1024-bound',height=2,width=2,audio_origin=origin)
        self.assertEqual(owner.committedF, 0)
