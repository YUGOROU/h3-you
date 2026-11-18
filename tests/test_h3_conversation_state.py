import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

from research.h3_personal.conversation_state import (
    Action as A, Event, Handoff, ListeningPolicy as L, Phase, SpeechSource,
    State, Suffix, reduce,
)


class ConversationTests(unittest.TestCase):
    def listen(self, **kwargs):
        return reduce(State(L.JOINT_GENERATED_DISCARDED, **kwargs), Event(A.LISTEN))[0]

    def handoff(self, s, owner, **kwargs):
        return Handoff(s.owner, owner, s.frame, "visual-tail", "guide-reset",
                       **kwargs)

    def speaking(self, s, source):
        s, _ = reduce(s, Event(A.PREPARE))
        return reduce(s, Event(A.PREPARED, source=source,
                      handoff=self.handoff(s, source.owner,
                                           audio_guide_reset_supported=True)))[0]

    def reserve(self, s, frames):
        return reduce(s, Event(A.RESERVE, token=str(s.next_suffix), frames=frames))[0]

    def commit(self, s):
        return reduce(s, Event(A.COMMIT, token=s.pending[0].token))

    def test_policy_and_bounds(self):
        self.assertEqual(len(set(L)), 4)
        for bad in (0, 3, float('inf'), True):
            with self.assertRaises(ValueError):
                State(L.ZERO_LATENT, max_pending=bad)
        with self.assertRaises(ValueError):
            State("silence")
        s = self.listen()
        with self.assertRaises(ValueError):
            self.reserve(s, 120)
        s = self.reserve(self.reserve(s, 1), 1)
        with self.assertRaises(ValueError):
            self.reserve(s, 1)
        with self.assertRaises(ValueError):
            reduce(s, Event(A.COMMIT, token=s.pending[1].token))

    def test_source_validation(self):
        for digest in (None, 123, b'a' * 64, '', 'g' * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                SpeechSource(digest, 48000, 10)
        for field in ('sample_rate', 'samples', 'channels', 'sample_width'):
            for value in (True, 0, -1, 1.5, '1', None):
                args = dict(digest='a' * 64, sample_rate=48000, samples=10)
                args[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    SpeechSource(**args)
        for channels, width in ((3, 2), (1, 5)):
            with self.assertRaises(ValueError):
                SpeechSource('a' * 64, 48000, 10, channels, width)
        for channels in (1, 2):
            for width in range(1, 5):
                SpeechSource('a' * 64, 48000, 10, channels, width)

    def test_handoff_validation(self):
        valid = dict(previous_owner='old', next_owner='new', frame=0,
                     visual_receipt='v', audio_guide_reset_receipt='a')
        for field, values in (
            ('previous_owner', ('', None, 1)), ('next_owner', ('', None, 1)),
            ('frame', (True, -1, 0.0, '0')),
            ('visual_receipt', (None, 1)),
            ('audio_guide_reset_receipt', (None, 1)),
            ('audio_guide_reset_supported', (1, 0, None, 'true')),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    Handoff(**(valid | {field: value}))
        receipt = Handoff(**valid)
        with self.assertRaises(FrozenInstanceError):
            receipt.frame = 1
        # Empty attestations can be represented, but cannot authorize switching.
        s, _ = reduce(self.listen(), Event(A.PREPARE))
        src = SpeechSource('e' * 64, 48000, 10)
        for visual, audio in (('', 'a'), ('v', '')):
            with self.assertRaises(ValueError):
                reduce(s, Event(A.PREPARED, source=src, handoff=Handoff(
                    s.owner, src.owner, s.frame, visual, audio, True)))

    def test_event_validation(self):
        for kwargs in (
            {'action': 'listen'}, {'action': 'typo'}, {'token': 0},
            {'frames': True}, {'frames': -1}, {'frames': 1.0},
            {'source': object()}, {'handoff': object()}, {'handoff': {}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Event(**({'action': A.LISTEN} | kwargs))

    def test_state_validation(self):
        for field in ('frame', 'next_suffix', 'source_frame', 'sample_cursor'):
            for value in (True, -1, 0.0, '0', None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    State(L.ZERO_LATENT, **{field: value})
        suffix = Suffix('0', 'owner', 0, 1)
        for kwargs in (
            {'phase': 'listening'}, {'phase': 'typo'}, {'fps': True},
            {'fps': Fraction(0)}, {'fps': Fraction(-1)},
            {'max_window_frames': True}, {'owner': None}, {'source': object()},
            {'pending': []}, {'pending': (object(),)},
            {'pending': (suffix, suffix, suffix)},
            {'retired_sources': []}, {'retired_sources': (None,)},
            {'retired_sources': ('',)}, {'retired_sources': ('pcm:fake',)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                State(L.ZERO_LATENT, **kwargs)
        State(L.ZERO_LATENT, retired_sources=('pcm:' + 'a' * 64,))
        # Policy acceptance is a CPU contract, not execution of a latent policy.
        self.assertIs(State(L.ZERO_LATENT).policy, L.ZERO_LATENT)

    def test_priority_preparation_failure_no_reset(self):
        s = self.reserve(self.listen(), 4)
        s, _ = reduce(s, Event(A.PREPARE))
        with self.assertRaises(ValueError):
            self.reserve(s, 1)
        s, effects = self.commit(s)
        self.assertIsNone(effects[0].source)
        self.assertEqual(effects[0].sample_stop, 0)
        s, _ = reduce(s, Event(A.PREPARATION_FAILED))
        self.assertEqual((s.phase, s.frame), (Phase.LISTENING, 4))

    def test_switch_refuses_missing_unsupported_stale_and_pending(self):
        src = SpeechSource('a' * 64, 48000, 10)
        s, _ = reduce(self.listen(), Event(A.PREPARE))
        for receipt in (None, self.handoff(s, src.owner),
                        Handoff(s.owner, src.owner, 1, 'v', 'a', True)):
            with self.assertRaises(ValueError):
                reduce(s, Event(A.PREPARED, source=src, handoff=receipt))
        pending = self.reserve(self.listen(), 1)
        with self.assertRaises(ValueError):
            self.speaking(pending, src)
        with self.assertRaises(FrozenInstanceError):
            src.samples = 20

    def test_rational_clocks_pcm_once_no_padding_and_source_switch(self):
        s = self.listen(fps=Fraction(30000, 1001))
        s, _ = self.commit(self.reserve(s, 3))
        src = SpeechSource('b' * 64, 48000, 7000)
        s = self.speaking(s, src)
        chunks = []
        for frames in (1, 2, 5):
            s, out = self.commit(self.reserve(s, frames))
            chunks.extend(out)
        self.assertEqual([(o.sample_start, o.sample_stop) for o in chunks],
                         [(0, 1601), (1601, 4804), (4804, 7000)])
        self.assertEqual(chunks[0].video_pts, Fraction(3003, 30000))
        self.assertEqual(chunks[1].audio_pts,
                         Fraction(3003, 30000) + Fraction(1601, 48000))
        last_token = chunks[-1].suffix.token
        with self.assertRaises(ValueError):
            reduce(s, Event(A.COMMIT, token=last_token))
        with self.assertRaises(ValueError):
            reduce(s, Event(A.LISTEN))
        s, _ = reduce(s, Event(A.LISTEN, handoff=self.handoff(
            s, s.listening_owner, audio_guide_reset_supported=True)))
        with self.assertRaises(ValueError):
            self.speaking(s, src)
        new = SpeechSource('c' * 64, 44100, 12)
        s = self.speaking(s, new)
        s, out = self.commit(self.reserve(s, 1))
        self.assertEqual((out[0].source, out[0].sample_start, out[0].sample_stop),
                         (new, 0, 12))

    def test_cancel_only_committed_and_late_completion_refused(self):
        s = self.speaking(self.listen(), SpeechSource('d' * 64, 48000, 10000))
        s, out = self.commit(self.reserve(s, 1))
        self.assertEqual(out[0].sample_stop, 2000)
        s = self.reserve(s, 2)
        old = s.pending[0].token
        s, effects = reduce(s, Event(A.CANCEL, handoff=self.handoff(
            s, s.listening_owner, audio_guide_reset_supported=True)))
        self.assertEqual((s.frame, effects, s.pending), (1, (), ()))
        s = self.reserve(s, 1)
        with self.assertRaises(ValueError):
            reduce(s, Event(A.COMMIT, token=old))
        s, out = self.commit(s)
        self.assertIsNone(out[0].source)

    def test_fault_terminal_and_close(self):
        s = self.reserve(self.listen(), 1)
        s, out = reduce(s, Event(A.INFERENCE_FAILED))
        self.assertEqual((s.phase, s.pending, out), (Phase.FAULTED, (), ()))
        for action in (A.LISTEN, A.PREPARE, A.CANCEL, A.PREPARATION_FAILED):
            with self.assertRaises(ValueError):
                reduce(s, Event(action))
        s, _ = reduce(s, Event(A.CLOSE))
        with self.assertRaises(ValueError):
            reduce(s, Event(A.LISTEN))


if __name__ == '__main__':
    unittest.main()
