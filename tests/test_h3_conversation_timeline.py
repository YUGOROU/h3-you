import unittest
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256

from research.h3_personal.conversation_timeline import (
    FrozenPCMSource, TimelineSpan, ConversationBinding, ConversationSegment,
)


def source(rate=48000, channels=1, samples=10001, name='original'):
    pcm = bytes(range(256)) * ((samples*channels*2+255)//256)
    pcm = pcm[:samples*channels*2]
    return FrozenPCMSource(name, pcm, rate, channels, sha256(pcm).hexdigest())


def binding(rate=48000, channels=1, onset=124, fps=Fraction(24)):
    s = source(rate, channels)
    return ConversationBinding('conversation', (s,), (
        TimelineSpan(0, onset, 'listening'),
        TimelineSpan(onset, onset+102, 'speech', s.source_id),
        TimelineSpan(onset+102, onset+204, 'listening'),
    ), rate, channels, fps)


class TimelineTests(unittest.TestCase):
    def test_global_clock_and_original_reconstruction(self):
        for rate in (48000, 44100):
            for onset in (124, 430):
                for fps in (Fraction(24), Fraction(24000, 1001)):
                    for channels in (1, 2):
                        with self.subTest(rate=rate, onset=onset, fps=fps, channels=channels):
                            b = binding(rate, channels, onset, fps)
                            p = b.producer()
                            result = []
                            frame = index = 0
                            end = b.spans[-1].frame_stop
                            while frame < end:
                                stop = min(end, frame+(119 if index == 0 else 102))
                                result.extend(p.emit(ConversationSegment(b, index, frame, stop)))
                                frame, index = stop, index+1
                            p.finish()
                            self.assertEqual(b''.join(x.pcm for x in result if x.kind == 'speech'), b.sources[0].pcm)
                            self.assertEqual(sum(len(x.pcm) for x in result), b.boundary(end)*channels*2)
                            for a, z in zip(result, result[1:]):
                                self.assertEqual(a.sample_stop, z.sample_start)
                            speech = [x for x in result if x.kind == 'speech']
                            self.assertEqual(speech[0].sample_start, b.boundary(onset))
                            self.assertEqual(speech[0].source_start, 0)
                            self.assertEqual(speech[-1].source_stop, b.sources[0].samples)
                            self.assertTrue(all(x.source_id == 'original' for x in speech))
                            self.assertTrue(all(not any(x.pcm) for x in result if x.kind != 'speech'))
                            with self.assertRaises(RuntimeError):
                                b.producer()

    def test_old_five_frames_retained_new_source_onset(self):
        for f in (124, 430):
            rate = 44100
            fps = Fraction(24000, 1001)
            boundary = lambda n: n*rate*fps.denominator//fps.numerator
            old = source(rate, samples=boundary(f), name='old')
            new = source(rate, samples=boundary(f+102)-boundary(f), name='new')
            b = ConversationBinding('c', (old, new), (
                TimelineSpan(0, f, 'speech', 'old'),
                TimelineSpan(f, f+102, 'speech', 'new')), rate, 1, fps)
            p = b.producer()
            cursor = index = 0
            while cursor < f-5:
                stop = min(f-5, cursor+119)
                p.emit(ConversationSegment(b, index, cursor, stop))
                cursor, index = stop, index+1
            slices = p.emit(ConversationSegment(b, index, f-5, f+97))
            self.assertEqual([s.source_id for s in slices], ['old', 'new'])
            self.assertEqual(slices[0].source_stop, old.samples)
            self.assertEqual(slices[0].pcm, old.pcm[boundary(f-5)*2:])
            self.assertEqual(slices[1].sample_start, boundary(f))
            self.assertEqual(slices[1].source_start, 0)
            p.emit(ConversationSegment(b, index+1, f+97, f+102))
            p.finish()

    def test_source_rejections(self):
        s = source()
        for changes in ({'pcm': s.pcm[:-1]}, {'pcm': bytearray(s.pcm)},
                        {'sha256': '0'*64}, {'sample_width': 4},
                        {'format': 'S16BE'}, {'sample_rate': True}, {'channels': 3}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    replace(s, **changes)

    def test_binding_rejections(self):
        b = binding()
        for changes in ({'sample_rate': 44100}, {'channels': 2}, {'fps': 24.0},
                        {'max_sources': 5}, {'sources': list(b.sources)},
                        {'spans': (TimelineSpan(0, 1, 'speech', 'original'),)},
                        {'spans': (TimelineSpan(0, 10, 'listening'),)},
                        {'sources': (b.sources[0], b.sources[0])}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    replace(b, **changes)
        with self.assertRaises(ValueError):
            TimelineSpan(0, 10, 'conditioningpreroll')

    def test_foreign_replay_and_incomplete(self):
        b = binding()
        p = b.producer()
        foreign = binding()
        with self.assertRaises(ValueError):
            p.emit(ConversationSegment(foreign, 0, 0, 119))
        with self.assertRaises(ValueError):
            ConversationSegment(object(), 0, 0, 119)
        with self.assertRaises(ValueError):
            p.finish()
        segment = ConversationSegment(b, 0, 0, 119)
        p.emit(segment)
        with self.assertRaises(ValueError):
            p.emit(segment)
        with self.assertRaises(ValueError):
            ConversationSegment(b, 1, 119, 239)


if __name__ == '__main__':
    unittest.main()
