import hashlib
from pathlib import Path
import unittest
import wave

from research.h3_personal.incremental_speech import TextRevision, text_hash
from research.h3_personal.utterance_incremental_tts import (
    UtteranceIncrementalTTS, ResidentSentenceAdapter, SentenceAudio, SOURCE, MODEL, CODEC,
)


def event(text, *, revision=1, offset=0, prefix='', eos=True, epoch=7, origin='human_reply'):
    return TextRevision(epoch, revision, offset, text, len(prefix+text),
                        text_hash(prefix), text_hash(prefix+text), eos, origin)


class SentenceTests(unittest.TestCase):
    def owner(self, callback=None, **kw):
        return UtteranceIncrementalTTS(7, callback or self.synth, deadline=100,
                                       clock=lambda: 1, **kw)

    @staticmethod
    def synth(text, **kw):
        return SentenceAudio(b'\x01\x80\xff\x7f', 24000, 1, 2, True)

    def test_incremental_stable_reply_and_exact_samples(self):
        calls = []
        chunks = [b'\x01\x80\xff\x7f', b'\x02\x00\xfe\xff']
        def synth(text, **kw):
            calls.append(text)
            return SentenceAudio(chunks[len(calls)-1], 24000, 1, 2, True)
        owner = self.owner(synth)
        owner.feed(event('最初。', eos=False))
        first = owner.produce()
        self.assertEqual(calls, ['最初。'])
        self.assertEqual(first.production, 'sentence_incremental')
        self.assertFalse(owner.reply.closed)
        with self.assertRaises(BufferError): owner.produce()
        with self.assertRaises(BufferError): owner.feed(event('後。'))
        owner.ack(first)
        self.assertIsNone(owner.pending)
        owner.feed(event('後。', revision=2, offset=3, prefix='最初。'))
        second = owner.produce()
        self.assertEqual(calls, ['最初。', '後。'])
        self.assertEqual(first.audio.pcm + second.audio.pcm, b''.join(chunks))
        owner.ack(second)
        self.assertTrue(owner.done)
        self.assertEqual(owner.total_bytes, 8)

    def test_whole_text_still_produces_only_one_sentence_per_pull(self):
        calls = []
        def synth(text, **kw):
            calls.append(text)
            return self.synth(text)
        owner = self.owner(synth)
        owner.feed(event('一。二！三？末尾'))
        for expected in ['一。', '二！', '三？', '末尾']:
            out = owner.produce()
            self.assertEqual(calls[-1], expected)
            self.assertEqual(len(calls), out.index)
            owner.ack(out)
        self.assertTrue(owner.done)

    def test_unterminated_stable_text_waits_for_eos(self):
        owner = self.owner()
        owner.feed(event('途中', eos=False))
        self.assertIsNone(owner.produce())

    def test_semantic_epoch_text_limits(self):
        for e in [event('x', origin='asr_partial'), event('{"reply":"x"}'),
                  event('x', epoch=8), event('x'*161), event('あ。'*5)]:
            owner = self.owner()
            with self.assertRaises((ValueError, BufferError)): owner.feed(e)
            self.assertTrue(owner.cancelled)
            self.assertEqual(owner.reply.text, '')

    def test_watermark_pins_and_format_rejection(self):
        for audio in [SentenceAudio(b'\0\0', 24000, 1, 2, False),
                      SentenceAudio(b'\0\0', 24000, 1, 2, True, source_revision='bad'),
                      SentenceAudio(b'x', 24000, 1, 2, True)]:
            owner = self.owner(lambda *a, **kw: audio)
            owner.feed(event('一。'))
            with self.assertRaises(ValueError): owner.produce()
            self.assertTrue(owner.cancelled)
            self.assertIsNone(owner.pending)

    def test_stale_and_deadline_after_callback_discard(self):
        for kind in ['stale', 'deadline', 'cancel']:
            current = [True]; now = [1]
            def synth(*a, **kw):
                if kind == 'stale': current[0] = False
                elif kind == 'deadline': now[0] = 101
                else: owner.cancel()
                return self.synth('')
            owner = UtteranceIncrementalTTS(7, synth, deadline=100,
                                           clock=lambda: now[0], is_current=lambda: current[0])
            owner.feed(event('一。'))
            with self.assertRaises(RuntimeError): owner.produce()
            self.assertIsNone(owner.pending)
            self.assertEqual(owner.total_bytes, 0)

    def test_memory_aggregate_bytes_duration_and_format(self):
        owner = self.owner(max_bytes=4)
        owner.feed(event('一。二。'))
        owner.ack(owner.produce())
        with self.assertRaises(BufferError): owner.produce()
        self.assertIsNone(owner.pending)
        owner = self.owner(lambda *a, **kw: SentenceAudio(b'\0\0'*16, 1, 1, 2, True))
        owner.feed(event('一。二。'))
        owner.ack(owner.produce())
        with self.assertRaises(BufferError): owner.produce()
        rates = iter([24000, 48000])
        owner = self.owner(lambda *a, **kw: SentenceAudio(b'\0\0', next(rates), 1, 2, True))
        owner.feed(event('一。二。'))
        owner.ack(owner.produce())
        with self.assertRaises(ValueError): owner.produce()

    def test_four_plus_unterminated_fifth_is_rejected(self):
        owner = self.owner()
        owner.feed(event('一。二。三。四。五'))
        for _ in range(4): owner.ack(owner.produce())
        with self.assertRaises(ValueError): owner.produce()

    def test_adapter_exact_native_wav_and_cleanup(self):
        paths = []
        pcm = b'\x01\x80\xfe\x7f'
        def resident(text, destination, **kw):
            paths.append(destination)
            with wave.open(str(destination), 'wb') as wav:
                wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                wav.writeframes(pcm)
            return dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                        samples=2, sample_rate=24000, channels=1, watermark_applied=True,
                        watermark_detected=None, source_revision=SOURCE, model_revision=MODEL,
                        codec_revision=CODEC, text_sha256=text_hash(text))
        adapter = ResidentSentenceAdapter(resident)
        audio = adapter('文。', is_current=lambda: True, max_bytes=4)
        self.assertEqual(audio.pcm, pcm)
        self.assertIsNone(audio.watermark_detected)
        self.assertFalse(paths[0].exists())
        def bad(*a, **kw):
            result = resident(*a, **kw)
            result['watermark_applied'] = False
            return result
        with self.assertRaises(ValueError):
            ResidentSentenceAdapter(bad)('文。', is_current=lambda: True, max_bytes=4)
        self.assertFalse(paths[-1].exists())


if __name__ == '__main__':
    unittest.main()
