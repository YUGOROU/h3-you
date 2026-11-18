import hashlib
import unittest

from research.h3_personal.incremental_speech import (
    IncrementalText, TextRevision, IncrementalSpeech, ProducedPCM, text_hash)
from research.h3_personal.streaming_pcm import PCMStreamBinding, PCMChunk, PCMBackpressure


def event(seam, text, *, offset=0, stable=None, eos=False, origin=None, revision=None):
    whole = seam.text[:offset]+text
    return TextRevision(seam.epoch, revision or seam.revision+1, offset, text,
                        len(whole) if stable is None else stable,
                        text_hash(seam.text[:offset]), text_hash(whole), eos,
                        origin or ('human_reply' if seam.channel == 'reply' else 'asr_partial'))


class SpeechTests(unittest.TestCase):
    def reply(self):
        r = IncrementalText(3, channel='reply')
        r.accept(event(r, 'Hello', eos=True))
        return r

    def stream(self, **kw):
        b = PCMStreamBinding('conversation', 'speech', 3, 'a'*64)
        return IncrementalSpeech(b, self.reply(), **kw)

    def pcm(self, s, data=b'\x01\x00', **kw):
        values = dict(chunk=PCMChunk(s.ingress.binding, s.ingress._index, s.samples, data),
                      text_revision=s.revision, text_sha256=s.text_sha256,
                      prefix_sha256=s.ingress.prefix_digest(s.samples),
                      chunk_sha256=hashlib.sha256(data).hexdigest(),
                      production=s.production, watermark_applied=True)
        values.update(kw)
        return ProducedPCM(**values)

    def test_partial_revision_and_explicit_commit(self):
        s = IncrementalText(3, channel='asr')
        s.accept(event(s, 'cat', stable=1))
        with self.assertRaises(RuntimeError): s.committed_input()
        s.accept(event(s, 'ar', offset=1, stable=1))
        self.assertEqual(s.text, 'car')
        with self.assertRaises(ValueError): s.accept(event(s, 'dog'))
        s.accept(event(s, 'ar', offset=1, eos=True, origin='application_endpoint'))
        self.assertEqual(s.committed_input(), 'car')
        self.assertEqual(s.commit_origin, 'application_endpoint')
        with self.assertRaises(RuntimeError): s.accept(event(s, 'x'))

    def test_tool_arguments_are_not_reply(self):
        s = IncrementalText(3, channel='reply')
        for origin in ('tool_call_json', 'reasoning', 'asr_final'):
            with self.assertRaises(ValueError):
                s.accept(event(s, '{"reply":"hi"}', eos=True, origin=origin))
        self.assertEqual(s.revision, 0)

    def test_stale_hash_and_bounds_atomic(self):
        s = IncrementalText(3, channel='reply', max_chars=3)
        for e in (TextRevision(2, 1, 0, 'x', 1, text_hash(''), text_hash('x')),
                  TextRevision(3, 1, 0, 'x', 1, text_hash(''), '0'*64, origin='human_reply'),
                  event(s, 'long')):
            with self.assertRaises((ValueError, BufferError)): s.accept(e)
        self.assertEqual(s.text, '')
        s.accept(event(s, 'hi', stable=0))
        with self.assertRaises(ValueError): s.accept(event(s, 'hi', revision=1))
        s.cancel()
        with self.assertRaises(RuntimeError): s.accept(event(s, 'x'))

    def test_produced_pcm_wait_eos_original_bytes(self):
        s = self.stream()
        self.assertEqual(s.read(0, 2).status, 'wait')
        s.append(self.pcm(s)); s.append(self.pcm(s, b'\x02\x00'))
        data = b'\x01\x00\x02\x00'
        source = s.seal(epoch=3, revision=1, total_samples=2,
                        sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(source.pcm, data)
        self.assertEqual(s.read(0, 3).status, 'end')
        self.assertTrue(s.genuinely_incremental)

    def test_reject_unmarked_stale_segmentation(self):
        s = self.stream()
        for kw in (dict(watermark_applied=False), dict(text_revision=2),
                   dict(production='complete_wav_segment'), dict(prefix_sha256='0'*64),
                   dict(chunk_sha256='0'*64)):
            with self.assertRaises(ValueError): s.append(self.pcm(s, **kw))
        self.assertEqual(s.samples, 0)
        e = self.pcm(s); s.append(e)
        with self.assertRaises(ValueError): s.append(e)
        with self.assertRaises(ValueError): s.seal(epoch=2, revision=1, total_samples=1, sha256='0'*64)

    def test_bounded_backpressure_no_drop(self):
        s = self.stream(max_bytes=2, max_chunk_bytes=2, max_read_bytes=2)
        s.append(self.pcm(s))
        with self.assertRaises(PCMBackpressure): s.append(self.pcm(s))
        self.assertEqual(s.samples, 1)
        s.cancel()
        with self.assertRaises(RuntimeError): s.read(0, 1)

    def test_deadline_and_complete_wav_label(self):
        now = [0.0]
        s = self.stream(production='complete_wav_segment', clock=lambda: now[0], timeout=1)
        self.assertFalse(s.genuinely_incremental)
        now[0] = 1.0
        with self.assertRaises(TimeoutError): s.append(self.pcm(s))

    def test_no_premature_reply_or_pcm_repr(self):
        r = IncrementalText(3, channel='reply')
        b = PCMStreamBinding('c', 's', 3, 'a'*64)
        with self.assertRaises(ValueError): IncrementalSpeech(b, r)
        s = self.stream()
        self.assertNotIn('secret', repr(self.pcm(s, b'secret')))


if __name__ == '__main__':
    unittest.main()
