"""CPU protocol tests only; no installs, weights, private files, or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from research.h3_personal.qwen3_streaming_candidate import Qwen3StreamingCandidate


def event(kind, **kw):
    return json.dumps(dict(type=kind, utterance_index=0, **kw))


def frames():
    return [event("audio.start", sentence_index=0, format="pcm", sample_rate=24000, channels=1),
            b"\x01\x00", b"\x02\x00", event("audio.done", sentence_index=0, total_bytes=4, error=False),
            event("session.done", total_sentences=1)]


class Socket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.closed = False
        self.received = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, timeout):
        assert timeout > 0
        self.received += 1
        return next(self.messages)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.wav"
        with wave.open(str(self.path), "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * 24)
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def make(self, messages=None, **kw):
        self.socket = Socket(frames() if messages is None else messages)
        return Qwen3StreamingCandidate(self.path, "参照", expected_reference_sha256=self.sha,
                                       _connect=lambda *a, **k: self.socket, **kw)

    def test_incremental_single_full_text_and_native_bytes(self):
        tts = self.make()
        stream = tts.stream("こんにちは。元気ですか？")
        self.assertEqual(next(stream), b"\x01\x00")
        self.assertEqual(self.socket.received, 2)  # not drained to EOS before yield
        config, text, flush = self.socket.sent
        self.assertEqual(text, {"type": "input.text", "text": "こんにちは。元気ですか？"})
        self.assertEqual(flush, {"type": "input.done"})
        self.assertTrue(config["stream_audio"])
        self.assertEqual(config["split_granularity"], "none")
        self.assertEqual(config["language"], "Japanese")
        self.assertEqual(config["task_type"], "Base")
        self.assertFalse(config["x_vector_only_mode"])
        self.assertEqual(list(stream), [b"\x02\x00"])
        self.assertTrue(self.socket.closed)

    def test_explicit_xvector_and_release_start_without_channels(self):
        messages=frames();start=json.loads(messages[0]);start.pop('channels');messages[0]=json.dumps(start)
        sock=Socket(messages)
        tts=Qwen3StreamingCandidate(self.path,expected_reference_sha256=self.sha,x_vector_only=True,
                                   _connect=lambda *a,**k:sock)
        self.assertEqual(list(tts.stream('一括の全文です。')), [b'\x01\x00',b'\x02\x00'])
        self.assertTrue(sock.sent[0]['x_vector_only_mode'])
        self.assertIsNone(sock.sent[0]['ref_text'])
        self.assertEqual(sock.sent[0]['seed'],42)
        with self.assertRaises(ValueError):
            Qwen3StreamingCandidate(self.path,'文字列',expected_reference_sha256=self.sha,x_vector_only=True)

    def test_original_bytes_local_file_config_small_and_bound(self):
        tts=Qwen3StreamingCandidate(self.path,expected_reference_sha256=self.sha,x_vector_only=True,
                                   reference_transport='local-file')
        self.assertEqual(tts._config['ref_audio'],self.path.resolve().as_uri())
        self.assertLess(len(json.dumps(tts._config)),4096)
        big=Path(self.tmp.name)/'large-synthetic.wav'
        with wave.open(str(big),'wb') as w:
            w.setparams((2,2,48000,0,'NONE','not compressed'));w.writeframes(b'\0'*4*48000*18)
        digest=hashlib.sha256(big.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'4MiB'):
            Qwen3StreamingCandidate(big,expected_reference_sha256=digest,x_vector_only=True)
        local=Qwen3StreamingCandidate(big,expected_reference_sha256=digest,x_vector_only=True,reference_transport='local-file')
        self.assertLess(len(json.dumps(local._config)),4096)

    def test_cancel_closes_and_serial_guard(self):
        tts = self.make()
        stream = tts.stream("こんにちは")
        next(stream)
        with self.assertRaises(RuntimeError):
            next(tts.stream("別"))
        stream.close()
        self.assertTrue(self.socket.closed)
        self.assertFalse(tts._busy)

    def test_reject_remote_and_proxy_paths(self):
        for url in ["ws://example.com:8091/v1/audio/speech/stream",
                    "ws://127.0.0.1:8091/v1/audio/speech/stream?secret=x",
                    "ws://user@127.0.0.1:8091/v1/audio/speech/stream"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.make(url=url)

    def test_hash_required(self):
        with self.assertRaises(ValueError):
            Qwen3StreamingCandidate(self.path, "参照", expected_reference_sha256="0" * 64)

    def test_limits(self):
        for kw in [{"max_new_tokens": 0}, {"max_new_tokens": True}, {"timeout": float("inf")},
                   {"timeout": 301}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.make(**kw)
        for text in ["", " " , "あ" * 4097]:
            with self.assertRaises(ValueError):
                list(self.make().stream(text))

    def test_invalid_protocol_never_succeeds(self):
        cases = [[b"\0\0"], frames()[:1] + [b"\0"],
                 [event("audio.start", sentence_index=0, format="pcm", sample_rate=48000, channels=1)],
                 frames()[:3] + [event("audio.done", sentence_index=0, total_bytes=5)],
                 frames()[:3] + [event("audio.done", sentence_index=0, total_bytes=4, error=True)],
                 frames()[:3] + [event("session.done", total_sentences=1)],
                 [event("error", message="PRIVATE DETAILS")]]
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(RuntimeError) as cm:
                list(self.make(messages).stream("こんにちは"))
            self.assertNotIn("PRIVATE DETAILS", str(cm.exception))
            self.assertTrue(self.socket.closed)


if __name__ == "__main__":
    unittest.main()
