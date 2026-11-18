import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from research import run_h3_split_av_long_review as review
from research import run_h3_split_av_probe as probe
from research.h3_personal.paired_av import LiveAVBinding


class LongReviewTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(len(review.RANGES), 23)
        self.assertEqual([b-a for a,b in review.RANGES], [119]+[102]*21+[107])
        self.assertEqual(review.RANGES[-1][1], 2368)
        self.assertTrue(all(x[1] == y[0] for x,y in zip(review.RANGES, review.RANGES[1:])))
        self.assertEqual(probe.RANGES, ((0,119),(119,221),(221,240)))

    def test_regions(self):
        pcm = bytes(review.SAMPLES*2)
        for a,b in review.RANGES:
            regions = review.regions(a,b,pcm)
            self.assertEqual(regions[0].sample_start, a*2000)
            self.assertEqual(regions[-1].sample_stop, b*2000)
            for r in regions:
                if r.kind == 'speech':
                    self.assertEqual(r.source_offset, r.sample_start-review.ONSET)
                    self.assertEqual(r.source_sha256, probe.sha(pcm[r.sample_start*2:r.sample_stop*2]))
                else:
                    self.assertIsNone(r.source_id)

    def test_pcm_rejects_tamper_and_wrong_source(self):
        with patch.multiple(review, SAMPLES=8, ONSET=2, SPEECH_SAMPLES=3):
            pcm = bytes(4)+b'abcdef'+bytes(6)
            with patch.object(review, 'PCM_SHA', probe.sha(pcm)):
                review.validate_pcm(pcm, b'abcdef')
                with self.assertRaises(ValueError):
                    review.validate_pcm(pcm, b'abcdeg')
                with self.assertRaises(ValueError):
                    review.validate_pcm(pcm[:-1]+b'x', b'abcdef')

    def test_permission_gate(self):
        with patch.object(review, 'inspect') as inspect:
            with self.assertRaises(ValueError):
                review.run('unused', audio_binary='unused')
            inspect.assert_not_called()

    def test_relative_path_and_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.chmod(0o755)
            with self.assertRaises(ValueError):
                review.inspect(root)
            root.chmod(0o700)
            (root/'manifest.json').write_bytes(b'{}')
            (root/'manifest.json').chmod(0o400)
            (root/'manifest.sha256').write_text('0'*64+'\n')
            (root/'manifest.sha256').chmod(0o400)
            with self.assertRaisesRegex(ValueError, 'seal'):
                review.inspect(root)
            self.assertEqual(probe.private_directory(root), root.resolve())

    def test_max_frames_and_one_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = LiveAVBinding('test', 'a'*64, 0, max_segments=23, max_total_frames=2368)
            sink = MagicMock()
            sink.audio_evidence = {'source_pcm_samples': review.SAMPLES}
            sink.timing = {'frames':2368}
            pin = dict(audio_binary='/fake/binary', binary_sha256='b'*64)
            with patch.object(review, 'inspect', return_value=(root,'c'*64,binding,[(root/'segment-0.mp4',object())])), patch('research.run_h3_relay_conversation.direct_audio_pin', return_value=pin), patch('research.h3_personal.split_av_sink.SplitAVSink', return_value=sink) as factory:
                result = review.run(root, audio_binary='relative-binary', playback_approved=True)
                self.assertEqual(result['status'], 'finished')
                self.assertEqual(factory.call_args.kwargs['max_total_frames'],2368)
                sink.accept.assert_called_once()
                sink.close.assert_called_once()
                with self.assertRaises(FileExistsError):
                    review.run(root, audio_binary='relative-binary', playback_approved=True)

    def test_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = LiveAVBinding('test','a'*64,0,max_segments=23,max_total_frames=2368)
            sink = MagicMock()
            sink.finish.side_effect = RuntimeError('private detail')
            sink.audio_evidence = {}
            sink.timing = {}
            with patch.object(review,'inspect',return_value=(root,'c'*64,binding,[])), patch('research.run_h3_relay_conversation.direct_audio_pin',return_value=dict(audio_binary='/fake',binary_sha256='b'*64)), patch('research.h3_personal.split_av_sink.SplitAVSink',return_value=sink):
                result = review.run(root,audio_binary='relative',playback_approved=True)
            self.assertEqual(result['error'],'long-review-playback-failed')
            self.assertIn('audio_evidence', result)
            sink.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
