import hashlib
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

from research.h3_personal.conversation_av import (
    Backpressure, ConversationAVSegment, ConversationAVStream, ConversationDelivery,
)
from research.h3_personal.conversation_timeline import (
    ConversationBinding, ConversationSegment, FrozenPCMSource, TimelineSpan,
)


RATE = 48000


def pcm_source(name, samples):
    pcm = bytes((i * 7 + 3) % 251 for i in range(samples * 2))
    return FrozenPCMSource(name, pcm, RATE, 1, hashlib.sha256(pcm).hexdigest())


def binding(frames=(2, 2, 2), onset=2, name='speech'):
    boundary = lambda n: n * RATE // 24
    source = pcm_source(name, boundary(onset + frames[1]) - boundary(onset))
    spans = (TimelineSpan(0, onset, 'listening'),
             TimelineSpan(onset, onset + frames[1], 'speech', name),
             TimelineSpan(onset + frames[1], onset + frames[1] + frames[2], 'listening'))
    return ConversationBinding('conversation', (source,), spans, RATE, 1, Fraction(24))


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class ConversationAVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.videos = []

    def tearDown(self):
        self.tmp.cleanup()

    def video(self, frames, color='red'):
        path = self.root / f'v{len(self.videos)}.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            f'color={color}:s=252x448:r=24', '-frames:v', str(frames),
            '-c:v', 'ffv1', str(path)], check=True, timeout=20)
        self.videos.append(path)
        return path

    def event(self, path, index, start, stop, final=False):
        return dict(path=str(path), index=index, frame_count=stop - start,
            frame_start=start, frame_stop=stop, time_base_numerator=1,
            time_base_denominator=24, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            video_only=True, committed=True, conversation_final=final)

    def test_three_span_continuous_clock_and_original_pcm(self):
        b = binding()
        stream = ConversationAVStream(self.root, b, width=252, height=448, timeout=40)
        stream.accept(self.event(self.video(2), 0, 0, 2))
        stream.accept(self.event(self.video(2), 1, 2, 4))
        stream.accept(self.event(self.video(2, 'blue'), 2, 4, 6, final=True))
        output = stream.finish(conversation_final=True)
        receipt = stream.receipt()
        self.assertEqual(receipt['frames'], 6)
        self.assertLessEqual(receipt['bounded_queue_highwater'], 2)
        self.assertEqual(receipt['audio_producers'], 1)
        self.assertEqual([r['frame_start'] for r in receipt['ranges']], [0, 2, 4])
        self.assertEqual([[s['kind'] for s in r['slices']] for r in receipt['ranges']],
                         [['listening'], ['speech'], ['listening']])
        audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
            '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=20)
        boundary = b.boundary
        expected = (bytes((boundary(2) * 2)) + b.sources[0].pcm
                    + bytes((boundary(2) * 2)))
        self.assertEqual(audio, expected)
        self.assertEqual(receipt['ranges'][1]['slices'][0]['source_start'], 0)
        self.assertEqual(receipt['ranges'][1]['slices'][0]['source_stop'], b.sources[0].samples)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', str(output)], timeout=20)
        times = [float(x) for x in probe.split()]
        self.assertEqual(len(times), 6)
        for index, value in enumerate(times):
            self.assertAlmostEqual(value, index / 24, places=6)
        audio_pts = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'packet=pts', '-of', 'csv=p=0', str(output)], timeout=20)
        self.assertTrue(audio_pts.split())
        stream.cancel()

    def test_segment_spanning_source_boundary(self):
        b = binding()
        stream = ConversationAVStream(self.root, b, width=252, height=448, timeout=40)
        stream.accept(self.event(self.video(4), 0, 0, 4))
        stream.accept(self.event(self.video(2), 1, 4, 6, final=True))
        output = stream.finish(conversation_final=True)
        receipt = stream.receipt()
        self.assertEqual([[s['kind'] for s in r['slices']] for r in receipt['ranges']],
                         [['listening', 'speech'], ['listening']])
        audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
            '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=20)
        self.assertEqual(audio, bytes(b.boundary(2) * 2) + b.sources[0].pcm
                         + bytes(b.boundary(2) * 2))
        self.assertEqual(receipt['ranges'][0]['slices'][1]['source_id'], 'speech')
        stream.cancel()

    def test_failure_is_terminal_and_unpublished(self):
        b = binding()
        stream = ConversationAVStream(self.root, b, width=252, height=448, timeout=40)
        bad = self.event(self.video(2), 0, 0, 2)
        bad['sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            stream.accept(bad)
        stream.cancel()
        self.assertFalse(any(t.name == 'h3-conversation-av' for t in threading.enumerate()))

    def test_final_and_coverage_required(self):
        b = binding()
        stream = ConversationAVStream(self.root, b, width=252, height=448, timeout=40)
        stream.accept(self.event(self.video(2), 0, 0, 2))
        with self.assertRaises(ValueError):
            stream.finish(conversation_final=False)
        with self.assertRaises(ValueError):
            stream.finish(conversation_final=True)  # incomplete coverage
        stream.cancel()


class ConversationDeliveryTests(unittest.TestCase):
    def delivery(self):
        b = binding()
        return b, ConversationDelivery(b)

    def segment(self, b, index, start, stop, final=False):
        return ConversationAVSegment(ConversationSegment(b, index, start, stop),
                                     'b' * 64, final)

    def test_two_deep_bound_and_pcm_derivation(self):
        b, d = self.delivery()
        d.enqueue(self.segment(b, 0, 0, 2))
        d.enqueue(self.segment(b, 1, 2, 4))
        self.assertEqual(d.outstanding, 2)
        with self.assertRaises(Backpressure):
            d.enqueue(self.segment(b, 2, 4, 6))
        lease = d.lease()
        self.assertEqual(lease.segment.index, 0)
        self.assertEqual(lease.audio_pcm, bytes(b.boundary(2) * 2))
        d.ack(lease)
        lease = d.lease()
        self.assertEqual(lease.segment.index, 1)
        self.assertEqual(lease.audio_pcm, b.sources[0].pcm)
        self.assertEqual(lease.segment.sample_start, b.boundary(2))
        d.ack(lease)
        self.assertEqual(d.acked_frame, 4)

    def test_rejections_and_uncertain_ack(self):
        b, d = self.delivery()
        other = binding(name='speech')
        foreign_binding = ConversationBinding('other', other.sources, other.spans,
                                              other.sample_rate, other.channels, other.fps)
        with self.assertRaises(ValueError):
            d.enqueue(self.segment(foreign_binding, 0, 0, 2))
        with self.assertRaises(ValueError):
            d.enqueue(ConversationSegment(b, 0, 0, 2))
        d.enqueue(self.segment(b, 0, 0, 2))
        with self.assertRaises(ValueError):
            ConversationAVSegment(ConversationSegment(b, 0, 0, 2), 'b' * 64, 'yes')
        with self.assertRaises(ValueError):
            d.enqueue(self.segment(b, 1, 4, 6))
        lease = d.lease()
        with self.assertRaises(ValueError):
            d.ack(None)
        d.ack(lease, certain=False)
        self.assertEqual(d.terminal, 'ACK uncertain; replay refused')
        with self.assertRaises(RuntimeError):
            d.lease()

    def test_final_flag_must_end_conversation(self):
        b, d = self.delivery()
        with self.assertRaises(ValueError):
            d.enqueue(self.segment(b, 0, 0, 2, True))
        d.enqueue(self.segment(b, 0, 0, 2))
        d.ack(d.lease())
        d.enqueue(self.segment(b, 1, 2, 4))
        d.enqueue(self.segment(b, 2, 4, 6, True))
        d.ack(d.lease())
        d.ack(d.lease())
        d.close_producer()
        self.assertTrue(d.final_flush())

    def test_incomplete_coverage_refused(self):
        b, d = self.delivery()
        d.enqueue(self.segment(b, 0, 0, 2))
        with self.assertRaises(ValueError):
            d.close_producer()


if __name__ == '__main__':
    unittest.main()
