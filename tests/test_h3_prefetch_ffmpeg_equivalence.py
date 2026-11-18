"""Device-free FFmpeg equivalence: serialized receive vs bounded prefetch.

No GPU, provider, device or external network beyond a local socketpair and local
files. Prefetch must not change decoded PCM, frame order/count, PTS or final ACK.
"""
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest

from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av_pump import ProducerPump, ReceiverPump

SEGMENTS = (5, 7, 6)


def thread_run(target, results, errors):
    def body():
        try:
            results.append(target())
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=body)
    thread.start()
    return thread


def build(root, binding):
    from research.run_h3_realtime_first import encode_rgb_segment
    from research.h3_personal.conversation_scenario import segment_timing
    packets = []
    cursor = 0
    for index, frames in enumerate(SEGMENTS):
        path = root / f'src{index}.mp4'
        encode_rgb_segment(bytes([index * 80 + 25, 50, 100]) * (252 * 448 * frames), path,
                           segment_timing(index, cursor, cursor + frames), width=252, height=448)
        video = path.read_bytes()
        pcm = bytes([index + 1]) * (frames * 2000 * 2)
        region = AudioRegion('speech', cursor * 2000, (cursor + frames) * 2000,
                             'source', 1, cursor * 2000, hashlib.sha256(pcm).hexdigest())
        envelope = encode(binding, index, cursor, cursor + frames, video, pcm,
                          (region,), index == len(SEGMENTS) - 1)
        packets.append(Packet(envelope, video, pcm))
        cursor += frames
    return packets


def decoded(output):
    audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                                     '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=60)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v',
                                               '-show_frames', '-show_streams', '-of', 'json',
                                               str(output)], timeout=60))
    tick = Fraction(info['streams'][0]['time_base'])
    return audio, len(info['frames']), [Fraction(f['pts']) * tick for f in info['frames']]


class DelayedSink:
    """Real headless sink plus a bounded delay standing in for paced submission."""

    def __init__(self, sink, delay):
        self.sink, self.delay = sink, delay
        self.root, self.binding, self.timing = sink.root, sink.binding, sink.timing
        self.overlap_observed = False
        self.receiver = None

    def accept(self, path, lease):
        self.sink.accept(path, lease)
        if lease.segment.index == 0 and self.receiver is not None:
            self.overlap_observed = self.receiver._cv.wait_for(
                lambda: self.receiver.receiver.snapshot()['verified'] >= 2, 10)
            time.sleep(self.delay)

    def finish(self):
        return self.sink.finish()

    def cancel(self):
        return self.sink.cancel()


class EquivalenceTests(unittest.TestCase):
    def exercise(self, prefetch, accept_delay=0.0):
        from research.h3_personal.headless_av_sink import HeadlessAVSink
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as sink_dir:
            binding = LiveAVBinding('prefetch-equivalence', 'a' * 64, 1)
            packets = build(Path(src_dir), binding)
            expected_audio = b''.join(packet.pcm for packet in packets)
            deadline = time.monotonic() + 120
            sink = HeadlessAVSink(Path(sink_dir), binding, width=252, height=448,
                                  max_frames=124, timeout=60)
            wrapped = DelayedSink(sink, accept_delay) if accept_delay else sink
            a, c = socket.socketpair()
            producer = ProducerPump(a, binding, deadline=deadline)
            receiver = ReceiverPump(c, binding, sink_dir, wrapped, deadline=deadline,
                                    prefetch=prefetch)
            if accept_delay:
                wrapped.receiver = receiver
            errors, produced, consumed = [], [], []
            receiver_thread = thread_run(receiver.run, consumed, errors)
            producer_thread = thread_run(lambda: producer.run(iter(packets)), produced, errors)
            try:
                producer_thread.join(150)
                receiver_thread.join(150)
            finally:
                if receiver_thread.is_alive() or producer_thread.is_alive():
                    receiver.cancel()
                producer_thread.join(10)
                receiver_thread.join(10)
            try:
                self.assertFalse(producer_thread.is_alive())
                self.assertFalse(receiver_thread.is_alive())
                self.assertFalse(errors, errors)
                self.assertTrue(receiver.receiver.snapshot()['complete'])
                self.assertEqual(consumed[0]['sink_acked'], len(SEGMENTS))
                self.assertEqual(produced[0]['acked'], len(SEGMENTS))
                self.assertEqual(sink.timing['ack_count'], len(SEGMENTS))
                audio, frames, pts = decoded(consumed[0]['output'])
                self.assertEqual(audio, expected_audio)
                self.assertEqual(frames, sum(SEGMENTS))
                self.assertEqual(pts, [Fraction(i, 24) for i in range(sum(SEGMENTS))])
                timing = consumed[0]['timing']
                self.assertEqual(timing['prefetch'], prefetch)
                marks = timing['receive']
                self.assertEqual([m['index'] for m in marks], list(range(len(SEGMENTS))))
                for mark in marks:
                    self.assertEqual(set(mark), {'index', 'received', 'verified',
                                                 'accept_start', 'accepted', 'sink_acked'})
                    self.assertLessEqual(mark['received'], mark['verified'])
                    self.assertLessEqual(mark['verified'], mark['accept_start'])
                    self.assertLessEqual(mark['accept_start'], mark['accepted'])
                    self.assertLessEqual(mark['accepted'], mark['sink_acked'])
                if prefetch:
                    # Next window arrives while the current one is still submitting.
                    self.assertLess(marks[1]['received'], marks[0]['accepted'])
                else:
                    self.assertGreater(marks[1]['received'], marks[0]['accepted'])
            finally:
                sink.close()
            return wrapped

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_prefetch_matches_serialized_exact_pcm_and_pts(self):
        self.exercise(False)
        self.exercise(True)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_prefetch_receives_next_window_during_sink_accept(self):
        wrapped = self.exercise(True, accept_delay=0.5)
        # Window 1 was already received and verified while window 0 was still
        # inside sink.accept; that is the overlap, not an early sink ACK.
        self.assertTrue(wrapped.overlap_observed)


if __name__ == '__main__':
    unittest.main()
