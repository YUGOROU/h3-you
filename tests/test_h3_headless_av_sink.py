import hashlib
import io
import json
import os
import signal
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

from research.h3_personal.av_delivery import Binding, Delivery, Segment
from research.h3_personal.conversation_state import SpeechSource
from research.h3_personal.headless_av_sink import HeadlessAVSink


def run(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE, timeout=15)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for i, color in enumerate(('red', 'blue')):
            run('ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                f'color={color}:s=16x16:r=24', '-frames:v', '2',
                '-c:v', 'ffv1', str(self.root / f'{i}.mkv'))
        # Generated in-memory WAV; lease receives its unmodified PCM payload.
        buf = io.BytesIO()
        pcm = b''.join(struct.pack('<h', i % 30000) for i in range(1200))
        with wave.open(buf, 'wb') as w:
            w.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            w.writeframes(pcm)
        with wave.open(io.BytesIO(buf.getvalue()), 'rb') as w:
            self.pcm = w.readframes(w.getnframes())
        self.binding = Binding('test', 'source', SpeechSource(
            hashlib.sha256(self.pcm).hexdigest(), 8000, 1200), 'a'*64)
        self.delivery = Delivery(self.binding, self.pcm)

    def tearDown(self):
        self.tmp.cleanup()

    def lease(self, i):
        path = self.root / f'{i}.mkv'
        self.delivery.enqueue(Segment(self.binding, i, i*2, i*2+2,
                                      hashlib.sha256(path.read_bytes()).hexdigest()))
        return self.delivery.lease()

    def sink(self, **kw):
        return HeadlessAVSink(self.root, self.binding, width=16, height=16, **kw)

    def assert_reaped(self, sink):
        self.assertTrue(all(p.poll() is not None for p in sink._processes))
        self.assertTrue(all(not t.is_alive() for t in sink._writers))

    def test_continuous_original_pcm_and_pts(self):
        with self.sink() as sink:
            first = sink.accept('0.mkv', self.lease(0))
            self.assertFalse(self.delivery.closed)
            self.assertFalse(sink.finished)
            self.assertIsNone(sink._encoder.poll())
            self.delivery.ack(self.delivery._lease)
            second_lease = self.lease(1)
            second = sink.accept('1.mkv', second_lease)
            self.assertEqual(first.first_av_submission_monotonic,
                             second.first_av_submission_monotonic)
            self.assertLessEqual(first.accepted_monotonic, time.monotonic())
            output = sink.finish()
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            pcm = run('ffmpeg', '-v', 'error', '-i', str(output), '-map', '0:a',
                      '-f', 's16le', 'pipe:1')
            self.assertEqual(pcm, self.pcm + bytes(second_lease.segment.padding_samples*2))
            probe = json.loads(run('ffprobe', '-v', 'error', '-select_streams', 'v',
                                  '-show_frames', '-of', 'json', str(output)))
            pts = [float(f['pts_time'] if 'pts_time' in f else f['pkt_pts_time']) for f in probe['frames']]
            self.assertEqual(len(pts), 4)
            for i, pts_value in enumerate(pts):
                self.assertAlmostEqual(pts_value, i/24, places=5)
            rgb = run('ffmpeg', '-v', 'error', '-i', str(output), '-map', '0:v',
                      '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
            self.assertGreater(rgb[0], rgb[2])
            self.assertGreater(rgb[2*16*16*3+2], rgb[2*16*16*3])
        self.assert_reaped(sink)

    def test_full_resolution_delayed_second_producer(self):
        for frames, deadline in ((2, 2), (119, 8)):
            with self.subTest(frames=frames):
                count = (2*frames*48000+23)//24
                pcm = b''.join(struct.pack('<h', (i*37)%60001-30000)
                               for i in range(count))
                binding = Binding('full', 'source', SpeechSource(
                    hashlib.sha256(pcm).hexdigest(), 48000, count), 'a'*64)
                delivery = Delivery(binding, pcm)
                run('ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc=s=252x448:r=24', '-frames:v', str(frames),
                    '-c:v', 'ffv1', str(self.root/'full0.mkv'))
                first_done = threading.Event()
                errors, results = [], []
                with HeadlessAVSink(self.root, binding) as sink:
                    def produce():
                        try:
                            for i in range(2):
                                # The second input does not exist until accept returns.
                                path = self.root/f'full{i}.mkv'
                                if i:
                                    self.assertTrue(first_done.is_set())
                                    time.sleep(.05)
                                    shutil.copyfile(self.root/'full0.mkv', path)
                                delivery.enqueue(Segment(binding, i, i*frames,
                                    (i+1)*frames, hashlib.sha256(path.read_bytes()).hexdigest()))
                                lease = delivery.lease()
                                started = time.monotonic()
                                results.append(sink.accept(path.name, lease))
                                if not i:
                                    results.append(time.monotonic()-started)
                                    first_done.set()
                                delivery.ack(lease)
                        except BaseException as exc:
                            errors.append(exc)
                    second = self.root/'full1.mkv'
                    second.unlink(missing_ok=True)
                    worker = threading.Thread(target=produce)
                    worker.start()
                    timely = first_done.wait(deadline)
                    if not timely:
                        sink.cancel()
                    worker.join(15)
                    if worker.is_alive():
                        sink.cancel()
                        worker.join(3)
                    self.assertFalse(worker.is_alive())
                    self.assertTrue(timely, f'first accept blocked: {errors}')
                    self.assertFalse(errors, errors)
                    self.assertLess(results[1], deadline)
                    output = sink.finish()
                    self.assertEqual(run('ffmpeg', '-v', 'error', '-i', str(output),
                        '-map', '0:a', '-f', 's16le', 'pipe:1'), pcm)
                    actual = run('ffmpeg', '-v', 'error', '-i', str(output),
                        '-map', '0:v', '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
                    expected = run('ffmpeg', '-v', 'error', '-i',
                        str(self.root/'full0.mkv'), '-vsync', '0', '-pix_fmt', 'rgb24',
                        '-f', 'rawvideo', 'pipe:1')
                    self.assertEqual(actual, expected+expected)
                    probe = json.loads(run('ffprobe', '-v', 'error',
                        '-show_frames', '-select_streams', 'v', '-of', 'json', str(output)))
                    self.assertEqual(len(probe['frames']), 2*frames)
                    for i, frame in enumerate(probe['frames']):
                        self.assertAlmostEqual(float(frame['pts_time'] if 'pts_time' in frame else frame['pkt_pts_time']), i/24, places=5)
                self.assert_reaped(sink)

    def test_paced_strict_bool(self):
        for value in (None, 0, 1, 'true'):
            with self.assertRaisesRegex(ValueError, 'bool'):
                self.sink(paced=value)
        with self.sink() as sink:
            self.assertFalse(sink.paced)
            self.assertEqual(sink.timing['ack_count'], 0)
            with self.assertRaises(TypeError):
                sink.timing['paced'] = True

    def test_real_clock_paced_exact_media(self):
        for frames in (6, 12):
            with self.subTest(frames=frames):
                count = frames * 1000  # 24kHz / 24fps
                pcm = b''.join(struct.pack('<h', i % 30000) for i in range(count))
                binding = Binding('paced', 'source', SpeechSource(
                    hashlib.sha256(pcm).hexdigest(), 24000, count), 'a'*64)
                delivery = Delivery(binding, pcm)
                path = self.root/'paced.mkv'
                run('ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc=s=16x16:r=24', '-frames:v', str(frames),
                    '-c:v', 'ffv1', str(path))
                delivery.enqueue(Segment(binding, 0, 0, frames,
                    hashlib.sha256(path.read_bytes()).hexdigest()))
                with HeadlessAVSink(self.root, binding, width=16, height=16,
                                    paced=True, timeout=8) as sink:
                    started = time.monotonic()
                    submission = sink.accept(path.name, delivery.lease())
                    output = sink.finish()
                    self.assertGreaterEqual(time.monotonic()-started, frames/24-.15)
                    self.assertLessEqual(submission.first_av_submission_monotonic,
                                         submission.accepted_monotonic)
                    timing = sink.timing
                    self.assertEqual(timing['first_av_submission_monotonic'], max(
                        timing['first_video_bytes_write_monotonic'],
                        timing['first_audio_bytes_write_monotonic']))
                    self.assertEqual(timing['ack_count'], 1)
                    self.assertEqual(run('ffmpeg', '-v', 'error', '-i', str(output),
                        '-map', '0:a', '-f', 's16le', 'pipe:1'), pcm)
                    actual = run('ffmpeg', '-v', 'error', '-i', str(output),
                        '-map', '0:v', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
                    expected = run('ffmpeg', '-v', 'error', '-i', str(path),
                        '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(actual), frames*16*16*3)
                self.assert_reaped(sink)

    def test_independent_producer_observes_first_write_before_ack(self):
        self.large_video()
        lease = self.lease(0)
        errors, results, arrivals = [], [], []
        with HeadlessAVSink(self.root, self.binding, paced=True, timeout=8) as sink:
            def submit():
                try:
                    results.append(sink.accept('0.mkv', lease))
                except BaseException as exc:
                    errors.append(exc)
            def ready():
                return getattr(sink, 'first_av_submission_monotonic', None)
            # Inject slow-consumer backpressure only here: after initial positive
            # AV writes, hold a subsequent video write until the independent
            # producer creates its artifact. Normal paced/full-119-frame tests
            # above exercise the unmocked route, without this scheduling barrier.
            release = threading.Event()
            both_written = threading.Event()
            positive = set()
            write_lock = threading.Lock()
            real_write = os.write

            def slow_write(fd, data):
                if (threading.current_thread().name != 'h3-headless-writer'
                        or fd not in sink._pipes):
                    return real_write(fd, data)
                with write_lock:
                    subsequent_video = fd == sink._pipes[0] and fd in positive
                if subsequent_video:
                    if not both_written.wait(2):
                        raise TimeoutError('initial AV writes did not arrive')
                    if not release.wait(2):
                        raise TimeoutError('independent producer did not release video')
                written = real_write(fd, data)
                if written > 0:
                    with write_lock:
                        positive.add(fd)
                        if len(positive) == 2:
                            both_written.set()
                return written

            worker = threading.Thread(target=submit)
            with patch('research.h3_personal.headless_av_sink.os.write', slow_write):
                try:
                    worker.start()
                    deadline = time.monotonic()+5
                    while not ready() and time.monotonic() < deadline:
                        time.sleep(.0005)
                    if ready():
                        # Independent producer creates the next artifact, not another accept.
                        shutil.copyfile(self.root/'0.mkv', self.root/'1.mkv')
                        arrivals.append(time.monotonic())
                finally:
                    release.set()
                    worker.join(8)
                    if worker.is_alive():
                        sink.cancel()
                        worker.join(2)
            # Restore os.write after joining, before the next accept and finish.
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors, errors)
            self.assertTrue(arrivals)
            self.assertLessEqual(ready(), arrivals[0])
            self.assertLess(arrivals[0], results[0].accepted_monotonic)
            self.delivery.ack(lease)
            sink.accept('1.mkv', self.lease(1))
            sink.finish()
        self.assert_reaped(sink)

    def test_process_failure_is_classified_without_private_stderr(self):
        import sys
        with self.sink() as sink:
            p = sink._spawn([sys.executable, '-c',
                "import sys; sys.stderr.write('Unrecognized option PRIVATE_PATH'); sys.exit(2)"])
            with self.assertRaisesRegex(RuntimeError,
                    'exit=2, category=unsupported-option') as raised:
                sink._wait(p, time.monotonic()+5)
            self.assertNotIn('PRIVATE_PATH', str(raised.exception))
        self.assertTrue(p._h3_errors.closed)
        self.assert_reaped(sink)

    def test_closed_sink_cannot_spawn_after_reaping(self):
        sink = self.sink()
        sink.cancel()
        count = len(sink._processes)
        with patch('research.h3_personal.headless_av_sink.subprocess.Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'process creation'):
                sink._spawn(['ffmpeg', '-version'])
            popen.assert_not_called()
        self.assertEqual(len(sink._processes), count)
        self.assert_reaped(sink)

    def test_bad_hash_symlink_and_path(self):
        for name in ('../0.mkv', 'link.mkv', 'bad.mkv'):
            with self.subTest(name=name):
                self.delivery = Delivery(self.binding, self.pcm)
                lease = self.lease(0)
                (self.root/'bad.mkv').write_bytes(b'not video')
                link = self.root/'link.mkv'
                if not link.exists(): link.symlink_to(self.root/'0.mkv')
                with self.sink() as sink:
                    with self.assertRaises(ValueError): sink.accept(name, lease)
                self.assert_reaped(sink)

    def test_count_mismatch_and_collision(self):
        lease = self.lease(0)
        with self.sink() as sink:
            (sink.directory/'segment.video').write_bytes(b'collision')
            with self.assertRaises(FileExistsError): sink.accept('0.mkv', lease)
        self.assert_reaped(sink)
        with self.sink(max_frames=1) as sink:
            with self.assertRaises(ValueError): sink.accept('0.mkv', lease)
        self.assert_reaped(sink)

    def test_wrong_decoded_frame_count(self):
        path = self.root/'0.mkv'
        segment = Segment(self.binding, 0, 0, 1, hashlib.sha256(path.read_bytes()).hexdigest())
        self.delivery.enqueue(segment)
        with self.sink() as sink:
            with self.assertRaisesRegex(ValueError, 'frame count'):
                sink.accept('0.mkv', self.delivery.lease())
        self.assert_reaped(sink)

    def test_rejects_shape_rate_and_embedded_audio(self):
        for shape, rate, audio in [('448x252', 24, False),
                                   ('16x16', 24, False),
                                   ('252x448', 30, False),
                                   ('252x448', 24, True)]:
            with self.subTest(shape=shape, rate=rate, audio=audio):
                args = ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi',
                        '-i', f'color=red:s={shape}:r={rate}']
                if audio:
                    args += ['-f', 'lavfi', '-i', 'anullsrc=r=8000:cl=mono',
                             '-c:a', 'pcm_s16le']
                run(*args, '-frames:v', '2', '-c:v', 'ffv1',
                    str(self.root/'0.mkv'))
                self.delivery = Delivery(self.binding, self.pcm)
                with HeadlessAVSink(self.root, self.binding) as sink:
                    with self.assertRaises(ValueError):
                        sink.accept('0.mkv', self.lease(0))
                    self.assertFalse(sink._writers)
                self.assert_reaped(sink)

    def test_global_source_pts_reset_to_continuous_clock(self):
        for i in range(2):
            run('ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                'color=red:s=16x16:r=24', '-vf', f'setpts=PTS+{5+i}/TB',
                '-frames:v', '2', '-c:v', 'ffv1', str(self.root/f'{i}.mkv'))
        self.test_continuous_original_pcm_and_pts_offset()

    def test_continuous_original_pcm_and_pts_offset(self):
        with self.sink() as sink:
            for i in range(2):
                lease = self.lease(i)
                sink.accept(f'{i}.mkv', lease)
                self.delivery.ack(lease)
            output = sink.finish()
            probe = json.loads(run('ffprobe', '-v', 'error', '-select_streams', 'v',
                                  '-show_frames', '-of', 'json', str(output)))
            self.assertEqual(len(probe['frames']), 4)
            for i, frame in enumerate(probe['frames']):
                self.assertAlmostEqual(float(frame['pts_time'] if 'pts_time' in frame else frame['pkt_pts_time']), i/24, places=5)

    def large_video(self):
        run('ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'color=red:s=252x448:r=24', '-frames:v', '2', '-c:v', 'ffv1',
            str(self.root/'0.mkv'))

    def test_blocked_pipe_timeout_reaps(self):
        self.large_video()
        lease = self.lease(0)
        with HeadlessAVSink(self.root, self.binding, timeout=.5) as sink:
            os.kill(sink._encoder.pid, signal.SIGSTOP)
            with self.assertRaises(TimeoutError):
                sink.accept('0.mkv', lease)
        self.assert_reaped(sink)

    def test_cancel_during_pipe_write(self):
        self.large_video()
        lease = self.lease(0)
        with HeadlessAVSink(self.root, self.binding, timeout=3) as sink:
            os.kill(sink._encoder.pid, signal.SIGSTOP)
            errors = []
            def submit():
                try: sink.accept('0.mkv', lease)
                except BaseException as exc: errors.append(exc)
            worker = threading.Thread(target=submit)
            worker.start()
            deadline = time.monotonic()+2
            while not sink._writers and time.monotonic() < deadline:
                time.sleep(.005)
            sink.cancel()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertTrue(errors)
        self.assert_reaped(sink)

    def test_timeout_and_cancel_reap(self):
        lease = self.lease(0)
        with self.sink(timeout=.000001) as sink:
            with self.assertRaises((TimeoutError, ValueError)):
                sink.accept('0.mkv', lease)
        self.assert_reaped(sink)
        with self.sink() as sink:
            sink.cancel()
            with self.assertRaises(RuntimeError): sink.accept('0.mkv', lease)
        self.assert_reaped(sink)
        self.assertFalse(any(t.name == 'h3-headless-writer' for t in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
