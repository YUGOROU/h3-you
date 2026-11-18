import itertools
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

from research.h3_personal.realtime_segments import (
    PipelineCancelled, SegmentPipeline, segment_timing,
)


class RealtimeSegmentsTests(unittest.TestCase):
    def test_integer_timeline(self):
        timings = [segment_timing(i) for i in range(3)]
        self.assertEqual([t.head_trim_frames for t in timings], [0, 1, 1])
        self.assertEqual([t.newly_emitted_frames for t in timings], [124, 123, 123])
        self.assertEqual([(t.pts_start, t.pts_end) for t in timings],
                         [(0, 124), (124, 247), (247, 370)])
        for invalid in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                segment_timing(invalid)

    def test_order_clock_and_publication(self):
        clock = itertools.count(100).__next__
        seen = []
        owner = threading.get_ident()

        def encode(job, event):
            self.assertNotEqual(threading.get_ident(), owner)
            seen.append(job)
            return {'value': job}

        pipe = SegmentPipeline(encode, publish=lambda result, event: result,
                               clock=clock)
        for i in range(5):
            pipe.submit(i)
        reports = pipe.close()
        self.assertEqual(seen, list(range(5)))
        self.assertEqual([r['segment_id'] for r in reports], list(range(5)))
        self.assertGreater(pipe.first_publish_seconds, 0)
        for r in reports:
            self.assertLessEqual(r['submitted_at'], r['encode_started_at'])
            self.assertLess(r['encode_started_at'], r['encode_finished_at'])
            self.assertLess(r['encode_finished_at'], r['publication_finished_at'])
        reports[0]['value'] = 'changed'
        self.assertEqual(pipe.reports[0]['value'], 0)
        self.assertEqual(len(pipe.close()), 5)
        with self.assertRaises(RuntimeError):
            pipe.submit(6)

    def test_bounded_queue_backpressure(self):
        entered, release, attempting = threading.Event(), threading.Event(), threading.Event()

        def encode(job, event):
            if job == 0:
                entered.set()
                self.assertTrue(release.wait(2))
            return {}

        def unblock():
            self.assertTrue(attempting.wait(2))
            # No third queued job fits while first callback is held.
            self.assertEqual(pipe._queue.qsize(), 1)
            release.set()

        pipe = SegmentPipeline(encode)
        pipe.submit(0)
        self.assertTrue(entered.wait(2))
        pipe.submit(1)
        helper = threading.Thread(target=unblock)
        helper.start()
        attempting.set()
        pipe.submit(2)
        helper.join(2)
        self.assertEqual(len(pipe.close()), 3)
        self.assertGreater(pipe.producer_blocked_seconds, 0)

    def test_failure_close_and_no_replay(self):
        calls = []
        error = ValueError('encode failed')

        def fail(job, event):
            calls.append(job)
            raise error

        pipe = SegmentPipeline(fail)
        try:
            pipe.submit(0)
        except ValueError:
            pass
        with self.assertRaisesRegex(ValueError, 'encode failed'):
            pipe.close()
        self.assertEqual(calls, [0])
        self.assertEqual(pipe.reports, [])

    def test_cancel_cooperative_callback(self):
        entered = threading.Event()

        def encode(job, event):
            entered.set()
            self.assertTrue(event.wait(2))
            return {}

        pipe = SegmentPipeline(encode)
        pipe.submit(None)
        self.assertTrue(entered.wait(2))
        pipe.cancel()
        with self.assertRaises(PipelineCancelled):
            pipe.close()
        self.assertFalse(pipe._thread.is_alive())
        self.assertEqual(pipe.reports, [])

    def test_bounded_join_uncooperative_callback(self):
        entered, release = threading.Event(), threading.Event()

        def encode(job, event):
            entered.set()
            release.wait(2)
            return {}

        pipe = SegmentPipeline(encode, join_timeout=0.01)
        pipe.submit(None)
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaises(TimeoutError):
                pipe.close()
            self.assertTrue(pipe.cancel_event.is_set())
        finally:
            release.set()
            pipe._thread.join(2)
        self.assertEqual(pipe.reports, [])

    def test_ids_and_callback_contract(self):
        pipe = SegmentPipeline(lambda job, event: None)
        with self.assertRaises(ValueError):
            pipe.submit(None, segment_id=1)
        try:
            pipe.submit(None)
        except TypeError:
            pass
        with self.assertRaises(TypeError):
            pipe.close()

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_cpu_ffmpeg_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            def encode(job, event):
                path = Path(directory) / f'{job}.mp4'
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                                'color=c=black:s=16x16:r=24', '-frames:v',
                                str(segment_timing(job).newly_emitted_frames),
                                '-c:v', 'mpeg4', '-y', str(path)],
                               check=True, timeout=15, capture_output=True)
                return {'path': str(path)}

            pipe = SegmentPipeline(encode)
            pipe.submit(0)
            pipe.submit(1)
            reports = pipe.close()
            for count, report in zip((124, 123), reports):
                result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
                                         '-select_streams', 'v:0', '-show_entries',
                                         'stream=nb_read_frames', '-of',
                                         'default=nw=1:nk=1', report['path']],
                                        check=True, capture_output=True, text=True, timeout=15)
                self.assertEqual(int(result.stdout.strip()), count)


if __name__ == '__main__':
    unittest.main()
