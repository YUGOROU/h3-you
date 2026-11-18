"""Synthetic device-free paired epoch invalidation, not acoustic acceptance."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

from research.h3_personal.epoch_av_sink import EpochAVSink, PresentationInvalidated
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, ReceivedAVLease, encode


class Child:
    def __init__(self, *args, **kwargs):
        self.origin = kwargs['discontinuity_origin']
        self.first_av_submission_monotonic = time.monotonic()
        self.stopped = False; self.accepted = []
    def accept(self, path, lease):
        self.accepted.append(lease.segment.index)
        return lease.segment.frame_stop
    def cancel(self):
        self.stopped = True
    close = cancel
    def finish(self):
        return 'private-output'


class EpochSinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binding = LiveAVBinding('session', 'a'*64, 1, max_segments=8)

    def lease(self, index, start, stop, epoch, *, source=None, video=b'video'):
        source = epoch if source is None else source
        pcm = int(source).to_bytes(2, 'little')*((stop-start)*2000)
        region = AudioRegion('speech', start*2000, stop*2000, f'speech-{source}',
                             source, 0, hashlib.sha256(pcm).hexdigest())
        envelope = encode(self.binding, index, start, stop, video, pcm, (region,), False,
                          presentation_epoch=epoch)
        return ReceivedAVLease(envelope, video, pcm)

    def test_invalidate_discard_and_reentry_without_clock_rewind(self):
        children = []
        def factory(*args, **kwargs):
            child = Child(*args, **kwargs); children.append(child); return child
        with EpochAVSink(self.root, self.binding, sink_factory=factory) as sink:
            sink.accept('unused', self.lease(0, 0, 2, 1))
            mark = sink.invalidate(2, interrupted_source_epoch=1)
            self.assertLess(mark['invalidated_monotonic']-mark['detected_monotonic'], .2)
            self.assertTrue(children[0].stopped)
            old = self.lease(1, 2, 4, 1)
            self.assertTrue(sink.should_discard(old)); sink.discard(old)
            sink.accept('unused', self.lease(2, 4, 6, 2))
            self.assertEqual(children[1].origin, (2, 4))
            self.assertEqual(sink.timing['global_frame'], 6)
            self.assertEqual(sink.timing['discarded_interrupted'], 1)
            self.assertFalse(sink.timing['presentation_complete'])
            self.assertEqual(sink.finish(), 'private-output')

    def test_mixed_old_tail_is_rejected_even_with_new_tag(self):
        with EpochAVSink(self.root, self.binding, sink_factory=Child) as sink:
            sink.invalidate(2, interrupted_source_epoch=1)
            mixed = self.lease(0, 0, 2, 2, source=1)
            self.assertTrue(sink.should_discard(mixed))
            with self.assertRaises(PresentationInvalidated):
                sink.accept('unused', mixed)
            sink.discard(mixed)
            with self.assertRaises(ValueError):
                sink.discard(self.lease(1, 2, 4, 2))

    def test_inflight_pair_writes_are_invalidated_not_terminal_session_cancel(self):
        entered, release = threading.Event(), threading.Event()
        class Blocked(Child):
            def accept(self, *args):
                entered.set(); release.wait(2)
                if self.stopped:
                    raise RuntimeError('encoder stopped')
                return super().accept(*args)
            def cancel(self):
                super().cancel(); release.set()
        with EpochAVSink(self.root, self.binding, sink_factory=Blocked) as sink:
            old = self.lease(0, 0, 2, 1); errors = []
            def accept():
                try:sink.accept('unused', old)
                except BaseException as exc:errors.append(exc)
            thread = threading.Thread(target=accept); thread.start()
            self.assertTrue(entered.wait(1))
            sink.invalidate(2, interrupted_source_epoch=1)
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], PresentationInvalidated)
            sink.discard(old)
            sink.accept('unused', self.lease(1, 2, 4, 2))
            self.assertEqual(sink.timing['accepted'], 1)

    def test_old_epoch_cannot_spawn_after_invalidation_during_creation(self):
        entered, release = threading.Event(), threading.Event(); children = []
        def factory(*args, **kwargs):
            entered.set(); release.wait(2)
            child = Child(*args, **kwargs); children.append(child); return child
        with EpochAVSink(self.root, self.binding, sink_factory=factory) as sink:
            errors = []
            def accept():
                try:sink.accept('unused', self.lease(0, 0, 2, 1))
                except BaseException as exc:errors.append(exc)
            thread = threading.Thread(target=accept); thread.start()
            self.assertTrue(entered.wait(1))
            sink.invalidate(2, interrupted_source_epoch=1)
            release.set(); thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(errors[0], PresentationInvalidated)
            self.assertTrue(children[0].stopped)
            self.assertFalse(children[0].accepted)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_actual_decoded_output_stops_and_new_source_is_exact(self):
        from research.run_h3_realtime_first import encode_rgb_segment
        from research.h3_personal.conversation_scenario import segment_timing
        videos = []
        for index, (start, stop) in enumerate(((0, 102), (102, 107))):
            path = self.root/f'observed-{index}.mp4'
            encode_rgb_segment(bytes([40,20,90])*(252*448*(stop-start)), path,
                segment_timing(index,start,stop),width=252,height=448)
            videos.append(path)
        with EpochAVSink(self.root,self.binding,max_frames=124,timeout=15,
                         paced=True,observe_output=True) as sink:
            old = self.lease(0,0,102,1,video=videos[0].read_bytes()); errors=[]
            def accept():
                try:sink.accept(videos[0].name,old)
                except BaseException as exc:errors.append(exc)
            thread=threading.Thread(target=accept);thread.start()
            sink.wait_for_speech_output(1,time.monotonic()+10)
            time.sleep(.1)
            mark=sink.invalidate(2,interrupted_source_epoch=1)
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue(mark['actual_decoded_sink_observed'])
            self.assertIsNotNone(mark['old_sink_eof_monotonic'])
            self.assertLess(mark['stop_upper_bound_seconds'],.2)
            self.assertEqual(len(errors),1)
            self.assertIsInstance(errors[0],PresentationInvalidated)
            sink.discard(old)
            new=self.lease(1,102,107,2,video=videos[1].read_bytes())
            sink.accept(videos[1].name,new);sink.finish()
            actual=sink.timing['decoded_source_output']['2']
            self.assertEqual(actual['samples'],10000)
            self.assertEqual(actual['sha256'],hashlib.sha256(new.audio_pcm).hexdigest())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_actual_encoder_reentry_has_exact_new_pcm_and_explicit_pts_origin(self):
        from research.run_h3_realtime_first import encode_rgb_segment
        from research.h3_personal.conversation_scenario import segment_timing
        import json
        from fractions import Fraction
        leases = []
        for index, start in enumerate((0, 5, 10)):
            path = self.root/f'{index}.mp4'
            encode_rgb_segment(bytes([40, 20, 90])*(252*448*5), path,
                segment_timing(index, start, start+5), width=252, height=448)
            leases.append(self.lease(index, start, start+5, 1 if index < 2 else 2,
                                     video=path.read_bytes()))
        with EpochAVSink(self.root, self.binding, max_frames=124, timeout=20) as sink:
            sink.accept('0.mp4', leases[0])
            sink.invalidate(2, interrupted_source_epoch=1)
            sink.discard(leases[1])
            sink.accept('2.mp4', leases[2])
            output = sink.finish()
            audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=20)
            self.assertEqual(audio, leases[2].audio_pcm)
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                '-select_streams', 'v', '-show_frames', '-show_streams', '-of', 'json',
                str(output)], timeout=20))
            self.assertEqual(len(probe['frames']), 5)
            self.assertEqual(Fraction(probe['frames'][0]['pts']) *
                             Fraction(probe['streams'][0]['time_base']), 0)
            self.assertEqual(sink._child.timing['global_frame_origin'], 10)
            self.assertEqual(sink.timing['global_frame'], 15)


if __name__ == '__main__':
    unittest.main()
