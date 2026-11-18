import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from unittest.mock import patch

from research.h3_personal.accepted_profiles import AcceptedQualityProfile
from research.h3_personal.headless_av_sink import HeadlessAVSink
from research.h3_personal.incremental_av import render_incremental_av
from research.h3_personal.live_voice_runtime import RenderResult


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.wav = self.root / 'source.wav'
        self.pcm = b'\x01\x00' * 9600  # .2s; six frames cover all samples.
        with wave.open(str(self.wav), 'wb') as w:
            w.setparams((1, 2, 48000, 0, 'NONE', ''))
            w.writeframes(self.pcm)
        self.digest = hashlib.sha256(self.wav.read_bytes()).hexdigest()
        self.fixture = self.root / 'fixture.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
            'color=red:s=252x448:r=24', '-frames:v', '2', '-c:v', 'ffv1',
            str(self.fixture)], check=True, timeout=10)
        self.accepted = threading.Event()
        self.sinks = []
        owner = self
        class Runtime:
            quality_profile = AcceptedQualityProfile('reference512', ('a'*64,), 'b'*64)
            paths = {'output': owner.root}
            finished = False
            bad = False
            def render_continuation(self, wav, digest, turn, *, on_segment):
                self.received_digest = digest
                turn.mkdir(mode=0o700)
                for i in range(3):
                    p = turn / f'{i}.mkv'
                    shutil.copyfile(owner.fixture, p)
                    on_segment(dict(path=str(p), session_id='actual-session', index=i,
                        frame_count=2, frame_offset=i*2, pts_start=i*2,
                        pts_end=i*2+2, time_base_numerator=1, time_base_denominator=24,
                        sha256=('0'*64 if self.bad else hashlib.sha256(p.read_bytes()).hexdigest()),
                        video_only=True))
                    if i == 0 and not self.bad:
                        if not owner.accepted.wait(10):
                            raise TimeoutError('first actual sink submission missing')
                self.finished = True
                return RenderResult(turn/'profile.json', turn/'audit.mp4')
        self.runtime = Runtime()
        class ObservedSink(HeadlessAVSink):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                owner.sinks.append(self)
            def accept(self, *a):
                try:
                    result = super().accept(*a)
                except BaseException as exc:
                    owner.accept_error = exc
                    owner.accepted.set()
                    raise
                if not owner.runtime.finished:
                    owner.accepted.set()
                return result
        self.patch = patch('research.h3_personal.incremental_av.HeadlessAVSink', ObservedSink)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def render(self):
        return render_incremental_av(self.runtime, self.wav, self.digest,
            self.root/'turn', self.root/'continuous.nut', timeout=20)

    def assert_clean(self):
        self.assertFalse(any(t.name == 'h3-incremental-av' for t in threading.enumerate()))
        for sink in self.sinks:
            self.assertTrue(all(p.poll() is not None for p in sink._processes))

    def test_real_three_segment_cpu_end_to_end(self):
        result = self.render()
        receipt = result.delivery_receipt
        self.assertTrue(receipt['firstAVsubmittedbefore_renderreturn'])
        self.assertLessEqual(receipt['bounded_queue_highwater'], 2)
        self.assertEqual(len(receipt['ranges']), 3)
        self.assertEqual(receipt['ranges'][-1]['sample_stop'], 9600)
        audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(result.output),
            '-map', '0:a', '-f', 's16le', 'pipe:1'], timeout=10)
        self.assertEqual(audio, self.pcm + bytes(2400*2))
        self.assertEqual(self.sinks[0].binding.source.digest, hashlib.sha256(self.pcm).hexdigest())
        self.assertEqual(result.output.stat().st_mode & 0o777, 0o400)
        self.assert_clean()

    def test_sink_failure_cleanup(self):
        self.runtime.bad = True
        with self.assertRaises(RuntimeError):
            self.render()
        self.assertFalse((self.root/'continuous.nut').exists())
        self.assert_clean()

    def test_wrong_wav_digest_no_runtime(self):
        self.digest = 'a'*64
        with self.assertRaises(ValueError):
            self.render()
        self.assertFalse((self.root/'turn').exists())
        self.assert_clean()
