"""Device-free local AAC regression, not OBS/BlackHole/Comet acceptance.

The production raw-input mapping and OBS codec/mux arguments are exercised;
only the final destination is replaced. Exact PCM input-boundary coverage lives
in test_h3_headless_av_sink; lossy AAC output is deliberately not byte-compared.
"""
import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from research.h3_personal.av_delivery import Binding, Delivery, Segment
from research.h3_personal.conversation_state import SpeechSource
from research.h3_personal.headless_av_sink import HeadlessAVSink
from research.h3_personal.obs_av_sink import output_arguments


def run(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE, timeout=15)


class FileEncodedSink(HeadlessAVSink):
    """Never construct OBSAVSink (its constructor starts UDP playback)."""

    def _encoder_tail(self):
        self.encoded_output = self.directory / 'encoded.ts'
        args = output_arguments(48124)
        return [*args[:-1], str(self.encoded_output)]

    def _validate_output(self, deadline):
        # Reuse the production headless decode validation against the TS file.
        self.output = self.encoded_output
        return super()._validate_output(deadline)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'FFmpeg required')
class EncodedAudioTests(unittest.TestCase):
    def test_local_aac_signal_silence_tail_and_pts(self):
        rate = 48000
        duration = 4
        # Leading silence, signal, interior silence, then signal through EOF.
        def active(t):
            return .5 <= t < 1.5 or 3 <= t < duration

        pcm = b''.join(struct.pack('<h', round(12000 * math.sin(2*math.pi*440*i/rate))
                                  if active(i/rate) else 0)
                       for i in range(rate*duration))
        binding = Binding('encoded-audio', 'synthetic', SpeechSource(
            hashlib.sha256(pcm).hexdigest(), rate, rate*duration), 'a'*64)
        delivery = Delivery(binding, pcm)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Same real encoded-video + Delivery lease pattern as the headless
            # sink fixture: no mock accept, encoder, pipes, or input mapping.
            video = root / 'synthetic.mkv'
            run('ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'color=red:s=16x16:r=24', '-frames:v', '96',
                '-c:v', 'ffv1', str(video))
            delivery.enqueue(Segment(binding, 0, 0, 96,
                hashlib.sha256(video.read_bytes()).hexdigest()))
            lease = delivery.lease()
            self.assertEqual(lease.segment.padding_samples, 0)
            with FileEncodedSink(root, binding, width=16, height=16,
                                 paced=False) as sink:
                sink.accept(video.name, lease)
                delivery.ack(lease)
                output = sink.finish()
                probe = json.loads(run('ffprobe', '-v', 'error', '-show_streams',
                    '-show_packets', '-of', 'json', str(output)))
                streams = {s['codec_type']: s for s in probe['streams']}
                self.assertEqual(set(streams), {'video', 'audio'})
                self.assertEqual(streams['video']['codec_name'], 'h264')
                self.assertEqual(streams['audio']['codec_name'], 'aac')
                self.assertEqual(int(streams['audio']['sample_rate']), rate)
                self.assertEqual(streams['audio']['channels'], 1)
                starts = {}
                for kind, stream in streams.items():
                    packets = [p for p in probe['packets']
                               if p['stream_index'] == stream['index']]
                    pts = [float(p['pts_time']) for p in packets]
                    self.assertTrue(pts)
                    self.assertTrue(all(b > a for a, b in zip(pts, pts[1:])))
                    starts[kind] = pts[0]
                    end = pts[-1] + float(packets[-1]['duration_time'])
                    self.assertAlmostEqual(end-pts[0], duration, delta=.05)
                    if kind == 'video':
                        self.assertEqual(len(pts), 96)
                        for i, value in enumerate(pts):
                            self.assertAlmostEqual(value-pts[0], i/24, delta=.00002)
                self.assertLessEqual(abs(starts['audio']-starts['video']), .05)
                decoded = run('ffmpeg', '-v', 'error', '-i', str(output),
                              '-map', '0:a:0', '-f', 'f32le', 'pipe:1')
                samples = array.array('f')
                samples.frombytes(decoded)
                import sys
                if sys.byteorder != 'little':
                    samples.byteswap()
                self.assertAlmostEqual(len(samples)/rate, duration, delta=.05)

                def rms(start, end):
                    values = samples[round(start*rate):round(end*rate)]
                    self.assertTrue(values)
                    return math.sqrt(sum(v*v for v in values)/len(values))

                self.assertLess(rms(.1, .4), .002)
                self.assertGreater(rms(.65, 1.35), .15)
                self.assertLess(rms(1.7, 2.8), .002)
                self.assertGreater(rms(3.15, 3.9), .15)
                # 5ms energy windows allow AAC priming/ringing but not a lost
                # onset or truncated final signal. Maximum timing slack: 50ms.
                step = rate//200
                energy = [(i/rate, math.sqrt(sum(v*v for v in samples[i:i+step])
                           /len(samples[i:i+step])))
                          for i in range(0, len(samples), step)]
                audible = [t for t, value in energy if value > .05]
                self.assertTrue(audible)
                self.assertAlmostEqual(audible[0], .5, delta=.05)
                self.assertAlmostEqual(audible[-1]+step/rate, duration, delta=.05)
            self.assertTrue(all(p.poll() is not None for p in sink._processes))
            self.assertTrue(all(not t.is_alive() for t in sink._writers))


if __name__ == '__main__':
    unittest.main()
