"""Actual paired wire + FFmpeg, interrupt/discard/re-entry. Device-free only."""
import hashlib
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest

from research.h3_personal.epoch_av_sink import EpochAVSink
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av import AudioRegion, LiveAVBinding, encode
from research.h3_personal.paired_av_pump import ProducerPump, ReceiverPump


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class EpochPumpTests(unittest.TestCase):
    def test_verify_drain_discard_credit_and_new_pcm_for_both_receive_modes(self):
        from research.run_h3_realtime_first import encode_rgb_segment
        from research.h3_personal.conversation_scenario import segment_timing
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); root.chmod(0o700)
                binding = LiveAVBinding('session', 'a'*64, 1, max_segments=8)
                packets = []
                for index in range(3):
                    start, stop = index*5, (index+1)*5
                    path = root/f'generated-{index}.mp4'
                    encode_rgb_segment(bytes([40, 20, 90])*(252*448*5), path,
                        segment_timing(index, start, stop), width=252, height=448)
                    epoch = 1 if index < 2 else 2
                    pcm = epoch.to_bytes(2, 'little')*10000
                    region = AudioRegion('speech', start*2000, stop*2000, f'speech-{epoch}',
                        epoch, start*2000 if epoch == 1 else 0, hashlib.sha256(pcm).hexdigest())
                    video = path.read_bytes()
                    envelope = encode(binding, index, start, stop, video, pcm, (region,),
                                      index == 2, presentation_epoch=epoch)
                    packets.append(Packet(envelope, video, pcm))
                class Sink(EpochAVSink):
                    def accept(self, path, lease):
                        result = super().accept(path, lease)
                        if lease.segment.index == 0:
                            self.invalidate(2, interrupted_source_epoch=1)
                        return result
                left, right = socket.socketpair(); deadline = time.monotonic()+25
                with Sink(root, binding, max_frames=124, timeout=15) as sink:
                    receiver = ReceiverPump(right, binding, root, sink, deadline=deadline,
                                            prefetch=prefetch)
                    results, errors = [], []
                    def receive():
                        try:results.append(receiver.run())
                        except BaseException as exc:errors.append(exc)
                    thread = threading.Thread(target=receive); thread.start()
                    producer = ProducerPump(left, binding, deadline=deadline, overlap_send=True)
                    try:
                        sent = producer.run(iter(packets))
                    finally:
                        thread.join(30)
                        left.close(); right.close()
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(errors, errors)
                    self.assertEqual(sent['acked'], 2)
                    self.assertEqual(sent['discarded'], 1)
                    self.assertTrue(sent['transport_complete'])
                    self.assertFalse(sent['presentation_complete'])
                    self.assertEqual(sent['pending'], 0)
                    self.assertEqual(results[0]['discarded'], 1)
                    self.assertTrue(results[0]['transport_complete'])
                    self.assertFalse(results[0]['presentation_complete'])
                    self.assertEqual(sink.timing['global_frame'], 15)
                    self.assertFalse(sink.timing['presentation_complete'])
                    audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i',
                        str(results[0]['output']), '-map', '0:a', '-f', 's16le', 'pipe:1'],
                        timeout=15)
                    self.assertEqual(audio, packets[2].pcm)


if __name__ == '__main__':
    unittest.main()
