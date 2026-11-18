import socket
import threading
import time
import unittest

from research.h3_personal import paired_av_wire as wire
from research.h3_personal.live_conversation import Packet
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, Receiver, encode
from research.h3_personal.webrtc_producer import ProducerPump


class RTCProducerTests(unittest.TestCase):
    def exercise(self, *, partial=True, mutation=None, early=False):
        binding = LiveAVBinding('rtc-test', 'a' * 64, 1, max_segments=8)
        a, b = socket.socketpair()
        deadline = time.monotonic() + 5
        pump = ProducerPump(a, binding, deadline=deadline)
        pulled = []
        closed = []
        errors = []
        results = []
        two = threading.Event()
        release = threading.Event()

        def packets():
            try:
                for i in range(3):
                    pulled.append(i)
                    if i == 1:
                        two.set()
                    video, pcm = b'v', bytes(8000)
                    e = encode(binding, i, i * 2, i * 2 + 2, video, pcm,
                               (AudioRegion('silence', i * 4000, (i + 1) * 4000),), i == 2)
                    yield Packet(e, video, pcm)
                raise AssertionError('pulled beyond final')
            finally:
                closed.append(True)

        def notice(e):
            value = dict(schema='h3-rtc-closing-tail-v1', session=binding.session,
                         profile_sha256=binding.profile_sha256, epoch=binding.epoch,
                         index=e.index, video_frame_stop=e.frame_stop - 1,
                         audio_sample_stop=e.frame_stop * 2000 - 2000,
                         source_complete=True, media_complete=False)
            if mutation:
                mutation(value)
            wire._write(b, wire._pack(value), deadline)

        def sink():
            receiver = Receiver(binding, deadline=deadline)
            try:
                for i in range(3):
                    receipt = wire.receive_pair(b, receiver, deadline=deadline)
                    e = receipt.envelope
                    wire.send_ack(b, e, stage='verified', deadline=deadline)
                    if i == 0:
                        self.assertTrue(two.wait(2))
                        self.assertEqual(pulled, [0, 1])
                        release.set()
                    if early or (i == 2 and partial):
                        notice(e)
                        return
                    lease = receiver.lease()
                    receiver.ack(lease, certain=True)
                    wire.send_ack(b, e, stage='sink-accepted', deadline=deadline)
            except BaseException as exc:
                errors.append(exc)
            finally:
                b.close()

        thread = threading.Thread(target=sink)
        thread.start()
        try:
            results.append(pump.run(packets()))
        except (RuntimeError, OSError) as exc:
            results.append(exc)
        finally:
            thread.join(6)
        self.assertFalse(thread.is_alive())
        self.assertFalse(pump._thread.is_alive())
        self.assertTrue(closed)
        self.assertFalse(errors, errors)
        self.assertEqual(pump.highwater, 2)
        with self.assertRaises(RuntimeError):
            pump.run([])
        return results[0], pump

    def test_partial_final_is_not_completion(self):
        result, pump = self.exercise()
        self.assertTrue(result['source_complete'])
        self.assertFalse(result['media_complete'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['closing_tail_unverified'])
        self.assertEqual(result['acked'], 2)
        self.assertEqual(result['terminal'], 'disconnected; no replay')
        self.assertTrue(result['local_ipc_source_verified'])
        self.assertFalse(result['remote_original_pcm_verified'])

    def test_full_sink_completion(self):
        result, _ = self.exercise(partial=False)
        self.assertTrue(result['complete'])
        self.assertTrue(result['media_complete'])
        self.assertEqual(result['acked'], 3)
        self.assertFalse(result['remote_original_pcm_verified'])

    def test_bad_notices_fail(self):
        for mutation in (lambda v: v.update(extra=1),
                         lambda v: v.update(epoch=True),
                         lambda v: v.update(session='other'),
                         lambda v: v.update(profile_sha256='b' * 64),
                         lambda v: v.update(index=0),
                         lambda v: v.update(video_frame_stop=6, audio_sample_stop=12000),
                         lambda v: v.update(audio_sample_stop=0)):
            with self.subTest(mutation=mutation):
                result, _ = self.exercise(mutation=mutation)
                self.assertIsInstance(result, RuntimeError)

    def test_early_notice_fails(self):
        result, _ = self.exercise(early=True)
        self.assertIsInstance(result, (RuntimeError, OSError))


if __name__ == '__main__':
    unittest.main()
