"""Offline finite ten-minute worker contracts; no devices or inference."""
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal.paired_av import LiveAVBinding, AudioRegion
from research.h3_personal.streaming_admission import StreamingAdmission
from research.h3_personal.streaming_windows import ResamplingContext
from research.h3_personal.streaming_native_owner import StreamingWindowOwner
from research.h3_personal.live_conversation import LiveConversation
from research.h3_personal.live_relay_worker import _conversation, run_worker


class TenMinuteWorkerTests(unittest.TestCase):
    def binding(self, **kw):
        return LiveAVBinding('endurance', 'a'*64, 0, **kw)

    def test_finite_caps_and_unchanged_defaults(self):
        b = self.binding()
        self.assertEqual((b.max_segments, b.max_total_frames), (128, 13100))
        self.binding(max_segments=144, max_total_frames=14710)
        for kw in ({'max_segments':145}, {'max_total_frames':14711}):
            with self.assertRaises(ValueError): self.binding(**kw)
        self.assertEqual(StreamingWindowOwner.MAX_IMMUTABLE_UNITS, 144)
        AudioRegion('silence', 29000000, 29420000)
        with self.assertRaises(ValueError): AudioRegion('silence', 0, 29420001)

    def test_142_window_binding_and_forwarding(self):
        with tempfile.TemporaryDirectory() as root:
            driver = LiveConversation(root+'/driver', binding=self.binding(
                max_segments=142, max_total_frames=14506), context=ResamplingContext(),
                max_idle_windows=141, timeout=720, extended_endurance=True)
            self.assertEqual(driver.control._max_windows, 142)

    def test_admission_timeout_requires_explicit_boolean(self):
        for timeout, flag, valid in ((300,False,True),(301,False,False),
                (720,True,True),(721,True,False),(720,1,False)):
            with self.subTest(timeout=timeout, flag=flag):
                def build():
                    return StreamingAdmission('endurance', context=ResamplingContext(),
                        timeout=timeout, extended_endurance=flag)
                if valid: build()
                else:
                    with self.assertRaises(ValueError): build()

    def test_worker_deadline_is_not_extended(self):
        b=self.binding(max_segments=142,max_total_frames=14506)
        with patch('research.h3_personal.live_relay_worker.time.monotonic',return_value=100), \
             patch('research.h3_personal.live_relay_worker.LiveConversation') as factory:
            for flag, deadline, expected in ((False,1300,300),(True,1300,720),
                    (True,700,600),(True,99,-1)):
                _conversation('unused',binding=b,context=ResamplingContext(),
                              deadline=deadline,ten_minute_test=flag)
                self.assertEqual(factory.call_args.kwargs['timeout'],expected)
                self.assertIs(factory.call_args.kwargs['extended_endurance'],flag)

    def test_worker_rejects_non_boolean_before_side_effects(self):
        with self.assertRaises(ValueError):
            run_worker(None,None,None,binding=None,context=None,root=None,
                       deadline=0,ten_minute_test=1)


if __name__ == '__main__':
    unittest.main()
