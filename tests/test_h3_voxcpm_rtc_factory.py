"""Offline isolated owner regressions: no cloud, media devices or TURN requests."""
import hashlib
import socket
import sys
import time
import unittest
from unittest.mock import patch

from research.h3_personal.voxcpm_rtc_factory import VoxRTCFactory, Turn
from research.h3_personal import paired_av_wire as wire


class FactoryTests(unittest.TestCase):
    def args(self):
        return dict(endpoint='https://example.endpoints.huggingface.cloud', token='token',
                    key='key', session='session', boot='a'*32, expires_at=time.time()+60,
                    text='hello', epoch=1, deadline=time.monotonic()+5)

    def test_missing_python(self):
        with self.assertRaises(ValueError):
            VoxRTCFactory('/no/such/python')

    def test_invalid_text_before_spawn(self):
        factory=VoxRTCFactory(sys.executable)
        args=self.args(); args['text']=' '
        with self.assertRaises(ValueError):factory.open_stream(**args)
        self.assertIsNone(factory._active)

    def test_child_exit_is_prompt_and_releases_slot(self):
        factory=VoxRTCFactory(sys.executable)
        # A real isolated process exits without retaining the socketpair endpoints.
        import subprocess
        real=subprocess.Popen
        def spawn(argv, **kwargs):
            return real([sys.executable, '-c', 'pass'], **kwargs)
        args=self.args()
        with patch('research.h3_personal.voxcpm_rtc_factory.subprocess.Popen', side_effect=spawn):
            with self.assertRaises(Exception):factory.open_stream(**args)
        self.assertIsNone(factory._active)

    def test_receipt_strict_response_only(self):
        for count, valid, consumed in ((0, True,b'test'), (1, False,b'test'), (False, False,b'test'),(0,False,b'other')):
            with self.subTest(count=count):
                factory=VoxRTCFactory(sys.executable)
                turn=Turn(factory,time.monotonic()+2); factory._active=turn
                a,b=socket.socketpair(); turn.control=a
                turn.record_read(consumed)
                receipt=dict(schema='h3-voxcpm-rtc-ipc-complete-v1',bytes=count,
                    sha256=hashlib.sha256(b'').hexdigest(),transport_complete=True,
                    native_eos_verified=False,local_eof_already_closed=False,
                    received_bytes=4,received_sha256=hashlib.sha256(b'test').hexdigest())
                wire._write(b,wire._pack(receipt),turn.deadline)
                class Process:
                    returncode=0
                    def wait(self,timeout):pass
                    def poll(self):return 0
                turn.process=Process()
                try:
                    if valid:turn.close()
                    else:
                        with self.assertRaises(ValueError):turn.close()
                    self.assertIsNone(factory._active)
                finally:b.close()


if __name__=='__main__':unittest.main()
