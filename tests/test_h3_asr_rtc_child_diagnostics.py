"""Real child processes, no cloud/network/device/model dependencies."""
import os
import socket
import subprocess
import sys
import time
import unittest
from research.h3_personal import paired_av_wire as wire
from research.h3_personal.asr_rtc_client import report_failure


class DiagnosticsTests(unittest.TestCase):
    def test_unknown_stage_never_leaks_text(self):
        a,b=socket.socketpair()
        try:
            report_failure(a,'secret-token-and-SDP')
            self.assertEqual(wire._json(b,time.monotonic()+1),
                {'schema':'h3-asr-rtc-ipc-error-v1','stage':'config'})
        finally:a.close();b.close()

    def child(self,body):
        a,b=socket.socketpair();c,d=socket.socketpair()
        script=('import asyncio,time; from research.h3_personal import asr_rtc_client as m\n'+body+
                '\nm.serve=serve\nraise SystemExit(m.main())\n')
        p=subprocess.Popen([sys.executable,'-c',script,str(b.fileno()),str(d.fileno())],
            pass_fds=(b.fileno(),d.fileno()),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        b.close();d.close()
        self.addCleanup(a.close);self.addCleanup(c.close)
        def reap():
            if p.poll() is None:p.kill()
            p.wait(timeout=3)
        self.addCleanup(reap)
        wire._write(a,wire._pack({'deadline':time.monotonic()+.7}),time.monotonic()+1)
        return p,a

    def test_http_status_is_fixed_vocabulary_only(self):
        p,a=self.child("async def serve(config,control,media,progress):\n progress('answer')\n raise m.SignalingFailure(504)")
        self.assertEqual(wire._json(a,time.monotonic()+2),
            {'schema':'h3-asr-rtc-ipc-error-v1','stage':'http-504'})
        self.assertEqual(p.wait(timeout=2),1)

    def test_sanitized_stage_on_exception(self):
        p,a=self.child("async def serve(config,control,media,progress):\n progress('answer')\n raise ValueError('private secret')")
        self.assertEqual(wire._json(a,time.monotonic()+2),
            {'schema':'h3-asr-rtc-ipc-error-v1','stage':'answer'})
        self.assertEqual(p.wait(timeout=2),1)

    def test_hard_deadline_stops_blocked_executor(self):
        p,a=self.child("async def serve(config,control,media,progress):\n progress('start')\n await asyncio.to_thread(time.sleep,30)")
        self.assertEqual(wire._json(a,time.monotonic()+2)['stage'],'start')
        self.assertEqual(p.wait(timeout=2),1)

if __name__=='__main__':unittest.main()
