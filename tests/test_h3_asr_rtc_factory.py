"""Subprocess IPC lifecycle contracts; stub child, no signaling/cloud/capture."""
import hashlib
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch
from research.h3_personal.asr_rtc_factory import ASRRTCFactory
from research.h3_personal import paired_av_wire as wire

class FactoryTests(unittest.TestCase):
    def fake_popen(self,bad=False):
        def popen(command,**kwargs):
            control=socket.socket(fileno=__import__('os').dup(kwargs['pass_fds'][0]))
            media=socket.socket(fileno=__import__('os').dup(kwargs['pass_fds'][1]))
            class Process:
                returncode=None
                def poll(self):return self.returncode
                def wait(self,timeout=None):
                    thread.join(timeout)
                    if thread.is_alive():raise __import__('subprocess').TimeoutExpired('stub',timeout)
                    return self.returncode
                def terminate(self):
                    try:media.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
                kill=terminate
            process=Process()
            def child():
                try:
                    row=wire._json(control,time.monotonic()+3)
                    deadline=row['deadline']
                    binding=dict(session=row['session'],boot='b'*32,epoch=row['epoch']+(1 if bad else 0),purpose='asr-native-event-wire-v1',max_bytes=4*1024*1024)
                    wire._write(control,wire._pack(dict(schema='h3-asr-rtc-ipc-ready-v1',binding=binding)),deadline)
                    data=bytearray()
                    while True:
                        body=media.recv(8192)
                        if not body:break
                        data.extend(body)
                    wire._write(control,wire._pack(dict(schema='h3-asr-rtc-ipc-complete-v1',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                        transport_complete=True,native_eos_verified=False,local_eof_already_closed=True)),deadline)
                    process.returncode=0
                except BaseException:process.returncode=1
                finally:control.close();media.close()
            thread=threading.Thread(target=child);thread.start()
            self.assertNotIn('token',repr(command));self.assertNotIn('k'*40,repr(command))
            self.assertEqual(kwargs['env']['H3_ASR_RTC_KEY'],'k'*40)
            return process
        return popen

    def args(self):
        return dict(endpoint='https://test.endpoints.huggingface.cloud',token='token',key='k'*40,
                    session='session',epoch=1,expires_at=time.time()+60,deadline=time.monotonic()+4)

    def test_owned_child_completion_releases_serial_slot(self):
        factory=ASRRTCFactory(sys.executable)
        with patch('research.h3_personal.asr_rtc_factory.subprocess.Popen',side_effect=self.fake_popen()):
            turn=factory.open_turn(**self.args())
            with self.assertRaises(RuntimeError):factory.open_turn(**self.args())
            connection,source=turn.transport('unused','token','key',1)
            connection.sendall(b'exact test frame');source.close()
            turn.close()
        self.assertEqual(turn.receipt['bytes'],len(b'exact test frame'))
        self.assertIsNone(factory._active);self.assertEqual(turn.process.poll(),0)

    def test_wrong_binding_kills_child_and_does_not_return_transport(self):
        factory=ASRRTCFactory(sys.executable)
        with patch('research.h3_personal.asr_rtc_factory.subprocess.Popen',side_effect=self.fake_popen(bad=True)):
            with self.assertRaises(ValueError):factory.open_turn(**self.args())
        self.assertIsNone(factory._active)

    def test_child_exit_before_ready_is_eof_not_deadline(self):
        import subprocess
        factory=ASRRTCFactory(sys.executable)
        real_popen=subprocess.Popen
        def exited_child(command,**kwargs):
            # Real process inherits then closes both descriptors without ready.
            return real_popen([sys.executable,'-c','import time; time.sleep(.1)'],**kwargs)
        start=time.monotonic()
        with patch('research.h3_personal.asr_rtc_factory.subprocess.Popen',side_effect=exited_child):
            with self.assertRaises((EOFError,ConnectionResetError)):
                factory.open_turn(**self.args())
        self.assertLess(time.monotonic()-start,2)
        self.assertIsNone(factory._active)

    def test_bad_endpoint_never_spawns_child(self):
        factory=ASRRTCFactory(sys.executable);args=self.args();args['endpoint']='http://untrusted/'
        with patch('research.h3_personal.asr_rtc_factory.subprocess.Popen',side_effect=AssertionError):
            with self.assertRaises(ValueError):factory.open_turn(**args)

if __name__=='__main__':unittest.main()
