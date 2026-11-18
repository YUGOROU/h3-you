import socket
import time
import unittest
from research.h3_personal.webrtc_upstream_adapter import RTCStream,asr_transport
from research.h3_personal.webrtc_exact_channel import Binding

class TailTests(unittest.TestCase):
    def check(self,tail):
        a,b=socket.socketpair()
        source=a.makefile('rb')
        try:
            b.sendall(tail);b.shutdown(socket.SHUT_WR)
            factory=asr_transport(RTCStream(a,Binding('s','b',1,'asr-native-event-wire-v1',1000)))
            factory.finish(a,source,time.monotonic()+1)
        finally:source.close();a.close();b.close()
    def test_optional_valid_close(self):
        for tail in (b'',b'\x88\x00',b'\x88\x02\x03\xe8'):self.check(tail)
    def test_exact_native_committed_marker(self):
        import json
        payload=json.dumps({'type':'input_audio_buffer.committed','event_id':'event_123'}).encode()
        frame=bytes([0x81,len(payload)])+payload
        self.check(frame+b'\x88\x00')
        with self.assertRaises(ValueError):self.check(frame+frame)

    def test_no_text_truncation_or_invalid_close(self):
        for tail in (b'\x81\x02hi',b'\x88',b'\x88\x01x',b'\x88\x80',b'\x88\x02\x03\xed',b'\x88\x03\x03\xe8\xff',b'\x88\x00junk'):
            with self.subTest(tail=tail),self.assertRaises((ValueError,UnicodeError,EOFError)):self.check(tail)
