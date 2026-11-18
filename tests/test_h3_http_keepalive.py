"""Device/provider-free authenticated finite HTTP connection reuse."""
import http.client
import threading
import time
import unittest
from research.live_relay_server import build_server
from research.h3_personal.live_relay import RelayConfig
from research.h3_personal.live_relay_client import SocketAdapter
from research.h3_personal.job_av_server import LocalProducer

class KeepaliveTests(unittest.TestCase):
    def exercise(self,enabled):
        expiry=time.time()+30
        server=build_server(('127.0.0.1',0),RelayConfig('session',expiry,'a'*64,'b'*64),keepalive=enabled)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});thread.start()
        producer=LocalProducer(server.relay)
        consumer=SocketAdapter(f'http://127.0.0.1:{server.server_port}',session='session',not_after=expiry,
            role='consumer',role_key='b'*64,hf_token='fake',allow_test_loopback=True,http_timeout=2).open()
        try:
            for i in range(40):
                payload=bytes([i])*4096
                self.assertEqual(producer.send(payload),len(payload))
                self.assertEqual(consumer.recv(8192),payload)
            count=server.timing()['connections']
            if enabled:
                # claim + read lane, and bounded rollover after32 responses.
                self.assertEqual(count,3)
            else:self.assertEqual(count,41)
            self.assertEqual(server.timing()['body_bytes']>=40*4096,True)
            consumer.send_input(b'fresh-control')
            self.assertEqual(producer.read_input(),b'fresh-control')
        finally:
            consumer.close();producer.close()
            server.shutdown();server.server_close();thread.join(2)
        self.assertFalse(thread.is_alive())
        return count

    def test_connection_count_reduction_and_exact_bytes(self):
        self.assertLess(self.exercise(True),self.exercise(False))

    def test_unauthenticated_and_errors_always_close(self):
        config=RelayConfig('session',time.time()+10,'a'*64,'b'*64)
        server=build_server(('127.0.0.1',0),config,keepalive=True)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});thread.start()
        try:
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
            for path,expected in (('/health',200),('/v1/av',401)):
                conn.request('GET',path);response=conn.getresponse()
                self.assertEqual(response.status,expected)
                self.assertEqual(response.getheader('Connection'),'close')
                response.read()
            conn.close()
        finally:server.shutdown();server.server_close();thread.join(2)

    def test_explicit_flag(self):
        for value in (1,None,'true'):
            with self.assertRaises(ValueError):
                build_server(('127.0.0.1',0),RelayConfig('session',time.time()+10,'a'*64,'b'*64),keepalive=value)

if __name__=='__main__':unittest.main()
