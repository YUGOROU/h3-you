import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest

from research.live_relay_server import build_server
from research.h3_personal.live_relay import RelayConfig
from research.h3_personal.webrtc_job_gateway import RTCGateway


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.seen = (self.path, dict(self.headers), b'')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'{}')

    def do_POST(self):
        payload = self.rfile.read(int(self.headers['Content-Length']))
        self.do_GET()
        self.server.seen = (self.path, dict(self.headers), payload)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.stub = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        self.config = RelayConfig('rtc-test', time.time() + 30, 'a' * 64, 'b' * 64, wait_timeout=0)
        self.gateway = RTCGateway(self.stub.server_port, self.config.consumer_key,
                                  self.config.session, self.config.not_after)
        self.server = build_server(('127.0.0.1', 0), self.config, rtc_gateway=self.gateway)
        self.threads = []
        for server in (self.stub, self.server):
            t = threading.Thread(target=server.serve_forever)
            t.start()
            self.threads.append(t)

    def tearDown(self):
        for server in (self.server, self.stub):
            server.shutdown()
            server.server_close()
        for t in self.threads:
            t.join()

    def request(self, path, method='GET', body=None, extra=None):
        headers = {'X-H3-RTC-Key': self.config.consumer_key,
                   'X-H3-Session': self.config.session, 'Content-Length': '0'}
        headers.update(extra or {})
        c = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            c.request(method, path, body, headers)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def test_forward_and_no_credentials(self):
        self.assertEqual(self.request('/rtc/status', extra={'Cookie': 'private',
                         'Authorization': 'Bearer private'})[0], 200)
        path, headers, _ = self.stub.seen
        self.assertEqual(path, '/status')
        self.assertNotIn('Cookie', headers)
        self.assertNotIn('Authorization', headers)
        self.assertEqual(headers['X-H3-RTC-Key'], self.config.consumer_key)
        self.assertEqual(self.request('/rtc/offer', 'POST', b'{}', {
            'Content-Length': '2', 'Content-Type': 'application/json'})[0], 200)
        self.assertEqual(self.stub.seen[0], '/offer')

    def test_discarded_proxy_duplicates_but_not_binding_duplicates(self):
        def raw(duplicate):
            c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
            try:
                c.putrequest('POST','/rtc/offer')
                for name,value in [('Content-Length','2'),('Content-Type','application/json'),
                        ('X-H3-RTC-Key',self.config.consumer_key),('X-H3-Session',self.config.session),
                        *duplicate]:c.putheader(name,value)
                c.endheaders(b'{}');r=c.getresponse();r.read();return r.status
            finally:c.close()
        self.assertEqual(raw([('X-Forwarded-For','ignored'),('X-Forwarded-For','also-ignored')]),200)
        self.assertNotIn('X-Forwarded-For',self.stub.seen[1])
        for name,value in [('Content-Length','2'),('Content-Type','application/json'),
                           ('X-H3-RTC-Key',self.config.consumer_key),('X-H3-Session',self.config.session)]:
            self.assertEqual(raw([(name,value)]),400)

    def test_binding_framing_expiry(self):
        for extra in ({'X-H3-RTC-Key': 'wrong'}, {'X-H3-Session': 'wrong'}):
            self.assertEqual(self.request('/rtc/status', extra=extra)[0], 403)
        for extra in ({'Transfer-Encoding': 'chunked'}, {'Upgrade': 'websocket'},
                      {'Content-Length': '-1'}, {'Origin': 'https://example.com'}):
            self.assertEqual(self.request('/rtc/status', extra=extra)[0], 400)
        self.assertEqual(self.request('/rtc/status?target=other')[0], 404)
        self.assertEqual(self.request('/rtc/offer', 'POST', extra={'Content-Length': '131073'})[0], 413)
        self.gateway.expires_at = time.time() - 1
        self.assertEqual(self.request('/rtc/status')[0], 410)

    def test_media_closed_control_and_legacy(self):
        for route in ('av', 'av-stream', 'ack'):
            self.assertEqual(self.request('/v1/' + route)[0], 404)
        auth = {'X-H3-Role': 'consumer', 'X-H3-Key': self.config.consumer_key}
        self.assertEqual(self.request('/v1/claim', 'POST', extra=auth)[0], 200)
        self.assertEqual(self.request('/v1/status', extra=auth)[0], 200)
        self.assertEqual(self.request('/v1/input', 'POST', b'x', {
            **auth, 'Content-Length': '1', 'Content-Type': 'application/octet-stream'})[0], 202)
        self.server.rtc_gateway = None
        self.assertNotEqual(self.request('/v1/av', extra=auth)[0], 404)
        self.assertEqual(self.request('/rtc/status')[0], 404)
        self.assertEqual(self.request('/v1/abort', 'POST', extra=auth)[0], 202)


if __name__ == '__main__':
    unittest.main()
