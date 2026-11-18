"""Loopback HTTP + mocked upstream only; NOT real-browser acceptance."""
import http.client
import json
import time
import unittest
from unittest.mock import patch
from research.h3_personal.webrtc_browser_proxy import BrowserProxy


class ProxyTests(unittest.TestCase):
    def binding(self):
        return dict(job_id='a'*24, session='session', boot='boot',
                    expires_at=time.time()+60, ice_profile='host-only-v1')

    def request(self, p, method, path, value=None, cap=True, origin=True):
        c = http.client.HTTPConnection('127.0.0.1', p._server.server_port, timeout=4)
        headers = {'Content-Type': 'application/json'}
        if cap:
            headers['X-H3-Capability'] = p.url.split('#')[1]
        if origin:
            headers['Origin'] = p.origin
        c.request(method, path, None if value is None else json.dumps(value), headers)
        r = c.getresponse()
        result = r.status, r.read()
        c.close()
        return result

    def test_owner_required(self):
        with self.assertRaises(ValueError):
            BrowserProxy(self.binding(), hf_token='secret', app_key='key')

    def test_auth_config_and_html(self):
        with BrowserProxy(self.binding(), hf_token='HFSECRET', app_key='APPSECRET', owner_attested=True) as p:
            self.assertEqual(self.request(p, 'GET', '/config', cap=False)[0], 403)
            code, body = self.request(p, 'GET', '/config')
            self.assertEqual(code, 200)
            self.assertNotIn(b'HFSECRET', body)
            self.assertNotIn(b'APPSECRET', body)
            self.assertEqual(json.loads(body)['iceServers'], [])
            code, html = self.request(p, 'GET', '/', cap=False)
            self.assertEqual(code, 200)
            self.assertIn(b'requestVideoFrameCallback', html)
            self.assertNotIn(b'getUserMedia', html)
            self.assertNotIn(b'https://', html)

    def test_fixed_offer_single_use_and_events(self):
        b = self.binding()
        offer = dict(session='session', boot='boot', description=dict(type='offer', sdp='private'))
        answer = dict(session='session', boot='boot', description=dict(type='answer', sdp='private-answer'))
        with patch('research.h3_personal.webrtc_connect_client.request', return_value=answer) as upstream:
            with BrowserProxy(b, hf_token='HF', app_key='APP', owner_attested=True) as p:
                self.assertEqual(self.request(p, 'POST', '/offer', offer, origin=False)[0], 403)
                upstream.assert_not_called()
                self.assertEqual(self.request(p, 'POST', '/offer', offer)[0], 200)
                args = upstream.call_args.args
                self.assertEqual(args[:4], ('a'*24+'--7860.hf.jobs', '/offer', 'HF', 'APP'))
                self.assertEqual(self.request(p, 'POST', '/offer', offer)[0], 409)
                self.assertEqual(self.request(p, 'POST', '/event', {'event':'error','sdp':'secret'})[0], 400)
                self.assertEqual(self.request(p, 'POST', '/event', {'event':'opened'})[0], 200)
                self.assertEqual(p.receipts, ({'event':'opened'},))
        self.assertEqual(p._token, '')

    def test_bad_binding_and_body(self):
        b = self.binding()
        b['expires_at'] = float('inf')
        with self.assertRaises(ValueError):
            BrowserProxy(b, hf_token='x', app_key='y', owner_attested=True)
        with BrowserProxy(self.binding(), hf_token='x', app_key='y', owner_attested=True) as p:
            self.assertEqual(self.request(p, 'POST', '/event', {'event':'x'*300})[0], 413)
            self.assertEqual(self.request(p, 'POST', '/arbitrary', {})[0], 403)


if __name__ == '__main__':
    unittest.main()
