import concurrent.futures
import http.client
import json
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch

from research.h3_personal.h3_deferred_boot import serve


class DeferredDispatchTests(unittest.TestCase):
    def test_fixed_dataset_download_pins_and_original_deadline(self):
        import hashlib,sys,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from research.h3_personal.h3_deferred_boot import _execute
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bootstrap=root/'bootstrap.py';bootstrap.write_text('# pinned\n')
            spec=root/'spec.json';spec.write_text(json.dumps({'live_direct_session':{'native_tts':{
                'schema':'h3-native-voxcpm-network-v2','connection':{'group':'group','server_job_id':'a'*24}}}}))
            calls=[]
            def download(**kwargs):
                calls.append(kwargs)
                return str(spec if kwargs['filename']=='spec.json' else bootstrap)
            capsule=dict(repo='YUGOROU/h3-fresh',revision='b'*40,group='group',expires_at=time.time()+120,
                bootstrap_sha256=hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
                spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),allow_sha256='c'*64,
                producer_key='p'*40,consumer_key='c'*40,native_key='n'*40)
            with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=download)}),\
                 patch.dict(os.environ,{},clear=True),patch.object(sys,'argv',[]),\
                 patch('research.h3_personal.h3_deferred_boot.runpy.run_path') as execute:
                _execute(capsule)
                self.assertEqual(float(os.environ['H3_SESSION_NOT_AFTER']),capsule['expires_at'])
                self.assertEqual(sys.argv,[str(bootstrap),capsule['repo'],capsule['revision'],'c'*64])
            self.assertTrue(all(c['repo_type']=='dataset' and c['revision']=='b'*40 for c in calls))
            execute.assert_called_once_with(str(bootstrap),run_name='__main__')


class DeferredBootTests(unittest.TestCase):
    def start(self, lifetime=3, execute=None):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.expiry = time.time() + lifetime
        self.calls = []
        self.errors = []
        def worker():
            try:
                serve(key='K'*32, campaign='campaign', group='group',
                      expires_at=self.expiry, bootstrap_sha256='a'*64,
                      host='127.0.0.1', port=self.port,
                      execute=execute or self.calls.append)
            except Exception as exc:
                self.errors.append(exc)
        with patch.dict(os.environ, JOB_ID='job'):
            self.thread = threading.Thread(target=worker)
            self.thread.start()
            for _ in range(100):
                try:
                    status, self.ready = self.request('GET', '/ready')
                    if status == 200:
                        break
                except OSError:
                    time.sleep(.01)
            else:
                self.fail('listener unavailable')
        self.addCleanup(self.thread.join, 4)

    def request(self, method, path, body=None, key='K'*32, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
        try:
            conn.request(method, path, body=body,
                         headers=headers or {'X-H3-Launch-Key': key,'Content-Type':'application/json'})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def capsule(self):
        return dict(schema='h3-deferred-launch-v1', campaign='campaign', group='group',
                    boot=self.ready['boot'], job_id='job', expires_at=self.expiry,
                    repo='YUGOROU/h3-test', revision='b'*40, allow_sha256='c'*64,
                    bootstrap_sha256='a'*64, spec_sha256='d'*64,
                    producer_key='P'*32, consumer_key='C'*32, native_key='N'*32)

    def launch(self, c):
        return self.request('POST', '/launch', json.dumps(c))[0]

    def test_rejections_then_one_shot(self):
        self.start()
        self.assertEqual(self.ready['state'], 'waiting')
        self.assertEqual(len(self.ready['boot']), 32)
        self.assertEqual(self.request('GET', '/ready', key='bad')[0], 403)
        for field, value in [('boot', '0'*32), ('job_id', 'other'),
                             ('group', 'wrong'), ('campaign', 'other'),
                             ('expires_at', self.expiry+1), ('expires_at', time.time()-1),
                             ('repo', 'other/h3-test'), ('revision', 'main'),
                             ('bootstrap_sha256', 'b'*64), ('producer_key', 'short'),
                             ('extra', True)]:
            c = self.capsule()
            c[field] = value
            self.assertEqual(self.launch(c), 400, field)
        self.assertEqual(self.request('POST', '/launch', '{broken')[0], 400)
        raw = json.dumps(self.capsule())[:-1] + ',"schema":"duplicate"}'
        self.assertEqual(self.request('POST', '/launch', raw)[0], 400)
        self.assertEqual(self.launch(self.capsule()), 202)
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.calls, [self.capsule()])
        self.assertFalse(self.errors)

    def test_concurrent_claim_and_closed_before_execute(self):
        def execute(c):
            probe = socket.socket()
            try:
                self.assertNotEqual(probe.connect_ex(('127.0.0.1', self.port)), 0)
            finally:
                probe.close()
            self.calls.append(c)
            raise RuntimeError('execution failure')
        self.start(execute=execute)
        c = self.capsule()
        def attempt(_):
            try:
                return self.launch(c)
            except (OSError, http.client.HTTPException):
                return None
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            results = list(pool.map(attempt, range(4)))
        self.thread.join(2)
        self.assertEqual(results.count(202), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIsInstance(self.errors[0], RuntimeError)

    def test_duplicate_headers_and_transfer_encoding(self):
        self.start(lifetime=.5)
        for extra in ('X-H3-Launch-Key: '+ 'K'*32, 'Transfer-Encoding: chunked'):
            with socket.create_connection(('127.0.0.1', self.port)) as sock:
                sock.sendall(('GET /ready HTTP/1.1\r\nHost: localhost\r\n'
                              'X-H3-Launch-Key: '+ 'K'*32 +'\r\n'+extra+'\r\n\r\n').encode())
                self.assertIn(b'400', sock.recv(4096).split(b'\r\n')[0])
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.calls)

    def test_shared_deadline_not_extended_by_requests(self):
        self.start(lifetime=.35)
        while time.time() < self.expiry-.05:
            self.assertEqual(self.request('GET', '/ready')[0], 200)
            time.sleep(.02)
        self.thread.join(.5)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
