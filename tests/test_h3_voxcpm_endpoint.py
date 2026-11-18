import io
import json
from email.message import Message
from unittest.mock import patch
import struct
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from research.h3_personal.voxcpm_endpoint_protocol import decode, frame, metadata, MAX_FRAME
from research.h3_personal.voxcpm_endpoint_service import Gate, create_server
from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM


class Candidate:
    sample_rate = 48000
    def stream(self, text):
        yield [0.0, .25, -.5]
        yield [.125]


class Tests(unittest.TestCase):
    def gate(self, candidate=None, **kw):
        return Gate(candidate or Candidate(), key='secret', session='session',
                    boot='boot', expires_at=time.time()+60, **kw)

    def request(self, epoch=1):
        return dict(session='session', boot='boot', epoch=epoch, text='hello')

    def wire(self, gate):
        parts = []
        gate.stream(self.request(), parts.append)
        return b''.join(parts)

    def test_incremental_exact_end(self):
        gate = self.gate()
        parts = []
        class Incremental(Candidate):
            def stream(self, text):
                yield [.25]
                self_check.assertTrue(any(p[:1] == b'P' for p in parts))
                yield [-.5]
        self_check = self
        gate.candidate = Incremental()
        gate.stream(self.request(), parts.append)
        chunks = list(decode(io.BytesIO(b''.join(parts)).read, {
            k:v for k,v in gate.ready(1).items() if k in ('session','boot','expires_at','epoch')}))
        self.assertEqual(chunks, [struct.pack('<f', .25), struct.pack('<f', -.5)])
        self.assertIn(b'iterator_exhausted', parts[-1])
        self.assertIn(b'"model_eos_verified":false', parts[-1])

    def test_truncated_and_wrong_binding(self):
        gate = self.gate()
        wire = self.wire(gate)
        expected = dict(session=gate.session, boot=gate.boot, expires_at=gate.expires_at, epoch=1)
        for data in (wire[:-1], wire[:5], b'P'+struct.pack('!I', MAX_FRAME+1)):
            with self.assertRaises(ValueError):
                list(decode(io.BytesIO(data).read, expected))
        with self.assertRaises(ValueError):
            list(decode(io.BytesIO(wire).read, dict(expected, boot='wrong')))

    def test_terminal_frame_and_metadata_types_are_strict(self):
        gate=self.gate();parts=[];gate.stream(self.request(),parts.append)
        wire=b''.join(parts)
        expected=dict(session=gate.session,boot=gate.boot,expires_at=gate.expires_at,epoch=1)
        for trailing in (b'garbage',parts[-1],frame(b'P',struct.pack('<f',.5))):
            with self.subTest(trailing=trailing[:1]),self.assertRaises(ValueError):
                list(decode(io.BytesIO(wire+trailing).read,expected))
        invalid_ready=dict(gate.ready(1),channels=True)
        with self.assertRaises(ValueError):
            list(decode(io.BytesIO(metadata(b'R',invalid_ready)+b''.join(parts[1:])).read,expected))
        end=json.loads(parts[-1][5:]);end['model_eos_verified']=0
        with self.assertRaises(ValueError):
            list(decode(io.BytesIO(b''.join(parts[:-1])+metadata(b'E',end)).read,expected))

    def test_authenticated_completion_evidence_is_separate_from_stream_end(self):
        class WithEvidence(Candidate):
            def export_evidence(self):return {'runtime':{'device_count':1},'completions':[]}
        server=create_server(WithEvidence(),key='secret',session='session',boot='boot',expires_at=time.time()+60)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_address[1]}/evidence'
        try:
            with self.assertRaises(HTTPError) as failure:urlopen(url,timeout=2)
            self.assertEqual(failure.exception.code,401)
            with urlopen(Request(url,headers={'X-H3-TTS-Key':'secret'}),timeout=2) as response:
                row=json.load(response)
            self.assertEqual(row['binding'],server.gate.ready(0))
            self.assertEqual(row['evidence']['completions'],[])
            self.assertFalse(row['binding']['watermark_applied'])
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_replay_limit_busy(self):
        gate = self.gate(max_requests=1)
        self.wire(gate)
        with self.assertRaises(ValueError):
            gate.stream(self.request(), lambda _: None)
        with self.assertRaises(ValueError):
            gate.stream(self.request(2), lambda _: None)
        gate = self.gate()
        gate.slot.acquire()
        try:
            with self.assertRaises(ValueError):
                gate.stream(self.request(), lambda _: None)
        finally:
            gate.slot.release()

    def test_error_and_disconnect_quarantine(self):
        for values in ([float('nan')], [], [0.]*(MAX_FRAME//4+1)):
            class Bad(Candidate):
                def stream(self, text):
                    yield values
            gate = self.gate(Bad())
            out = []
            with self.assertRaises(ValueError):
                gate.stream(self.request(), out.append)
            self.assertFalse(gate.healthy())
            self.assertEqual(out[-1][:1], b'X')
        gate = self.gate()
        def fail(_):
            raise OSError()
        with self.assertRaises(OSError):
            gate.stream(self.request(), fail)
        self.assertFalse(gate.healthy())

    def test_expiry(self):
        gate = self.gate()
        gate.expires_at = time.time()-1
        with self.assertRaises(ValueError):
            self.wire(gate)
        with self.assertRaises(ValueError):
            Gate(Candidate(), key='a', session='s', boot='b', expires_at=time.time()+1300)

    def test_local_http_auth_stream(self):
        server = create_server(Candidate(), key='secret', session='session', boot='boot',
                               expires_at=time.time()+60, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:'+str(server.server_port)
        try:
            with self.assertRaises(HTTPError) as caught:
                urlopen(base+'/ready', timeout=2)
            self.assertEqual(caught.exception.code, 401)
            import json
            req = Request(base+'/stream', data=json.dumps(self.request()).encode(),
                          headers={'X-H3-TTS-Key':'secret'})
            with urlopen(req, timeout=2) as response:
                expected = dict(session='session', boot='boot', epoch=1,
                                expires_at=server.gate.expires_at)
                self.assertEqual(len(list(decode(response.read, expected))), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_client_rejects_non_endpoint(self):
        with self.assertRaises(Exception):
            RemoteVoxCPM('http://localhost', token='x', key='k', session='s',
                         boot='b', expires_at=time.time()+60)


class RemoteTests(unittest.TestCase):
    def setup_client(self, mutate=None):
        self.gate = Gate(Candidate(), key='secret', session='session', boot='boot',
                         expires_at=time.time()+60)
        self.calls = []
        def transport(method, path, body, headers, deadline):
            self.calls.append((method, path, body, headers, deadline))
            value = self.gate.ready(0)
            if mutate:
                mutate(value)
            return 200, json.dumps(value).encode()
        self.client = RemoteVoxCPM('https://test.endpoints.huggingface.cloud',
            token='token', key='secret', session='session', boot='boot',
            expires_at=self.gate.expires_at, transport=transport)
        return self.client

    def ready(self):
        return self.client.read('tts_ready', time.monotonic()+10)

    def connection(self, wire, fail=False):
        owner = self
        class Socket:
            def shutdown(self, *_): pass
            def close(self): pass
        class Response:
            status = 200
            headers = Message()
            headers['Content-Type'] = 'application/x-h3-voxcpm-stream-v1'
            def __init__(self): self.data = io.BytesIO(wire)
            def getheader(self, key): return self.headers.get(key)
            def read1(self, n):
                if fail: raise OSError('disconnect')
                return self.data.read(min(n, 3))
        class Connection:
            def __init__(self, host, timeout):
                owner.assertEqual(host, 'test.endpoints.huggingface.cloud')
                owner.assertGreater(timeout, 0)
                self.sock = Socket()
                self.closed = False
            def connect(self): pass
            def request(self, method, path, body, headers):
                owner.assertEqual((method, path), ('POST', '/stream'))
                owner.assertEqual(headers['Authorization'], 'Bearer token')
                owner.assertEqual(headers['X-H3-TTS-Key'], 'secret')
                owner.assertEqual(json.loads(body), dict(session='session', boot='boot',
                                                       epoch=1, text='hello'))
            def getresponse(self): return Response()
            def close(self): self.closed = True
        return Connection

    def wire(self):
        parts = []
        self.gate.stream(dict(session='session', boot='boot', epoch=1, text='hello'), parts.append)
        return b''.join(parts)

    def test_ready_exact_auth_and_stream(self):
        client = self.setup_client()
        self.assertEqual(self.ready(), self.gate.ready(0))
        method, path, body, headers, deadline = self.calls[0]
        self.assertEqual((method, path, body), ('GET', '/ready', None))
        self.assertEqual(headers['Authorization'], 'Bearer token')
        self.assertEqual(headers['X-H3-TTS-Key'], 'secret')
        with patch('research.h3_personal.voxcpm_endpoint_client.http.client.HTTPSConnection',
                   self.connection(self.wire())):
            self.assertEqual(len(list(client.stream('hello', 1, time.monotonic()+10))), 2)
        self.assertFalse(client._closed)
        with self.assertRaises(RuntimeError):
            list(client.stream('hello', 1, time.monotonic()+10))

    def test_close_failure_preserves_stream_error_and_releases_slot(self):
        client=self.setup_client();self.ready()
        base=self.connection(self.wire(),fail=True)
        class BrokenClose(base):
            def close(self):raise OSError('cleanup failure')
        with patch('research.h3_personal.voxcpm_endpoint_client.http.client.HTTPSConnection',BrokenClose):
            with self.assertRaisesRegex(RuntimeError,'Vox endpoint stream failed'):
                list(client.stream('hello',1,time.monotonic()+10))
        self.assertTrue(client._closed)
        self.assertTrue(client._lock.acquire(False))
        client._lock.release()

    def test_cancel_during_connect_and_late_connect(self):
        for late in (False, True):
            with self.subTest(late=late):
                client = self.setup_client()
                self.ready()
                entered = threading.Event()
                closed = threading.Event()
                release = threading.Event()
                errors = []
                instances = []
                owner = self

                class Connection:
                    sock = None
                    def __init__(self, host, timeout):
                        owner.assertGreater(timeout, 0)
                        owner.assertLessEqual(timeout, 1)
                        self.posts = 0
                        self.closes = 0
                        instances.append(self)
                    def connect(self):
                        entered.set()
                        if not (release if late else closed).wait(2):
                            raise AssertionError('test connect was not released')
                        if not late:
                            raise OSError('closed during connect')
                        # Simulate DNS/TLS finishing despite close().
                        self.sock = Socket()
                    def close(self):
                        self.closes += 1
                        if self.sock is not None:
                            self.sock.close()
                        closed.set()
                    def request(self, *args, **kwargs):
                        self.posts += 1
                        raise AssertionError('POST after cancellation')

                class Socket:
                    closed = False
                    def close(self): self.closed = True

                def consume():
                    try:
                        list(client.stream('hello', 1, time.monotonic()+120))
                    except RuntimeError as exc:
                        errors.append(exc)

                with patch('research.h3_personal.voxcpm_endpoint_client.http.client.HTTPSConnection',
                           Connection):
                    worker = threading.Thread(target=consume)
                    worker.start()
                    try:
                        self.assertTrue(entered.wait(1))
                        client.close()
                        self.assertTrue(closed.is_set())
                        if late:
                            self.assertTrue(worker.is_alive())
                            release.set()
                        worker.join(1)
                        self.assertFalse(worker.is_alive())
                    finally:
                        release.set()
                        closed.set()
                        worker.join(3)
                    self.assertEqual(len(errors), 1)
                    self.assertEqual(instances[0].posts, 0)
                    if late:
                        self.assertTrue(instances[0].sock.closed)
                    self.assertIsNone(client._active_connection)
                    self.assertIsNone(client._active_socket)
                    self.assertFalse(client._lock.locked())
                    with self.assertRaises(RuntimeError):
                        list(client.stream('hello', 2, time.monotonic()+10))
                    self.assertEqual(len(instances), 1)

    def test_ready_strict(self):
        mutations = [lambda r: r.update(extra=1), lambda r: r.pop('boot'),
                     lambda r: r.update(boot='wrong'), lambda r: r.update(session='wrong'),
                     lambda r: r.update(epoch=False), lambda r: r.update(channels=True),
                     lambda r: r.update(watermark_applied=0),
                     lambda r: r.update(sample_rate=48000.0),
                     lambda r: r.update(expires_at=True),
                     lambda r: r.update(expires_at=float('inf'))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                client = self.setup_client(mutate)
                with self.assertRaises(RuntimeError): self.ready()
                self.assertTrue(client._closed)
                self.assertFalse(client._ready)

    def test_no_post_before_readiness_or_synthesize(self):
        client = self.setup_client()
        with patch('research.h3_personal.voxcpm_endpoint_client.http.client.HTTPSConnection') as https:
            with self.assertRaises(RuntimeError):
                client.synthesize('hello', 1, '/unused', time.monotonic()+10)
            self.assertFalse(client._closed)
            with self.assertRaises(RuntimeError):
                list(client.stream('hello', 1, time.monotonic()+10))
            https.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_failure_deadline_cancel_abandon(self):
        for mode in ('invalid_end', 'disconnect', 'deadline', 'cancel', 'abandon'):
            with self.subTest(mode=mode):
                client = self.setup_client()
                self.ready()
                wire = self.wire()
                if mode == 'invalid_end': wire = wire[:-1]
                with patch('research.h3_personal.voxcpm_endpoint_client.http.client.HTTPSConnection',
                           self.connection(wire, fail=mode == 'disconnect')):
                    stream = client.stream('hello', 1, time.monotonic()+10)
                    if mode in ('invalid_end', 'disconnect'):
                        with self.assertRaises(RuntimeError): list(stream)
                    else:
                        next(stream)
                        if mode == 'cancel': client.close()
                        if mode == 'deadline':
                            with patch('research.h3_personal.voxcpm_endpoint_client.time.monotonic',
                                       return_value=time.monotonic()+200):
                                with self.assertRaises(RuntimeError): next(stream)
                        else:
                            stream.close()
                self.assertTrue(client._closed)
                self.assertIsNone(client._active_socket)
                self.assertFalse(client._lock.locked())


if __name__ == '__main__':
    unittest.main()
