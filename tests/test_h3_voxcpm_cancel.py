"""CPU protocol/ownership proofs only; no native model cache-health claim."""
import io
import json
import struct
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL, decode, metadata
from research.h3_personal.voxcpm_endpoint_service import Gate, create_server


class Candidate:
    sample_rate = 48000
    def __init__(self):
        self.advances = 0
        self.closed = 0
        self.active = False
    def stream(self, text):
        if self.active:
            raise RuntimeError('overlap')
        self.active = True
        try:
            for value in (.25, -.5, .125):
                self.advances += 1
                yield [value]
        finally:
            self.closed += 1
            self.active = False


class Tests(unittest.TestCase):
    def gate(self, candidate=None, **kw):
        return Gate(candidate or Candidate(), key='key', session='s', boot='b',
                    expires_at=time.time()+60, cancel_protocol=CANCEL_PROTOCOL, **kw)
    def request(self, epoch=1):
        return dict(session='s', boot='b', epoch=epoch, text='hello', cancel_protocol=CANCEL_PROTOCOL)
    def cancel(self, epoch=1):
        return dict(session='s', boot='b', epoch=epoch, cancel_protocol=CANCEL_PROTOCOL)
    def unpack(self, gate, parts, epoch=1):
        expected = dict(session='s', boot='b', expires_at=gate.expires_at, epoch=epoch)
        iterator = decode(io.BytesIO(b''.join(parts)).read, expected, cancel_protocol=CANCEL_PROTOCOL)
        chunks = []
        while True:
            try:
                chunks.append(next(iterator))
            except StopIteration as end:
                return b''.join(chunks), end.value

    def test_cancel_between_yields_then_same_gate_next_complete(self):
        gate = self.gate(); parts = []; acks = []
        def write(data):
            parts.append(data)
            if data[:1] == b'P':
                acks.extend((gate.cancel(self.cancel()), gate.cancel(self.cancel())))
        gate.stream(self.request(), write)
        prefix, end = self.unpack(gate, parts)
        self.assertEqual(prefix, struct.pack('<f', .25))
        self.assertEqual(end['finish_reason'], 'cancelled')
        self.assertFalse(end['model_eos_verified'])
        self.assertTrue(end['native_closed'])
        self.assertEqual(acks[0], acks[1])
        self.assertEqual(gate.candidate.advances, 1)
        self.assertEqual(gate.candidate.closed, 1)
        self.assertTrue(gate.healthy())
        self.assertEqual(gate.cancel(self.cancel())['state'], 'cancelled')
        parts = []; gate.stream(self.request(2), parts.append)
        pcm, end = self.unpack(gate, parts, 2)
        self.assertEqual(pcm, struct.pack('<fff', .25, -.5, .125))
        self.assertEqual(end['finish_reason'], 'iterator_exhausted')
        self.assertEqual(gate.candidate.closed, 2)
        with self.assertRaises(ValueError):
            gate.cancel(self.cancel(1))

    def test_cancel_before_first_advance_zero_prefix_terminal(self):
        gate = self.gate(); parts = []
        def write(data):
            parts.append(data)
            if data[:1] == b'R':
                gate.cancel(self.cancel())
        gate.stream(self.request(), write)
        pcm, end = self.unpack(gate, parts)
        self.assertEqual(pcm, b'')
        self.assertEqual(end['samples'], 0)
        self.assertEqual(end['finish_reason'], 'cancelled')
        self.assertEqual(gate.candidate.advances, 0)

    def test_cancel_during_native_next_only_owner_closes(self):
        entered = threading.Event(); release = threading.Event(); close_threads = []
        class Blocking(Candidate):
            def stream(self, text):
                try:
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError('test release missing')
                    yield [.25]
                    raise AssertionError('cancel must stop next advance')
                finally:
                    close_threads.append(threading.get_ident())
        gate = self.gate(Blocking()); parts = []; errors = []
        def run():
            try: gate.stream(self.request(), parts.append)
            except BaseException as error: errors.append(error)
        owner = threading.Thread(target=run); owner.start()
        try:
            self.assertTrue(entered.wait(2))
            ack = gate.cancel(self.cancel())
            self.assertEqual(ack['state'], 'requested')
            self.assertEqual(close_threads, [])
            self.assertIsNone(ack['native_closed_monotonic_ns'])
        finally:
            release.set(); owner.join(3)
        self.assertFalse(owner.is_alive()); self.assertEqual(errors, [])
        self.assertEqual(close_threads, [owner.ident])
        pcm, end = self.unpack(gate, parts)
        self.assertEqual(pcm, struct.pack('<f', .25))
        self.assertEqual(end['finish_reason'], 'cancelled')

    def test_eos_wins_late_cancel_and_next_request_is_not_cancelled(self):
        gate = self.gate(); parts = []; acks = []
        def write(data):
            if data[:1] == b'E': acks.append(gate.cancel(self.cancel()))
            parts.append(data)
        gate.stream(self.request(), write)
        self.assertEqual(acks[0]['state'], 'settling')
        self.assertIsNone(acks[0]['cancel_monotonic_ns'])
        self.assertEqual(self.unpack(gate, parts)[1]['finish_reason'], 'iterator_exhausted')
        self.assertEqual(gate.cancel(self.cancel())['state'], 'iterator_exhausted')
        parts = []; gate.stream(self.request(2), parts.append)
        self.assertEqual(self.unpack(gate, parts, 2)[1]['finish_reason'], 'iterator_exhausted')

    def test_invalid_cancel_does_not_change_owner(self):
        gate = self.gate(); parts = []
        with self.assertRaises(ValueError): gate.cancel(self.cancel())
        def write(data):
            parts.append(data)
            if data[:1] == b'P':
                for change in ({'boot':'wrong'}, {'session':'wrong'}, {'epoch':True},
                               {'epoch':2}, {'cancel_protocol':'v0'}, {'extra':1}):
                    with self.assertRaises(ValueError): gate.cancel(dict(self.cancel(), **change))
        gate.stream(self.request(), write)
        self.assertEqual(self.unpack(gate, parts)[1]['finish_reason'], 'iterator_exhausted')

    def test_disconnect_or_consumer_failure_is_not_successful_cancel(self):
        for failed_kind in (b'P', b'E'):
            gate = self.gate(); parts = []
            def write(data):
                if data[:1] == failed_kind: raise BrokenPipeError()
                parts.append(data)
            with self.assertRaises(BrokenPipeError): gate.stream(self.request(), write)
            self.assertFalse(gate.healthy())
            self.assertFalse(any(p[:1] == b'E' for p in parts))
            with self.assertRaises(ValueError): gate.cancel(self.cancel())

    def test_close_failure_poisoned_no_success_terminal(self):
        class Broken(Candidate):
            def stream(self, text):
                try: yield [.25]
                finally: raise RuntimeError('native cleanup failed')
        gate = self.gate(Broken()); parts = []
        def write(data):
            parts.append(data)
            if data[:1] == b'P': gate.cancel(self.cancel())
        with self.assertRaises(RuntimeError): gate.stream(self.request(), write)
        self.assertFalse(gate.healthy())
        self.assertFalse(any(p[:1] == b'E' for p in parts))

    def test_uncloseable_iterator_cannot_claim_native_closed(self):
        class Iterator:
            def __iter__(self): return self
            def __next__(self):
                if getattr(self, 'done', False): raise StopIteration
                self.done = True
                return [.25]
        class Uncloseable(Candidate):
            def stream(self, text): return Iterator()
        gate = self.gate(Uncloseable()); parts = []
        with self.assertRaises(RuntimeError): gate.stream(self.request(), parts.append)
        self.assertFalse(gate.healthy())
        self.assertFalse(any(p[:1] == b'E' for p in parts))

    def test_terminal_prefix_hash_and_native_closed_are_strict(self):
        gate = self.gate(); parts = []
        def write(data):
            parts.append(data)
            if data[:1] == b'P': gate.cancel(self.cancel())
        gate.stream(self.request(), write)
        for change in ({'samples':0}, {'sha256':'0'*64}, {'epoch':2},
                       {'native_closed':False}, {'native_closed':1}, {'model_eos_verified':True}):
            end = json.loads(parts[-1][5:]); end.update(change)
            with self.assertRaises(ValueError):
                self.unpack(gate, parts[:-1]+[metadata(b'E', end)])
        with self.assertRaises(ValueError): self.unpack(gate, parts[:-1])
        with self.assertRaises(ValueError): self.unpack(gate, parts+[b'x'])
        with self.assertRaises(ValueError):
            list(decode(io.BytesIO(b''.join(parts)).read,
                        dict(session='s', boot='b', epoch=1, expires_at=gate.expires_at)))

    def test_legacy_cancel_disabled_ready_unchanged(self):
        gate = Gate(Candidate(), key='key', session='s', boot='b', expires_at=time.time()+60)
        self.assertNotIn('cancel_protocol', gate.ready(0))
        with self.assertRaises(ValueError): gate.cancel(self.cancel())
        request = self.request(); del request['cancel_protocol']
        parts = []; gate.stream(request, parts.append)
        expected = dict(session='s', boot='b', epoch=1, expires_at=gate.expires_at)
        self.assertEqual(len(list(decode(io.BytesIO(b''.join(parts)).read, expected))), 3)

    def test_http_cancel_authenticated_and_independent_of_busy_media(self):
        entered = threading.Event(); release = threading.Event()
        class Blocking(Candidate):
            def stream(self, text):
                entered.set()
                if not release.wait(3): raise TimeoutError()
                yield [.25]
        server = create_server(Blocking(), host='127.0.0.1', port=0, key='key',
            session='s', boot='b', expires_at=time.time()+60, cancel_protocol=CANCEL_PROTOCOL)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        url = f'http://127.0.0.1:{server.server_address[1]}'
        parts = []; errors = []
        def media():
            try: server.gate.stream(self.request(), parts.append)
            except BaseException as error: errors.append(error)
        owner = threading.Thread(target=media); owner.start()
        try:
            self.assertTrue(entered.wait(2))
            body = json.dumps(self.cancel()).encode()
            with self.assertRaises(HTTPError) as denied:
                urlopen(Request(url+'/cancel', data=body), timeout=2)
            self.assertEqual(denied.exception.code, 401)
            denied.exception.close()
            with urlopen(Request(url+'/cancel', data=body,
                         headers={'X-H3-TTS-Key':'key'}), timeout=2) as response:
                self.assertEqual(json.load(response)['state'], 'requested')
            self.assertTrue(owner.is_alive())
        finally:
            release.set(); owner.join(3); server.shutdown(); server.server_close(); thread.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(self.unpack(server.gate, parts)[1]['finish_reason'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
