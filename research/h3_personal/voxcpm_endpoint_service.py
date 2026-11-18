"""Finite serial native Vox endpoint; deployment/termination owned externally."""
import hashlib
import math
import threading
import time

from .irodori_endpoint_service import _Server, _Handler
from .voxcpm_endpoint_protocol import (PINS, CONTENT_TYPE, MAX_SAMPLES,
    MAX_CHUNKS, loads, metadata, frame, pcm_bytes, CANCEL_PROTOCOL, CANCEL_CONTENT_TYPE)


class Gate:
    def __init__(self, candidate, *, key, session, boot, expires_at, max_requests=2,
                 cancel_protocol=None, session_seconds=1200):
        if type(session_seconds) is not int or session_seconds not in (1200, 10800):
            raise ValueError('explicit finite session duration required')
        request_limit = 1000 if session_seconds == 10800 else 16
        if (any(not isinstance(v, str) or not v or '\r' in v or '\n' in v
                for v in (key, session, boot)) or type(expires_at) not in (int, float)
                or not math.isfinite(expires_at) or not time.time() < expires_at <= time.time()+session_seconds
                or type(max_requests) is not int or not 1 <= max_requests <= request_limit
                or candidate.sample_rate != 48000):
            raise ValueError('invalid service configuration')
        if cancel_protocol not in (None, CANCEL_PROTOCOL):
            raise ValueError('unsupported cancel protocol')
        self.cancel_protocol = cancel_protocol
        self._state_lock = threading.Lock()
        self._cancel_row = None
        self.candidate = candidate
        self.key, self.session, self.boot = key, session, boot
        self.expires_at, self.max_requests = expires_at, max_requests
        self.slot = threading.Lock()
        self.epoch = self.count = 0
        self.closed = False

    def healthy(self):
        return not self.closed and time.time() < self.expires_at

    def close(self):
        self.closed = True

    def ready(self, epoch):
        result = dict(PINS, session=self.session, boot=self.boot,
                      expires_at=self.expires_at, epoch=epoch)
        if self.cancel_protocol is not None:
            result['cancel_protocol'] = self.cancel_protocol
        return result

    def progress(self):
        """Authenticated protocol progress, not a promise of future activity.

        Counts are whole yielded chunks already sent. Native EOS can win just
        after this snapshot; only a subsequent verified cancelled END is success.
        """
        if self.cancel_protocol is None or not self.healthy():
            raise ValueError('native progress unavailable')
        with self._state_lock:
            row=self._cancel_row
            return dict(schema='h3-native-progress-v1',session=self.session,boot=self.boot,
                cancel_protocol=self.cancel_protocol,epoch=0 if row is None else row['epoch'],
                state=('idle' if row is None else row['decided'] if row['decided'] is not None
                       else 'cancel_requested' if row['event'].is_set() else 'streaming'),
                terminal=False if row is None else row['terminal'],
                samples=0 if row is None else row['samples'],chunks=0 if row is None else row['chunks'],
                observed_monotonic_ns=time.monotonic_ns())

    def cancel(self, request):
        """Request only; never close/advance an iterator on the control thread.

        Boot/session/epoch uniquely name a request. Repeated current-epoch calls
        are idempotent; once a newer request is admitted even old retries fail.
        A requested/settling response is NOT verified media-terminal evidence.
        """
        if (self.cancel_protocol is None or type(request) is not dict
                or set(request) != {'session', 'boot', 'epoch', 'cancel_protocol'}
                or request['session'] != self.session or request['boot'] != self.boot
                or request['cancel_protocol'] != self.cancel_protocol
                or type(request['epoch']) is not int):
            raise ValueError('invalid cancel binding')
        with self._state_lock:
            row = self._cancel_row
            if not self.healthy() or row is None or request['epoch'] != row['epoch']:
                raise ValueError('stale or unavailable cancel')
            if row['decided'] is None and not row['event'].is_set():
                row['cancel_monotonic_ns'] = time.monotonic_ns()
                row['event'].set()
            state = (row['decided'] if row['terminal'] else
                     'settling' if row['decided'] is not None else 'requested')
            return dict(cancel_protocol=self.cancel_protocol, session=self.session,
                        boot=self.boot, epoch=row['epoch'], state=state,
                        cancel_monotonic_ns=row['cancel_monotonic_ns'],
                        native_closed_monotonic_ns=row['native_closed_monotonic_ns'],
                        terminal_monotonic_ns=row['terminal_monotonic_ns'])

    def stream(self, request, write):
        keys = {'session', 'boot', 'epoch', 'text'}
        if self.cancel_protocol is not None:
            keys.add('cancel_protocol')
        if (not isinstance(request, dict) or set(request) != keys
                or (self.cancel_protocol is not None
                    and request['cancel_protocol'] != self.cancel_protocol)
                or request['session'] != self.session or request['boot'] != self.boot
                or type(request['epoch']) is not int or request['epoch'] <= self.epoch
                or not isinstance(request['text'], str) or not request['text'].strip()
                or len(request['text']) > 160):
            raise ValueError('invalid request')
        if not self.slot.acquire(False):
            raise ValueError('busy')
        iterator = None
        try:
            if not self.healthy() or self.count >= self.max_requests or request['epoch'] <= self.epoch:
                raise ValueError('admission closed')
            with self._state_lock:
                self.epoch = request['epoch']
                self.count += 1
                row = dict(epoch=self.epoch, event=threading.Event(), decided=None,
                           terminal=False, samples=0, chunks=0, cancel_monotonic_ns=None,
                           native_closed_monotonic_ns=None, terminal_monotonic_ns=None)
                self._cancel_row = row
            deadline = min(time.monotonic()+120, time.monotonic()+self.expires_at-time.time())
            def emit(data):
                if not self.healthy() or time.monotonic() >= deadline:
                    raise ValueError('expired')
                write(data)
            emit(metadata(b'R', self.ready(self.epoch)))
            digest = hashlib.sha256()
            samples = chunks = 0
            iterator = iter(self.candidate.stream(request['text']))
            reason = 'iterator_exhausted'
            while True:
                # Only the owning media thread advances or closes native state.
                # If cancel arrives in next(), transmit that entire yielded chunk
                # unchanged, then stop before another native advancement.
                with self._state_lock:
                    if self.cancel_protocol is not None and row['event'].is_set():
                        reason = 'cancelled'
                        row['decided'] = reason
                        break
                try:
                    chunk = next(iterator)
                except StopIteration:
                    with self._state_lock:
                        row['decided'] = reason
                    break
                data = pcm_bytes(chunk)
                samples += len(data)//4
                chunks += 1
                if samples > MAX_SAMPLES or chunks > MAX_CHUNKS:
                    raise ValueError('stream bound')
                digest.update(data)
                emit(frame(b'P', data))
                with self._state_lock:
                    row['samples'],row['chunks']=samples,chunks
            if not samples and reason != 'cancelled':
                raise ValueError('empty stream')
            if self.cancel_protocol is not None:
                # Cleanup failure must poison the Gate, never produce a successful
                # cancelled terminal. This proves iterator closure, NOT GPU cache
                # health; that still needs the real subsequent native request.
                close = getattr(iterator, 'close', None)
                if not callable(close):
                    raise RuntimeError('cancel protocol requires owned iterator close')
                close()
                iterator = None
                with self._state_lock:
                    row['native_closed_monotonic_ns'] = time.monotonic_ns()
            end = dict(samples=samples, chunks=chunks, sha256=digest.hexdigest(),
                       finish_reason=reason, model_eos_verified=False)
            if self.cancel_protocol is not None:
                end.update(cancel_protocol=self.cancel_protocol, epoch=self.epoch,
                           native_closed=True)
            emit(metadata(b'E', end))
            with self._state_lock:
                row['terminal'] = True
                row['terminal_monotonic_ns'] = time.monotonic_ns()
        except BaseException:
            self.close()
            try:
                write(metadata(b'X', {'error':'stream failed'}))
            except BaseException:
                pass
            raise
        finally:
            try:
                if iterator is not None and hasattr(iterator, 'close'):
                    iterator.close()
            finally:
                self.slot.release()


class Handler(_Handler):
    def do_GET(self):
        gate = self.server.gate
        if self.path == '/health':
            return self.reply(200 if gate.healthy() else 503, {'ready':gate.healthy()})
        if not self.authorized():
            return self.reply(401, {'error':'unauthorized'})
        if self.path == '/progress':
            try:return self.reply(200,gate.progress())
            except ValueError:return self.reply(503,{'error':'progress unavailable'})
        if self.path == '/evidence':
            export=getattr(gate.candidate,'export_evidence',None)
            if not callable(export):return self.reply(503,{'error':'unavailable'})
            return self.reply(200,dict(binding=gate.ready(0),evidence=export()))
        if self.path != '/ready' or not gate.healthy():
            return self.reply(503, {'error':'unavailable'})
        self.reply(200, gate.ready(0))

    def do_POST(self):
        if not self.authorized():
            return self.reply(401, {'error':'unauthorized'})
        lengths = self.headers.get_all('Content-Length', [])
        if (self.path not in ('/stream', '/cancel') or self.headers.get_all('Transfer-Encoding')
                or self.headers.get_all('Upgrade') or len(lengths) != 1
                or not lengths[0].isascii() or not lengths[0].isdigit()
                or not 1 <= int(lengths[0]) <= 8192):
            return self.reply(400, {'error':'invalid request'})
        try:
            end = time.monotonic()+5
            body = bytearray()
            while len(body) < int(lengths[0]):
                self.connection.settimeout(max(.001, end-time.monotonic()))
                part = self.rfile.read1(int(lengths[0])-len(body))
                if not part or time.monotonic() >= end:
                    raise ValueError()
                body.extend(part)
            request = loads(body)
        except Exception:
            return self.reply(400, {'error':'invalid request'})
        if self.path == '/cancel':
            try:
                result = self.server.gate.cancel(request)
            except ValueError:
                return self.reply(400, {'error': 'invalid cancel'})
            return self.reply(200, result)
        self.send_response(200)
        self.send_header('Content-Type', CANCEL_CONTENT_TYPE
                         if self.server.gate.cancel_protocol is not None else CONTENT_TYPE)
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        end = min(time.monotonic()+120, time.monotonic()+self.server.gate.expires_at-time.time())
        def write(data):
            remaining = end-time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            self.connection.settimeout(remaining)
            self.wfile.write(data)
            self.wfile.flush()
        try:
            self.server.gate.stream(request, write)
        except Exception:
            pass  # partial/missing end is terminal; never log text or secrets


def create_server(candidate, *, host='127.0.0.1', port=7860, **config):
    # Reuse Irodori's four-handler bound and five-second accepted socket timeout.
    server = _Server((host, port), Gate(candidate, **config))
    server.RequestHandlerClass = Handler
    return server


def serve(candidate, **config):
    server = create_server(candidate, **config)
    timer = threading.Timer(max(0, server.gate.expires_at-time.time()),
                            lambda: (server.gate.close(), server.shutdown()))
    timer.daemon = True
    timer.start()
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        timer.cancel()
        server.gate.close()
        server.server_close()
