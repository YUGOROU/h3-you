"""Native PCM iterator client; no local fallback or provider lifecycle actions."""
import http.client
import json
import time
import threading
import socket

from .irodori_endpoint_client import RemoteIrodoriTTS
from .live_relay_client import _DeadlineSocket
from .voxcpm_endpoint_protocol import (CONTENT_TYPE, PINS, decode, loads,
    CANCEL_PROTOCOL, CANCEL_CONTENT_TYPE)


class RemoteVoxCPM(RemoteIrodoriTTS):
    """HF HTTPS/app-auth conventions shared with Irodori, streaming API distinct.

    Consume stream() fully to verify end. Always close the generator on early
    abandonment. PCM yielded before end remains provisional, not accepted audio.

    Native stream connect uses a <=1s socket timeout (not an aggregate deadline
    across addresses/TCP/TLS). Python getaddrinfo is not bounded by that timeout;
    close cannot interrupt DNS or a connect whose socket is not yet published.
    No resolver substitution or background connector is used. A late connection
    is closed without sending a request after cancellation/deadline expiry.
    """
    def __init__(self, base_url, *, boot, cancel_protocol=None, session_seconds=1200, **config):
        if type(session_seconds) is not int or session_seconds not in (1200, 10800):
            raise ValueError('explicit finite Vox session duration required')
        if not isinstance(boot, str) or not boot:
            raise ValueError('boot required')
        if cancel_protocol not in (None, CANCEL_PROTOCOL):
            raise ValueError('unsupported cancel protocol')
        super().__init__(base_url, **config)
        self.boot = boot
        self.session_seconds = session_seconds
        self._max_completion_receipts = 1000 if session_seconds == 10800 else 2
        self.cancel_protocol = cancel_protocol
        self.last_completion = None
        self._active_connection = None
        self._control_connection = None
        self._control_lock = threading.Lock()

    def close(self):
        # Publish cancellation before releasing the lock: admission and connection
        # registration use this same lock, so close cannot miss a new connection.
        with self._socket_lock:
            self._closed = True
            conn = self._active_connection
            control = self._control_connection
        try:
            super().close()
        except Exception:
            pass  # cancellation is terminal; cleanup cannot replace its cause
        finally:
            for connection in (conn, control):
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass

    def request_cancel(self, epoch, deadline):
        """Independent bounded control lane; ACK is not verified stream END.

        Failure leaves media readable for bounded drain. The stream lock still
        prevents any next request until verified terminal/EOF or session failure.
        """
        if not self._control_lock.acquire(False):
            raise RuntimeError('cancel control busy')
        conn = None
        try:
            if (self.cancel_protocol is None or not self._ready
                    or type(epoch) is not int or epoch != self._epoch
                    or type(deadline) not in (int, float)
                    or not __import__('math').isfinite(deadline)):
                raise ValueError('invalid cancel epoch')
            deadline = min(deadline, time.monotonic()+5)
            self._remaining(deadline)
            # Reuse only the already-authenticated media TCP peer, not DNS or
            # the media socket itself. HTTPSConnection still performs TLS with
            # the ORIGINAL hostname/SNI/certificate validation. No new resolver,
            # provider or unbounded getaddrinfo on this deadline-critical lane.
            with self._socket_lock:
                self._remaining(deadline)
                media_socket = self._active_socket
                if media_socket is None:
                    raise ValueError('no active media peer; use verified terminal or bounded drain')
                peer, family = media_socket.getpeername(), media_socket.family
            if family not in (socket.AF_INET, socket.AF_INET6):
                raise ValueError('unsupported authenticated peer family')
            conn = http.client.HTTPSConnection(self._host,
                timeout=min(1.0, self._remaining(deadline)))
            def dial_peer(_address, timeout=None, source_address=None):
                raw = socket.socket(family, socket.SOCK_STREAM)
                try:
                    raw.settimeout(min(1.0, self._remaining(deadline)))
                    raw.connect(peer)
                    self._remaining(deadline)
                    return raw
                except BaseException:
                    raw.close()
                    raise
            conn._create_connection = dial_peer
            with self._socket_lock:
                self._remaining(deadline)
                self._control_connection = conn
            conn.connect()
            with self._socket_lock:
                self._remaining(deadline)
                if epoch != self._epoch:
                    raise ValueError('late cancel connection')
            conn.sock = _DeadlineSocket(conn.sock, lambda: self._remaining(deadline))
            conn.request('POST', '/cancel', body=json.dumps(dict(session=self.session,
                boot=self.boot, epoch=epoch, cancel_protocol=self.cancel_protocol)).encode(),
                headers=self._headers)
            response = conn.getresponse()
            if (response.status != 200 or response.getheader('Content-Type') != 'application/json'
                    or response.headers.get_all('Content-Encoding')
                    or response.headers.get_all('Transfer-Encoding', []) not in ([], ['chunked'])):
                raise ValueError('cancel not accepted')
            body = bytearray()
            while True:
                self._remaining(deadline)
                part = response.read1(min(4097-len(body), 1024))
                if not part: break
                body.extend(part)
                if len(body) > 4096: raise ValueError('cancel response bound')
            self._remaining(deadline)
            result = loads(body)
            fields = {'cancel_protocol','session','boot','epoch','state',
                      'cancel_monotonic_ns','native_closed_monotonic_ns','terminal_monotonic_ns'}
            if (type(result) is not dict or set(result) != fields
                    or result['cancel_protocol'] != self.cancel_protocol
                    or result['session'] != self.session or result['boot'] != self.boot
                    or type(result['epoch']) is not int or result['epoch'] != epoch
                    or result['state'] not in ('requested','settling','cancelled','iterator_exhausted')
                    or any(result[k] is not None and (type(result[k]) is not int or result[k] <= 0)
                           for k in fields if k.endswith('_ns'))):
                raise ValueError('cancel acknowledgment binding')
            return result
        except Exception:
            raise RuntimeError('Vox cancel control failed; terminal still required') from None
        finally:
            with self._socket_lock:
                self._control_connection = None
            try:
                if conn is not None: conn.close()
            except Exception:
                pass
            finally:
                self._control_lock.release()

    def read(self, kind, deadline):
        def operation():
            if kind != 'tts_ready' or self._ready:
                raise ValueError()
            result = self._request('GET', '/ready', None, deadline)
            expected = dict(PINS, session=self.session, boot=self.boot,
                            expires_at=self.expires_at, epoch=0)
            if self.cancel_protocol is not None:
                expected['cancel_protocol'] = self.cancel_protocol
            if (result.keys() != expected.keys()
                    or any(type(result[k]) is not type(v) or result[k] != v
                           for k, v in expected.items())):
                raise ValueError()
            self._ready = True
            return result
        return self._run(operation)

    def progress(self, epoch, deadline):
        """One bounded authenticated snapshot; never replay cancel or inference."""
        def operation():
            if self.cancel_protocol is None or type(epoch) is not int or epoch<=0:
                raise ValueError('explicit native progress epoch required')
            result=self._request('GET','/progress',None,min(deadline,time.monotonic()+2))
            keys={'schema','session','boot','cancel_protocol','epoch','state','terminal',
                  'samples','chunks','observed_monotonic_ns'}
            if (type(result) is not dict or set(result)!=keys
                    or result['schema']!='h3-native-progress-v1'
                    or result['session']!=self.session or result['boot']!=self.boot
                    or result['cancel_protocol']!=self.cancel_protocol
                    or type(result['epoch']) is not int or result['epoch'] not in (0,epoch)
                    or result['state'] not in ('idle','streaming','cancel_requested','cancelled','iterator_exhausted')
                    or type(result['terminal']) is not bool
                    or type(result['samples']) is not int or not 0<=result['samples']<=48000*30
                    or type(result['chunks']) is not int or not 0<=result['chunks']<=1024
                    or type(result['observed_monotonic_ns']) is not int or result['observed_monotonic_ns']<=0
                    or (result['epoch']==0 and (result['state']!='idle' or result['samples'] or result['chunks'] or result['terminal']))
                    or (result['epoch']==epoch and result['state']=='idle')
                    or (result['terminal'] and result['state'] not in ('cancelled','iterator_exhausted'))):
                raise ValueError('native progress binding mismatch')
            return result
        return self._run(operation)

    def evidence(self, deadline):
        """Bounded authenticated metadata only; independent of media END proof."""
        def operation():
            result = self._request('GET', '/evidence', None, deadline)
            expected = dict(PINS, session=self.session, boot=self.boot,
                            expires_at=self.expires_at, epoch=0)
            if self.cancel_protocol is not None:expected['cancel_protocol']=self.cancel_protocol
            from .voxcpm_endpoint_protocol import _exact_metadata
            if (type(result) is not dict or set(result)!={'binding','evidence'}
                    or not _exact_metadata(result['binding'],expected)
                    or type(result['evidence']) is not dict
                    or set(result['evidence'])!={'runtime','completions'}
                    or type(result['evidence']['runtime']) is not dict
                    or type(result['evidence']['completions']) is not list
                    or len(result['evidence']['completions'])>self._max_completion_receipts):
                raise ValueError('native evidence binding mismatch')
            return result
        return self._run(operation)

    def synthesize(self, text, turn_id, directory, deadline):
        raise RuntimeError('Vox endpoint is native streaming-only; use stream')

    def stream(self, text, epoch, deadline):
        if not self._lock.acquire(False):
            raise RuntimeError('busy')
        conn = None
        complete = False
        try:
            if (not self._ready or not isinstance(text, str) or not text.strip() or len(text) > 160
                    or type(epoch) is not int or epoch <= self._epoch
                    or type(deadline) not in (int, float) or not __import__('math').isfinite(deadline)):
                raise ValueError()
            deadline = min(deadline, time.monotonic()+120)
            self._remaining(deadline)
            self._epoch = epoch  # reserve once; no replay after uncertain result
            self.last_completion = None
            conn = http.client.HTTPSConnection(
                self._host, timeout=min(1.0, self._remaining(deadline)))
            with self._socket_lock:
                self._remaining(deadline)
                self._active_connection = conn
            conn.connect()
            with self._socket_lock:
                self._remaining(deadline)
                self._active_socket = conn.sock
            conn.sock = _DeadlineSocket(conn.sock, lambda: self._remaining(deadline))
            request = dict(session=self.session, boot=self.boot, epoch=epoch, text=text)
            if self.cancel_protocol is not None:
                request['cancel_protocol'] = self.cancel_protocol
            conn.request('POST', '/stream', body=json.dumps(request).encode(), headers=self._headers)
            response = conn.getresponse()
            if (response.status != 200 or response.getheader('Content-Type') != (
                        CANCEL_CONTENT_TYPE if self.cancel_protocol is not None else CONTENT_TYPE)
                    or response.headers.get_all('Content-Encoding')
                    or response.headers.get_all('Content-Length')
                    or response.headers.get_all('Transfer-Encoding', []) not in ([], ['chunked'])):
                raise ValueError()
            expected = dict(session=self.session, boot=self.boot,
                            expires_at=self.expires_at, epoch=epoch)
            def read(size):
                self._remaining(deadline)
                data = response.read1(size)
                self._remaining(deadline)
                return data
            terminal = yield from decode(read, expected, cancel_protocol=self.cancel_protocol)
            self._remaining(deadline)
            self.last_completion = terminal
            complete = True
        except Exception:
            raise RuntimeError('Vox endpoint stream failed') from None
        finally:
            if not complete:
                self.close()
            with self._socket_lock:
                self._active_socket = None
                self._active_connection = None
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            finally:
                self._lock.release()
