"""Bounded, capture-free native endpoint ASR (stdlib only).

Call transcribe(endpoint, token=..., owner_key=..., pcm=iterable_or_binary_stream).
PCM is signed little-endian 16-bit, 16000 Hz mono; no resampling or silence is
invented. Iterable/read operations must be cooperative: Python cannot interrupt
arbitrary user code. A stuck input worker is daemonized and reported in cleanup.
Credentials are headers only. No CLI, persistence, capture or fallback exists.
"""
import base64
import copy
import hashlib
import json
import math
import re
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

from research.probe_h3_asr_cpu_space import frame, recv_frame


PATH = '/v1/audio/transcriptions/realtime'
PARTIAL = 'conversation.item.input_audio_transcription.delta'
FINAL = 'conversation.item.input_audio_transcription.completed'


def _host(endpoint):
    p = urlsplit(endpoint)
    if (p.scheme != 'https' or not p.hostname or p.username or p.password
            or p.port not in (None, 443) or p.query or p.fragment
            or p.path not in ('', '/', PATH)
            or not re.fullmatch(r'[a-z0-9-]+(?:\.[a-z0-9-]+)*\.huggingface\.cloud', p.hostname)):
        raise ValueError('HTTPS huggingface.cloud endpoint required')
    return p.hostname


def _credential(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ValueError('invalid credential')


def _connect(host, token, owner_key, timeout):
    connection = socket.create_connection((host, 443), timeout=timeout)
    source = None
    try:
        connection = ssl.create_default_context().wrap_socket(connection, server_hostname=host)
        connection.settimeout(timeout)
        key = base64.b64encode(__import__('os').urandom(16)).decode()
        connection.sendall((f'GET {PATH} HTTP/1.1\r\nHost: {host}\r\n'
                            f'Authorization: Bearer {token}\r\nX-H3-ASR-Key: {owner_key}\r\n'
                            'Connection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\n'
                            f'Sec-WebSocket-Key: {key}\r\n\r\n').encode())
        source = connection.makefile('rb')
        status = source.readline(4096).split()
        headers = {}
        for _ in range(64):
            line = source.readline(8192)
            if line == b'\r\n':
                break
            if not line.endswith(b'\r\n') or b':' not in line:
                raise ValueError('invalid upgrade')
            k, v = line.split(b':', 1)
            headers[k.lower()] = v.strip()
        else:
            raise ValueError('invalid upgrade')
        expected = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        if (len(status) < 2 or status[1] != b'101'
                or headers.get(b'sec-websocket-accept') != expected
                or headers.get(b'upgrade', b'').lower() != b'websocket'
                or b'upgrade' not in headers.get(b'connection', b'').lower().split(b', ')):
            raise ValueError('invalid upgrade')
        return connection, source
    except BaseException:
        if source is not None:
            source.close()
        connection.close()
        raise


def finalized_covers(processed, sent_seconds):
    """Compare 16k PCM sample counts, not lossy native float32 seconds.

    Rounding to nearest sample accommodates <half-sample representation error,
    not missing audio frames or a decoder-chunk-sized tolerance.
    """
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in (processed, sent_seconds)):
        return False
    return round(processed * 16000) >= round(sent_seconds * 16000)


def native_completed_text(events):
    """Join segments, not deltas; repeated spoken segments are preserved."""
    finals = [e for e in events if e.get('final') is True]
    if any(e.get('endpoint_source') != 'native_completed' or
           not isinstance(e.get('text'), str) or not e['text'].strip() for e in finals):
        raise ValueError('genuine native segments required')
    return ' '.join(e['text'].strip() for e in finals)


def transcribe(endpoint, *, token, owner_key, pcm, language='ja-JP',
               timeout=50.0, max_audio_seconds=45.0, join_timeout=1.0,
               max_events=512, completion_policy='first-final', on_event=None, on_input_stop=None,
               _transport=_connect):
    """Return ordered partial/final metadata and bounded cleanup diagnostics.

    Only a nonblank native completed event is final. Deltas are NOT promoted on
    EOF/deadline. Input is paced, limited to max_audio_seconds, and committed on
    EOF/limit. Legacy first-final cancels further sends. input-finalized collects
    ALL native segments until commit plus finalized coverage of all sent PCM;
    partials/EOF/deadline never complete a turn. This is bounded-window completion,
    not natural end-of-utterance detection. No claim is made about
    capture stopping: the caller owns capture and must stop it independently.
    A callable pcm is invoked only after authenticated transport/session setup;
    it must return the caller-owned binary stream/iterable (no eager capture).
    _transport is an injectable (host, token, owner_key, timeout) test factory.
    """
    if on_event is not None and not callable(on_event):
        raise ValueError('on_event must be callable')
    if on_input_stop is not None and not callable(on_input_stop):
        raise ValueError('on_input_stop must be callable')
    if completion_policy not in ('first-final', 'input-finalized'):
        raise ValueError('invalid completion policy')
    host = _host(endpoint)
    _credential(token)
    _credential(owner_key)
    if (not isinstance(language, str) or not re.fullmatch(r'[A-Za-z-]{2,32}', language)
            or any(not math.isfinite(v) or v <= 0 for v in (timeout, max_audio_seconds, join_timeout))
            or timeout > 120 or max_audio_seconds > 90 or join_timeout > 5
            or not isinstance(max_events, int) or not 1 <= max_events <= 4096):
        raise ValueError('invalid bounds')
    started = time.monotonic()
    deadline = started + timeout
    stop = threading.Event()
    lock = threading.Lock()
    state_lock = threading.Lock()
    resources = []
    events = []
    errors = []
    sent_bytes = 0
    committed = False
    final_text = None
    finalized_audio_seconds = 0.0

    def send(connection, payload, opcode):
        with lock:
            if stop.is_set():
                return False
            connection.sendall(frame(payload, opcode))
            return True

    def sender(connection):
        nonlocal sent_bytes, committed
        try:
            input_pcm = pcm() if callable(pcm) else pcm
            chunks = iter(lambda: input_pcm.read(5120), b'') if hasattr(input_pcm, 'read') else iter([input_pcm] if isinstance(input_pcm, bytes) else input_pcm)
            next_send = time.monotonic()
            limit = int(max_audio_seconds * 16000) * 2
            while not stop.is_set() and sent_bytes < limit:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    break
                if not isinstance(chunk, bytes) or len(chunk) % 2 or len(chunk) > 65536:
                    raise ValueError('invalid PCM chunk')
                for offset in range(0, min(len(chunk), limit - sent_bytes), 5120):
                    if stop.wait(max(0, next_send - time.monotonic())):
                        return
                    block = chunk[offset:offset + min(5120, limit - sent_bytes)]
                    if not send(connection, block, 2):
                        return
                    sent_bytes += len(block)
                    next_send = time.monotonic() + len(block) / 32000
            if stop.wait(max(0, next_send - time.monotonic())):
                return
            if sent_bytes == 0:
                # A local VAD may reject an entire authorized input window.
                # Close without a fabricated empty EOS/commit or final transcript.
                errors.append('no_input_audio')
                stop.set()
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # caller/receiver may already own intentional shutdown
                return
            with lock:
                if not stop.is_set():
                    connection.sendall(frame(b'{"type":"input_audio_buffer.commit"}', 1))
                    committed = True
        except Exception:
            errors.append('input_or_send_error')
            stop.set()

    workers = []

    def receiver():
        nonlocal final_text, finalized_audio_seconds
        connection = source = None
        try:
            connection, source = _transport(host, token, owner_key, min(timeout, 10))
            with state_lock:
                if stop.is_set():
                    return
                resources.extend((connection, source))
            connection.settimeout(max(.001, deadline - time.monotonic()))
            send(connection, json.dumps({'type': 'session.update', 'session': {
                'sample_rate': 16000, 'language': language, 'endpointing_ms': 800}}).encode(), 1)
            worker = threading.Thread(target=sender, args=(connection,), daemon=True)
            workers.append(worker)
            worker.start()
            fragments = bytearray()
            fragmented = False
            count = 0
            while not stop.is_set():
                fin, opcode, payload = recv_frame(source)
                if opcode in (8, 9, 10):
                    if not fin or len(payload) > 125:
                        raise ValueError('invalid control')
                    if opcode == 8:
                        break
                    if opcode == 9:
                        send(connection, payload, 10)
                    continue
                if opcode not in (0, 1) or (opcode == 0) != fragmented:
                    raise ValueError('invalid fragmentation')
                fragments.extend(payload)
                if len(fragments) > 65536:
                    raise ValueError('message bound')
                fragmented = not fin
                if not fin:
                    continue
                row = json.loads(fragments.decode())
                fragments.clear()
                count += 1
                if count > max_events or not isinstance(row, dict):
                    raise ValueError('event bound')
                kind = row.get('type')
                if kind == 'error':
                    errors.append('server_error')
                    break
                if kind not in (PARTIAL, FINAL):
                    continue
                text = row.get('delta' if kind == PARTIAL else 'transcript')
                if not isinstance(text, str):
                    raise ValueError('invalid transcript')
                processed = row.get('audio_processed')
                if processed is not None and (type(processed) not in (int, float)
                        or not math.isfinite(processed) or processed < 0):
                    raise ValueError('invalid audio coverage')
                genuine = kind == FINAL and bool(text.strip())
                events.append(dict(type='final' if kind == FINAL else 'partial', text=text,
                                   final=genuine, native_event_type=kind,
                                   endpoint_source='native_completed' if genuine else None,
                                   at_seconds=time.monotonic() - started,
                                   audio_processed=row.get('audio_processed'), on_device=False,
                                   capture_stopped=False))
                if on_event is not None:
                    snapshot = copy.deepcopy(events[-1])
                    snapshot.update(event='native_completed' if kind == FINAL else 'partial',
                                    semantic_label='native_segment' if kind == FINAL else 'hypothesis',
                                    provisional=True, turn_final=False)
                    try:
                        on_event(snapshot)
                    except BaseException:
                        errors.append('event_callback_error')
                        return
                processed = row.get('audio_processed')
                if kind == FINAL and type(processed) in (int, float):
                    if not math.isfinite(processed) or processed < 0:
                        raise ValueError('invalid finalized coverage')
                    finalized_audio_seconds = max(finalized_audio_seconds, processed)
                with lock:
                    input_committed = committed
                if completion_policy == 'input-finalized':
                    if (kind == FINAL and input_committed and sent_bytes > 0
                            and type(processed) in (int, float)
                            and finalized_covers(processed, sent_bytes / 32000)):
                        final_text = native_completed_text(events) or None
                        if final_text is None:
                            errors.append('no_native_final')
                        break
                    continue
                if genuine:
                    final_text = text
                    break
                if (kind == FINAL and committed
                        and isinstance(row.get('audio_processed'), (int, float))
                        and finalized_covers(row['audio_processed'], sent_bytes / 32000)):
                    # Empty flushed completion is not speech, and not a transport
                    # timeout. Never promote it or keep waiting for invented text.
                    errors.append('no_native_final')
                    break
        except Exception as exc:
            if not stop.is_set():
                errors.append('deadline' if isinstance(exc, TimeoutError)
                              and time.monotonic() >= deadline else 'transport_or_protocol_error')
        finally:
            # Explicit RTC adapter owns graceful carrier shutdown after native
            # final. Legacy WSS transport has no hook and is unchanged.
            finish=getattr(_transport,'finish',None)
            if final_text is not None and not errors and callable(finish):
                try:finish(connection,source,deadline)
                except Exception:errors.append('transport_close_error')
            stop.set()
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    if source is not None:
                        source.close()
                except Exception:
                    errors.append('transport_close_error')
                finally:
                    try:
                        connection.close()
                    except Exception:
                        errors.append('transport_close_error')

    reader = threading.Thread(target=receiver, daemon=True)
    reader.start()
    try:
        if not stop.wait(max(0, deadline - time.monotonic())):
            errors.append('deadline')
    finally:
        # Also runs on caller KeyboardInterrupt/SIGTERM converted by the owner.
        stop.set()
        with state_lock:
            if resources:
                try:
                    resources[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        # Stop caller-owned capture before joining a sender blocked in read().
        # Callback must itself be bounded; no arbitrary stream is hard-preempted.
        if on_input_stop is not None:
            try:
                on_input_stop()
            except Exception:
                errors.append('input_stop_error')
        join_end = time.monotonic() + join_timeout
        reader.join(max(0, join_end - time.monotonic()))
        for worker in list(workers):
            worker.join(max(0, join_end - time.monotonic()))
    alive = reader.is_alive() or any(w.is_alive() for w in workers)
    if alive:
        errors.append('cleanup_incomplete')
    if final_text is None and not errors:
        errors.append('no_native_final')
    result = dict(text=final_text, final=final_text is not None, events=list(events),
                errors=list(errors), cleanup_complete=not alive and 'transport_close_error' not in errors and 'input_stop_error' not in errors,
                audio_sent_seconds=sent_bytes / 32000, input_committed=committed,
                completion_policy=completion_policy, finalized_audio_seconds=finalized_audio_seconds,
                capture_stopped=False, audio_saved=False, on_device=False)
    if (on_event is not None and completion_policy == 'input-finalized'
            and result['final'] and not errors and result['cleanup_complete']
            and committed and sent_bytes > 0
            and finalized_covers(finalized_audio_seconds, sent_bytes / 32000)):
        try:
            on_event(dict(event='input_finalized', semantic_label='bounded_input_turn',
                          provisional=False, turn_final=True, text=final_text,
                          completion_policy=completion_policy, input_committed=True,
                          audio_sent_seconds=sent_bytes / 32000,
                          finalized_audio_seconds=finalized_audio_seconds,
                          cleanup_complete=True, errors=[], capture_stopped=False))
        except BaseException:
            result['errors'].append('event_callback_error')
    return result
