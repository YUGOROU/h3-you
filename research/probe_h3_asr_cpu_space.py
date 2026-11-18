"""Explicit PUBLIC-file-only paced WebSocket probe of an owned private CPU Space.
No microphone, model installation, GPU allocation or paid upgrade. HF token stays
in memory/auth headers. The Space is paused on success/failure unless --keep-running
is explicitly chosen. Private and anonymous-denied checks precede all audio.
"""
import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
import struct
import threading
import time
from urllib.parse import urlsplit
import wave


def read_exact(stream, n):
    out = bytearray()
    while len(out) < n:
        value = stream.read(n-len(out))
        if not value:
            raise EOFError('WebSocket peer ended')
        out.extend(value)
    return bytes(out)


def frame(payload, opcode=2):
    if len(payload) > 65536:
        raise ValueError('bounded client frame required')
    key = os.urandom(4)
    n = len(payload)
    header = bytes([0x80 | opcode])
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack('!H', n)
    else:
        header += bytes([0x80 | 127]) + struct.pack('!Q', n)
    return header + key + bytes(v ^ key[i % 4] for i, v in enumerate(payload))


def recv_frame(stream):
    a, b = read_exact(stream, 2)
    if a & 0x70 or b & 0x80:
        raise ValueError('unnegotiated extension/masked server frame')
    n = b & 127
    if n == 126:
        n = struct.unpack('!H', read_exact(stream, 2))[0]
    elif n == 127:
        n = struct.unpack('!Q', read_exact(stream, 8))[0]
    if n > 65536:
        raise ValueError('bounded server frame required')
    return bool(a & 0x80), a & 15, read_exact(stream, n)


def handshake(host, token, extra_headers=None):
    connection = ssl.create_default_context().wrap_socket(socket.create_connection((host, 443), timeout=15), server_hostname=host)
    connection.settimeout(90)
    key = base64.b64encode(os.urandom(16)).decode()
    path = '/v1/audio/transcriptions/realtime'
    extras = extra_headers or {}
    if any(k != 'X-H3-ASR-Key' or '\r' in v or '\n' in v for k, v in extras.items()):
        connection.close()
        raise ValueError('only bounded owner-key header allowed')
    extra = ''.join(f'{k}: {v}\r\n' for k, v in extras.items())
    request = f'GET {path} HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {token}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {key}\r\n{extra}\r\n'
    try:
        connection.sendall(request.encode())
        stream = connection.makefile('rb')
        status = stream.readline(4096).decode().strip()
        headers = {}
        for _ in range(64):
            line = stream.readline(8192)
            if line == b'\r\n':
                break
            if not line or b':' not in line:
                raise ValueError('invalid upgrade headers')
            k, v = line.decode().split(':', 1)
            headers[k.lower()] = v.strip()
        else:
            raise ValueError('too many upgrade headers')
        expected = base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        if status.split()[1] != '101' or headers.get('sec-websocket-accept') != expected:
            raise ValueError('WebSocket upgrade did not succeed')
        return connection, stream
    except BaseException:
        connection.close()
        raise


def http_ready(host, token=None):
    c = http.client.HTTPSConnection(host, timeout=15)
    try:
        c.request('GET', '/ready', headers={'Authorization':'Bearer '+token} if token else {})
        r = c.getresponse()
        body = r.read(16384)
        return r.status, body
    finally:
        c.close()


def summarize(events, audio_seconds, input_end):
    finals = [r for r in events if r['event'].get('type') == 'conversation.item.input_audio_transcription.completed' and r['event'].get('transcript', '').strip()]
    partials = [r for r in events if r['event'].get('type') == 'conversation.item.input_audio_transcription.delta' and r['event'].get('delta', '').strip()]
    progress = [r for r in events if isinstance(r['event'].get('audio_processed'), (int, float))]
    ratio = None
    if len(progress) >= 2:
        duration = progress[-1]['event']['audio_processed'] - progress[0]['event']['audio_processed']
        if duration > 0:
            ratio = (progress[-1]['at_seconds']-progress[0]['at_seconds']) / duration
    last = finals[-1]['at_seconds'] if finals else None
    return dict(audio_seconds=audio_seconds, input_end_seconds=input_end,
                first_partial_seconds=partials[0]['at_seconds'] if partials else None,
                first_final_seconds=finals[0]['at_seconds'] if finals else None,
                last_final_seconds=last, final_after_input_end_seconds=max(0, last-input_end) if last is not None else None,
                observed_progress_wall_per_audio_second=ratio,
                partial_before_input_end=bool(partials and partials[0]['at_seconds'] < input_end),
                final_received=bool(finals), final_segments=[r['event']['transcript'] for r in finals])


def stream_public(host, token, pcm, rate, extra_headers=None, language='en-US'):
    connection, source = handshake(host, token, extra_headers)
    lock = threading.Lock()
    started = time.monotonic()
    events = []
    failures = []
    done = threading.Event()
    def send(payload, opcode):
        with lock:
            connection.sendall(frame(payload, opcode))
    def receiver():
        fragments = bytearray()
        try:
            while time.monotonic()-started < 90:
                final, opcode, payload = recv_frame(source)
                if opcode == 8:
                    break
                if opcode == 9:
                    send(payload, 10)
                    continue
                if opcode == 10:
                    continue
                if opcode not in (0, 1):
                    raise ValueError('unexpected server frame')
                fragments.extend(payload)
                if len(fragments) > 65536:
                    raise ValueError('message size bound')
                if not final:
                    continue
                row = json.loads(fragments.decode())
                fragments.clear()
                events.append(dict(at_seconds=time.monotonic()-started, event=row))
                if row.get('type') == 'error':
                    failures.append('server_error')
                    break
                if row.get('type') == 'conversation.item.input_audio_transcription.completed':
                    # Endpointing may emit more than one final; collect until input commit.
                    if sent.is_set() and row.get('audio_processed', 0) >= len(pcm)/(rate*2):
                        break
        except Exception as exc:
            failures.append(type(exc).__name__)
        finally:
            done.set()
    sent = threading.Event()
    reader = threading.Thread(target=receiver, name='cpu-asr-public-results')
    reader.start()
    try:
        send(json.dumps(dict(type='session.update', session=dict(sample_rate=rate, language=language, endpointing_ms=800))).encode(), 1)
        audio_start = time.monotonic()
        step = int(rate * .16) * 2
        # One second of appended silence exercises endpointing; no microphone.
        samples = pcm + bytes(rate*2)
        for offset in range(0, len(samples), step):
            if done.is_set() and failures:
                raise ValueError('server failed before complete public sample')
            wait = audio_start + offset/(rate*2) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            send(samples[offset:offset+step], 2)
        input_end = time.monotonic()-started
        sent.set()
        send(b'{"type":"input_audio_buffer.commit"}', 1)
        if not done.wait(max(0, 90-(time.monotonic()-started))):
            failures.append('receiver_deadline')
        coverage = max((r['event'].get('audio_processed', 0) for r in events), default=0)
        finalized_coverage = max((r['event'].get('audio_processed', 0) for r in events
                                  if r['event'].get('type') == 'conversation.item.input_audio_transcription.completed'), default=0)
        if coverage < len(pcm)/(rate*2):
            failures.append('incomplete_audio_coverage')
        if finalized_coverage < len(pcm)/(rate*2):
            failures.append('unfinalized_audio_coverage')
        result = summarize(events, len(pcm)/(rate*2), input_end)
        result['audio_processed_coverage_seconds'] = coverage
        result['finalized_audio_coverage_seconds'] = finalized_coverage
        result.update(errors=failures, events=events)
        return result
    finally:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        reader.join(timeout=3)
        if reader.is_alive():
            failures.append('receiver_thread_survived_shutdown')
        source.close()
        connection.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--sample-directory', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--keep-running', action='store_true')
    args = p.parse_args()
    if not args.run:
        print(json.dumps(dict(public_sample_only=True, device_capture=False, pause_after_test=not args.keep_running, max_seconds=90)))
        return
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    from research.launch_h3_asr_cpu_space import stage
    plan, _ = stage(args.directory)
    api = HfApi(token=cached_token())
    repo = plan['repo_id']
    output = args.directory/'public-stream-result.json'
    if output.exists():
        raise ValueError('no automatic replay of completed test')
    try:
        info = api.space_info(repo)
        runtime = api.get_space_runtime(repo)
        if not info.private or str(runtime.hardware) != 'cpu-basic' or str(runtime.stage) != 'RUNNING' or plan['expires_at']-time.time() < 120:
            raise ValueError('owned private CPU Basic ready window required')
        host = urlsplit(info.host).hostname
        if not host or not host.endswith('.hf.space'):
            raise ValueError('HF Space host required')
        anonymous_status, _ = http_ready(host)
        if anonymous_status not in (401, 403, 404, 302, 307):
            raise ValueError('anonymous access not denied; no audio sent')
        status, body = http_ready(host, cached_token())
        if status != 200:
            raise ValueError('authenticated model not ready; no audio sent')
        ready = json.loads(body)
        receipt = json.loads((args.sample_directory/'model-sample-receipt.json').read_text())
        sample = args.sample_directory/'jfk.wav'
        if receipt['sample_url'] != 'https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/v0.2.0/test_files/asr/wav/test/jfk.wav' or hashlib.sha256(sample.read_bytes()).hexdigest() != receipt['sample_sha256']:
            raise ValueError('pinned public JFK sample required')
        with wave.open(str(sample), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError('public mono16kPCM16 sample required')
            pcm = wav.readframes(wav.getnframes())
        result = stream_public(host, cached_token(), pcm, 16000)
        result.update(repo_id=repo, ready=ready, anonymous_status=anonymous_status, public_sample_only=True,
                      japanese_verified=False, comet_verified=False, gpu_used=False)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k != 'events'}))
    finally:
        if not args.keep_running:
            runtime = api.pause_space(repo)
            print(json.dumps(dict(repo_id=repo, paused_stage=str(runtime.stage))))


if __name__ == '__main__':
    main()
