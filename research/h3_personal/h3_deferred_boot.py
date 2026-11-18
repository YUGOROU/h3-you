"""Finite, single-claim deferred bootstrap; safe to inline as source text."""
import hashlib
import hmac
import json
import math
import os
import re
import runpy
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

_FIELDS = set('schema campaign group boot job_id expires_at repo revision allow_sha256 bootstrap_sha256 spec_sha256 producer_key consumer_key native_key'.split())
_MAX = 32768


def _loads(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON field')
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))


def _expiry(value):
    return type(value) in (int, float) and math.isfinite(value) and value > time.time()


def _secret(value):
    return type(value) is str and re.fullmatch(r'[\x20-\x7e]{32,8192}', value) is not None


def _execute(c):
    # This is the only optional dependency, imported only after the claim.
    from huggingface_hub import hf_hub_download
    def download(path, digest):
        name = hf_hub_download(repo_id=c['repo'], filename=path,
                               revision=c['revision'], repo_type='dataset')
        with open(name, 'rb') as f:
            raw = f.read()
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), digest):
            raise ValueError('download hash mismatch')
        return name, raw
    bootstrap, _ = download('research/bootstrap_h3_fast_ref2va.py', c['bootstrap_sha256'])
    _, raw = download('spec.json', c['spec_sha256'])
    spec = _loads(raw)
    session = spec.get('live_direct_session')
    native = session.get('native_tts') if type(session) is dict else None
    connection = native.get('connection') if type(native) is dict else None
    if (type(native) is not dict or native.get('schema') != 'h3-native-voxcpm-network-v2'
            or type(connection) is not dict or connection.get('group') != c['group']
            or type(connection.get('server_job_id')) is not str
            or re.fullmatch('[a-f0-9]{24}',connection['server_job_id']) is None):
        raise ValueError('native network source mismatch')
    if not _expiry(c['expires_at']):
        raise ValueError('expired launch')
    for field, env in (('producer_key', 'H3_DIRECT_PRODUCER_KEY'),
                       ('consumer_key', 'H3_DIRECT_CONSUMER_KEY'),
                       ('native_key', 'H3_NATIVE_TTS_KEY')):
        os.environ[env] = c[field]
    os.environ['H3_SESSION_NOT_AFTER'] = str(c['expires_at'])
    sys.argv = [bootstrap, c['repo'], c['revision'], c['allow_sha256']]
    runpy.run_path(bootstrap, run_name='__main__')


def serve(*, key, campaign, group, expires_at, bootstrap_sha256,
          host='0.0.0.0', port=7860, execute=None):
    """Block until expiry or one launch; close the listener before dispatch.

    The callback receives the validated capsule, without the authorization key.
    A claimed launch is never retried, including on response or execution failure.
    """
    if (not _secret(key) or not _expiry(expires_at)
            or not all(type(x) is str and x and len(x) <= 256 for x in (campaign, group))
            or type(bootstrap_sha256) is not str
            or not re.fullmatch('[0-9a-f]{64}', bootstrap_sha256)):
        raise ValueError('invalid wrapper configuration')
    job_id = os.environ['JOB_ID']
    boot = secrets.token_hex(16)
    claimed = None
    deadline = time.monotonic() + expires_at - time.time()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(max(.001, min(2, deadline-time.monotonic())))
            connection = self.connection
            def interrupt():
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self.deadline_timer = threading.Timer(max(0, deadline-time.monotonic()), interrupt)
            self.deadline_timer.daemon = True
            self.deadline_timer.start()

        def finish(self):
            try:
                super().finish()
            finally:
                self.deadline_timer.cancel()

        def reply(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def handle_request(self, launch):
            nonlocal claimed
            try:
                names = [k.lower() for k in self.headers.keys()]
                checked=('host','content-length','content-type','x-h3-launch-key','authorization','connection','content-encoding')
                if any(names.count(n)>1 for n in checked) or any(n in names for n in ('transfer-encoding','expect','upgrade','origin')):
                    raise ValueError('headers')
                supplied = self.headers.get('X-H3-Launch-Key', '')
                if not hmac.compare_digest(supplied.encode(), key.encode()):
                    self.reply(403, {})
                    return
                if time.monotonic() >= deadline or not _expiry(expires_at):
                    self.reply(410, {})
                    return
                if not launch:
                    if self.path != '/ready' or self.headers.get('Content-Length', '0') != '0':
                        raise ValueError('request')
                    self.reply(200, dict(schema='h3-deferred-ready-v1', campaign=campaign,
                        group=group, boot=boot, job_id=job_id, expires_at=expires_at,
                        state='claimed' if claimed is not None else 'waiting'))
                    return
                if self.path != '/launch':
                    raise ValueError('path')
                if self.headers.get('Content-Type')!='application/json':raise ValueError('JSON required')
                length = self.headers.get('Content-Length', '')
                if not re.fullmatch('[1-9][0-9]{0,4}', length) or int(length) > _MAX:
                    raise ValueError('length')
                raw = self.rfile.read(int(length))
                if len(raw) != int(length):
                    raise ValueError('short body')
                c = _loads(raw)
                if type(c) is not dict or set(c) != _FIELDS:
                    raise ValueError('fields')
                for field, expected in (('schema', 'h3-deferred-launch-v1'), ('campaign', campaign),
                        ('group', group), ('boot', boot), ('job_id', job_id),
                        ('bootstrap_sha256', bootstrap_sha256)):
                    if type(c[field]) is not str or not hmac.compare_digest(c[field].encode(), expected.encode()):
                        raise ValueError('identity')
                if not _expiry(c['expires_at']) or c['expires_at'] > expires_at or time.monotonic() >= deadline:
                    raise ValueError('expiry')
                for field, pattern in (('repo', r'YUGOROU/h3-[A-Za-z0-9][A-Za-z0-9._-]{0,90}'),
                        ('revision', '[a-f0-9]{40}'), ('allow_sha256', '[a-f0-9]{64}'),
                        ('spec_sha256', '[a-f0-9]{64}')):
                    if type(c[field]) is not str or not re.fullmatch(pattern, c[field]):
                        raise ValueError('pin')
                if not all(_secret(c[k]) for k in ('producer_key', 'consumer_key', 'native_key')):
                    raise ValueError('secret')
                if claimed is not None:
                    self.reply(409, {})
                    return
                claimed = c  # HTTPServer serializes handlers: the claim is atomic.
                self.reply(202, {'state': 'claimed'})
            except (ValueError, UnicodeError, TypeError):
                self.reply(400, {})

        def do_GET(self):
            self.handle_request(False)

        def do_POST(self):
            self.handle_request(True)

    server = HTTPServer((host, port), Handler)
    server.timeout = .1
    try:
        while claimed is None and time.monotonic() < deadline and _expiry(expires_at):
            try:
                server.handle_request()
            except (OSError, TimeoutError):
                pass
    finally:
        server.server_close()
    if claimed is not None:
        if not _expiry(claimed['expires_at']):
            raise TimeoutError('claimed capsule expired')
        (execute or _execute)(claimed)


def main():
    """Standalone wrapper only: absolute process lifetime includes bootstrap."""
    expiry = float(os.environ.pop('H3_DEFERRED_EXPIRES_AT'))
    timer = threading.Timer(max(0, expiry-time.time()), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    serve(key=os.environ.pop('H3_DEFERRED_KEY'),
          campaign=os.environ.pop('H3_DEFERRED_CAMPAIGN'),
          group=os.environ.pop('H3_DEFERRED_GROUP'), expires_at=expiry,
          bootstrap_sha256=os.environ.pop('H3_DEFERRED_BOOTSTRAP_SHA256'))


if __name__ == '__main__':
    main()
