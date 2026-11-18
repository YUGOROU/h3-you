"""Finite owned RTC child. Parent imports only stdlib and paired-wire contracts.

root is the staged project; workspace (optional) is private scratch OUTSIDE root.
Creation must occur after H3 model readiness. H264/Opus is a lossy candidate.
"""
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
import zipfile

from .paired_av import LiveAVBinding
from . import paired_av_wire as wire

UV_URL = 'https://files.pythonhosted.org/packages/3a/ad/3c2a81d66d1839063662e5b6c0002ec31f8ee987dbc5295747627ca6160a/uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946'


def clean_env():
    # Allowlist, not a blacklist: inherited provider/application secrets stay out.
    return {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'TMPDIR', 'SSL_CERT_FILE') if k in os.environ}


def bootstrap(root, workspace, deadline):
    """Install only into private scratch; no root environment mutation or retry."""
    root, workspace = Path(root), Path(workspace)
    with urllib.request.urlopen(UV_URL, timeout=min(30, wire._remaining(deadline))) as response:
        body = response.read(32 * 1024 * 1024 + 1)
    if len(body) > 32 * 1024 * 1024 or hashlib.sha256(body).hexdigest() != UV_SHA256:
        raise ValueError('uv pin')
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = [n for n in archive.namelist() if n.endswith('/uv')]
        if len(names) != 1:
            raise ValueError('uv inventory')
        uv = workspace / 'uv'
        uv.write_bytes(archive.read(names[0]))
        uv.chmod(0o700)
    python = workspace / 'venv/bin/python'
    req = workspace / 'requirements.txt'
    commands = [
        ['venv', '--python', '3.12', workspace / 'venv'],
        ['export', '--project', root / 'research/webrtc_experiment', '--frozen', '--no-dev',
         '--no-emit-project', '--python', python, '--output-file', req],
        ['pip', 'install', '--python', python, '--require-hashes', '-r', req],
    ]
    env = dict(clean_env(), UV_NO_CACHE='1')
    for args in commands:
        subprocess.run([str(uv), *map(str, args)], env=env, check=True,
                       timeout=wire._remaining(deadline), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return python


class LiveRTCSidecar:
    def __init__(self, root, binding, expires_at, app_key, hf_token, deadline,
                 port=7861, *, workspace=None):
        self.root = Path(root).resolve()
        if type(binding) is not LiveAVBinding:
            raise ValueError('strict binding required')
        binding.__post_init__()
        if (type(expires_at) not in (int, float) or not math.isfinite(expires_at)
                or not 0 < expires_at - time.time() <= 1200
                or type(port) is not int or not 0 <= port <= 65535
                or not isinstance(app_key, str) or len(app_key) < 32
                or not isinstance(hf_token, str) or not hf_token):
            raise ValueError('finite authenticated RTC configuration required')
        wire._remaining(deadline)
        self.binding = binding
        # Signaling must not advertise a lease longer than the caller deadline.
        self.expires_at = min(expires_at, time.time() + wire._remaining(deadline) - 0.1)
        self.deadline = min(deadline, time.monotonic() + self.expires_at - time.time() + 0.05)
        self.port, self.boot = port, None
        self.media_adapter = self.process = self._child = self._scratch = None
        self._used = False
        self._workspace = workspace
        self._secrets = (app_key, hf_token)

    def __enter__(self):
        if self._used:
            raise RuntimeError('sidecar cannot replay')
        self._used = True
        try:
            base = Path(self._workspace).resolve() if self._workspace else Path(tempfile.gettempdir()).resolve()
            if base == self.root or self.root in base.parents:
                raise ValueError('workspace must be outside published project')
            self._scratch = tempfile.TemporaryDirectory(prefix='h3-rtc-', dir=base)
            scratch = Path(self._scratch.name)
            startup = min(self.deadline, time.monotonic() + 90)
            python = bootstrap(self.root, scratch, startup)
            config = scratch / 'binding.json'
            fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump(dict(binding=asdict(self.binding), expires_at=self.expires_at,
                               deadline=self.deadline, port=self.port), f, allow_nan=False)
            self.media_adapter, self._child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            key, token = self._secrets
            env = dict(clean_env(), PYTHONPATH=str(self.root), H3_RTC_KEY=key, HF_TOKEN=token)
            self.process = subprocess.Popen(
                [str(python), '-m', 'research.run_h3_rtc_sidecar', str(config), str(self._child.fileno())],
                cwd=self.root, env=env, pass_fds=(self._child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._child.close()
            self._child = None
            ready = wire._json(self.media_adapter, startup)
            if (set(ready) != {'schema', 'session', 'boot', 'port', 'expires_at'}
                    or ready['schema'] != 'h3-rtc-sidecar-ready-v1'
                    or ready['session'] != self.binding.session
                    or type(ready['boot']) is not str or not re.fullmatch('[a-f0-9]{32}', ready['boot'])
                    or type(ready['port']) is not int or not 1 <= ready['port'] <= 65535
                    or (self.port and ready['port'] != self.port)
                    or type(ready['expires_at']) not in (int,float)
                    or not time.time() < ready['expires_at'] <= self.expires_at):
                raise ValueError('sidecar ready binding differs')
            wire._remaining(startup)
            if self.process.poll() is not None:
                raise RuntimeError('sidecar exited during startup')
            self.port, self.boot = ready['port'], ready['boot']
            self.expires_at=ready['expires_at']
            self.deadline=min(self.deadline,time.monotonic()+self.expires_at-time.time())
            return self
        except BaseException:
            self.close()
            raise
        finally:
            self._secrets = None

    def close(self):
        self._secrets = None
        for name in ('media_adapter', '_child'):
            sock = getattr(self, name)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
                setattr(self, name, None)
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            finally:
                # Retain ownership if reaping itself fails; a later close may retry.
                if self.process.poll() is not None:
                    self.process = None
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def __exit__(self, *exc):
        self.close()


async def serve(config, fd):
    # Imports intentionally child-only.
    import asyncio
    from .webrtc_rtp_bridge import RTPBridge
    from .webrtc_live_peer import Session
    from .webrtc_signal_server import SignalServer
    from .webrtc_turn import fetch_turn_credentials
    from .webrtc_ice import HF_CLOUDFLARE_TURN_TLS

    sock = socket.socket(fileno=fd)
    bridge = peer = server = None
    try:
        binding = LiveAVBinding(**config['binding'])
        deadline = config['deadline']
        wire._remaining(deadline)
        token = os.environ.pop('HF_TOKEN')
        key = os.environ.pop('H3_RTC_KEY')
        turn_started=time.time()
        try:
            async with asyncio.timeout_at(deadline):
                turn = await asyncio.to_thread(fetch_turn_credentials, token, ttl=600)
        finally:
            del token
        # Do not advertise a peer lease beyond the requested credential TTL.
        expires=min(config['expires_at'],turn_started+590)
        deadline=min(deadline,time.monotonic()+expires-time.time()+.05)
        bridge = RTPBridge(binding, deadline)
        gate, start = asyncio.Event(), [None]
        peer = Session(deadline=deadline, tracks=bridge.source.make_tracks(gate, lambda: start[0]),
                       ice_profile=HF_CLOUDFLARE_TURN_TLS, turn_config=turn)
        server = SignalServer(session=peer, loop=asyncio.get_running_loop(),
                              session_id=binding.session, key=key, expires_at=expires,
                              host='127.0.0.1', port=config['port'], max_lifetime=1200)
        server.start()
        await asyncio.to_thread(wire._write, sock, wire._pack(dict(
            schema='h3-rtc-sidecar-ready-v1', session=binding.session, boot=server.boot,
            port=server.address[1], expires_at=expires)), deadline)
        await peer.wait_control()
        if await peer.receive() != 'start':
            raise ValueError('expected start')
        await peer.send('started')
        start[0] = time.monotonic()
        gate.set()
        await bridge.run(peer, sock)
    finally:
        sock.close()
        try:
            if peer is not None:
                await peer.close()
        finally:
            try:
                if bridge is not None:
                    await bridge.source.aclose()
            finally:
                if server is not None:
                    await asyncio.to_thread(server.close)
