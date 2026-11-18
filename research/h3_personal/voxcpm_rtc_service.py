"""Finite native Vox turns over exact RTC; HTTP carries control/SDP only."""
import asyncio
from dataclasses import asdict
import hmac
import socket
import time

from .asr_rtc_gateway import Server as _Server, Handler as _Handler
from .webrtc_exact_channel import Binding, ExactChannel
from .webrtc_exact_ipc import bridge
from .webrtc_live_peer import Session
from .webrtc_signal_server import SignalServer
from .webrtc_turn import validate_turn_config
from .webrtc_ice import HF_CLOUDFLARE_TURN_TLS
from .webrtc_job_gateway import RTCGateway
from . import webrtc_voxcpm_source
from .voxcpm_endpoint_protocol import loads

# The existing native adapter is named stream_native (no alternate carrier).
serve_gate = webrtc_voxcpm_source.stream_native
# Original 30s float32 PCM is 5.76MB plus bounded framing/metadata.
# ASR's 4MiB cap would truncate valid native TTS; no native limit changes.
MAX_WIRE_BYTES=8*1024*1024


class VoxRTCManager:
    def __init__(self, gate):
        self.gate = gate
        self.key, self.expires_at = gate.key, gate.expires_at
        self._epoch = 0
        self._task = self._signal = self._peer = None
        self._closed = self._failed = self._allocating = False
        self._timer = None

    async def start(self, request):
        if (self._closed or self._failed or self._allocating or not self.gate.healthy()
                or self._task is not None and not self._task.done()
                or type(request) is not dict
                or set(request) != {'session', 'boot', 'epoch', 'text', 'turn_config'}):
            raise ValueError('serial finite Vox turn required')
        epoch = request['epoch']
        if (request['session'] != self.gate.session or request['boot'] != self.gate.boot
                or type(epoch) is not int or not self._epoch < epoch <= self.gate.max_requests
                or type(request['text']) is not str or not request['text'].strip()
                or len(request['text']) > 160):
            raise ValueError('invalid native turn binding')
        binding = Binding(self.gate.session, self.gate.boot, epoch, 'voxcpm-native-wire-v1', MAX_WIRE_BYTES)
        turn = validate_turn_config(request['turn_config'])
        seconds = min(120, self.expires_at-time.time())
        if seconds <= 0:
            raise TimeoutError('expired Vox lease')
        self._allocating = True
        try:
            if self._signal is not None:
                await asyncio.to_thread(self._signal.close)
            if self._peer is not None:
                await self._peer.close()
            deadline = time.monotonic()+seconds
            expires = min(self.expires_at, time.time()+seconds)-.01
            peer = self._peer = Session(deadline=deadline,
                ice_profile=HF_CLOUDFLARE_TURN_TLS, turn_config=turn)
            signal = self._signal = SignalServer(session=peer, loop=asyncio.get_running_loop(),
                session_id=self.gate.session, key=self.key, expires_at=expires, host='127.0.0.1', port=0)
            signal.__enter__()
            self._epoch = epoch
            native_request = {k: request[k] for k in ('session', 'boot', 'epoch', 'text')}
            self._task = asyncio.create_task(self._serve(peer, binding, native_request))
            if self._timer is None:
                self._timer = asyncio.get_running_loop().call_later(
                    max(0, self.expires_at-time.time()), lambda: asyncio.create_task(self.close()))
            return dict(schema='h3-voxcpm-rtc-turn-v1', binding=asdict(binding),
                signal_boot=signal.boot, expires_at=expires, ice_profile=HF_CLOUDFLARE_TURN_TLS,
                signal_port=signal.address[1])
        except BaseException:
            self._failed = True
            await self.close()
            raise
        finally:
            self._allocating = False

    async def _serve(self, peer, binding, request):
        channel = ExactChannel(peer, binding)
        native = ipc = None
        tasks = []
        completed = False
        try:
            async with asyncio.timeout_at(peer.deadline):
                await peer.wait_control()
                await channel.start()
                native, ipc = socket.socketpair()
                tasks = [asyncio.create_task(bridge(channel, ipc, keep_peer=True)),
                         asyncio.create_task(asyncio.to_thread(serve_gate, self.gate, request, native, peer.deadline))]
                await asyncio.gather(*tasks)
            completed = True
        except BaseException:
            self._failed = True
            self.gate.close()
            raise
        finally:
            for sock in (native, ipc):
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await channel.close(close_peer=not completed)

    async def close(self):
        self._closed = True
        self.gate.close()
        if self._timer is not None:
            self._timer.cancel()
        if self._signal is not None:
            await asyncio.to_thread(self._signal.close)
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._peer is not None:
            await self._peer.close()


class Server(_Server):
    def __init__(self, address, manager, loop):
        super().__init__(address, manager, loop)
        self.RequestHandlerClass = Handler


class Handler(_Handler):
    def authorized(self):
        manager = self.server.manager
        control=self.path in ('/ready','/evidence')
        keys = self.headers.get_all('X-H3-TTS-Key' if control else 'X-H3-RTC-Key', [])
        return (len(keys) == 1 and keys[0].isascii()
                and hmac.compare_digest(keys[0], manager.key)
                and (control or self.headers.get_all('X-H3-Session', []) == [manager.gate.session]))

    def route(self):
        future = None
        try:
            manager = self.server.manager
            gate = manager.gate
            allowed = {('GET', '/health'), ('GET', '/ready'), ('GET', '/evidence'),
                       ('POST', '/rtc/start'), ('POST', '/rtc/offer'), ('GET', '/rtc/status')}
            if (self.command, self.path) not in allowed:
                return self.reply(404, {'error': 'control only'})
            if any(name in self.headers for name in ('Transfer-Encoding', 'Upgrade', 'Expect', 'Trailer', 'Content-Encoding')):
                return self.reply(400, {'error': 'control only'})
            checked = ('Host', 'Content-Length', 'Content-Type', 'X-H3-RTC-Key',
                       'X-H3-Session', 'X-H3-TTS-Key', 'Authorization', 'Connection')
            if any(len(self.headers.get_all(name, [])) > 1 for name in checked):
                return self.reply(400, {'error': 'ambiguous framing'})
            healthy = gate.healthy() and not manager._closed and not manager._failed
            if self.path == '/health':
                return self.reply(200 if healthy else 503, {'ready': healthy})
            if not self.authorized():
                return self.reply(401, {'error': 'unauthorized'})
            if not healthy:
                return self.reply(503, {'error': 'unavailable'})
            if self.path in ('/rtc/offer', '/rtc/status'):
                if self.server.gateway is None:
                    return self.reply(409, {'error': 'no active turn'})
                return self.server.gateway.handle(self)
            if self.command == 'GET':
                if self.headers.get('Content-Length', '0') != '0':
                    return self.reply(400, {'error': 'no body allowed'})
                if self.path == '/ready':
                    return self.reply(200, gate.ready(0))
                export = getattr(gate.candidate, 'export_evidence', None)
                if not callable(export):
                    return self.reply(503, {'error': 'unavailable'})
                return self.reply(200, dict(binding=gate.ready(0), evidence=export()))
            size = self.headers.get('Content-Length', '')
            if (not size.isascii() or not size.isdigit() or len(size) > 6
                    or not 0 < int(size) <= 8192 or self.headers.get('Content-Type') != 'application/json'):
                return self.reply(400, {'error': 'bounded JSON required'})
            deadline = time.monotonic()+5
            body = bytearray()
            while len(body) < int(size):
                self.connection.settimeout(max(.001, deadline-time.monotonic()))
                part = self.rfile.read1(int(size)-len(body))
                if not part or time.monotonic() >= deadline:
                    raise ValueError('body deadline')
                body.extend(part)
            row = loads(body)
            future = asyncio.run_coroutine_threadsafe(manager.start(row), self.server.loop)
            result = future.result(timeout=10)
            self.server.gateway = RTCGateway(result['signal_port'], manager.key, gate.session, result['expires_at'],operation_timeout=35)
            return self.reply(200, {k: v for k, v in result.items() if k != 'signal_port'})
        except Exception:
            if future is not None:
                future.cancel()
            return self.reply(503, {'error': 'RTC control unavailable; no fallback'})
