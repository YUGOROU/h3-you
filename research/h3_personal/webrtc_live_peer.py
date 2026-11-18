"""Finite, single-negotiation RTC peer with externally transported SDP.

Use in separate processes: create_offer() -> JSON -> accept_offer() -> JSON ->
set_answer(). No signaling transport, logging or playback claim. Host-only by
default; explicit approved Cloudflare STUN or HF-provided TURN/TLS profiles.
Deadline is an absolute time.monotonic() value. Use async with for ownership.
H264/Opus are explicit lossy media candidates, not original PCM acceptance.
"""
import asyncio
import math
import time

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, RTCRtpSender
from .webrtc_ice import HOST_ONLY, HF_CLOUDFLARE_TURN_TLS, ice_urls
from .webrtc_turn import validate_turn_config


class PeerError(RuntimeError):
    """Payload-free terminal peer failure."""


def description(value, expected):
    """Validate the complete JSON signaling envelope without retaining/logging it."""
    if (type(value) is not dict or set(value) != {"type", "sdp"}
            or type(value["type"]) is not str or value["type"] != expected
            or type(value["sdp"]) is not str):
        raise PeerError("invalid description envelope")
    try:
        size = len(value["sdp"].encode("utf-8"))
    except UnicodeError:
        raise PeerError("invalid description encoding") from None
    if not 0 < size <= 65536 or expected not in ("offer", "answer"):
        raise PeerError("invalid description size or type")
    return RTCSessionDescription(sdp=value["sdp"], type=expected)


class Session:
    def __init__(self, *, deadline, tracks=(), receive_audio=False,
                 receive_video=False, control=False, ice_profile=HOST_ONLY, turn_config=None):
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError("future finite monotonic deadline required")
        urls = ice_urls(ice_profile)
        if ice_profile == HF_CLOUDFLARE_TURN_TLS:
            servers = [RTCIceServer(**validate_turn_config(turn_config))]
        else:
            if turn_config is not None:
                raise ValueError('TURN credentials require explicit TURN profile')
            servers = [RTCIceServer(urls=url) for url in urls]
        self.ice_profile = ice_profile
        self.deadline = deadline
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        self._tracks = list(tracks)
        self._incoming = asyncio.Queue(16)
        self._messages = asyncio.Queue(16)
        self._opened = asyncio.Event()
        self._connected = asyncio.Event()
        self._failed = asyncio.Event()
        self._channel = None
        self._phase = "new"
        self._closing = None
        self._timer = asyncio.get_running_loop().call_later(
            max(0, deadline - time.monotonic()), self._fail)
        try:
            for track in self._tracks:
                self.pc.addTrack(track)
            for kind, enabled in (("audio", receive_audio), ("video", receive_video)):
                if enabled and not any(t.kind == kind for t in self.pc.getTransceivers()):
                    self.pc.addTransceiver(kind, direction="recvonly")
            self._codecs()
            if control:
                self._bind(self.pc.createDataChannel("control", ordered=True))
        except BaseException:
            self._fail()
            raise

        @self.pc.on("datachannel")
        def channel(ch):
            self._bind(ch)

        @self.pc.on("track")
        def track_received(track):
            self._tracks.append(track)
            try:
                self._incoming.put_nowait(track)
            except asyncio.QueueFull:
                self._fail()

        @self.pc.on("connectionstatechange")
        def state_changed():
            if self.pc.connectionState == "connected":
                self._connected.set()
            elif self.pc.connectionState in ("failed", "closed", "disconnected"):
                self._fail()

    def _codecs(self):
        for transceiver in self.pc.getTransceivers():
            mime = "audio/opus" if transceiver.kind == "audio" else "video/H264"
            codecs = [c for c in RTCRtpSender.getCapabilities(transceiver.kind).codecs
                      if c.mimeType == mime]
            if not codecs:
                raise PeerError("required codec unavailable")
            transceiver.setCodecPreferences(codecs)

    def _fail(self):
        self._failed.set()
        if self._closing is None:
            self._closing = asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        self._timer.cancel()
        for track in self._tracks:
            try:
                track.stop()
            except Exception:
                pass
        try:
            await asyncio.wait_for(self.pc.close(), 5)
        except Exception:
            pass

    async def close(self):
        self._fail()
        await asyncio.shield(self._closing)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _run(self, operation):
        task = asyncio.ensure_future(operation)
        failure = asyncio.create_task(self._failed.wait())
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self._failed.is_set():
                raise PeerError("peer unavailable")
            done, _ = await asyncio.wait((task, failure), timeout=remaining,
                                         return_when=asyncio.FIRST_COMPLETED)
            if failure in done or task not in done:
                raise PeerError("peer failed or deadline expired")
            return task.result()
        except BaseException as exc:
            self._fail()
            await asyncio.shield(self._closing)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise PeerError("peer operation failed") from None
        finally:
            task.cancel()
            failure.cancel()
            await asyncio.gather(task, failure, return_exceptions=True)

    async def create_offer(self):
        async def operation():
            if self._phase != "new":
                raise PeerError("negotiation already used")
            self._phase = "offer"
            await self.pc.setLocalDescription(await self.pc.createOffer())
            return self._local("offer")
        return await self._run(operation())

    def _local(self, kind):
        value = {"type": self.pc.localDescription.type, "sdp": self.pc.localDescription.sdp}
        description(value, kind)
        return value

    async def accept_offer(self, value):
        async def operation():
            if self._phase != "new":
                raise PeerError("negotiation already used")
            self._phase = "answer"
            await self.pc.setRemoteDescription(description(value, "offer"))
            self._codecs()
            await self.pc.setLocalDescription(await self.pc.createAnswer())
            return self._local("answer")
        return await self._run(operation())

    async def set_answer(self, value):
        async def operation():
            if self._phase != "offer":
                raise PeerError("unexpected answer")
            self._phase = "done"
            await self.pc.setRemoteDescription(description(value, "answer"))
        await self._run(operation())

    def _bind(self, channel):
        if (self._channel is not None or channel.label != "control" or not channel.ordered
                or channel.maxRetransmits is not None or channel.maxPacketLifeTime is not None):
            channel.close()
            self._fail()
            return
        self._channel = channel
        channel.bufferedAmountLowThreshold = 0
        self._drained = asyncio.Event()
        self._drained.set()

        @channel.on("bufferedamountlow")
        def drained():
            self._drained.set()

        @channel.on("open")
        def opened():
            self._opened.set()

        @channel.on("close")
        def closed():
            self._fail()

        @channel.on("message")
        def message(value):
            try:
                self._validate_message(value)
                self._messages.put_nowait(value)
            except (PeerError, asyncio.QueueFull):
                self._fail()

        if channel.readyState == "open":
            self._opened.set()

    @staticmethod
    def _validate_message(value):
        try:
            size = len(value.encode("utf-8")) if type(value) is str else len(value)
        except (TypeError, UnicodeError):
            raise PeerError("invalid control message") from None
        if type(value) not in (str, bytes) or size > 16384:
            raise PeerError("invalid control message")

    async def wait_connected(self):
        """ICE/DTLS connectivity only, NOT playback acceptance."""
        await self._run(self._connected.wait())

    async def wait_control(self):
        await self._run(self._opened.wait())

    async def send(self, value):
        async def operation():
            self._validate_message(value)
            await self._opened.wait()
            # No application send backlog: fail rather than silently enqueue.
            if self._channel.bufferedAmount:
                raise PeerError("control send overflow")
            self._drained.clear()
            self._channel.send(value)
            if self._channel.bufferedAmount:
                await self._drained.wait()
        await self._run(operation())

    async def receive(self):
        return await self._run(self._messages.get())

    async def receive_track(self):
        return await self._run(self._incoming.get())
