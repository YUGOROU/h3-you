"""Opt-in finite, same-host synthetic WebRTC media experiment.

No files, devices, external signaling/STUN/TURN or private H3 media are used.
This is reception instrumentation, not actual playout or a WAN benchmark.
Optional libraries are imported only inside the explicit run entry point.
"""
import asyncio
from fractions import Fraction
import math
import struct
import sys
import time


class WebRTCProbeError(RuntimeError):
    pass


def validate_duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 10:
        raise WebRTCProbeError('finite duration from 1 to 10 seconds required')
    return float(value)


def sanitized_stats(report):
    allowed = ('type', 'kind', 'packetsSent', 'packetsReceived', 'packetsLost',
               'bytesSent', 'bytesReceived', 'jitter', 'roundTripTime', 'dtlsState')
    rows = []
    for item in report.values():
        if len(rows) >= 64:
            raise WebRTCProbeError('stats bound exceeded')
        if item.type not in ('outbound-rtp', 'inbound-rtp', 'remote-inbound-rtp', 'transport'):
            continue
        row = {}
        for key in allowed:
            value = getattr(item, key, None)
            if type(value) is str and key in ('type', 'kind', 'dtlsState'):
                accepted = {'outbound-rtp', 'inbound-rtp', 'remote-inbound-rtp',
                            'transport', 'audio', 'video', 'new', 'connecting',
                            'connected', 'closed', 'failed'}
                if value in accepted:
                    row[key] = value
            elif type(value) in (int, float) and math.isfinite(value):
                row['jitter_rtp_timestamp_units' if key == 'jitter' else key] = value
        rows.append(row)
    return rows


class Reception:
    def __init__(self, kind, *, max_frames=1024):
        self.kind = kind
        self.max_frames = max_frames
        self.frames = self.samples = 0
        self.first = self.last = self.first_pts = self.last_pts = None
        self.max_gap = 0.0
        self.formats = set()

    def observe(self, frame, elapsed):
        if frame.pts is None or frame.time_base is None:
            raise WebRTCProbeError('explicit received media timestamp required')
        pts = float(frame.pts * frame.time_base)
        if not math.isfinite(pts) or (self.last_pts is not None and pts <= self.last_pts):
            raise WebRTCProbeError('nonincreasing received media clock')
        if self.last is not None:
            self.max_gap = max(self.max_gap, elapsed - self.last)
        else:
            self.first, self.first_pts = elapsed, pts
        self.last, self.last_pts = elapsed, pts
        self.frames += 1
        if self.frames > self.max_frames:
            raise WebRTCProbeError('received frame bound exceeded')
        if self.kind == 'audio':
            self.samples += frame.samples
            self.formats.add((frame.sample_rate, len(frame.layout.channels)))
        else:
            self.formats.add((frame.width, frame.height))

    def summary(self):
        return dict(frames=self.frames, samples=self.samples,
                    first_receive_seconds=self.first, last_receive_seconds=self.last,
                    first_pts_seconds=self.first_pts, last_pts_seconds=self.last_pts,
                    max_receive_gap_seconds=self.max_gap, formats=sorted(self.formats))


async def close_owned(peers, tracks, consumers, *, timeout=5):
    """Attempt every close even if a consumer resists cancellation.

    Pending cleanup is failure, not a complete receipt. The CLI parent remains
    the hard process boundary for uncooperative native/task shutdown.
    """
    failed = False
    end = time.monotonic() + timeout
    for track in tracks:
        try:
            track.stop()
        except Exception:
            failed = True
    tasks = list(consumers)
    for task in consumers:
        task.cancel()
    for peer in peers:
        try:
            tasks.append(asyncio.create_task(peer.close()))
        except Exception:
            failed = True
    try:
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=max(0, end - time.monotonic()))
            failed |= bool(pending)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    failed = True
        # Sender shutdown must finish before closing a source async generator;
        # a suspended yield can otherwise retain private compressed media.
        if all(task.done() for task in tasks) and all(peer.connectionState == 'closed' for peer in peers):
            closers = []
            for track in tracks:
                close = getattr(track, 'aclose', None)
                if close is not None:
                    try:
                        closers.append(asyncio.create_task(close()))
                    except Exception:
                        failed = True
            tasks.extend(closers)
            if closers:
                done, pending = await asyncio.wait(closers, timeout=max(0, end - time.monotonic()))
                failed |= bool(pending)
                for task in done:
                    if task.cancelled() or task.exception() is not None:
                        failed = True
        return not failed and all(peer.connectionState == 'closed' for peer in peers)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
                task.add_done_callback(lambda t: None if t.cancelled() else t.exception())


async def run_media_probe(track_factory, *, run_seconds, source_summary,
                          schema='h3-webrtc-synthetic-probe-v1', private_media=False,
                          operation_timeout=20):
    if (type(run_seconds) not in (int, float) or not math.isfinite(run_seconds)
            or not 0 < run_seconds <= 85 or operation_timeout not in (20, 90)):
        raise WebRTCProbeError('bounded media probe required')
    try:
        import av
        import aiortc
        from aiortc import MediaStreamTrack, RTCPeerConnection, RTCConfiguration, RTCRtpSender
        from aiortc.mediastreams import MediaStreamError
    except ImportError:
        raise WebRTCProbeError('run in the locked uv WebRTC environment') from None

    gate = asyncio.Event()
    start = None

    peers = []
    tracks = []
    reception = {kind: Reception(kind, max_frames=max(1024, math.ceil(run_seconds * 50) + 64))
                 for kind in ('audio', 'video')}
    consumers = []
    opened = asyncio.Event()
    pong = asyncio.Event()
    completed = False
    cleanup_complete = False
    result = None

    async def consume(track):
        try:
            while True:
                frame = await track.recv()
                if start is None:
                    raise WebRTCProbeError('media before explicit source start')
                reception[track.kind].observe(frame, time.monotonic() - start)
        except MediaStreamError:
            return

    def on_track(track):
        if track.kind not in reception or len(consumers) >= 2:
            raise WebRTCProbeError('unexpected media track')
        consumers.append(asyncio.create_task(consume(track)))

    def on_channel(channel):
        @channel.on('message')
        def on_message(message):
            if message == b'H3 synthetic ping':
                channel.send(b'H3 synthetic pong')

    def on_open():
        opened.set()

    def on_pong(message):
        if message == b'H3 synthetic pong':
            pong.set()

    try:
        async with asyncio.timeout(operation_timeout):
            # Register ownership immediately, before the next acquisition.
            # [] is deliberate: None would permit default external ICE servers.
            for _ in range(2):
                peers.append(RTCPeerConnection(RTCConfiguration(iceServers=[])))
            sender, receiver = peers
            for track in track_factory(gate, lambda: start):
                tracks.append(track)
            if [t.kind for t in tracks] != ['audio', 'video']:
                raise WebRTCProbeError('exactly audio/video tracks required')
            receiver.on('track', on_track)
            receiver.on('datachannel', on_channel)
            channel = sender.createDataChannel('synthetic-control', ordered=True)
            channel.on('open', on_open)
            channel.on('message', on_pong)
            for track in tracks:
                sender.addTrack(track)
            selected = {}
            for transceiver in sender.getTransceivers():
                mime = 'audio/opus' if transceiver.kind == 'audio' else 'video/H264'
                codecs = [c for c in RTCRtpSender.getCapabilities(transceiver.kind).codecs
                          if c.mimeType.lower() == mime.lower()]
                if not codecs:
                    raise WebRTCProbeError('explicit codec unavailable; no fallback')
                transceiver.setCodecPreferences(codecs[:1])
                selected[transceiver.kind] = dict(mime_type=codecs[0].mimeType,
                    clock_rate=codecs[0].clockRate, channels=codecs[0].channels,
                    parameters=dict(codecs[0].parameters))
            await sender.setLocalDescription(await sender.createOffer())
            await receiver.setRemoteDescription(sender.localDescription)
            await receiver.setLocalDescription(await receiver.createAnswer())
            await sender.setRemoteDescription(receiver.localDescription)
            await opened.wait()
            channel.send(b'H3 synthetic ping')
            await pong.wait()
            start = time.monotonic()
            gate.set()
            # No extra source frames/samples to flush codec/jitter tails. Receive
            # losses/tails are reported, not concealed or called sample-exact.
            await asyncio.sleep(run_seconds + 0.5)
            for task in consumers:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    raise WebRTCProbeError('headless media consumer failed')
            if len(consumers) != 2 or any(r.frames == 0 for r in reception.values()):
                raise WebRTCProbeError('both media tracks required')
            if any(p.connectionState != 'connected' for p in peers):
                raise WebRTCProbeError('peer disconnected before probe completed')
            source = source_summary(tracks)
            result = dict(schema=schema,
                aiortc_version=aiortc.__version__, av_version=av.__version__,
                same_host=True, loopback_only_verified=False,
                external_ice_servers=[], signaling='in-memory',
                private_media_used=private_media, device_access=False, wan_verified=False,
                playout_verified=False, sample_exact=False, source_tail_flushed=False,
                source_duration_seconds=source['media_seconds'], source=source,
                source_generated_frames={t.kind: t.generated for t in tracks},
                source_audio_samples=source['generated_audio_samples'],
                source_video_geometry=[252, 448], source_audio_format='mono/s16/48000',
                codec_preferences=selected, control_roundtrip=True,
                receive={kind: r.summary() for kind, r in reception.items()},
                sender_stats=sanitized_stats(await sender.getStats()),
                receiver_stats=sanitized_stats(await receiver.getStats()))
            completed = True
    finally:
        primary = sys.exc_info()[1]
        try:
            cleanup_complete = await close_owned(peers, tracks, consumers)
        except BaseException:
            if primary is None:
                raise WebRTCProbeError('finite peer cleanup failed') from None
        if completed and not cleanup_complete:
            raise WebRTCProbeError('peer cleanup incomplete')
    result['cleanup_complete'] = cleanup_complete
    return result


async def run_synthetic(*, duration_seconds=4):
    duration = validate_duration(duration_seconds)
    try:
        import av
        from aiortc import MediaStreamTrack
        from aiortc.mediastreams import MediaStreamError
    except ImportError:
        raise WebRTCProbeError('run in the locked uv WebRTC environment') from None

    def factory(gate, get_start):
        class SyntheticTrack(MediaStreamTrack):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
                self.generated = 0

            async def recv(self):
                await gate.wait()
                rate = 50 if self.kind == 'audio' else 24
                index = self.generated
                if index / rate >= duration:
                    self.stop()
                    raise MediaStreamError
                await asyncio.sleep(max(0, get_start() + index / rate - time.monotonic()))
                if self.kind == 'audio':
                    frame = av.AudioFrame(format='s16', layout='mono', samples=960)
                    values = [int(3000 * math.sin(2 * math.pi * 440 * (index * 960 + n) / 48000))
                              for n in range(960)]
                    frame.planes[0].update(struct.pack('<960h', *values))
                    frame.sample_rate = 48000
                    frame.pts, frame.time_base = index * 960, Fraction(1, 48000)
                else:
                    frame = av.VideoFrame(252, 448, format='yuv420p')
                    for i, plane in enumerate(frame.planes):
                        plane.update(bytes([16 + index % 219 if i == 0 else 128]) * plane.buffer_size)
                    frame.pts, frame.time_base = index * 3750, Fraction(1, 90000)
                self.generated += 1
                return frame
        return [SyntheticTrack('audio'), SyntheticTrack('video')]

    def summary(tracks):
        if [t.generated for t in tracks] != [math.ceil(duration * 50), math.ceil(duration * 24)]:
            raise WebRTCProbeError('incomplete synthetic source')
        return dict(kind='synthetic', media_seconds=duration,
                    generated_audio_samples=tracks[0].generated * 960)

    return await run_media_probe(factory, run_seconds=duration, source_summary=summary)
