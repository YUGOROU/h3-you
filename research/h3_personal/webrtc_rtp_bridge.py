"""Owned local H3AV1 -> H264/Opus candidate; never remote original-PCM proof.

The caller constructs Session tracks from bridge.source and owns its gate/session.
Local sink ACK means local original-byte verification plus remote RTP sink report.
It does NOT mean remote original hashes or decoded Opus equality were verified.
"""
import asyncio
from collections import deque
import json
import socket
import threading
from types import SimpleNamespace

from .paired_av import Receiver, ProtocolError
from . import paired_av_wire as wire
from .webrtc_live_source import LiveSource


class RTPBridge:
    def __init__(self, binding, deadline):
        self.receiver = Receiver(binding, deadline=deadline)
        self.binding, self.deadline = binding, deadline
        self.source = LiveSource(binding)
        self.state_lock = threading.Lock()
        self._used = False
        self._threads = set()
        self._pending = deque()
        self._changed = asyncio.Event()
        self._speech_end = 0
        self._coverage = (0, 0)

    def _identity(self):
        b = self.binding
        return dict(session=b.session, profile_sha256=b.profile_sha256, epoch=b.epoch)

    def _message(self, kind, **fields):
        return dict(schema='h3-rtc-media-v1', kind=kind, **self._identity(), **fields)

    async def _io(self, fn, *args, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        self._threads.add(task)
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._threads.discard(task)

    def _event(self, raw):
        if type(raw) is not str or len(raw.encode('utf8')) > 4096:
            raise ProtocolError('bounded JSON control required')
        v = json.loads(raw, object_pairs_hook=wire._pairs)
        keys = {'schema', 'kind', 'session', 'profile_sha256', 'epoch',
                'index', 'video_frame_stop', 'audio_sample_stop'}
        if (type(v) is not dict or set(v) != keys or
                v['schema'] != 'h3-rtc-media-v1' or
                v['kind'] not in ('sink-accepted', 'end-partial') or
                any(type(v[k]) is not type(x) or v[k] != x
                    for k, x in self._identity().items()) or
                any(type(v[k]) is not int or v[k] < 0
                    for k in ('index', 'video_frame_stop', 'audio_sample_stop'))):
            raise ProtocolError('invalid bound sink event')
        return v

    async def run(self, peer, sock):
        """Own the socket and terminate the single-use supplied peer on exit.

        Remote coverage is an explicit sink observation, not source consumption.
        Only two receiver credits, including bytes already consumed by codecs.
        """
        if self._used:
            raise ProtocolError('bridge cannot replay')
        self._used = True
        writer = asyncio.Lock()
        tasks = []

        async def local(fn, *args, **kwargs):
            async with writer:
                return await self._io(fn, sock, *args, **kwargs)

        async def ingest():
            while True:
                while True:
                    self._changed.clear()
                    with self.state_lock:
                        s = self.receiver.snapshot()
                    reserve = min(self.binding.max_inflight_bytes, 16 * 1024 * 1024 + 124 * 4000)
                    if (s['pending'] + s['leased'] < 2 and
                            s['inflight_bytes'] + reserve <= self.binding.max_inflight_bytes):
                        break
                    # A header's size is unknown before reading. Reserve a full
                    # window byte budget: conservatively serialize admission.
                    await self._changed.wait()
                receipt = await self._io(wire.receive_pair, sock, self.receiver,
                                         deadline=self.deadline, state_lock=self.state_lock)
                e = receipt.envelope
                with self.state_lock:
                    lease = self.receiver._pending[-1]
                await self.source.append(SimpleNamespace(envelope=e, video=lease.video,
                                                        pcm=lease.audio_pcm))
                self._pending.append(e)
                self._speech_end = max(self._speech_end, max(
                    (r.sample_stop for r in e.regions if r.kind == 'speech'), default=0))
                await local(wire.send_ack, e, stage='verified', deadline=self.deadline)
                await peer.send(json.dumps(self._message('window', index=e.index,
                    frame_start=e.frame_start, frame_stop=e.frame_stop,
                    sample_start=e.sample_start, sample_stop=e.sample_stop,
                    fps=self.binding.fps, sample_rate=self.binding.sample_rate,
                    final=e.conversation_final)))
                if e.conversation_final:
                    await peer.send(json.dumps(self._message('source_eos', index=e.index,
                        video_frame_stop=e.frame_stop, audio_sample_stop=e.sample_stop)))
                    return

        async def events():
            while True:
                v = self._event(await peer.receive())
                if not self._pending or v['index'] != self._pending[0].index:
                    raise ProtocolError('sink event not FIFO')
                e = self._pending[0]
                vf, af = v['video_frame_stop'], v['audio_sample_stop']
                if vf < self._coverage[0] or af < self._coverage[1]:
                    raise ProtocolError('coverage regressed')
                if v['kind'] == 'end-partial':
                    if (not e.conversation_final or not self.source.finished or
                            not e.frame_start <= vf <= e.frame_stop or
                            not e.sample_start <= af <= e.sample_stop or
                            (vf == e.frame_stop and af == e.sample_stop) or
                            self._speech_end > min(af, vf * self.binding.sample_rate // self.binding.fps)):
                        raise ProtocolError('not a bounded silent closing tail')
                    await self.source.wait_for_consumed()
                    notice = dict(schema='h3-rtc-closing-tail-v1', **self._identity(),
                                  index=e.index, video_frame_stop=vf, audio_sample_stop=af,
                                  source_complete=True, media_complete=False)
                    await local(wire._write, wire._pack(notice), self.deadline)
                    return dict(video_frame_stop=vf, audio_sample_stop=af,
                                source_complete=True, media_complete=False)
                if vf < e.frame_stop or af < e.sample_stop or vf > self.source.frame_stop or af > self.source.sample_stop:
                    raise ProtocolError('sink coverage outside declared source')
                with self.state_lock:
                    lease = self.receiver.lease()
                    if lease is None or lease.segment != e:
                        raise ProtocolError('missing FIFO receiver lease')
                    self.receiver.ack(lease)
                await local(wire.send_ack, e, stage='sink-accepted', deadline=self.deadline)
                self._pending.popleft()
                self._coverage = vf, af
                self._changed.set()
                if e.conversation_final:
                    await self.source.wait_for_consumed()
                    return dict(video_frame_stop=vf, audio_sample_stop=af,
                                source_complete=True, media_complete=True)

        try:
            async with asyncio.timeout_at(self.deadline):
                tasks = [asyncio.create_task(ingest()), asyncio.create_task(events())]
                _, result = await asyncio.gather(*tasks)
                return result
        finally:
            # Shutdown wakes blocking reads/writes before joining their threads.
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._threads:
                await asyncio.gather(*self._threads, return_exceptions=True)
            sock.close()
            with self.state_lock:
                self.receiver.disconnect()
            # Stop RTP senders before closing their active async generators.
            # Session.close is idempotent; the caller's context still owns final
            # cleanup. Fake control peers in protocol tests may omit close().
            close = getattr(peer, 'close', None)
            if close is not None:
                await close()
            await self.source.aclose()
