"""Explicit RTC candidate over an owned local IPC socketpair, never a media socket.

The sidecar verifies original bytes locally. Its legacy sink ACK represents the
remote RTP sink, not remote verification of original PCM (Opus is not lossless).
"""
import socket

from .paired_av import ProtocolError
from .paired_av_pump import ProducerPump as PairedProducerPump
from . import paired_av_wire as wire


class ClosingTailNotice(ProtocolError):
    """Validated source completion with an unverified remote closing tail."""


class ProducerPump(PairedProducerPump):
    def __init__(self, sock, binding, *, deadline, overlap_send=False):
        if not isinstance(sock, socket.socket) or sock.family != socket.AF_UNIX:
            raise ValueError('owned local socketpair required')
        sock.getpeername()  # Reject an unconnected local socket.
        super().__init__(sock, binding, deadline=deadline, overlap_send=overlap_send)
        self._native_final = None
        self._speech_end = 0
        self._closing_notice = None

    def _closing(self, value):
        keys = {'schema', 'session', 'profile_sha256', 'epoch', 'index',
                'video_frame_stop', 'audio_sample_stop', 'source_complete', 'media_complete'}
        if type(value) is not dict or set(value) != keys:
            raise ProtocolError('strict closing-tail fields required')
        for k, v in value.items():
            expected = bool if k in ('source_complete', 'media_complete') else (
                int if k in ('epoch', 'index', 'video_frame_stop', 'audio_sample_stop') else str)
            if type(v) is not expected:
                raise ProtocolError('strict closing-tail types required')
        e = self._native_final
        state = self.sender.snapshot()
        b = self.sender.binding
        if (e is None or not state['final'] or state['pending'] != 1
                or state['acked'] != e.index or state['reserved'] != e.index + 1
                or state['verified'] != state['reserved']
                or value['schema'] != 'h3-rtc-closing-tail-v1'
                or value['session'] != b.session or value['epoch'] != b.epoch
                or value['profile_sha256'] != b.profile_sha256 or value['index'] != e.index
                or value['source_complete'] is not True or value['media_complete'] is not False):
            raise ProtocolError('unbound or premature closing tail')
        v, a = value['video_frame_stop'], value['audio_sample_stop']
        if (not e.frame_start <= v <= e.frame_stop
                or not e.frame_start * 2000 <= a <= e.frame_stop * 2000
                or min(a, v * 2000) < self._speech_end
                or (v == e.frame_stop and a == e.frame_stop * 2000)):
            raise ProtocolError('invalid closing-tail coverage')
        notice = ClosingTailNotice('remote closing tail unverified; no replay')
        self._closing_notice = notice
        raise notice

    def _acks(self):
        try:
            while True:
                with self._cv:
                    if self._stop:
                        return
                value = wire._json(self.sock, self.deadline)
                with self._cv:
                    if type(value) is dict and value.get('schema') == 'h3-rtc-closing-tail-v1':
                        self._closing(value)
                    msg = wire.parse_ack_message(value)
                    self.sender.ack(msg)
                    self._mark(msg['index'], 'ack_' + msg['stage'].replace('-', '_'))
                    state = self.sender.snapshot()
                    self._cv.notify_all()
                    if state['final'] and not state['pending']:
                        return
        except BaseException as exc:
            with self._cv:
                if not self._stop:
                    self._error = exc
                    self.sender.disconnect()
                self._cv.notify_all()
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def run(self, packets):
        def tracked():
            iterator = iter(packets)
            try:
                for packet in iterator:
                    with self._cv:
                        for region in packet.envelope.regions:
                            if region.kind == 'speech':
                                self._speech_end = max(self._speech_end, region.sample_stop)
                        if packet.envelope.conversation_final:
                            self._native_final = packet.envelope
                    yield packet
            finally:
                if hasattr(iterator, 'close'):
                    iterator.close()
        try:
            result = super().run(tracked())
        except RuntimeError as exc:
            if self._closing_notice is None or exc.__cause__ is not self._closing_notice:
                raise
            result = dict(self.sender.snapshot(), queue_highwater=self.highwater)
            result.update(source_complete=True, media_complete=False, complete=False,
                          closing_tail_unverified=True)
        else:
            result.update(source_complete=True, media_complete=True, complete=True,
                          closing_tail_unverified=False)
        result.update(local_ipc_source_verified=True, remote_original_pcm_verified=False)
        return result


RTCProducerPump = ProducerPump
