"""Model-client seams for exact RTC sidecar IPC. No HTTP/WSS audio fallback.

The caller owns peer negotiation, authenticated identity and sidecar lifecycle.
Only connected AF_UNIX socketpairs are accepted; this is not an arbitrary TCP
connector. Existing native model parsers/coverage checks remain authoritative.
"""
from dataclasses import dataclass
import socket
import time
from .webrtc_exact_channel import Binding
from .voxcpm_endpoint_client import RemoteVoxCPM
from .voxcpm_endpoint_protocol import decode

@dataclass(frozen=True)
class RTCStream:
    sock:socket.socket
    binding:Binding
    finish_callback:object=None
    read_observer:object=None

    def __post_init__(self):
        if (type(self.binding) is not Binding or not isinstance(self.sock,socket.socket)
                or self.sock.family!=socket.AF_UNIX or self.sock.type!=socket.SOCK_STREAM):raise ValueError('bound owned RTC IPC required')
        if self.finish_callback is not None and not callable(self.finish_callback):raise ValueError('owned RTC completion callback required')
        if self.read_observer is not None and not callable(self.read_observer):raise ValueError('owned RTC read observer required')
        self.sock.getpeername()

    def close(self,*,require_complete=False):
        try:
            if self.finish_callback is not None:self.finish_callback(require_complete=require_complete)
        finally:
            try:self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self.sock.close()


def asr_transport(stream):
    """Use native ASR application framing over RTC bytes, NOT WAN WebSocket.

    No WebSocket HTTP Upgrade or remote connection occurs here. The endpoint
    RTC adapter must deliver these frames to the native ASR event engine.
    """
    if type(stream) is not RTCStream or stream.binding.purpose!='asr-native-event-wire-v1':
        raise ValueError('ASR RTC binding required')
    used=False
    def connect(host,token,key,timeout):
        nonlocal used
        if used:raise RuntimeError('no audio stream replay')
        used=True;stream.sock.settimeout(timeout)
        return stream.sock,stream.sock.makefile('rb')
    connect.tail_status={'state':'not-started'}
    def finish(connection,source,deadline):
        # Native final is not carrier EOF. Preserve buffered protocol bytes and
        # accept only the pinned server's committed marker and optional CLOSE;
        # additional transcription text is a protocol failure.
        connect.tail_status={'state':'draining','bytes':0,'opcode':None}
        from research.probe_h3_asr_cpu_space import frame
        # Finish the native WebSocket protocol too: TCP half-close alone need
        # not make the C++ event server finish its application session.
        connection.sendall(frame(b'\x03\xe8',opcode=8))
        connection.shutdown(socket.SHUT_WR)
        tail=bytearray()
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('ASR RTC closing deadline')
            connection.settimeout(remaining)
            body=source.read1(4096-len(tail))
            if not body:break
            tail.extend(body)
            connect.tail_status.update(bytes=len(tail),opcode=tail[0]&15)
            if len(tail)>=4096:raise ValueError('ASR RTC closing frame bound')
        import io,re
        from research.probe_h3_asr_cpu_space import recv_frame
        pending=io.BytesIO(tail);committed=False;closed=False
        while pending.tell()<len(tail):
            final,opcode,payload=recv_frame(pending)
            if not final or closed:raise ValueError('ASR RTC unexpected trailing native data')
            if opcode==1 and not committed:
                from .voxcpm_endpoint_protocol import loads
                row=loads(payload)
                if (type(row) is not dict or set(row) not in ({'type'},{'type','event_id'})
                        or row.get('type')!='input_audio_buffer.committed'
                        or 'event_id' in row and (type(row['event_id']) is not str
                            or re.fullmatch(r'event_[0-9]{1,24}',row['event_id']) is None)):
                    raise ValueError('ASR RTC unexpected trailing native event')
                committed=True
            elif opcode==8:
                if len(payload)>125 or len(payload)==1:raise ValueError('ASR RTC invalid close')
                if len(payload)>=2:
                    code=int.from_bytes(payload[:2],'big')
                    if not (1000<=code<=1014 and code not in (1004,1005,1006) or 3000<=code<=4999):
                        raise ValueError('ASR RTC invalid close status')
                    payload[2:].decode('utf-8')
                closed=True
            else:raise ValueError('ASR RTC unexpected trailing native data')
        connect.tail_status['committed_marker']=committed
        connect.tail_status['state']='verified'
    connect.finish=finish
    return connect


class RemoteVoxCPMRTC(RemoteVoxCPM):
    """READY/control remains HTTPS; all stream bytes come from exact RTC IPC.

    open_stream(text,epoch,deadline) creates one authenticated, bound stream.
    The endpoint receives text as control. No parent aiortc/model dependency.
    Production selection is intentionally not changed merely by this class.
    """
    def __init__(self,base_url,*,open_stream,**config):
        if not callable(open_stream):raise ValueError('explicit RTC stream owner required')
        super().__init__(base_url,**config);self._open_stream=open_stream

    def stream(self,text,epoch,deadline):
        if not self._lock.acquire(False):raise RuntimeError('busy')
        stream=None;complete=False
        try:
            if (not self._ready or type(text) is not str or not text.strip() or len(text)>160
                    or type(epoch) is not int or epoch<=self._epoch
                    or type(deadline) not in (int,float) or not __import__('math').isfinite(deadline)):
                raise ValueError('native RTC request')
            deadline=min(deadline,time.monotonic()+120);self._remaining(deadline)
            self._epoch=epoch  # Reserve before uncertain creation; never replay.
            stream=self._open_stream(text,epoch,deadline)
            if (type(stream) is not RTCStream or stream.binding.purpose!='voxcpm-native-wire-v1'
                    or stream.binding.session!=self.session or stream.binding.boot!=self.boot
                    or stream.binding.epoch!=epoch):raise ValueError('native RTC stream identity')
            with self._socket_lock:
                self._remaining(deadline);self._active_socket=stream.sock
            stream.sock.shutdown(socket.SHUT_WR)  # Native response-only carrier.
            def read(size):
                stream.sock.settimeout(self._remaining(deadline))
                body=stream.sock.recv(size)
                if stream.read_observer is not None:stream.read_observer(body)
                self._remaining(deadline);return body
            yield from decode(read,dict(session=self.session,boot=self.boot,expires_at=self.expires_at,epoch=epoch))
            self._remaining(deadline);complete=True
        except Exception:
            raise RuntimeError('Vox RTC stream failed; no fallback') from None
        finally:
            if not complete:self.close()
            with self._socket_lock:self._active_socket=None
            try:
                if stream is not None:stream.close(require_complete=complete)
            except BaseException:
                self.close();raise
            finally:self._lock.release()
