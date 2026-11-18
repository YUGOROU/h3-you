"""Bounded exact-byte WebRTC carrier for upstream audio; no network fallback.

Uses the existing reliable ordered RTC Session channel, never a media codec.
Receipts mean application byte consumption, NOT native ASR/TTS EOS or playback.
RTC imports stay in the isolated runtime; this module itself is stdlib-only.
"""
import asyncio
from dataclasses import asdict,dataclass
import hashlib
import json
import re
import struct

MAX_CHUNK=12288

def _json(value):
    def unique(pairs):
        result={}
        for key,item in pairs:
            if key in result:raise ValueError('duplicate control key')
            result[key]=item
        return result
    def invalid(_):raise ValueError('nonfinite control')
    if type(value) is not str or len(value.encode())>16384:raise ValueError('bounded control required')
    return json.loads(value,object_pairs_hook=unique,parse_constant=invalid)

HEADER=struct.Struct('!4sQI')

@dataclass(frozen=True)
class Binding:
    session:str
    boot:str
    epoch:int
    purpose:str
    max_bytes:int

    def __post_init__(self):
        if (any(type(v) is not str or not re.fullmatch('[a-zA-Z0-9_-]{1,128}',v) for v in (self.session,self.boot))
                or type(self.epoch) is not int or not 1<=self.epoch<=1000
                or self.purpose not in ('asr-native-event-wire-v1','voxcpm-native-wire-v1')
                or type(self.max_bytes) is not int or not 1<=self.max_bytes<=64*1024*1024):
            raise ValueError('finite exact audio binding required')

class ExactChannel:
    """One bounded duplex stream per negotiated peer/immutable binding.

    At most one unconsumed inbound fragment and one unacknowledged outbound
    fragment. send and receive must run concurrently for a duplex application.
    A second writer/reader is rejected, never queued. EOF includes independent
    byte count/SHA; it does not establish native model completion.
    """
    def __init__(self,peer,binding):
        if type(binding) is not Binding:raise ValueError('exact binding required')
        self.peer=peer;self.binding=binding
        self._write=asyncio.Lock();self._read=asyncio.Lock();self._wire=asyncio.Lock()
        self._in=asyncio.Queue(1);self._pending=False;self._failure=None;self._failed=asyncio.Event()
        self._ack=None;self._seq=0;self._inseq=0;self._sent=0;self._received=0
        self._tx=hashlib.sha256();self._rx=hashlib.sha256()
        self._tx_end=False;self._rx_end=False;self._started=False;self._task=None
        self._expected_ack=None;self._input_closed=False;self._peer_done=asyncio.Event();self._done_sent=False;self._tx_finished=False

    async def _send(self,value):
        async with self._wire:await self.peer.send(value)

    async def start(self):
        if self._started:raise ValueError('no stream replay')
        self._started=True
        hello=dict(schema='h3-exact-rtc-v1',kind='hello',**asdict(self.binding))
        try:
            await self._send(json.dumps(hello,separators=(',',':')))
            row=_json(await self.peer.receive())
            if (type(row) is not dict or row!=hello
                    or any(type(row[k]) is not type(v) for k,v in hello.items())):
                raise ValueError('exact RTC identity mismatch')
        except BaseException:
            await self.peer.close();raise
        self._task=asyncio.create_task(self._pump())
        return self

    async def _pump(self):
        try:
            while True:
                value=await self.peer.receive()
                if type(value) is bytes:
                    if len(value)<HEADER.size:raise ValueError('short fragment')
                    magic,seq,size=HEADER.unpack_from(value)
                    body=value[HEADER.size:]
                    if (magic!=b'H3B1' or seq!=self._inseq or not 0<size<=MAX_CHUNK
                            or size!=len(body) or self._pending or self._input_closed
                            or self._received+size>self.binding.max_bytes):raise ValueError('invalid fragment')
                    self._pending=True;self._in.put_nowait(('data',seq,body))
                elif type(value) is str:
                    row=_json(value)
                    if type(row) is not dict:raise ValueError('invalid control')
                    if row.get('kind')=='ack':
                        if (row!=self._expected_ack or self._ack is None or self._ack.done()
                                or type(row.get('seq')) is not int or type(row.get('bytes')) is not int
                                or type(row.get('end')) is not bool):raise ValueError('unexpected ack')
                        self._ack.set_result(None)
                    elif row.get('kind')=='end':
                        if (set(row)!={'kind','seq','bytes','sha256'} or type(row['seq']) is not int
                                or type(row['bytes']) is not int or type(row['sha256']) is not str
                                or self._pending or self._input_closed or row['seq']!=self._inseq
                                or row['bytes']!=self._received or row['sha256']!=self._rx.hexdigest()):
                            raise ValueError('invalid exact EOF')
                        self._input_closed=True;self._pending=True;self._in.put_nowait(('end',row['seq'],b''))
                    elif row.get('kind')=='done':
                        if row!={'kind':'done'} or self._peer_done.is_set() or not self._rx_end or not self._tx_end:
                            raise ValueError('premature duplex completion')
                        self._peer_done.set()
                        # Both directions have verified EOF. No further peer
                        # messages are valid/needed; normal peer teardown must
                        # not turn this latched completion into a read failure.
                        return
                    else:raise ValueError('unexpected control')
                else:raise ValueError('unexpected message')
        except asyncio.CancelledError:raise
        except BaseException as error:
            self._failure=error;self._failed.set();await self.peer.close()

    async def _wait(self,operation):
        task=asyncio.ensure_future(operation);failed=asyncio.create_task(self._failed.wait())
        try:
            done,_=await asyncio.wait((task,failed),return_when=asyncio.FIRST_COMPLETED)
            if failed in done:raise RuntimeError('exact RTC failed; no fallback or replay') from None
            return task.result()
        finally:
            task.cancel();failed.cancel();await asyncio.gather(task,failed,return_exceptions=True)

    async def _exchange(self,value,ack):
        self._expected_ack=ack;self._ack=asyncio.get_running_loop().create_future()
        try:
            await self._send(value);await self._wait(self._ack)
        except BaseException:
            self._failed.set();await self.peer.close();raise
        finally:self._expected_ack=None;self._ack=None

    async def send(self,body):
        if (not self._started or self._failed.is_set() or self._write.locked() or self._tx_end
                or type(body) is not bytes or not body or len(body)>65536
                or self._sent+len(body)>self.binding.max_bytes):raise ValueError('bounded serial bytes required')
        async with self._write:
            for offset in range(0,len(body),MAX_CHUNK):
                chunk=body[offset:offset+MAX_CHUNK]
                ack=dict(kind='ack',seq=self._seq,bytes=self._sent+len(chunk),end=False)
                await self._exchange(HEADER.pack(b'H3B1',self._seq,len(chunk))+chunk,ack)
                self._tx.update(chunk);self._sent+=len(chunk);self._seq+=1

    async def finish(self):
        if not self._started or self._failed.is_set() or self._write.locked() or self._tx_end:raise ValueError('serial one-time EOF required')
        async with self._write:
            self._tx_end=True
            await self._exchange(json.dumps(dict(kind='end',seq=self._seq,bytes=self._sent,sha256=self._tx.hexdigest())),
                                 dict(kind='ack',seq=self._seq,bytes=self._sent,end=True))
        self._tx_finished=True
        return dict(bytes=self._sent,sha256=self._tx.hexdigest(),transport_complete=True,native_eos_verified=False)

    async def receive_into(self,sink):
        """Deliver one fragment; ACK only after awaited sink accepts it.

        Returns False for verified carrier EOF. Sink failures are terminal.
        """
        if not self._started or self._read.locked() or self._rx_end:raise ValueError('serial live reader required')
        async with self._read:
            kind,seq,body=await self._wait(self._in.get())
            try:
                if kind=='data':
                    await self._wait(sink(body))
                    self._received+=len(body);self._rx.update(body);self._inseq+=1
                else:self._rx_end=True
                # Clear before ACK: a conforming sender may immediately send
                # its next fragment when this ACK reaches it.
                self._pending=False
                await self._send(json.dumps(dict(kind='ack',seq=seq,bytes=self._received,end=kind=='end')))
                return kind!='end'
            except BaseException:
                self._failed.set();await self.peer.close();raise

    async def barrier(self):
        if not self._tx_finished or not self._rx_end or self._done_sent:raise ValueError('duplex EOF required')
        self._done_sent=True
        try:
            await self._send(json.dumps(dict(kind='done')))
        except Exception:
            # The remote may receive our done and close before local SCTP
            # drain resolves. Its already-validated done independently proves
            # both directions' EOF/ACK; never forgive a pre-completion failure.
            if not self._peer_done.is_set() or self._failed.is_set():raise
        await self._wait(self._peer_done.wait())

    async def close(self,*,close_peer=True):
        self._failed.set()
        if self._task is not None:
            self._task.cancel();await asyncio.gather(self._task,return_exceptions=True)
        if close_peer:await self.peer.close()
