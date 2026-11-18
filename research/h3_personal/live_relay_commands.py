"""Bounded ordered command framing on the relay's separate RAM input lane.

No model/tool execution. Turn payload validation (fresh ASR/Pi/TTS provenance,
model/voice/watermark contracts) belongs to the actor. Never log/store payloads.
"""
import json
import struct
import threading

MAX_MESSAGE=16384
MAX_INPUT=131072


def _pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate command field')
        result[k]=v
    return result


class Commands:
    def __init__(self,transport,session,*,turn_interruptions=False,long_session=False):
        if type(session) is not str or not 0<len(session)<=128:raise ValueError('bounded session required')
        if type(turn_interruptions) is not bool:raise ValueError('explicit turn interruption opt-in required')
        if type(long_session) is not bool or (long_session and turn_interruptions):
            raise ValueError('explicit non-interruptible long session required')
        self._max_epoch=1000 if long_session else 16
        self._max_commands=2001 if long_session else 32
        self.transport=transport;self.session=session;self._sent=0;self._received=0
        self._epoch=0;self._buffer=bytearray();self._failed=False;self._stopped=False
        self._turn_interruptions=turn_interruptions
        self._presentation_epoch=0;self._interrupted_epoch=0
        self._send_lock=threading.Lock()

    def _validate(self,value,expected):
        if (type(value) is not dict or set(value)!={'schema','session','sequence','kind','epoch','payload'}
                or value['schema']!='h3-live-command-v1' or value['session']!=self.session
                or type(value['sequence']) is not int or value['sequence']!=expected or expected>self._max_commands
                or type(value['epoch']) is not int or not 0<=value['epoch']<=self._max_epoch
                or value['kind'] not in ('turn','cancel_pending','stop','interrupt_turn') or type(value['payload']) is not dict):
            raise ValueError('strict ordered command required')
        if self._stopped:raise ValueError('command after stop')
        if value['kind']=='turn':
            if value['epoch']<=self._epoch:raise ValueError('stale input epoch')
        elif value['kind']=='interrupt_turn':
            p=value['payload']
            if (not self._turn_interruptions or value['epoch']!=self._epoch
                    or value['epoch']<=self._interrupted_epoch
                    or set(p)!={'schema','presentation_epoch'}
                    or p['schema']!='h3-turn-interruption-v1'
                    or type(p['presentation_epoch']) is not int
                    or not self._presentation_epoch<p['presentation_epoch']<=2**63-1):
                raise ValueError('explicit current-turn monotonic presentation interruption required')
        elif value['payload'] or value['epoch']!=self._epoch:
            raise ValueError('bound cancel/stop required')

    def _advance(self,value):
        if value['kind']=='turn':self._epoch=value['epoch']
        if value['kind']=='interrupt_turn':
            self._presentation_epoch=value['payload']['presentation_epoch']
            self._interrupted_epoch=value['epoch']
        if value['kind']=='stop':self._stopped=True

    def send(self,kind,epoch,payload):
        """Serialize control submissions, independently of any media read lane.

        Ordering is lock acquisition order, not caller scheduling order. The
        transport still owns finite blocking IO deadlines. Receiver has one owner.
        """
        with self._send_lock:
            if self._failed:raise RuntimeError('terminal command stream')
            value=dict(schema='h3-live-command-v1',session=self.session,sequence=self._sent+1,
                       kind=kind,epoch=epoch,payload=payload)
            self._validate(value,self._sent+1)
            raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            if not 0<len(raw)<=MAX_MESSAGE:raise ValueError('command byte bound')
            try:
                n=self.transport.send_input(struct.pack('!I',len(raw))+raw)
                if n!=len(raw)+4:raise ValueError('partial command submission')
                self._sent+=1;self._advance(value)
            except BaseException:
                self._failed=True
                raise

    def receive(self):
        if self._failed or self._stopped:raise RuntimeError('terminal command stream')
        try:
            while True:
                if len(self._buffer)>=4:
                    n=struct.unpack('!I',self._buffer[:4])[0]
                    if not 0<n<=MAX_MESSAGE:raise ValueError('command byte bound')
                    if len(self._buffer)>=n+4:
                        raw=bytes(self._buffer[4:n+4]);del self._buffer[:n+4]
                        value=json.loads(raw,object_pairs_hook=_pairs,
                            parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite command')))
                        self._validate(value,self._received+1)
                        self._received+=1;self._advance(value)
                        return value
                block=self.transport.read_input()
                if type(block) is not bytes or not 0<len(block)<=MAX_INPUT:
                    raise ValueError('invalid input lane block')
                if len(self._buffer)+len(block)>MAX_INPUT+MAX_MESSAGE+4:
                    raise ValueError('command retention bound')
                self._buffer.extend(block)
        except BaseException:
            self._failed=True;self._buffer.clear()
            raise

    def close(self):self._failed=True;self._buffer.clear()
