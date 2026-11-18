"""Bounded framed paired AV over an already-authenticated owned socket.

No connection/listener/provider/device is opened here. Caller must supply TLS or
an authenticated private carrier; this framing alone is NOT authentication.
No automatic retry after partial write/read or ambiguous sink acknowledgement.
"""
from contextlib import nullcontext
from dataclasses import asdict,fields
import json
import math
import struct
import time

from .paired_av import (LiveAVBinding,AudioRegion,AVEnvelope,PresentationCut,ProtocolError,
                        Receiver,_bodies)

MAGIC=b'H3AV1\x00'
MAX_HEADER=16384
# v1 identities/framing remain unchanged. The explicit new disposition releases
# verified credit without making the legacy sink-accepted presentation claim.
ACK_STAGES=('verified','sink-accepted','discarded-interrupted','partially-presented-interrupted')


def _cut_record(value):
    if type(value) is not dict or set(value)!={f.name for f in fields(PresentationCut)}:
        raise ProtocolError('exact non-null cut record required')
    return PresentationCut(**value)


def _ack_stage(envelope, stage):
    if type(stage) is not str or stage not in ACK_STAGES:
        raise ProtocolError('explicit disposition required')
    if ((stage=='partially-presented-interrupted' and envelope.presentation_cut is None)
            or (stage=='sink-accepted' and envelope.presentation_cut is not None)):
        raise ProtocolError('cut requires partial presentation acknowledgement')


def ack_record(envelope, stage):
    if type(envelope) is not AVEnvelope:raise ProtocolError('envelope required')
    envelope.__post_init__();_ack_stage(envelope,stage)
    value=dict(schema='h3-paired-av-ack-v1',session=envelope.binding.session,
        epoch=envelope.binding.epoch,profile_sha256=envelope.binding.profile_sha256,
        index=envelope.index,video_sha256=envelope.video_sha256,pcm_sha256=envelope.pcm_sha256,stage=stage)
    if envelope.presentation_epoch:value['presentation_epoch']=envelope.presentation_epoch
    if envelope.presentation_cut is not None:value['presentation_cut']=asdict(envelope.presentation_cut)
    return value


def _remaining(deadline):
    if type(deadline) not in (int,float) or not math.isfinite(deadline):
        raise ProtocolError('finite absolute deadline required')
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('paired AV deadline')
    return value


def _read(sock,count,deadline):
    result=bytearray()
    while len(result)<count:
        sock.settimeout(_remaining(deadline))
        block=sock.recv(min(65536,count-len(result)))
        if not block:raise EOFError('incomplete paired AV; no replay')
        result.extend(block)
    return bytes(result)


def _write(sock,body,deadline):
    view=memoryview(body)
    while view:
        sock.settimeout(_remaining(deadline))
        n=sock.send(view[:65536])
        if n<=0:raise EOFError('incomplete paired AV write; no replay')
        view=view[n:]


def _pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ProtocolError('duplicate JSON field')
        result[k]=v
    return result


def _pack(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if not 0<len(raw)<=MAX_HEADER:raise ProtocolError('bounded wire header required')
    return MAGIC+struct.pack('!I',len(raw))+raw


def _json(sock,deadline):
    if _read(sock,len(MAGIC),deadline)!=MAGIC:raise ProtocolError('paired AV magic differs')
    n=struct.unpack('!I',_read(sock,4,deadline))[0]
    if not 0<n<=MAX_HEADER:raise ProtocolError('wire header exceeds bound')
    value=json.loads(_read(sock,n,deadline),object_pairs_hook=_pairs,
                     parse_constant=lambda x:(_ for _ in ()).throw(ProtocolError('nonfinite JSON')))
    if type(value) is not dict:raise ProtocolError('object header required')
    return value


def parse_envelope(value):
    if type(value) is not dict or set(value)!={'schema','envelope'} or value['schema']!='h3-paired-av-wire-v1':
        raise ProtocolError('exact paired header required')
    e=value['envelope']
    legacy_fields={f.name for f in fields(AVEnvelope)}-{'presentation_epoch','presentation_cut'}
    if (type(e) is not dict or not legacy_fields<=set(e)
            or set(e)-legacy_fields-{'presentation_epoch','presentation_cut'}):
        raise ProtocolError('exact envelope fields required')
    if ('presentation_epoch' in e and
            (type(e['presentation_epoch']) is not int or not 1<=e['presentation_epoch']<=2**63-1)):
        raise ProtocolError('explicit wire presentation epoch must be a positive integer')
    b=e['binding'];regions=e['regions']
    if type(b) is not dict or set(b)!={f.name for f in fields(LiveAVBinding)}:
        raise ProtocolError('exact binding fields required')
    if type(regions) is not list or not 1<=len(regions)<=8:
        raise ProtocolError('bounded explicit audio regions required')
    if any(type(r) is not dict or set(r)!={f.name for f in fields(AudioRegion)} for r in regions):
        raise ProtocolError('exact region fields required')
    parsed=dict(e,binding=LiveAVBinding(**b),regions=tuple(AudioRegion(**r) for r in regions))
    if 'presentation_cut' in e:parsed['presentation_cut']=_cut_record(e['presentation_cut'])
    return AVEnvelope(**parsed)


def envelope_record(envelope):
    """Exact v1 wire envelope; omit the default presentation generation.

    dataclasses.asdict() is NOT the wire serializer: its explicit zero field is
    deliberately rejected by parse_envelope, preventing two default encodings.
    """
    if type(envelope) is not AVEnvelope:raise ProtocolError('envelope required')
    envelope.__post_init__()
    record=asdict(envelope)
    record['regions']=[asdict(region) for region in envelope.regions]
    if envelope.presentation_epoch==0:del record['presentation_epoch']
    if envelope.presentation_cut is None:del record['presentation_cut']
    return record


def send_pair(sock,envelope,video,pcm,*,deadline):
    if type(envelope) is not AVEnvelope:raise ProtocolError('envelope required')
    envelope.__post_init__();_bodies(envelope,video,pcm)
    _write(sock,_pack(dict(schema='h3-paired-av-wire-v1',envelope=envelope_record(envelope))),deadline)
    _write(sock,video,deadline);_write(sock,pcm,deadline)
    # These writes are not receiver verification and never release credits.


def receive_pair(sock,receiver,*,deadline,state_lock=None):
    if type(receiver) is not Receiver:raise ProtocolError('receiver required')
    try:
        _remaining(deadline)
        with state_lock if state_lock is not None else nullcontext():
            state=receiver.snapshot()
        if state['terminal'] is not None:raise ProtocolError('terminal receiver')
        deadline=min(deadline,state['deadline'])
        e=parse_envelope(_json(sock,deadline))
        with state_lock if state_lock is not None else nullcontext():
            state=receiver.snapshot()
        size=e.video_size+e.pcm_size
        if (e.binding!=receiver.binding or e.index!=state['verified'] or e.frame_start!=state['frame_stop']
                or state['final'] or state['terminal'] is not None or state['pending']+state['leased']>=2
                or state['inflight_bytes']+size>receiver.binding.max_inflight_bytes):
            raise ProtocolError('peer sent without receiver credits')
        video=_read(sock,e.video_size,deadline);pcm=_read(sock,e.pcm_size,deadline)
        with state_lock if state_lock is not None else nullcontext():
            return receiver.offer(e,video,pcm)
    except BaseException:
        with state_lock if state_lock is not None else nullcontext():
            receiver.disconnect()
        raise


def send_ack(sock,envelope,*,stage,deadline):
    if type(envelope) is not AVEnvelope or type(stage) is not str or stage not in ACK_STAGES:
        raise ProtocolError('explicit verification or presentation disposition required')
    _write(sock,_pack(ack_record(envelope,stage)),deadline)


def parse_ack_message(value):
    if type(value) is not dict:raise ProtocolError('acknowledgement object required')
    keys={'schema','session','epoch','profile_sha256','index','video_sha256','pcm_sha256','stage'}
    if (not keys<=set(value) or set(value)-keys-{'presentation_epoch','presentation_cut'}
            or value.get('schema')!='h3-paired-av-ack-v1' or value.get('stage') not in ACK_STAGES
            or any(type(v) is not (int if k in ('epoch','index','presentation_epoch') else str)
                   for k,v in value.items() if k!='presentation_cut')
            or ('presentation_epoch' in value and not 1<=value['presentation_epoch']<=2**63-1)):
        raise ProtocolError('strict acknowledgement fields required')
    if 'presentation_cut' in value:
        cut=_cut_record(value['presentation_cut'])
        if cut.previous_epoch>=value.get('presentation_epoch',0) or value['stage']=='sink-accepted':
            raise ProtocolError('cut acknowledgement generation/stage differs')
    elif value['stage']=='partially-presented-interrupted':
        raise ProtocolError('partial acknowledgement requires cut')
    return value


def receive_ack_message(sock,*,deadline):
    return parse_ack_message(_json(sock,deadline))


def receive_ack(sock,expected,*,stage,deadline):
    if type(expected) is not AVEnvelope or type(stage) is not str or stage not in ACK_STAGES:
        raise ProtocolError('bound expected acknowledgement required')
    wanted=ack_record(expected,stage)
    actual=receive_ack_message(sock,deadline=deadline)
    if actual!=wanted or type(actual.get('index')) is not int or type(actual.get('epoch')) is not int:
        raise ProtocolError('ambiguous or mismatched acknowledgement; no replay')
    return actual
