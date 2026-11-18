"""Bounded Vox native PCM wire contract (stdlib, no model imports)."""
import array
import hashlib
import json
import math
import struct
import sys

from .voxcpm2_streaming_candidate import MODEL_REVISION, SOURCE_REVISION
from .director_tts_job import REFERENCE_SHA

CONTENT_TYPE = 'application/x-h3-voxcpm-stream-v1'
CANCEL_PROTOCOL = 'h3-voxcpm-cancel-v2'
CANCEL_CONTENT_TYPE = 'application/x-h3-voxcpm-stream-v2'
MAX_FRAME = 256 * 1024
MAX_SAMPLES = 48000 * 30
MAX_CHUNKS = 1024
PINS = dict(source_revision=SOURCE_REVISION, model_revision=MODEL_REVISION,
            reference_sha256=REFERENCE_SHA, sample_rate=48000, channels=1,
            format='float32le', watermark_applied=False)


def loads(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate field')
            out[k] = v
        return out
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def frame(kind, payload):
    if kind not in (b'R', b'P', b'E', b'X') or not 0 < len(payload) <= MAX_FRAME:
        raise ValueError('invalid frame')
    return struct.pack('!cI', kind, len(payload)) + payload


def metadata(kind, value):
    return frame(kind, json.dumps(value, allow_nan=False, separators=(',', ':')).encode())


def pcm_bytes(chunk):
    # Native yields a 1D numpy array; avoid importing numpy on CPU control hosts.
    if getattr(chunk, 'ndim', 1) != 1:
        raise ValueError('non-mono PCM')
    if len(chunk) * 4 > MAX_FRAME:
        raise ValueError('native chunk exceeds wire bound')
    values = array.array('f', chunk)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid PCM')
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def validate_pcm(data):
    if not data or len(data) % 4:
        raise ValueError('invalid PCM')
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite PCM')


def _exact_metadata(actual, expected):
    return (type(actual) is dict and actual.keys()==expected.keys()
            and all(type(actual[key]) is type(value) and actual[key]==value
                    for key,value in expected.items()))


def decode(read, expected, *, cancel_protocol=None):
    """Yield original PCM bytes; normal return requires verified explicit end.

    read(n) must obey an absolute deadline. Early caller abandonment is NOT
    completion; the caller must close its transport context.
    """
    if cancel_protocol not in (None, CANCEL_PROTOCOL):
        raise ValueError('unsupported cancel protocol')
    expected = dict(expected)
    if cancel_protocol is not None:
        expected['cancel_protocol'] = cancel_protocol
    def exact(n):
        out = bytearray()
        while len(out) < n:
            part = read(n - len(out))
            if not part:
                raise ValueError('truncated stream')
            out.extend(part)
        return bytes(out)
    digest = hashlib.sha256()
    samples = chunks = 0
    ready = False
    while True:
        kind, size = struct.unpack('!cI', exact(5))
        if not 0 < size <= MAX_FRAME:
            raise ValueError('frame bound')
        data = exact(size)
        if not ready:
            if kind != b'R' or not _exact_metadata(loads(data), dict(PINS, **expected)):
                raise ValueError('ready mismatch')
            ready = True
        elif kind == b'P':
            validate_pcm(data)
            samples += len(data) // 4
            chunks += 1
            if samples > MAX_SAMPLES or chunks > MAX_CHUNKS:
                raise ValueError('stream bound')
            digest.update(data)
            yield data
        elif kind == b'E':
            end = loads(data)
            reason = end.get('finish_reason') if type(end) is dict else None
            wanted = dict(samples=samples, chunks=chunks, sha256=digest.hexdigest(),
                          finish_reason='iterator_exhausted', model_eos_verified=False)
            if cancel_protocol is not None:
                if reason not in ('iterator_exhausted', 'cancelled'):
                    raise ValueError('invalid terminal reason')
                wanted.update(finish_reason=reason, cancel_protocol=cancel_protocol,
                              epoch=expected['epoch'], native_closed=True)
            if ((not samples and reason != 'cancelled')
                    or (reason == 'cancelled' and cancel_protocol is None)
                    or not _exact_metadata(end, wanted)):
                raise ValueError('invalid end')
            if read(1):
                raise ValueError('trailing bytes after END')
            return end if cancel_protocol is not None else None
        else:
            raise ValueError('endpoint stream failed')
