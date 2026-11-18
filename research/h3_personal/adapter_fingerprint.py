"""CPU-only canonical fingerprints; never decode or export tensor values."""
import hashlib
import json
import math
import os
import struct

SCHEMA = 'h3-safetensors-canonical-sha256-v1'
_MAX_HEADER = 16 * 1024 * 1024
_CHUNK = 1024 * 1024
# Explicit allowlist: unsupported/new dtypes fail closed.
_SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
          'I32': 4, 'U32': 4, 'I64': 8, 'U64': 8,
          'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8,
          'F8_E4M3': 1, 'F8_E5M2': 1}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def fingerprint_file(path):
    """Return schema and digest binding sorted names, dtypes, shapes and raw bytes.

    Header is bounded; payload reads use at most 1 MiB. Offsets must describe
    the entire payload without gaps/overlaps (including valid empty tensors).
    """
    with open(path, 'rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(_MAX_HEADER, size - 8):
            raise ValueError('invalid safetensors header length')
        raw = stream.read(length)
        if not raw.startswith(b'{'):
            raise ValueError('invalid safetensors header')
        try:
            header = json.loads(raw.decode('utf-8'), object_pairs_hook=_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('invalid JSON constant')))
        except (UnicodeError, RecursionError) as exc:
            raise ValueError('invalid safetensors JSON') from exc
        if not isinstance(header, dict):
            raise ValueError('header must be an object')
        metadata = header.pop('__metadata__', {})
        if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
            raise ValueError('invalid metadata')
        payload_size = size - 8 - length
        intervals = []
        for key, tensor in header.items():
            if not isinstance(tensor, dict) or set(tensor) != {'dtype', 'shape', 'data_offsets'}:
                raise ValueError('invalid tensor descriptor')
            dtype, shape, offsets = (tensor[k] for k in ('dtype', 'shape', 'data_offsets'))
            if not isinstance(dtype, str) or dtype not in _SIZES:
                raise ValueError('unsupported dtype')
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError('invalid shape')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int for v in offsets)):
                raise ValueError('invalid offsets')
            start, end = offsets
            if not 0 <= start <= end <= payload_size or end - start != math.prod(shape) * _SIZES[dtype]:
                raise ValueError('tensor bounds or byte size mismatch')
            intervals.append((start, end))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError('payload gap or overlap')
            cursor = end
        if cursor != payload_size:
            raise ValueError('unclaimed payload bytes')
        digest = hashlib.sha256(SCHEMA.encode('ascii') + b'\0')
        for key in sorted(header):
            tensor = header[key]
            descriptor = json.dumps([key, tensor['dtype'], tensor['shape']],
                                    ensure_ascii=True, separators=(',', ':')).encode('ascii')
            digest.update(struct.pack('<Q', len(descriptor)))
            digest.update(descriptor)
            start, end = tensor['data_offsets']
            digest.update(struct.pack('<Q', end - start))
            stream.seek(8 + length + start)
            remaining = end - start
            while remaining:
                chunk = stream.read(min(_CHUNK, remaining))
                if not chunk:
                    raise ValueError('truncated tensor payload')
                digest.update(chunk)
                remaining -= len(chunk)
        return {'canonical_tensor_sha256': digest.hexdigest(),
                'canonical_tensor_schema': SCHEMA}
