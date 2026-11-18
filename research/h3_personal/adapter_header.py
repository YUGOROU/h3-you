"""Bounded safetensors header inspection without loading tensor payloads.

Reports structure only; no license, schedule, base or identity compatibility claim.
"""
import json
import struct

MAX_HEADER_BYTES = 1_048_576


def header_length(prefix):
    if len(prefix) != 8:
        raise ValueError('exact eight-byte safetensors prefix required')
    length = struct.unpack('<Q', prefix)[0]
    if not 2 <= length <= MAX_HEADER_BYTES:
        raise ValueError('safetensors header size outside bounds')
    return length


def inspect_header(prefix, body):
    if len(body) != header_length(prefix):
        raise ValueError('header byte count differs')
    def unique(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate safetensors key')
            out[k] = v
        return out
    header = json.loads(body, object_pairs_hook=unique)
    if not isinstance(header, dict):
        raise ValueError('safetensors object required')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError('safetensors string metadata required')
    pairs = {}
    other = []
    for name, tensor in header.items():
        shape = tensor.get('shape') if isinstance(tensor, dict) else None
        if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('invalid tensor shape')
        if '.lora_A.' in name or '.lora_B.' in name:
            group, tail = name.split('.lora_', 1)
            side, suffix = tail.split('.', 1)
            if suffix not in ('weight', 'default.weight') or side not in ('A', 'B') or len(shape) != 2:
                raise ValueError('unsupported PEFT key layout')
            pair = pairs.setdefault(group, {})
            if side in pair:
                raise ValueError('duplicate adapter side')
            pair[side] = shape
        else:
            other.append(name)
    summaries = []
    for name, pair in sorted(pairs.items()):
        if set(pair) != {'A', 'B'} or pair['A'][0] != pair['B'][1]:
            raise ValueError('adapter rank mismatch/incomplete pair')
        summaries.append(dict(module=name, rank=pair['A'][0],
                              input_features=pair['A'][1], output_features=pair['B'][0]))
    return dict(schema='h3-adapter-header-audit-v1', metadata=metadata,
                tensor_count=len(header), peft_modules=summaries, other_tensor_names=sorted(other),
                alpha_verified=False, base_compatible_verified=False,
                identity_composition_verified=False,
                note='metadata alpha must be checked against author source; never infer alpha=rank')
