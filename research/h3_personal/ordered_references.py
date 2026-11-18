"""Local-only two-reference identity contract; no model, staging or upload.

PNG checks validate signature/IHDR/CRC and exact full-file hash, NOT complete
pixel decoding, scene grounding, native tokens, LoRA support or cloud consent.
Default single-reference callers are unaffected; this module has no side effects.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import zlib

SCHEMA = 'h3-ordered-references-v1'
SINGLE_SCHEMA = 'h3-single-reference-v1'
ROLES = ('person', 'surroundings')


def reference_roles(schema):
    """Cardinality is explicit in the schema; legacy ordered v1 stays exactly two."""
    if type(schema) is not str or schema not in (SCHEMA, SINGLE_SCHEMA):
        raise ValueError('explicit supported reference schema required')
    return ROLES if schema == SCHEMA else ('person',)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_DIMENSION = 32768
MAX_PIXELS = 100_000_000
MAX_EXIF_BYTES = 256 * 1024
MAX_PNG_CHUNKS = 65536
ORIENTATION_POLICY = 'png-exif-identity-only-v1'
_REPO = Path(__file__).resolve().parents[2]
_FIELDS = {'role', 'path', 'sha256', 'width', 'height'}


def _path(value):
    if (type(value) is not str or not value or '\x00' in value
            or not Path(value).is_absolute() or os.path.normpath(value) != value):
        raise ValueError('canonical absolute local reference path required')
    path = Path(value)
    if path == _REPO or _REPO in path.parents:
        raise ValueError('private references must remain outside repository')
    return path


def validate_ordered_references(value, *, check_files=False):
    """Detached exact descriptor. check_files=False is structural, without IO.

    An expected digest must accompany downstream native/cache bindings: changing
    both a file and its declared hash creates a DIFFERENT descriptor, not proof
    of the previously selected identity. No preprocessing is performed here.
    """
    if type(check_files) is not bool:
        raise ValueError('explicit local-file check boolean required')
    if type(value) is not dict or set(value) != {'schema', 'references'}:
        raise ValueError('exact explicit reference schema required')
    roles = reference_roles(value['schema'])
    if type(value['references']) is not list or len(value['references']) != len(roles):
        raise ValueError('reference cardinality must exactly match explicit schema')
    references = []
    for role, record in zip(roles, value['references']):
        if type(record) is not dict or set(record) != _FIELDS or record['role'] != role:
            raise ValueError('ordered person then surroundings descriptors required')
        _path(record['path'])
        if (type(record['role']) is not str or type(record['sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}', record['sha256']) is None
                or any(type(record[k]) is not int or not 1 <= record[k] <= MAX_DIMENSION
                       for k in ('width', 'height'))
                or record['width'] * record['height'] > MAX_PIXELS):
            raise ValueError('bounded original PNG geometry/hash required')
        if record['width']>4*record['height'] or record['height']>4*record['width']:
            raise ValueError('native reference aspect must be within1:4..4:1')
        references.append({k: record[k] for k in ('role', 'path', 'sha256', 'width', 'height')})
    if len(references) == 2 and (references[0]['path'] == references[1]['path'] or references[0]['sha256'] == references[1]['sha256']):
        raise ValueError('two distinct reference files and hashes required')
    result = dict(schema=value['schema'], references=references)
    if check_files:
        for record in references:
            _verify_file(record)
    return result


def config_sha256(value):
    """Canonical ordered descriptor identity; clock/file independent."""
    value = validate_ordered_references(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _preprocessing(short_edge):
    if type(short_edge) is not int or short_edge not in (512, 1024):
        raise ValueError('explicit supported shortedge1024 or512 required')
    from .ordered_qwen_tokens import ordered_token_policy
    return dict(schema='h3-reference-shortedge-round32-v1', short_edge=short_edge,
                round_multiple=32, rounding='python-nearest-ties-even',
                aspect='preserve-with-rounding', crop='none',
                orientation_policy=ORIENTATION_POLICY, admission=ordered_token_policy(short_edge))


def content_identity(value, *, short_edge=1024):
    """Portable content/order/policy identity, intentionally excluding paths.

    This does NOT verify assets or authorize staging. Fresh local checks still
    require verify_local_files; native manifests must compare this identity and
    actual prepared geometry/tokens rather than trusting a filename.
    """
    value=validate_ordered_references(value)
    return dict(schema=value['schema'], identity_kind='content-order-preprocess-v1',
                preprocessing=_preprocessing(short_edge),
                references=[{k:r[k] for k in ('role','sha256','width','height')}
                            for r in value['references']])


def validate_content_identity(value):
    """Strict portable identity parser, without files or cloud authorization."""
    if type(value) is not dict or set(value)!={'schema','identity_kind','preprocessing','references'}:
        raise ValueError('exact portable reference identity required')
    refs=value['references']
    roles=reference_roles(value['schema'])
    if type(refs) is not list or len(refs)!=len(roles):
        raise ValueError('portable reference cardinality must match explicit schema')
    local=[]
    for role,record in zip(roles,refs):
        if type(record) is not dict or set(record)!={'role','sha256','width','height'}:
            raise ValueError('exact portable reference descriptor required')
        local.append(dict(record,path='/private/h3-portable-reference-'+role+'.png'))
    policy=value['preprocessing']
    if type(policy) is not dict or type(policy.get('short_edge')) is not int:
        raise ValueError('explicit preprocessing policy required')
    expected=content_identity(dict(schema=value['schema'],references=local),short_edge=policy['short_edge'])
    if json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('noncanonical portable reference identity')
    return expected


def content_identity_sha256(value):
    identity=validate_content_identity(value)
    return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def content_sha256(value, *, short_edge=1024):
    """Portable identity digest; config_sha256 remains local/path-bound."""
    identity=content_identity(value,short_edge=short_edge)
    return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def config_digest(value, *, short_edge=1024):
    """Explicit portable content/order/preprocess digest (no local paths)."""
    return content_sha256(value,short_edge=short_edge)


def expected_prepared_sizes(value, *, short_edge=1024):
    """Expected native round32 no-crop WxH, not executed processor evidence.

    Matches conditioning_experiment.experimental_resolver's rounding rule.
    R/T depend on actual native layout/conditioner and are deliberately NOT
    inferred from PNG headers here; compare them from both-rank receipts.
    """
    value=validate_ordered_references(value);policy=_preprocessing(short_edge)
    sizes=[]
    for r in value['references']:
        width,height=r['width'],r['height']
        if width>4*height or height>4*width:
            raise ValueError('native reference aspect must be within1:4..4:1')
        scale=short_edge/min(width,height)
        w,h=(max(32,round(n*scale/32)*32) for n in (width,height))
        sizes.append(dict(role=r['role'],width=w,height=h))
    return dict(preprocessing=policy,references=sizes,R=None,T=None,
                token_evidence='requires-actual-native-layout-and-conditioner')


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _exif_orientation(data):
    """Bounded classic TIFF directory parsing, not a general EXIF decoder.

    Orientation must be a unique SHORT/count1 and identity when present. Extra
    IFDs are traversed only to reject conflicting/duplicate orientation. No
    metadata is modified and no thumbnails/pixels are decoded.
    """
    if not 8 <= len(data) <= MAX_EXIF_BYTES or data[:2] not in (b'II', b'MM'):
        raise ValueError('bounded classic TIFF eXIf required')
    endian = '<' if data[:2] == b'II' else '>'
    def number(offset, code):
        size = struct.calcsize(code)
        if offset < 0 or offset + size > len(data):
            raise ValueError('EXIF offset outside payload')
        return struct.unpack_from(endian + code, data, offset)[0]
    if number(2, 'H') != 42:
        raise ValueError('classic TIFF EXIF magic required')
    offset = number(4, 'I')
    if offset < 8:
        raise ValueError('EXIF first IFD required')
    seen = set(); orientation = None
    sizes = {1:1, 2:1, 3:2, 4:4, 5:8, 6:1, 7:1, 8:2, 9:4, 10:8, 11:4, 12:8, 13:4}
    while offset:
        if offset in seen or len(seen) >= 8 or offset < 8:
            raise ValueError('cyclic or unsupported EXIF IFD chain')
        seen.add(offset)
        count = number(offset, 'H')
        if count > 512 or offset + 2 + count * 12 + 4 > len(data):
            raise ValueError('bounded complete EXIF IFD required')
        tags = set()
        for index in range(count):
            entry = offset + 2 + index * 12
            tag = number(entry, 'H'); kind = number(entry + 2, 'H'); length = number(entry + 4, 'I')
            if tag in tags or kind not in sizes or length == 0:
                raise ValueError('ambiguous or malformed EXIF entry')
            tags.add(tag)
            byte_count = sizes[kind] * length
            if byte_count > 4:
                start = number(entry + 8, 'I')
                if start < 8 or start + byte_count > len(data):
                    raise ValueError('EXIF value outside payload')
            if tag == 0x0112:
                if orientation is not None or kind != 3 or length != 1:
                    raise ValueError('unique SHORT/count1 EXIF orientation required')
                orientation = number(entry + 8, 'H')
                if orientation != 1:
                    raise ValueError('nonidentity EXIF orientation prohibited')
        offset = number(offset + 2 + count * 12, 'I')
    return orientation


def _scan_png_orientation(stream, size, digest):
    """Walk bounded PNG chunks/CRCs to IEND; never search arbitrary bytes."""
    count = 33; exif = None; exif_seen = False; idat_seen = False
    for _ in range(MAX_PNG_CHUNKS):
        head = stream.read(8)
        if len(head) != 8:
            raise ValueError('complete PNG chunk stream/IEND required')
        length, kind = struct.unpack('!I4s', head)
        if (count + 12 + length > size or any(not (65 <= c <= 90 or 97 <= c <= 122) for c in kind)
                or kind == b'IHDR' or length > MAX_FILE_BYTES):
            raise ValueError('invalid bounded PNG chunk')
        if kind == b'eXIf' and (exif_seen or length > MAX_EXIF_BYTES):
            raise ValueError('one bounded PNG eXIf chunk required')
        crc = zlib.crc32(kind); digest.update(head); remaining = length
        payload = bytearray() if kind == b'eXIf' else None
        keyword = bytearray() if kind in (b'tEXt', b'zTXt', b'iTXt') else None
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError('truncated PNG chunk data')
            digest.update(chunk); crc = zlib.crc32(chunk, crc); remaining -= len(chunk)
            if payload is not None:payload.extend(chunk)
            if keyword is not None and len(keyword) < 80:keyword.extend(chunk[:80-len(keyword)])
        raw_crc = stream.read(4)
        if len(raw_crc) != 4 or struct.unpack('!I', raw_crc)[0] != crc:
            raise ValueError('PNG chunk CRC mismatch')
        digest.update(raw_crc); count += length + 12
        if keyword is not None:
            if b'\x00' not in keyword or not 1 <= keyword.index(0) <= 79:
                raise ValueError('valid bounded PNG text keyword required')
            if bytes(keyword).split(b'\x00', 1)[0] in (b'Raw profile type exif', b'XML:com.adobe.xmp'):
                raise ValueError('alternate orientation metadata unsupported by identity-only policy')
        if kind == b'eXIf':
            exif_seen = True; exif = _exif_orientation(bytes(payload))
        if kind == b'IDAT':idat_seen = True
        if kind == b'IEND':
            if length or not idat_seen or count != size or stream.read(1):
                raise ValueError('PNG IDAT/final IEND required; trailing bytes prohibited')
            return dict(exif_present=exif_seen, exif_orientation=exif)
    raise ValueError('PNG chunk count bound exceeded')


def _verify_file(record):
    path = _path(record['path'])
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('nonsymlink local reference and ancestors required')
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 33 <= before.st_size <= MAX_FILE_BYTES:
                raise ValueError('bounded regular reference file required')
            header = stream.read(33)
            if (len(header) != 33 or header[:8] != b'\x89PNG\r\n\x1a\n'
                    or header[8:16] != b'\x00\x00\x00\rIHDR'
                    or struct.unpack('!I', header[29:33])[0] != zlib.crc32(header[12:29])):
                raise ValueError('valid PNG IHDR header required')
            width, height, depth, color, compression, filtering, interlace = struct.unpack('!IIBBBBB', header[16:29])
            if ((width, height) != (record['width'], record['height'])
                    or (depth, color, compression, filtering) != (8, 2, 0, 0)
                    or interlace not in (0, 1)):
                raise ValueError('original RGB8 PNG geometry/header mismatch')
            digest = hashlib.sha256(header)
            orientation = _scan_png_orientation(stream, before.st_size, digest)
            after = os.fstat(stream.fileno())
            if (_identity(before) != _identity(after)
                    or digest.hexdigest() != record['sha256']):
                raise ValueError('reference changed or hash mismatch')
        if (any(p.is_symlink() for p in (path, *path.parents))
                or _identity(path.stat(follow_symlinks=False)) != _identity(before)):
            raise ValueError('reference path changed during verification')
        return dict(role=record['role'], source_sha256=record['sha256'],
                    stored_size=[width, height], oriented_size=[width, height], **orientation)
    except OSError:
        raise ValueError('local reference unavailable or unsafe') from None


def verify_orientation(value):
    """Fresh exact-byte identity-only orientation evidence; no pixel decode.

    Stored/oriented dimensions are equal ONLY after bounded metadata scanning
    has rejected nonidentity/ambiguous orientation and alternate channels.
    """
    selected = validate_ordered_references(value)
    return dict(schema='h3-png-orientation-evidence-v1', orientation_policy=ORIENTATION_POLICY,
                images=[_verify_file(record) for record in selected['references']])


def verify_local_files(value, *, expected_sha256=None, expected_content_sha256=None, short_edge=1024):
    """Fresh bounded file verification; never copy or upload either image."""
    result = validate_ordered_references(value)
    if expected_sha256 is not None:
        if (type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
                or config_sha256(result) != expected_sha256):
            raise ValueError('ordered reference selection changed')
    if expected_content_sha256 is not None:
        if (type(expected_content_sha256) is not str
                or re.fullmatch('[0-9a-f]{64}', expected_content_sha256) is None
                or content_sha256(result,short_edge=short_edge) != expected_content_sha256):
            raise ValueError('portable ordered reference content/policy changed')
    return validate_ordered_references(result, check_files=True)


def role_prompt(value):
    """Natural-language role guidance, NOT dedicated person/scene control."""
    selected=validate_ordered_references(value)
    if selected['schema'] == SINGLE_SCHEMA:
        return ('Reference image 1 supplies the person and the surroundings of the same physical place. '
                'Keep this single reference throughout the continuous smartphone video, '
                'preserving the person and the stable background together.')
    return ('Reference image 1 supplies the appearance of the person on the video call. '
            'Reference image 2 shows the surroundings of the same physical place. '
            'Keep both references throughout the continuous handheld smartphone video; '
            'camera framing may reveal the person or the surroundings.')
