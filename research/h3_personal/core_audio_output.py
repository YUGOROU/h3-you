"""Explicit, readonly default-output check. Importing performs no device calls."""
from __future__ import annotations

import ctypes as C
import sys


def _fourcc(value):
    return int.from_bytes(value.encode('ascii'), 'big')


class _Address(C.Structure):
    _fields_ = [('selector', C.c_uint32), ('scope', C.c_uint32),
                ('element', C.c_uint32)]


def _query():
    """Read only the default output and its bounded metadata; never enumerate."""
    audio = C.CDLL('/System/Library/Frameworks/CoreAudio.framework/CoreAudio')
    cf = C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    get = audio.AudioObjectGetPropertyData
    get.argtypes = [C.c_uint32, C.POINTER(_Address), C.c_uint32, C.c_void_p,
                    C.POINTER(C.c_uint32), C.c_void_p]
    get.restype = C.c_int32
    size = audio.AudioObjectGetPropertyDataSize
    size.argtypes = [C.c_uint32, C.POINTER(_Address), C.c_uint32, C.c_void_p,
                     C.POINTER(C.c_uint32)]
    size.restype = C.c_int32
    cf.CFStringGetCString.argtypes = [C.c_void_p, C.c_void_p, C.c_long, C.c_uint32]
    cf.CFStringGetCString.restype = C.c_bool
    cf.CFRelease.argtypes = [C.c_void_p]
    cf.CFRelease.restype = None

    def read(obj, selector, scope='glob', kind=C.c_uint32):
        address = _Address(_fourcc(selector), _fourcc(scope), 0)
        value = kind()
        length = C.c_uint32(C.sizeof(value))
        if get(obj, C.byref(address), 0, None, C.byref(length), C.byref(value)) or length.value != C.sizeof(value):
            raise RuntimeError('property query failed')
        return value.value

    device = read(1, 'dOut')
    if not device:
        raise RuntimeError('no default output')
    uid = read(device, 'uid ', kind=C.c_void_p)
    if not uid:
        raise RuntimeError('missing UID')
    try:
        text = C.create_string_buffer(256)
        if not cf.CFStringGetCString(uid, text, len(text), 0x08000100):
            raise RuntimeError('UID conversion failed')
        builtin_uid = text.value == b'BuiltInSpeakerDevice'
    finally:
        cf.CFRelease(uid)
    transport = read(device, 'tran')
    alive = read(device, 'livn')
    address = _Address(_fourcc('stm#'), _fourcc('outp'), 0)
    length = C.c_uint32()
    if size(device, C.byref(address), 0, None, C.byref(length)):
        raise RuntimeError('stream query failed')
    if length.value > 4096 or length.value % 4:
        raise RuntimeError('invalid stream metadata')
    streams = (C.c_uint32 * (length.value // 4))()
    if length.value and get(device, C.byref(address), 0, None, C.byref(length), streams):
        raise RuntimeError('stream query failed')
    return builtin_uid and transport == _fourcc('bltn') and alive == 1 and len(streams) > 0


def output_status(*, query=None, platform=None):
    """Stable redacted contract; query injection allows device-free CPU tests."""
    if (sys.platform if platform is None else platform) != 'darwin':
        return dict(default_is_builtin_speaker=False, reason='unsupported_platform')
    try:
        verified = (query or _query)() is True
    except Exception:
        return dict(default_is_builtin_speaker=False, reason='query_failed')
    return dict(default_is_builtin_speaker=verified,
                reason='verified' if verified else 'not_builtin_speaker')
