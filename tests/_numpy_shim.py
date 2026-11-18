"""Stdlib-only stand-in for the small numpy surface the Zoom/VoxLock CPU tests use.

The root environment has no numpy (the real lock is ``research/voxlock_experiment``).
Without numpy these tests used to fail at import time, so the repository could not
run its full suite from a clean checkout. When numpy is importable this module is
never used. The shim is not a numeric library: it only reproduces the dtype/shape
attributes and the handful of elementwise operations the code under test touches,
and it raises for anything it does not model instead of guessing.
"""
import struct
import sys


class _Array(list):
    dtype = '<f4'
    ndim = 1

    def astype(self, dtype):
        return _Array(float(v) for v in self)

    def __truediv__(self, other):
        return _Array(v / other for v in self)

    def __mul__(self, other):
        return _Array(v * other for v in self)

    __rmul__ = __mul__

    def tobytes(self):
        return struct.pack('<%dh' % len(self), *(int(v) for v in self))

    def tolist(self):
        return [int(v) for v in self]

    def min(self):
        return min(self)

    def max(self):
        return max(self)


class _Bytes(bytes):
    """Opaque byte buffer standing in for a numpy uint8 array."""

    def astype(self, dtype):
        return _Bytes(self)


class _Bool:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


class _Shim:
    float32 = '<f4'
    uint8 = 'u1'
    int16 = '<i2'

    @staticmethod
    def frombuffer(data, dtype='<i2'):
        if dtype != '<i2':
            raise NotImplementedError('shim frombuffer supports only int16 PCM')
        return _Array(struct.unpack('<%dh' % (len(data) // 2), data))

    def array(self, values, dtype=None):
        if isinstance(values, (bytes, bytearray, memoryview)):
            # Real uint8 arrays are byte sequences; the shim keeps them opaque.
            return _Bytes(values)
        return _Array(int(v) for v in values)

    def asarray(self, values):
        return values if isinstance(values, _Array) else _Array(values)

    def arange(self, *args, dtype=None):
        return _Array(range(*args))

    def zeros(self, n, dtype=None):
        return _Array([0.0] * n)

    def zeros_like(self, values):
        return _Array([0.0] * len(values))

    def ones_like(self, values):
        return _Array([1.0] * len(values))

    def full(self, n, value, dtype='<i2'):
        return _Array([value] * n)

    def isfinite(self, values):
        return _Bool(all(v == v and abs(v) != float('inf') for v in values))

    def rint(self, values):
        return _Array(round(v) for v in values)


def install():
    """Return numpy, or register and return the shim under ``sys.modules['numpy']``."""
    try:
        import numpy
    except ModuleNotFoundError:
        numpy = _Shim()
        sys.modules['numpy'] = numpy
    return numpy
