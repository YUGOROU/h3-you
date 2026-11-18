"""Finite append-only original PCM ingress. No synthesis, model, network or devices.

Receiving chunks does NOT imply causal H3 conditioning. The seal-only adapter
exists for explicit complete-source compatibility, never as a streaming claim.
Persistent PCM retention <= max_bytes; reads <= max_read_bytes; sealing may
transiently allocate another max_bytes while joining immutable chunks.
Deadlines are checked on operations (no background timer); owners must cancel
on abandonment or use the context manager for deterministic local cleanup.
"""
from dataclasses import dataclass, field
import hashlib
import math
import re
import threading
import time

from .conversation_timeline import FrozenPCMSource, MAX_PCM_BYTES


def _int(n, low, high):
    if type(n) is not int or not low <= n <= high:
        raise ValueError('bounded strict integer required')


def _id(value):
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError('bounded nonempty ID required')


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('SHA256 required')


@dataclass(frozen=True)
class PCMStreamBinding:
    conversation_id: str
    source_id: str
    epoch: int
    voice_binding_sha256: str
    sample_rate: int = 48000
    channels: int = 1
    sample_width: int = 2
    format: str = 'S16LE'

    def __post_init__(self):
        _id(self.conversation_id); _id(self.source_id)
        _int(self.epoch, 0, 1_000_000); _sha(self.voice_binding_sha256)
        _int(self.sample_rate, 1, 192000); _int(self.channels, 1, 2)
        if type(self.sample_width) is not int or self.sample_width != 2 or self.format != 'S16LE':
            raise ValueError('original signed S16LE only')

    @property
    def stride(self):
        return self.channels*self.sample_width


@dataclass(frozen=True)
class PCMChunk:
    binding: PCMStreamBinding
    index: int
    sample_start: int
    pcm: bytes = field(repr=False)


@dataclass(frozen=True)
class PCMRange:
    status: str  # ready / wait / end; no synthesized silence
    sample_start: int
    requested_stop: int
    available_stop: int
    pcm: bytes = field(repr=False)


class PCMBackpressure(BufferError):
    """Finite retention exhausted; stop this source, never drop or pad PCM.

    This first adapter retains the entire bounded source; it has no eviction or
    credit-release path. A future streaming consumer must add explicit credits.
    """


class PCMIngress:
    def __init__(self, binding, *, max_bytes=16*1024*1024,
                 max_chunk_bytes=256*1024, max_read_bytes=1024*1024,
                 max_samples=None, max_chunks=4096, timeout=120, clock=time.monotonic):
        if type(binding) is not PCMStreamBinding:
            raise ValueError('immutable PCM binding required')
        _int(max_bytes, binding.stride, MAX_PCM_BYTES)
        _int(max_chunk_bytes, binding.stride, max_bytes)
        _int(max_read_bytes, binding.stride, max_bytes)
        _int(max_chunks, 1, 65536)
        if max_samples is None:
            max_samples = min(binding.sample_rate*300, max_bytes//binding.stride)
        _int(max_samples, 1, min(binding.sample_rate*300, max_bytes//binding.stride))
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 0 < timeout <= 300 or not callable(clock)):
            raise ValueError('finite <=300s deadline required')
        self.binding = binding
        self.max_bytes, self.max_chunk_bytes = max_bytes, max_chunk_bytes
        self.max_read_bytes, self.max_samples = max_read_bytes, max_samples
        self.max_chunks = max_chunks
        self._clock = clock
        now = clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('finite monotonic clock required')
        self._last_clock = now
        self._deadline = now+timeout
        self._lock = threading.RLock()
        self._chunks = []
        self._hash = hashlib.sha256()
        self._received = self._index = self._highwater = 0
        self._state = 'open'
        self._source = None

    def _live(self):
        if self._state in ('cancelled', 'expired', 'failed'):
            raise RuntimeError('terminal PCM ingress')
        if self._state in ('open', 'interrupted'):
            now = self._clock()
            if (type(now) not in (int, float) or not math.isfinite(now)
                    or now < self._last_clock):
                self._terminal('failed')
                raise ValueError('invalid monotonic clock')
            self._last_clock = now
            if now >= self._deadline:
                self._terminal('expired')
                raise TimeoutError('PCM ingress deadline')

    def _terminal(self, state):
        self._state = state
        self._chunks.clear()
        self._source = None
        self._hash = None

    def append(self, chunk):
        with self._lock:
            self._live()
            if self._state != 'open':
                raise RuntimeError('sealed ingress refuses append')
            if (type(chunk) is not PCMChunk or type(chunk.binding) is not PCMStreamBinding
                    or chunk.binding != self.binding):
                raise ValueError('stale epoch or PCM binding drift')
            _int(chunk.index, 0, self.max_samples)
            _int(chunk.sample_start, 0, self.max_samples)
            pcm = chunk.pcm
            if (type(pcm) is not bytes or not pcm or len(pcm) > self.max_chunk_bytes
                    or len(pcm) % self.binding.stride):
                raise ValueError('bounded immutable whole-sample chunk required')
            if chunk.index != self._index or chunk.sample_start != self._received:
                raise ValueError('ordered contiguous exactly-once chunks required')
            stop = self._received+len(pcm)//self.binding.stride
            if (self._index >= self.max_chunks or stop > self.max_samples
                    or stop*self.binding.stride > self.max_bytes):
                raise PCMBackpressure('finite retained PCM capacity exhausted')
            digest = self._hash.copy(); digest.update(pcm)
            # Allocate before publishing cursor/hash. Invalid or over-budget
            # input leaves the previous received watermark and digest intact.
            self._chunks.append((self._received, stop, pcm))
            self._received = stop
            self._index += 1
            self._hash = digest
            self._highwater = max(self._highwater, stop*self.binding.stride)
            return stop

    def read(self, sample_start, sample_stop):
        with self._lock:
            self._live()
            _int(sample_start, 0, self.max_samples)
            _int(sample_stop, sample_start+1, self.max_samples)
            if (sample_stop-sample_start)*self.binding.stride > self.max_read_bytes:
                raise ValueError('range read byte bound exceeded')
            if sample_stop > self._received and self._state in ('open', 'interrupted'):
                # Temporary starvation is neither EOS nor permission to pad.
                return PCMRange('wait', sample_start, sample_stop,
                                min(sample_start, self._received), b'')
            stop = min(sample_stop, self._received)
            if stop <= sample_start:
                return PCMRange('end', sample_start, sample_stop, stop, b'')
            stride = self.binding.stride
            if self._source is not None:
                pcm = self._source.pcm[sample_start*stride:stop*stride]
            else:
                pcm = b''.join(data[(max(sample_start,a)-a)*stride:(min(stop,b)-a)*stride]
                    for a,b,data in self._chunks if a < stop and b > sample_start)
            return PCMRange('ready' if stop == sample_stop else 'end',
                            sample_start, sample_stop, stop, pcm)

    def prefix_digest(self, sample_stop):
        """Hash an existing immutable prefix without allocating a whole copy."""
        with self._lock:
            self._live()
            _int(sample_stop, 0, self._received)
            stop = sample_stop*self.binding.stride
            digest = hashlib.sha256()
            if self._source is not None:
                digest.update(memoryview(self._source.pcm)[:stop])
            else:
                for _,_,data in self._chunks:
                    count = min(stop, len(data))
                    digest.update(memoryview(data)[:count])
                    stop -= count
                    if not stop:
                        break
            return digest.hexdigest()

    def seal(self, *, total_samples, sha256):
        with self._lock:
            self._live()
            if self._state != 'open':
                raise RuntimeError('seal exactly once')
            _int(total_samples, 1, self.max_samples); _sha(sha256)
            if total_samples != self._received or sha256 != self._hash.hexdigest():
                self._terminal('failed')
                raise ValueError('final PCM count/hash mismatch')
            pcm = b''.join(data for _,_,data in self._chunks)
            source = FrozenPCMSource(self.binding.source_id, pcm,
                                     self.binding.sample_rate, self.binding.channels, sha256)
            self._source = source
            self._chunks.clear()
            self._state = 'sealed'
            return source

    def finish_interrupted(self, *, total_samples, sha256):
        """Stop producer append, retaining verified prefix for native tail reads.

        Not natural EOS: out-of-prefix reads remain wait, never end/padding.
        The source owner must retire it at a safe boundary and later cancel it.
        It cannot be exposed through the complete-source adapter.
        """
        with self._lock:
            self._live()
            if self._state != 'open':
                raise RuntimeError('producer terminal exactly once')
            _int(total_samples, 0, self.max_samples); _sha(sha256)
            if total_samples != self._received or sha256 != self._hash.hexdigest():
                self._terminal('failed')
                raise ValueError('interrupted prefix count/hash mismatch')
            self._state = 'interrupted'

    def complete_source(self):
        """Explicit compatibility adapter: never returns an unfinished source."""
        with self._lock:
            self._live()
            if self._state != 'sealed':
                raise RuntimeError('complete-source adapter requires explicit EOS')
            return self._source

    def cancel(self):
        with self._lock:
            self._terminal('cancelled')

    def __enter__(self):
        with self._lock:
            self._live()
        return self

    def __exit__(self, *exc):
        self.cancel()

    def snapshot(self):
        with self._lock:
            self._live()
            return dict(state=self._state, received_samples=self._received,
                        chunks=self._index, retained_bytes=(0 if self._source is None
                            and not self._chunks else self._received*self.binding.stride),
                        highwater_bytes=self._highwater,
                        conditioning_ready_samples=None, committed_samples=None,
                        delivered_samples=None)
