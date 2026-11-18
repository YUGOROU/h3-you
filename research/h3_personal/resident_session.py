"""Single-owner inference model lifetime. No GPU/dependencies at import.

Model/pins/adapter/implementation identity is immutable. Warm conditioning only:
a new Qwen miss requires closing the resident DiT first. Decode likewise needs
explicit release until a measured combined VRAM budget exists.
"""
from contextlib import contextmanager
import hashlib
import threading
from .component_cache import canonical


class ResidentSession:
    def __init__(self, release, *, allow_measured_auxiliary=False, margin_bytes=8*1024**3):
        self._owner = threading.get_ident()
        self._release = release
        self._model = None
        self._key = None
        self._busy = False
        self._closed = False
        if type(margin_bytes) is not int or margin_bytes < 8*1024**3:
            raise ValueError('at least8GiB auxiliary safety margin required')
        self.allow_measured_auxiliary = bool(allow_measured_auxiliary)
        self.margin_bytes = margin_bytes
        self._auxiliary_peaks = {}
        self.loads = 0
        self.hits = 0

    def _check(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('resident session is single-owner; queue commands')
        if self._closed:
            raise RuntimeError('resident session closed')

    @property
    def resident(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('resident session is single-owner')
        return self._model is not None

    @contextmanager
    def turn(self):
        self._check()
        if self._busy:
            raise RuntimeError('overlapping resident turns refused')
        self._busy = True
        try:
            yield self
        except BaseException:
            # An interrupted/failed CUDA turn must not leave reusable state.
            self._dispose()
            raise
        finally:
            self._busy = False

    def release_model(self):
        """Release GPU ownership without losing immutable adapter/model binding."""
        self._check()
        self._dispose()

    def auxiliary_available(self, component, identity, free_bytes):
        """Opt-in reuse only after an isolated SAME-shape/component peak measurement.

        Memory fit is a safety preflight, not a promise against allocator/OOM races.
        Unknown/mismatched footprints release DiT, rather than assume co-residency.
        """
        self._check()
        if self._model is None:
            return True
        if not self.allow_measured_auxiliary:
            raise ValueError('release resident DiT before auxiliary component')
        key=(component,hashlib.sha256(canonical(identity)).hexdigest())
        peak=self._auxiliary_peaks.get(key)
        if type(free_bytes) is not int or free_bytes<0:
            raise ValueError('measured nonnegative free CUDA bytes required')
        if peak is not None and free_bytes>=peak+self.margin_bytes:
            return True
        self._dispose()
        return True

    def record_auxiliary_peak(self, component, identity, peak_bytes):
        self._check()
        if self._model is not None:
            raise ValueError('isolated auxiliary measurement required')
        if type(peak_bytes) is not int or peak_bytes<=0:
            raise ValueError('positive measured peak bytes required')
        key=(component,hashlib.sha256(canonical(identity)).hexdigest())
        self._auxiliary_peaks[key]=max(peak_bytes,self._auxiliary_peaks.get(key,0))

    def require_conditioner_available(self):
        self._check()
        if self._model is not None:
            if self.allow_measured_auxiliary:
                self._dispose()  # New Qwen misses remain separated, never guessed safe.
            else:
                raise ValueError('Qwen cache miss requires releasing resident DiT first')

    def get(self, identity, loader):
        self._check()
        if not self._busy:
            raise RuntimeError('model access requires an ordered turn')
        key = hashlib.sha256(canonical(identity)).hexdigest()
        if self._key is not None and key != self._key:
            raise ValueError('resident model/adapter/pins cannot change')
        if self._model is None:
            model = loader()
            if model is None:
                raise ValueError('model loader returned no model')
            self._model = model
            self._key = key
            self.loads += 1
        else:
            self.hits += 1
        return self._model

    def _dispose(self):
        had_model = self._model is not None
        self._model = None
        if had_model:
            # Drop the owner's reference before GC/empty_cache callback.
            self._release()

    def close(self):
        self._check()
        if self._busy:
            raise RuntimeError('cannot close during an active turn')
        try:
            self._dispose()
        finally:
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
