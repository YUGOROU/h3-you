"""Opt-in D2H candidate only; not connected to any runtime or GPU-accepted.

Input is already quantized NCTHW uint8. No quantization, model placement,
profiling, or source conversion is performed here. Not thread safe. Call abort
on abandoned writes; a failed drain deliberately retains the destination.
"""
from __future__ import annotations

import importlib


# Last-resort safety retention if an abandoned object's CUDA drain also fails.
# Never publish these allocations or retry copies after a CUDA failure.
_UNDRAINED_ALLOCATIONS = []


class SuffixPixelBuffer:
    """One private CPU output allocation, published only after completion.

    ``allocator(shape, dtype=..., pin_memory=...)`` may be injected, as may a
    torch-compatible backend. The native allocator is imported only on use.
    Offsets are explicit and must be contiguous, starting at zero.
    """

    def __init__(self, expected_shape, *, pin_memory=True, torch_backend=None,
                 allocator=None):
        if (type(expected_shape) not in (tuple, list)
                or len(expected_shape) != 5
                or any(type(x) is not int or x <= 0 for x in expected_shape)
                or expected_shape[:2] not in ((1, 3), [1, 3])
                or expected_shape[2] > 124):
            raise ValueError("expected positive [1, 3, frames<=124, h, w]")
        if type(pin_memory) is not bool:
            raise TypeError("pin_memory must be bool")
        self._torch = torch_backend or importlib.import_module("torch")
        if allocator is None:
            allocator = importlib.import_module(
                "fastvideo.utils").allocate_cpu_tensor_with_pin_fallback
        self._shape = tuple(expected_shape)
        self._buffer = allocator(self._shape, dtype=self._torch.uint8,
                                 pin_memory=pin_memory)
        self._frames = 0
        self._pending = False
        self._stream = None
        self._device = None
        self._state = "open"
        self._waits = 0
        self._mode = "none"
        self._pin_requested = pin_memory
        self._validate_tensor(self._buffer)
        if tuple(self._buffer.shape) != self._shape or self._buffer.device.type != "cpu":
            raise ValueError("allocator must return exact CPU output geometry")
        self._pinned = bool(self._buffer.is_pinned())

    def _validate_tensor(self, value):
        if not isinstance(value, self._torch.Tensor):
            raise TypeError("expected torch Tensor")
        if value.dtype != self._torch.uint8:
            raise ValueError("expected already-quantized uint8")
        if value.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA supported")
        if value.layout != self._torch.strided:
            raise ValueError("expected strided tensor")

    def _open(self):
        if self._state != "open":
            raise RuntimeError("buffer is finished or faulted")

    def _drain(self):
        if not self._pending:
            return
        self._waits += 1
        try:
            event = self._torch.cuda.Event(enable_timing=False)
            event.record(self._stream)
            event.synchronize()
        except BaseException:
            # Drain even if event construction/record/sync failed. This is NOT
            # a copy fallback: the operation remains faulted and unpublished.
            self._stream.synchronize()
            self._pending = False
            raise
        self._pending = False

    def _fail(self):
        self._state = "faulted"
        try:
            self._drain()
        finally:
            if not self._pending:
                self._buffer = None

    def append(self, chunk, *, offset):
        """Copy one exact next temporal chunk; never return destination aliases."""
        self._open()
        try:
            if type(offset) is not int or offset != self._frames:
                raise ValueError("offset must equal next frame")
            self._validate_tensor(chunk)
            shape = tuple(chunk.shape)
            if (len(shape) != 5 or shape[:2] != self._shape[:2]
                    or shape[3:] != self._shape[3:] or shape[2] <= 0
                    or offset + shape[2] > self._shape[2]):
                raise ValueError("chunk geometry/overflow")
            device = chunk.device
            if self._device is not None and device != self._device:
                raise ValueError("source device drift")
            self._device = device
            asynchronous = device.type == "cuda" and self._pinned
            if asynchronous:
                stream = self._torch.cuda.current_stream(device)
                if self._stream is not None and stream != self._stream:
                    raise ValueError("source stream drift")
                self._stream = stream
                self._mode = "cuda-pinned-async"
                for batch in range(shape[0]):
                    for channel in range(shape[1]):
                        plane = chunk[batch, channel].contiguous()
                        self._pending = True  # copy may enqueue then raise
                        self._buffer[batch, channel, offset:offset + shape[2]].copy_(
                            plane, non_blocking=True)
            else:
                self._mode = "synchronous"
                self._buffer[:, :, offset:offset + shape[2]].copy_(
                    chunk, non_blocking=False)
            self._frames += shape[2]
        except BaseException:
            self._fail()
            raise

    def finish(self):
        """Publish the stable CPU tensor exactly once, after all writes finish."""
        self._open()
        try:
            if self._frames != self._shape[2]:
                raise ValueError("incomplete frame count")
            self._drain()
        except BaseException:
            self._fail()
            raise
        self._state = "finished"
        return self._buffer

    def abort(self):
        """Fault permanently and drain before releasing private output storage."""
        self._open()
        self._fail()

    def __del__(self):
        # Best effort for abandoned instances, including an already-faulted
        # instance whose explicit drain failed. Keep storage alive on failure.
        if getattr(self, "_pending", False):
            try:
                self._drain()
            except BaseException:
                if self._pending:
                    _UNDRAINED_ALLOCATIONS.append((self._buffer, self._stream))

    @property
    def receipt(self):
        return {"pin_requested": self._pin_requested, "pin_actual": self._pinned,
                "copy_mode": self._mode, "frames": self._frames,
                "completion_waits": self._waits}
