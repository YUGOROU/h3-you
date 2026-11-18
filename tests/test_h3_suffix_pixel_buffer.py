"""CPU-only protocol doubles: no CUDA, torch import, or runtime activation."""
import unittest
from types import SimpleNamespace

from research.h3_personal.suffix_pixel_buffer import SuffixPixelBuffer


class Device:
    def __init__(self, kind, index=0):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return (self.type, self.index) == (other.type, other.index)


class Tensor:
    def __init__(self, backend, shape, device="cpu", pinned=False, dtype="uint8",
                 root=None, indices=None, value=7):
        self.backend, self.shape = backend, tuple(shape)
        self.device = Device(device) if isinstance(device, str) else device
        self.dtype, self.layout = dtype, "strided"
        self.pinned = pinned
        self.root = root if root is not None else self
        self.indices = indices
        self.values = [value] * (shape[2] if len(shape) == 5 else shape[0])

    def is_pinned(self):
        return self.pinned

    def __getitem__(self, key):
        if len(key) == 2:
            return Tensor(self.backend, self.shape[2:], self.device,
                          root=self.root, value=self.values[0])
        temporal = key[2]
        return Tensor(self.backend, (temporal.stop - temporal.start, 1, 1),
                      self.device, root=self.root,
                      indices=range(temporal.start, temporal.stop))

    def contiguous(self):
        return self

    def copy_(self, source, *, non_blocking):
        self.backend.copies.append(non_blocking)
        def write():
            for i in self.indices:
                self.root.values[i] = source.values[0]
        if non_blocking:
            self.backend.queue.append(write)
        else:
            write()
        if self.backend.copy_error:
            raise RuntimeError("copy failure")


class Backend:
    Tensor, uint8, strided = Tensor, "uint8", "strided"

    def __init__(self, pinned=True):
        self.pinned = pinned
        self.queue, self.copies = [], []
        self.events = self.syncs = self.stream_syncs = 0
        self.copy_error = self.event_error = False
        self.stream = SimpleNamespace(synchronize=self.stream_sync)
        self.cuda = SimpleNamespace(Event=self.event, current_stream=lambda d: self.stream)
        self.allocations = 0

    def allocate(self, shape, *, dtype, pin_memory):
        self.allocations += 1
        self.output = Tensor(self, shape, pinned=self.pinned, dtype=dtype, value=0)
        return self.output

    def drain(self):
        while self.queue:
            self.queue.pop(0)()

    def stream_sync(self):
        self.stream_syncs += 1
        self.drain()

    def event(self, *, enable_timing):
        assert enable_timing is False
        self.events += 1
        def record(stream):
            assert stream is self.stream
            if self.event_error:
                raise RuntimeError("event failure")
        def sync():
            self.syncs += 1
            self.drain()
        return SimpleNamespace(record=record, synchronize=sync)


class SuffixPixelBufferTests(unittest.TestCase):
    def make(self, pinned=True, frames=5):
        backend = Backend(pinned)
        buffer = SuffixPixelBuffer((1, 3, frames, 2, 2),
                                   torch_backend=backend, allocator=backend.allocate)
        return backend, buffer

    def chunk(self, backend, frames=3, device="cuda", **kwargs):
        return Tensor(backend, (1, 3, frames, 2, 2), device, **kwargs)

    def faulted(self, buffer):
        with self.assertRaises(RuntimeError):
            buffer.finish()
        with self.assertRaises(RuntimeError):
            buffer.append(None, offset=0)

    def test_async_partial_final_single_completion_and_no_early_publication(self):
        b, p = self.make()
        self.assertIsNone(p.append(self.chunk(b), offset=0))
        p.append(self.chunk(b, 2, value=9), offset=3)
        self.assertEqual(b.output.values, [0] * 5)
        self.assertEqual(b.events, 0)
        self.assertEqual(p.receipt["completion_waits"], 0)
        output = p.finish()
        self.assertIs(output, b.output)
        self.assertEqual(output.values, [7, 7, 7, 9, 9])
        self.assertEqual((b.allocations, b.events, b.syncs), (1, 1, 1))
        self.assertEqual(b.copies, [True] * 6)
        self.assertEqual(p.receipt["copy_mode"], "cuda-pinned-async")
        self.faulted(p)
        self.assertEqual(b.events, 1)

    def test_sync_cpu_and_unpinned_cuda(self):
        for pinned, device in ((False, "cuda"), (True, "cpu")):
            b, p = self.make(pinned, 3)
            p.append(self.chunk(b, device=device), offset=0)
            self.assertEqual(p.finish().values, [7] * 3)
            self.assertEqual(b.copies, [False])
            self.assertEqual(b.events, 0)
            self.assertEqual(p.receipt["pin_actual"], pinned)

    def test_abort_and_incomplete_finish_drain(self):
        for abort in (True, False):
            b, p = self.make()
            p.append(self.chunk(b), offset=0)
            if abort:
                p.abort()
            else:
                with self.assertRaises(ValueError):
                    p.finish()
            self.assertEqual(b.syncs, 1)
            self.assertFalse(b.queue)
            self.assertIsNone(p._buffer)
            self.faulted(p)

    def test_copy_and_event_failures_drain_no_fallback(self):
        for failure in ("copy_error", "event_error"):
            b, p = self.make(frames=3)
            setattr(b, failure, True)
            with self.assertRaises(RuntimeError):
                p.append(self.chunk(b), offset=0)
                p.finish()
            self.assertFalse(b.queue)
            self.assertTrue(all(b.copies))
            self.assertIsNone(p._buffer)
            self.faulted(p)
            if failure == "event_error":
                self.assertEqual(b.stream_syncs, 1)

    def test_invalid_chunks_fault_and_drain_prior_writes(self):
        cases = [lambda b: (self.chunk(b, 2), 0),  # reordered
                 lambda b: (self.chunk(b, 3), 3),  # overflow
                 lambda b: (self.chunk(b, 2, dtype="float32"), 3),
                 lambda b: (Tensor(b, (1, 3, 2, 3, 2), "cuda"), 3),
                 lambda b: (self.chunk(b, 2, "cpu"), 3),
                 lambda b: (self.chunk(b, 2, Device("cuda", 1)), 3),
                 lambda b: (self.chunk(b, 2, "mps"), 3),
                 lambda b: (None, 3),
                 lambda b: (self.chunk(b, 2), True)]
        for case in cases:
            b, p = self.make()
            p.append(self.chunk(b), offset=0)
            chunk, offset = case(b)
            with self.assertRaises((ValueError, TypeError)):
                p.append(chunk, offset=offset)
            self.assertFalse(b.queue)
            self.assertEqual(b.syncs, 1)
            self.faulted(p)

    def test_failed_drain_retains_buffer_until_explicit_drain_possible(self):
        b, p = self.make()
        p.append(self.chunk(b), offset=0)
        b.event_error = True
        b.stream.synchronize = lambda: (_ for _ in ()).throw(RuntimeError("drain"))
        with self.assertRaises(RuntimeError):
            p.abort()
        self.assertIs(p._buffer, b.output)
        self.assertTrue(p._pending)
        self.faulted(p)
        b.stream.synchronize = b.stream_sync
        with self.assertRaises(RuntimeError):
            p._drain()  # cleanup only; candidate remains faulted
        self.assertFalse(p._pending)

    def test_stream_drift_drains_original_stream(self):
        b, p = self.make()
        p.append(self.chunk(b), offset=0)
        b.cuda.current_stream = lambda d: object()
        with self.assertRaises(ValueError):
            p.append(self.chunk(b, 2), offset=3)
        self.assertEqual(b.syncs, 1)
        self.assertFalse(b.queue)
        self.faulted(p)

    def test_bad_allocator_output(self):
        for device, dtype, shape in (("cuda", "uint8", (1, 3, 3, 2, 2)),
                                     ("cpu", "float32", (1, 3, 3, 2, 2)),
                                     ("cpu", "uint8", (1, 3, 2, 2, 2))):
            b = Backend()
            with self.assertRaises(ValueError):
                SuffixPixelBuffer((1, 3, 3, 2, 2), torch_backend=b,
                                  allocator=lambda *a, **k: Tensor(b, shape, device,
                                                                  dtype=dtype))

    def test_real_torch_cpu_layout_if_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional local Torch unavailable; no install')
        source = torch.arange(60, dtype=torch.uint8).reshape(1, 3, 5, 2, 2)
        p = SuffixPixelBuffer(source.shape, pin_memory=False, torch_backend=torch,
            allocator=lambda shape, **kw: torch.empty(shape, **kw))
        p.append(source[:, :, :3], offset=0)
        p.append(source[:, :, 3:], offset=3)
        self.assertTrue(torch.equal(p.finish(), source))
        self.assertEqual(p.receipt['completion_waits'], 0)

    def test_constructor_validation(self):
        for shape in ((1, 3, 125, 2, 2), (True, 3, 2, 2, 2),
                      (1, 4, 2, 2, 2), (1, 3, 0, 2, 2)):
            with self.assertRaises(ValueError):
                SuffixPixelBuffer(shape)
        with self.assertRaises(TypeError):
            SuffixPixelBuffer((1, 3, 2, 2, 2), pin_memory=1)


if __name__ == "__main__":
    unittest.main()
