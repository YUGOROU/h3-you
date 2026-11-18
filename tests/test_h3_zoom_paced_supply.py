"""Realtime-supply tolerance for the Zoom capture->extract path. CPU only.

No capture device, app, network or provider is used: a paced writer thread
emits the same 64ms/1024-sample PCM16 quanta the Swift tap would produce, into
a bounded FIFO that mirrors the real Swift-side buffers (32x8192 input frames
plus the ~64KiB stdout pipe ~= 7.4s of 16kHz backlog). The consumer drives the
REAL hash-pinned VoxLock FP32 ONNX extraction at CPU4/batch8. If the writer
would overflow the bound it fails closed exactly like the Swift queue-full
path; the reader must then reject the turn as CaptureInterrupted instead of
masking with raw fallback, and the finite ASR lease design must keep running.

Scenario 1 uses the real buffer bound; scenario 2 shrinks the bound only to
make the fail-closed transition observable in bounded test time.
"""
import io
import threading
import time
import unittest

from research.h3_personal.zoom_input import (CaptureInterrupted, EchoGuard,
                                             ZoomPCMReader)
from research.h3_personal.zoom_turn_input import BoundedRaw

MODEL_DIR = '/tmp/h3-voxlock-model-a69f1607601789bf102202a1d7497b67c7cf7fb8'
QUANTUM = 1024          # samples per 64ms quantum (selected profile)
RATE = 16000
WINDOW_SECONDS = 4      # scenario-1 window; writer runs at true realtime pace
REAL_BOUND_BYTES = int(7.4 * RATE * 2)   # 32x8192@48k tap frames + 64KiB pipe


class BoundedPipe:
    """Bounded FIFO raw sink; overflow fails closed like the Swift tap."""

    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.lock = threading.Lock()
        self.data = bytearray()
        self.closed = False
        self.overflow = False
        self.max_backlog = 0

    def write(self, chunk):
        with self.lock:
            if self.closed or len(self.data) + len(chunk) > self.capacity:
                self.overflow = self.closed is False and len(self.data) + len(chunk) > self.capacity
                self.closed = True
                return False
            self.data.extend(chunk)
            self.max_backlog = max(self.max_backlog, len(self.data))
            return True

    def read(self, size):
        while True:
            with self.lock:
                if self.data:
                    out = bytes(self.data[:size])
                    del self.data[:size]
                    return out
                if self.closed:
                    return b''
            time.sleep(0.005)


@unittest.skipUnless(__import__('os').path.isdir(MODEL_DIR),
                     'local verified VoxLock ONNX model directory required')
class PacedSupplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from research.h3_personal.voxlock_streaming import VoxLockStream, load_cpu
        cls.np = np
        enroll, chunk = load_cpu(MODEL_DIR, threads=4)
        # Synthetic enrollment reference: throughput measurement only, never a
        # speaker identity claim and not any private voice.
        rng = np.random.default_rng(42)
        reference = (rng.standard_normal(48000) * 0.05).astype(np.float32)
        cls.stream = VoxLockStream(enroll, chunk, reference, batch_frames=8)

    def reader(self, pipe, samples, *, guard=None):
        guard = guard or EchoGuard(tail_seconds=1, tail_validated=True)
        guard.initial_idle()
        epoch = guard.arm(deadline=time.monotonic() + 40, manual_human_turn=True,
                          all_participants_consented=True, offline=True)
        return ZoomPCMReader(pipe, self.stream, guard, epoch)

    def paced_writer(self, pipe, samples, *, period=QUANTUM / RATE):
        np = self.np
        rng = np.random.default_rng(7)
        start = time.monotonic()
        written = 0
        while written < samples:
            target = start + written / RATE
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            count = min(QUANTUM, samples - written)
            if not pipe.write((rng.standard_normal(count) * 0.05)
                              .astype('<i2').tobytes()):
                break  # Swift-equivalent fail-closed overflow/abort
            written += count
        pipe.closed = True
        return written, time.monotonic() - start

    def test_realtime_supply_keeps_up_with_real_extraction(self):
        """RTF<1 profile: bounded realtime backlog, exact coverage, no failure."""
        samples = WINDOW_SECONDS * RATE
        pipe = BoundedPipe(REAL_BOUND_BYTES)
        reader = self.reader(pipe, samples)
        writer = threading.Thread(target=self.paced_writer, args=(pipe, samples))
        started = time.monotonic()
        writer.start()
        submitted = 0
        process_seconds = 0.0
        while True:
            t0 = time.monotonic()
            chunk = reader.read(5120)
            process_seconds += time.monotonic() - t0
            if not chunk:
                break
            submitted += len(chunk) // 2
        writer.join(timeout=WINDOW_SECONDS + 10)
        elapsed = time.monotonic() - started
        self.assertFalse(pipe.overflow, 'Swift-side bound overflowed; fail-closed expected only under RTF>1')
        self.assertTrue(writer.is_alive() is False)
        receipt = reader.receipt()
        self.assertFalse(receipt['failed'])
        self.assertTrue(receipt['eof'])
        self.assertEqual(receipt['captured_samples'], samples)
        self.assertEqual(receipt['extracted_samples'], samples)
        self.assertEqual(receipt['submitted_samples'], samples)
        self.assertEqual(submitted, samples)
        # Effective streaming RTF includes quantum-granular ONNX calls that the
        # unpaced file benchmark (large pushes) does not capture.
        rtf = elapsed / WINDOW_SECONDS
        print(f'\n[paced-supply] window={WINDOW_SECONDS}s elapsed={elapsed:.3f}s '
              f'streaming_rtf~{rtf:.3f} '
              f'max_backlog={pipe.max_backlog / (RATE * 2):.3f}s', flush=True)
        self.assertLess(pipe.max_backlog, REAL_BOUND_BYTES)

    def test_sustained_slow_consumer_fails_closed_as_interrupted_turn(self):
        """RTF>1 simulation: bound exhausts, turn rejected, no raw fallback.

        The bound is shrunk (0.5s) to observe the fail-closed transition in
        bounded test time; the real 7.4s bound only moves the transition time.
        """
        samples = WINDOW_SECONDS * RATE
        # The real fixed-window contract lives in BoundedRaw: a dead tap ends
        # the stream before the window is full, which must surface as an
        # interrupted turn rather than a clean short capture.
        pipe = BoundedPipe(int(0.5 * RATE * 2))
        reader = self.reader(BoundedRaw(pipe, samples), samples)
        writer = threading.Thread(target=self.paced_writer, args=(pipe, samples))
        writer.start()
        outcome = {}
        def slow_consumer():
            try:
                while True:
                    chunk = reader.read(5120)
                    if not chunk:
                        break
                    time.sleep(0.12)  # deterministic RTF ~1.75 consumer throttle
            except BaseException as exc:
                outcome['error'] = exc
        thread = threading.Thread(target=slow_consumer)
        thread.start()
        thread.join(timeout=WINDOW_SECONDS + 10)
        writer.join(timeout=WINDOW_SECONDS + 10)
        self.assertIsInstance(outcome.get('error'), CaptureInterrupted,
                              'slow consumer must fail closed as an interrupted '
                              'turn, never a raw fallback or silent drop')
        self.assertFalse(reader.receipt()['raw_audio_fallback'])
        self.assertTrue(pipe.overflow)


if __name__ == '__main__':
    unittest.main()