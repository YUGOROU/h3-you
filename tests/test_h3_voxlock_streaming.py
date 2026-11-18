"""CPU synthetic ONNX-interface tests, no model/device/network operations."""
import unittest
from types import SimpleNamespace
try:
    import numpy as np
except ImportError:
    np = None
from research.h3_personal.voxlock_streaming import VoxLockStream, load_cpu, MAX_PUSH


class Enroll:
    def run(self, names, feed):
        assert names == ['enroll_n', 'k', 'v']
        assert feed['enroll'].shape == (1, 48000)
        return [np.ones(sh, np.float32) for sh in ((257, 376), (32, 2, 64, 298), (32, 2, 298, 64))]


class Chunk:
    def __init__(self):
        self.calls = 0
        self.bad = False
        self.last = 0
        self.sizes = []
        self.wrong_shape = False

    def get_inputs(self):
        return [SimpleNamespace(name=n, shape=(6, 32, 256) if n in ('h', 'c') else ())
                for n in ('wav', 'enroll_n', 'k', 'v', 'h', 'c')]

    def run(self, names, feed):
        assert names == ['frames', 'h_out', 'c_out']
        assert feed['wav'].ndim == 2 and feed['wav'].shape[0] == 1
        samples = feed['wav'].shape[1]
        assert samples >= 512 and (samples - 512) % 128 == 0
        count = 1 + (samples - 512) // 128
        self.sizes.append(count)
        # Verify actual returned recurrent state is reused, not zeroed per chunk.
        assert feed['h'].flat[0] == self.last
        self.calls += 1
        self.last += count
        w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)).astype(np.float32) ** 2
        frames = np.stack([feed['wav'][0, i*128:i*128+512] for i in range(count)])
        fr = (-2 * frames * w)[None]
        if self.wrong_shape:
            fr = fr[:, :0]
        if self.bad:
            fr[:] = np.nan
        return fr, np.full((6, 32, 256), self.last, np.float32), feed['c'] + count


class LoaderThreadTests(unittest.TestCase):
    def test_threads_validation_and_cpu_options(self):
        import sys
        import tempfile
        import hashlib
        from pathlib import Path
        from unittest.mock import patch, Mock
        from research.h3_personal import voxlock_streaming as module
        ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Mock())
        with patch.dict(sys.modules, {'onnxruntime': ort}):
            for value in (0, 5, -1, True, False, 1.0, '2', None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_cpu('/nonexistent', threads=value)
            with tempfile.TemporaryDirectory() as d:
                data = b'local CPU mock'
                Path(d, 'mock.onnx').write_bytes(data)
                files = {'mock.onnx': (len(data), hashlib.sha256(data).hexdigest())}
                with patch.object(module, 'MODEL_FILES', files):
                    for value in (None, 1, 2, 3, 4):
                        load_cpu(d) if value is None else load_cpu(d, threads=value)
                        args, kwargs = ort.InferenceSession.call_args
                        self.assertEqual(args[1].intra_op_num_threads, value or 1)
                        self.assertEqual(kwargs['providers'], ['CPUExecutionProvider'])
                    Path(d, 'mock.onnx').write_bytes(b'x' * len(data))
                    before = ort.InferenceSession.call_count
                    with self.assertRaises(ValueError):
                        load_cpu(d, threads=2)
                    self.assertEqual(ort.InferenceSession.call_count, before)


@unittest.skipIf(np is None, 'numpy optional; run in isolated runtime')
class StreamingTests(unittest.TestCase):
    def make(self, batch_frames=1):
        c = Chunk()
        return VoxLockStream(Enroll(), c, np.ones(48000, np.float32), batch_frames=batch_frames), c

    def test_batch_partition_state_tail_and_reset(self):
        for batch in (1, 4, 8, 16):
            for length in (0, 1, 256, 257, 511, 512, 513, 2049, 16003):
                x = np.random.default_rng(length).normal(0, .1, length).astype(np.float32)
                for quantum in (127, 1024, 2048, MAX_PUSH):
                    s, c = self.make(batch)
                    baseline, _ = self.make()
                    def process(stream):
                        return np.concatenate([stream.push(x[i:i+quantum])
                            for i in range(0, length, quantum)] + [stream.flush()])
                    y = process(s)
                    np.testing.assert_array_equal(y, process(baseline))
                    np.testing.assert_allclose(y, x, atol=1e-6)
                    np.testing.assert_array_equal(s.h, baseline.h)
                    np.testing.assert_array_equal(s.c, baseline.c)
                    self.assertTrue(all(1 <= n <= batch for n in c.sizes))
                    self.assertEqual(s.n_in, s.n_out)
                    self.assertEqual(len(s.flush()), 0)
                    c.last = 0
                    s.reset()
                    np.testing.assert_array_equal(process(s), y)

    def test_batch_bounds_shape_and_available_only(self):
        for value in (0, 17, True, 1.0, '8', None):
            with self.assertRaises(ValueError):
                self.make(value)
        s, c = self.make(16)
        self.assertEqual(len(s.push(np.ones(512, np.float32))), 128)
        self.assertEqual(c.sizes, [3])
        s.push(np.ones(MAX_PUSH, np.float32))
        self.assertIn(16, c.sizes)
        self.assertLess(len(s.buf), 512)
        c.wrong_shape = True
        with self.assertRaises(ValueError):
            s.push(np.ones(1024, np.float32))
        self.assertTrue(s.failed)

    def test_partition_identity_and_exact_eos(self):
        for length in (0, 1, 2, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 16003):
            for partition in (1, 127, 512, MAX_PUSH):
                s, c = self.make()
                x = np.random.default_rng(length).normal(0, .1, length).astype(np.float32)
                ys = [s.push(x[i:i+partition]) for i in range(0, length, partition)]
                ys.append(s.flush())
                y = np.concatenate(ys)
                self.assertEqual(len(y), length)
                np.testing.assert_allclose(y, x, atol=1e-6)
                self.assertEqual(len(s.flush()), 0)
                self.assertEqual(s.n_out, s.n_in)
                with self.assertRaises(RuntimeError):
                    s.push(x[:0])

    def test_latency_and_bounded_retention(self):
        s, c = self.make()
        self.assertEqual(len(s.push(np.ones(511, np.float32))), 0)
        self.assertEqual(len(s.push(np.ones(1, np.float32))), 128)
        for _ in range(40):
            s.push(np.ones(MAX_PUSH, np.float32))
            self.assertLess(len(s.buf), 512)
            self.assertEqual(len(s.ola), 512)
        self.assertGreater(c.calls, 100)

    def test_reject_bad_inputs_and_reference(self):
        s, _ = self.make()
        for x in (np.ones(MAX_PUSH+1, np.float32), np.zeros((2, 2), np.float32),
                  np.ones(1, np.float64), np.array([np.nan], np.float32)):
            with self.assertRaises(ValueError):
                s.push(x)
        self.assertEqual(s.n_in, 0)
        for ref in (np.ones(47999, np.float32), np.ones(48001, np.float32)):
            with self.assertRaises(ValueError):
                VoxLockStream(Enroll(), Chunk(), ref)
        with self.assertRaises(ValueError):
            VoxLockStream(Enroll(), Chunk(), np.ones(48000, np.float32), sample_rate=48000)

    def test_failure_poison_and_explicit_reset(self):
        s, c = self.make()
        c.bad = True
        with self.assertRaises(ValueError):
            s.push(np.ones(512, np.float32))
        with self.assertRaises(RuntimeError):
            s.flush()
        c.bad = False
        c.last = 0
        s.reset()
        self.assertEqual(s.n_in, 0)
        self.assertTrue(np.all(s.h == 0))
        x = np.ones(600, np.float32)
        np.testing.assert_allclose(np.concatenate((s.push(x), s.flush())), x, atol=1e-6)

    def test_wrong_model_size_no_session(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'enroll.onnx').write_bytes(b'not a model')
            # runtime import is optional; mock it to exercise fail-closed loader.
            import sys
            from unittest.mock import patch
            with patch.dict(sys.modules, {'onnxruntime': SimpleNamespace()}):
                with self.assertRaises(ValueError):
                    load_cpu(d)


if __name__ == '__main__':
    unittest.main()
