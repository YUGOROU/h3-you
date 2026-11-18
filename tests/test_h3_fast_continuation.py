"""No GPU, cloud, media, or dependency installation. Optional pinned CPU packing parity."""
import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

from research.h3_personal.fast_continuation import (
    Session, SamplerContract, CleanTarget, WindowCancelled,
    build_layout, verify_packing, INTEGRATION_ACTIVATED,
)
from research.h3_personal.continuation_plan import audio_boundary, video_t


class Tensor:
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        count = 1
        for n in shape:
            count *= n
        self.values = list(values) if values is not None else [0] * count
        assert len(self.values) == count


class Ops:
    def shape(self, t):
        return t.shape

    def copy(self, t):
        return Tensor(t.shape, t.values)

    def slice(self, t, axis, start, end):
        shape = list(t.shape)
        shape[axis] = end-start
        values = []
        for coords in itertools.product(*(range(n) for n in shape)):
            original = list(coords)
            original[axis] += start
            index = 0
            for c, n in zip(original, t.shape):
                index = index*n+c
            values.append(t.values[index])
        return Tensor(shape, values)

    def digest(self, t):
        import hashlib
        return hashlib.sha256(repr((t.shape, t.values)).encode()).hexdigest()

    def equal(self, a, b):
        return a.shape == b.shape and a.values == b.values

    def finite_cpu(self, t):
        import math
        return all(math.isfinite(v) for v in t.values)


class BridgeTests(unittest.TestCase):
    def session(self, name='one', total=685, window=277, ops=None):
        source = Tensor((2, audio_boundary(total), 32),
                        range(2*audio_boundary(total)*32))
        return Session(name, source_audio=source, total_frames=total,
                       height=2, width=2, window_frames=window, ops=ops or Ops())

    def commit(self, s, token, **overrides):
        args = dict(clean_video=Tensor((video_t(token.window.frames), 96),
                                      [token.index+1]*(video_t(token.window.frames)*96)),
                    clean_audio=Ops().copy(token.locked_audio),
                    observed_sampler=SamplerContract())
        args.update(overrides)
        return s.commit_window(token, **args)

    def test_three_windows_actual_guide_and_suffix_ownership(self):
        s = self.session()
        for i, (vt, at, gv, ga, sv, sa, vr, ar) in enumerate((
            (82, 462, 0, 0, 82, 462, 82, 462),
            (82, 462, 7, 37, 75, 425, 157, 887),
            (52, 292, 7, 37, 45, 255, 202, 1142),
        )):
            t = s.begin_window(i)
            self.assertEqual(t.locked_audio.shape, (2, at, 32))
            if gv:
                self.assertEqual(t.guide_video.shape, (gv, 96))
                self.assertEqual(t.guide_audio.shape, (2, ga, 32))
                self.assertEqual(t.guide_video.values, [i]*(gv*96))
                expected = s.ops.slice(t.locked_audio, 1, 0, ga)
                self.assertTrue(s.ops.equal(t.guide_audio, expected))
            else:
                self.assertIsNone(t.guide_video)
                self.assertIsNone(t.guide_audio)
            r = self.commit(s, t)
            self.assertEqual(r.video.shape, (sv, 96))
            self.assertEqual(r.audio.shape, (2, sa, 32))
            self.assertEqual(s.committed_shape, (vr, ar))
            r.video.values[0] = -999  # cannot corrupt retained guide
            r.audio.values[0] = -999
            with self.assertRaises(ValueError):
                self.commit(s, t)
        self.assertIsNone(s._video_tail)
        self.assertIsNone(s._audio_tail)
        with self.assertRaises(ValueError):
            s.begin_window(3)

    def test_explicit_124frame_windows_preserve_native_overlap_counts(self):
        s = self.session(total=328, window=124)
        for i in range(3):
            t = s.begin_window(i)
            self.assertEqual(t.window.frames, 124)
            self.assertEqual(t.sampler.forwards, 4)
            self.commit(s, t)
        self.assertEqual(s.committed_shape, (video_t(328), audio_boundary(328)))
        # No silent padding/window-size switch for an incompatible final target.
        with self.assertRaises(ValueError):self.session(total=294, window=124)

    def test_global_rounding_not_local_duration_rounding(self):
        s = self.session(total=736, window=260)
        expected = ((0, 433, 0, 433), (397, 830, 36, 397), (793, 1227, 37, 397))
        from fractions import Fraction
        for i, (start, end, guide, suffix) in enumerate(expected):
            t = s.begin_window(i)
            self.assertEqual(t.window.audio_start, start)
            self.assertEqual(t.window.audio_end, end)
            self.assertEqual(t.temporal_offset, Fraction(t.window.start*5, 3))
            self.assertEqual(t.locked_audio.shape, (2, end-start, 32))
            self.assertEqual(t.window.guide_audio_t, guide)
            r = self.commit(s, t)
            self.assertEqual(r.audio.shape, (2, suffix, 32))
            self.assertEqual(r.video.shape, (77 if i == 0 else 70, 96))
        self.assertEqual(s.committed_shape, (217, 1227))
        # The last target is 434, while naive local rounding always returns433.
        self.assertNotEqual(end-start, audio_boundary(260))

    def test_order_foreign_token_duplicate_abort_and_cancel(self):
        s, other = self.session(), self.session()  # even same session-id cannot impersonate
        with self.assertRaises(ValueError):
            s.begin_window(1)
        t = s.begin_window(0)
        with self.assertRaises(RuntimeError):
            s.begin_window(0)
        with self.assertRaises(ValueError):
            self.commit(other, t)
        with self.assertRaises(ValueError):
            other.abort_window(t)
        s.abort_window(t)  # backend cancellation uses explicit abort, not model.close
        with self.assertRaises(ValueError):
            s.abort_window(t)
        fresh = s.begin_window(0)
        self.assertGreater(fresh.attempt, t.attempt)
        with self.assertRaises(ValueError):
            self.commit(s, t)
        self.commit(s, fresh)
        with self.assertRaises(ValueError):
            s.abort_window(fresh)
        self.assertEqual(s.committed_shape, (82, 462))

    def test_fault_requires_abort_preserves_last_committed_state(self):
        s = self.session()
        self.commit(s, s.begin_window(0))
        t = s.begin_window(1)
        bad = s.ops.copy(t.locked_audio)
        bad.values[-1] += 1
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.commit(s, t, clean_audio=bad)
        self.assertTrue(s.faulted)
        self.assertEqual(s.committed_shape, (82, 462))
        with self.assertRaises(RuntimeError):
            self.commit(s, t)
        with self.assertRaises(RuntimeError):
            s.begin_window(1)
        s.abort_window(t)
        t = s.begin_window(1)
        self.assertEqual(t.guide_video.values, [1]*(7*96))
        self.commit(s, t)

    def test_mutated_conditions_source_digest_and_failed_begin(self):
        s = self.session()
        self.commit(s, s.begin_window(0))
        t = s.begin_window(1)
        t.guide_video.values[0] = 3
        with self.assertRaisesRegex(ValueError, 'mutated'):
            self.commit(s, t)
        s.abort_window(t)
        s._source.values[0] = -10
        with self.assertRaisesRegex(ValueError, 'digest'):
            s.begin_window(1)
        self.assertTrue(s.faulted)
        with self.assertRaises(RuntimeError):
            s.begin_window(1)
        s.abort_window()  # failed begin had no returned token
        self.assertEqual(s.committed_shape, (82, 462))

    def test_bad_shapes_nonfinite_and_no_eight_forward_override(self):
        for bad in (Tensor((1, 96)), Tensor((82, 96), [float('nan')]*(82*96))):
            s = self.session()
            t = s.begin_window(0)
            with self.assertRaises(ValueError):
                self.commit(s, t, clean_video=bad)
            self.assertEqual(s.committed_shape, (0, 0))
            s.abort_window(t)
        s = self.session()
        t = s.begin_window(0)
        with self.assertRaisesRegex(ValueError, 'SMD4'):
            self.commit(s, t, observed_sampler=SamplerContract('Turbo8-shift12', 8))
        self.assertTrue(s.faulted)
        s.abort_window(t)
        self.assertFalse(INTEGRATION_ACTIVATED)

    def test_allocation_failure_is_transactional(self):
        class FailingOps(Ops):
            fail = False
            def copy(self, t):
                if self.fail:
                    raise MemoryError('injected')
                return super().copy(t)
        ops = FailingOps()
        s = self.session(ops=ops)
        t = s.begin_window(0)
        ops.fail = True
        with self.assertRaises(MemoryError):
            self.commit(s, t)
        self.assertEqual(s.committed_shape, (0, 0))
        self.assertTrue(s.faulted)
        s.abort_window(t)

    def test_close_only_disposes_local_state_no_underlying_shutdown(self):
        s = self.session()
        # No model argument, ownership, or generic lifecycle callback exists.
        t = s.begin_window(0)
        with self.assertRaises(RuntimeError):
            s.close()
        s.abort_window(t)
        s.close()
        with self.assertRaises(RuntimeError):
            s.begin_window(0)
        with self.assertRaises(RuntimeError):
            s.close()
        self.assertIsNone(s._source)

    def test_backend_hooks_carry_latents_fault_cancel_and_never_shutdown(self):
        class Backend:
            stopped = False
            fail = False
            current = True
            def check_current(self):
                return self.current
            def prepare(self, token):
                self.token = token
                return token
            def denoise(self, token):
                if self.fail:
                    raise RuntimeError('local backend failure')
                return CleanTarget(Tensor((video_t(token.window.frames), 96)),
                                   Ops().copy(token.locked_audio), SamplerContract())
            def shutdown(self):
                self.stopped = True
        s, backend = self.session(), Backend()
        first = s.run_window(0, backend)
        self.assertEqual(first.cumulative_video_rows, 82)
        backend.fail = True
        with self.assertRaisesRegex(RuntimeError, 'backend failure'):
            s.run_window(1, backend)
        self.assertTrue(s.faulted)
        self.assertIsNotNone(backend.token.guide_video)
        self.assertEqual(backend.token.guide_audio.shape, (2, 37, 32))
        with self.assertRaises(RuntimeError):
            self.commit(s, s.pending_window)
        s.abort_window(s.pending_window)
        backend.current = False
        with self.assertRaises(WindowCancelled):
            s.run_window(1, backend)
        self.assertEqual(s.committed_shape, (82, 462))
        s.abort_window(s.pending_window)
        backend.fail, backend.current = False, True
        s.run_window(1, backend)
        s.close()
        self.assertFalse(backend.stopped)

    def test_pure_import(self):
        script = ('import sys; import research.h3_personal.fast_continuation; '
                  'assert not any(n in sys.modules for n in ("torch", "fastvideo", "diffusers"))')
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_source_contract_rejects_unverified_module(self):
        with self.assertRaises(ValueError):
            verify_packing(types.SimpleNamespace(__file__=__file__), None)


# Optional CPU test loads ONLY packing/reference source with import stubs, never
# FastVideo package initialization. Source SHA checks remain active. No models.
@unittest.skipUnless(os.environ.get('H3_FASTVIDEO_SOURCE_DIR') and importlib.util.find_spec('torch'),
                     'optional pinned FastVideo source + existing CPU torch required')
class PackingTests(unittest.TestCase):
    def test_real_packing_shared_origin_fixed_images_global_clock(self):
        import torch
        from unittest.mock import patch
        base = Path(os.environ['H3_FASTVIDEO_SOURCE_DIR'])
        root = base / 'fastvideo/pipelines/basic/minimax_h3'
        packing_name = 'fastvideo.pipelines.basic.minimax_h3.packing'
        reference_name = 'fastvideo.pipelines.basic.minimax_h3.reference'
        vision = types.ModuleType('fastvideo.models.vision_utils')
        vision.load_image = lambda *a: None
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        with patch.dict(sys.modules, {'fastvideo.models.vision_utils': vision}):
            packing = load(packing_name, root/'packing.py')
            references = load(reference_name, root/'reference.py')
            images = [references.MiniMaxH3PreparedReference('image', latent_height=2, latent_width=4),
                      references.MiniMaxH3PreparedReference('image', latent_height=4, latent_width=2)]
            s = BridgeTests().session(total=736, window=260)
            tags = torch.tensor([1, 0], dtype=torch.long)
            for w in s.windows:
                layout = build_layout(packing, references, text_tags=tags, images=images,
                                      window=w, height=2, width=2, ranks=1)
                pos = layout.position_ids
                # Two fixed references, two rows each, at original clocks2/3.
                self.assertTrue(torch.equal(pos[layout.video_indices[:4], 0],
                                            torch.tensor([2., 2., 3., 3.], dtype=torch.float64)))
                origin = 4. + float(w.temporal_offset)
                target_v = layout.video_indices[layout.num_condition_video_rows:]
                target_a = layout.audio_indices[layout.num_condition_audio_rows:]
                self.assertEqual(float(pos[target_v[0], 0]), origin)
                self.assertEqual(float(pos[target_a[0], 0]), origin)
                self.assertEqual(layout.num_audio_latents, w.audio_end-w.audio_start)
                self.assertEqual(layout.num_condition_audio_rows, 2*w.guide_audio_t)
                if w.overlap:
                    self.assertEqual(float(pos[layout.video_indices[4], 0]), origin)
                    self.assertEqual(float(pos[layout.audio_indices[0], 0]), origin)
                self.assertEqual(len(images), 2)
                self.assertTrue(all(r.media_type == 'image' for r in images))


if __name__ == '__main__':
    unittest.main()
