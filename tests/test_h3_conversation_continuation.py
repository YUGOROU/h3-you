"""CPU tensor-value tests; no decoder/GPU/provenance acceptance claims."""
import unittest

from test_h3_fast_continuation import Tensor, Ops as ListOps
from research.h3_personal.fast_continuation import WindowInput, SamplerContract, Suffix
from research.h3_personal.continuation_plan import Window, audio_boundary, video_t
from research.h3_personal.conversation_continuation import ConversationOwner
from research.h3_personal.conversation_audio import audio_origin


class ConversationTests(unittest.TestCase):
    def owner(self, **kw):
        return ConversationOwner('conversation', profile='reference1024-bound',
                                 height=2, width=2, enabled=True,
                                 model_healthy=True, ops=ListOps(), **kw)

    def source(self, owner, sid, total=124, value=2):
        audio = Tensor((2, audio_boundary(total), 32),
                       [value]*(2*audio_boundary(total)*32))
        owner.start_source(sid, canonical_audio=audio, total_frames=total,
                           profile='reference1024-bound', height=2, width=2,
                           audio_origin=audio_origin(owner.committedF, total))
        return audio

    def commit(self, owner, token, final=False, video=None):
        return owner.commit_window(token,
                                   clean_video=video or Tensor((video_t(token.window.frames), 96),
                                                              [9]*(video_t(token.window.frames)*96)),
                                   clean_audio=owner.ops.copy(token.model.locked_audio),
                                   observed_sampler=SamplerContract(),
                                   conversation_final=final)

    def prepare(self, o, t):
        return o.prepare_commit(t, clean_video=Tensor((video_t(t.window.frames), 96)),
                                clean_audio=t.model.locked_audio,
                                observed_sampler=SamplerContract())

    def test_vote_rejection_single_use_and_no_postvote_ops(self):
        o = self.owner()
        self.source(o, 'a')
        t = o.begin_window()
        original = o.source_session
        ticket = self.prepare(o, t)
        self.assertEqual(original.committed_shape, (0, 0))
        self.assertIs(original.pending_window, t.local)
        self.assertEqual(o.committedF, 0)
        o.discard_prepared(ticket)  # external common MIN rejected
        with self.assertRaises(ValueError): o.apply_prepared(ticket)
        foreign = self.owner()
        with self.assertRaises(ValueError): foreign.apply_prepared(ticket)
        ticket = self.prepare(o, t)
        def forbidden(*args):
            raise AssertionError('tensor operation after vote')
        for name in ('copy', 'slice', 'digest', 'shape', 'equal', 'finite_cpu'):
            setattr(o.ops, name, forbidden)
        o.apply_prepared(ticket)
        self.assertEqual(o.committedF, 124)
        self.assertIsNone(original.pending_window)
        self.assertTrue(original._closed)
        with self.assertRaises(ValueError): o.apply_prepared(ticket)

    def test_local_allocation_failure_preserves_original(self):
        for fail_at in range(1, 6):
            o = self.owner()
            self.source(o, 'a', 226)
            t = o.begin_window()
            original = o.source_session
            copier = o.ops.copy
            calls = 0
            def failing(tensor):
                nonlocal calls
                calls += 1
                if calls == fail_at:
                    raise MemoryError('injected')
                return copier(tensor)
            o.ops.copy = failing
            with self.assertRaises(MemoryError): self.prepare(o, t)
            self.assertIs(o.source_session, original)
            self.assertEqual(original.committed_shape, (0, 0))
            self.assertFalse(original.faulted)
            self.assertEqual(o.committedF, 0)
            self.assertIsNone(o._prepared)

    def test_actual_window_waveform_phase(self):
        o = self.owner()
        self.source(o, 'a')
        self.commit(o, o.begin_window())
        o.ack_decoded_count(119)
        total = 226
        n = audio_boundary(total)
        audio = Tensor((2, n, 32),
                       [channel*100000+time*32+k
                        for channel in range(2) for time in range(n) for k in range(32)])
        o.start_source('wave', canonical_audio=audio, total_frames=total,
                       profile='reference1024-bound', height=2, width=2,
                       audio_origin=audio_origin(o.committedF, total))
        for _ in range(2):
            t = o.begin_window()
            start = t.local.window.audio_start
            expected = o.ops.slice(audio, 1, start, t.local.window.audio_end)
            self.assertTrue(o.ops.equal(t.model.locked_audio, expected))
            guide = o.ops.slice(audio, 1, start, start+t.window.guide_audio_t)
            self.assertTrue(o.ops.equal(t.model.guide_audio, guide))
            self.commit(o, t)
            o.ack_decoded_count(o.committedF-5)

    def test_ack_and_window_bound(self):
        o = self.owner()
        with self.assertRaises(RuntimeError): o.ack_decoded_count(0)
        with self.assertRaises(ValueError): self.source(o, 'too-long', 634)
        self.source(o, 'a', 226)
        self.commit(o, o.begin_window())
        with self.assertRaises(RuntimeError): o.begin_window()
        o.ack_decoded_count(119)
        with self.assertRaises(RuntimeError): o.ack_decoded_count(119)
        self.assertIs(o.pending_window, None)
        t = o.begin_window()
        self.assertIs(o.pending_window, t.model)

    def test_actual_windows_scenario_and_multiple_audio_phases(self):
        o = self.owner()
        self.source(o, 'listening')
        t = o.begin_window()
        self.assertIsInstance(t.model, WindowInput)
        self.assertEqual(t.window, Window(0, 124, 0))
        self.assertIsNone(t.model.guide_video)
        r = self.commit(o, t)
        self.assertIsInstance(r.suffix, Suffix)
        self.assertEqual(r.suffix.video.shape, (37, 96))
        self.assertFalse(r.conversation_final)
        self.assertEqual((o.committedF, o.emittedF), (124, 0))
        with self.assertRaises(RuntimeError): self.source(o, 'speech', 328)
        with self.assertRaises(ValueError): o.ack_decoded_count(124)
        o.ack_decoded_count(119)
        self.source(o, 'speech', 328, 4)
        for i, endpoint in enumerate((226, 328, 430)):
            t = o.begin_window()
            self.assertEqual(t.window, Window(endpoint-124, endpoint, 22))
            self.assertEqual(t.model.temporal_offset, t.window.temporal_offset)
            self.assertEqual(t.model.guide_video.shape, (7, 96))
            self.assertEqual(t.model.guide_audio.shape, (2, t.window.guide_audio_t, 32))
            self.assertEqual(t.model.guide_audio.values, [4]*(2*37*32))
            self.assertEqual(t.model.locked_audio.shape, (2, 207, 32))
            if i == 0:
                self.assertEqual(t.local.window, Window(0, 124, 0))
                self.assertIsNone(t.local.guide_video)
            r = self.commit(o, t)
            self.assertEqual(r.suffix.video.shape, (30, 96))
            self.assertEqual(r.source_audio_start, (0, 170, 340)[i]+37)
            self.assertEqual(r.suffix.video_row_offset, (37+30*i))
            self.assertEqual(r.suffix.cumulative_video_rows, 67+30*i)
            o.ack_decoded_count(endpoint-5)
        self.source(o, 'return', value=6)
        t = o.begin_window()
        self.assertEqual(t.window, Window(408, 532, 22))
        self.assertEqual(t.window.guide_audio_t,
                         audio_boundary(430)-audio_boundary(408))
        r = self.commit(o, t, final=True)
        self.assertEqual(r.suffix.cumulative_video_rows, 157)
        self.assertEqual(o.committedF, 532)
        with self.assertRaises(ValueError): o.ack_decoded_count(527)
        o.ack_decoded_count(532, final_flush=True)
        self.assertEqual(o.emittedF, 532)
        with self.assertRaises(RuntimeError): o.ack_decoded_count(532, final_flush=True)
        with self.assertRaises(RuntimeError): self.source(o, 'extra')

    def test_owned_tail_canonical_and_new_source_reset(self):
        o = self.owner()
        original = self.source(o, 'a', value=3)
        original.values[0] = -100  # canonical owned, not caller's mutable storage
        t = o.begin_window()
        clean = Tensor((37, 96), list(range(37*96)))
        r = self.commit(o, t, video=clean)
        digest = o.ops.digest(o._tail)
        clean.values[-1] = -200
        r.suffix.video.values[-1] = -300
        self.assertEqual(o.ops.digest(o._tail), digest)
        o.ack_decoded_count(119)
        self.source(o, 'b', value=17)
        t = o.begin_window()
        self.assertEqual(t.model.guide_video.values, list(range(30*96, 37*96)))
        self.assertEqual(set(t.model.guide_audio.values), {17})
        self.assertEqual(set(t.model.locked_audio.values), {17})
        t.model.guide_video.values[0] = -400
        with self.assertRaises(ValueError): self.commit(o, t)
        self.assertEqual(o.committedF, 124)
        self.assertEqual(o.ops.digest(o._tail), digest)
        o.abandon()

    def test_stale_foreign_shape_profile_and_double_handoff(self):
        o = self.owner()
        self.source(o, 'a')
        t = o.begin_window()
        foreign = self.owner()
        self.source(foreign, 'a')
        with self.assertRaises(ValueError): self.commit(o, foreign.begin_window())
        with self.assertRaises(ValueError): self.commit(o, t, video=Tensor((36, 96)))
        self.assertEqual(o._session.committed_shape, (0, 0))
        self.commit(o, t)
        with self.assertRaises(ValueError): self.commit(o, t)
        o.ack_decoded_count(119)
        with self.assertRaises(ValueError):
            o.start_source('b', canonical_audio=Tensor((2, 207, 32)), total_frames=124,
                           profile='reference512-bound', height=2, width=2)
        self.source(o, 'b')
        with self.assertRaises(RuntimeError): self.source(o, 'c')
        with self.assertRaises(ValueError): o.ack_decoded_count(124, final_flush=True)
        self.assertEqual(o.emittedF, 119)

    def test_allocation_failure_before_local_commit_and_abandon(self):
        o = self.owner()
        self.source(o, 'a')
        t = o.begin_window()
        original_copy = o.ops.copy
        audio = original_copy(t.model.locked_audio)
        def fail(tensor):
            raise MemoryError('allocation')
        o.ops.copy = fail
        with self.assertRaises(MemoryError):
            o.commit_window(t, clean_video=Tensor((37, 96)), clean_audio=audio,
                            observed_sampler=SamplerContract())
        self.assertEqual(o._session.committed_shape, (0, 0))
        self.assertEqual(o.committedF, 0)
        o.ops.copy = original_copy
        o.abandon()
        self.assertIsNone(o._session.pending_window)
        with self.assertRaises(RuntimeError): o.begin_window()
        with self.assertRaises(ValueError):
            ConversationOwner('new', profile='x', height=2, width=2,
                              enabled=True, model_healthy=None)
        self.owner()  # explicit external healthy admission, not inferred from abandon

    def test_global_offset_and_retained_tail_tampering(self):
        o = self.owner()
        self.source(o, 'a')
        t = o.begin_window()
        object.__setattr__(t.model, 'video_row_offset', 96)
        with self.assertRaises(ValueError): self.commit(o, t)
        self.assertEqual(o._session.committed_shape, (0, 0))
        o.abandon()
        o = self.owner()
        self.source(o, 'a')
        self.commit(o, o.begin_window())
        o.ack_decoded_count(119)
        o._tail.values[0] = -987
        with self.assertRaises(ValueError): self.source(o, 'b')
        self.assertEqual(o.committedF, 124)

    def test_real_tensor_digest_fault_and_finite_sources(self):
        o = self.owner(max_sessions=1)
        self.source(o, 'a')
        t = o.begin_window()
        t.model.locked_audio.values[0] = 123
        with self.assertRaises(ValueError): self.commit(o, t)
        self.assertEqual(o.committedF, 0)
        o.abandon()
        o = self.owner(max_sessions=1)
        self.source(o, 'a')
        self.commit(o, o.begin_window())
        o.ack_decoded_count(119)
        with self.assertRaises(ValueError): self.source(o, 'b')


if __name__ == '__main__':
    unittest.main()
