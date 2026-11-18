"""CPU synthetic lineage only: no model, decoder, or visual-quality acceptance."""
import unittest

from test_h3_fast_continuation import Tensor, Ops
from research.h3_personal.conversation_audio import audio_origin
from research.h3_personal.conversation_continuation import ConversationOwner
from research.h3_personal.continuation_plan import video_t
from research.h3_personal.fast_continuation import SamplerContract
from research.h3_personal.streaming_native_owner import StreamingWindowOwner


class TimeOps(Ops):
    def cat_time(self, a, b):
        values = []
        for channel in range(2):
            for tensor in (a, b):
                stride = tensor.shape[1] * 32
                values.extend(tensor.values[channel * stride:(channel + 1) * stride])
        return Tensor((2, a.shape[1] + b.shape[1], 32), values)


class IdleVisualLineageTests(unittest.TestCase):
    def exercise(self, owner, program, streaming):
        previous_tail = None
        previous_session = None
        committed = 0
        for sequence, (logical, index, role, source_final) in enumerate(program):
            with self.subTest(sequence=sequence, role=role, logical=logical):
                audio = Tensor((2, 207, 32), [sequence + 11] * (2 * 207 * 32))
                kwargs = dict(canonical_audio=audio, profile='p', height=2, width=2)
                if streaming:
                    owner.start_window('window-%d' % sequence,
                        logical_source_id=logical, logical_index=index,
                        role=role, source_final=source_final, **kwargs)
                else:
                    owner.start_source('source-%d' % sequence, total_frames=124,
                        audio_origin=audio_origin(committed, 124), **kwargs)
                session = owner.source_session
                self.assertIsNot(session, previous_session)
                if previous_session is not None:
                    self.assertTrue(previous_session._closed)
                token = owner.begin_window()
                self.assertEqual(token.epoch, sequence + 1)
                self.assertEqual(token.local.window.start, 0)
                self.assertIsNone(token.local.guide_video)
                self.assertEqual(token.window.overlap, 22 if sequence else 0)
                self.assertEqual(token.window.start, committed - 22 if sequence else 0)
                self.assertEqual(token.model.video_row_offset, video_t(committed) if sequence else 0)
                if previous_tail is None:
                    self.assertIsNone(token.model.guide_video)
                else:
                    self.assertEqual(token.model.guide_video.shape, (7, 96))
                    self.assertEqual(token.model.guide_video.values, previous_tail)
                    self.assertIsNot(token.model.guide_video, previous_clean)

                # Unique nonzero values identify window, latent row, and feature.
                # Model output retains the supplied guide, with fresh generated suffix.
                values = [(sequence + 1) * 100000 + row * 100 + feature + 1
                          for row in range(37) for feature in range(96)]
                if previous_tail is not None:
                    values[:7 * 96] = previous_tail
                clean = Tensor((37, 96), values)
                expected_tail = values[-7 * 96:]
                expected_suffix = values[7 * 96:] if sequence else values[:]
                final = sequence == len(program) - 1
                result = owner.commit_window(token, clean_video=clean,
                    clean_audio=owner.ops.copy(token.model.locked_audio),
                    observed_sampler=SamplerContract(), conversation_final=final)
                self.assertEqual(result.suffix.video.values, expected_suffix)
                self.assertEqual(result.suffix.video_row_offset, video_t(committed) if sequence else 0)
                committed = 124 + sequence * 102
                self.assertEqual(owner.committedF, committed)
                self.assertEqual(result.conversation_final, final)
                self.assertEqual(result.suffix.cumulative_video_rows, video_t(committed))
                # Neither caller output nor emitted suffix owns the persistent tail.
                clean.values[-1] = -1001
                result.suffix.video.values[-1] = -2002
                owner.ack_decoded_count(committed if final else committed - 5,
                                        final_flush=final)
                self.assertEqual(owner.emittedF, committed if final else committed - 5)
                previous_tail = expected_tail
                previous_clean = clean
                previous_session = session

    def test_conversation_owner_fresh_sessions_keep_generated_tail(self):
        owner = ConversationOwner('conversation', profile='p', height=2, width=2,
            enabled=True, model_healthy=True, ops=TimeOps(), max_sessions=4)
        self.exercise(owner, (
            ('idle-a', 0, 'listening', True),
            ('idle-b', 0, 'listening', True),
            ('speech', 0, 'speech', True),
            ('return', 0, 'listening', True),
        ), streaming=False)

    def test_streaming_repeated_idle_and_role_reentry_keep_generated_tail(self):
        program = (
            ('idle', 0, 'listening', False),
            ('idle', 1, 'listening', False),
            ('idle', 2, 'listening', False),
            ('speech-a', 0, 'speech', False),
            ('speech-a', 1, 'speech', True),
            ('idle', 0, 'listening', False),
            ('idle', 1, 'listening', False),
            ('speech-b', 0, 'speech', True),
            ('idle', 0, 'listening', False),
            ('idle', 1, 'listening', True),
        )
        owner = StreamingWindowOwner('stream', profile='p', height=2, width=2,
            enabled=True, model_healthy=True, ops=TimeOps(), max_windows=len(program))
        self.exercise(owner, program, streaming=True)


if __name__ == '__main__':
    unittest.main()
