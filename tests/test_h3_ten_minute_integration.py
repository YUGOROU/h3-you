"""CPU-only checks across the ten-minute native and Mac seams."""
import inspect
import unittest
from research import hf_h3_direct_conversation as gpu
from research import launch_h3_direct_conversation as launcher
from research.h3_personal.live_relay_worker import run_worker
from research.run_h3_relay_conversation import run_conversation
from research.h3_personal.streaming_native_stages import validate_manifest, SCHEMA


class IntegrationTests(unittest.TestCase):
    def test_flags_reach_both_owners(self):
        self.assertIn('ten_minute_test', inspect.signature(run_worker).parameters)
        self.assertIn('ten_minute_test', inspect.signature(run_conversation).parameters)
        self.assertIn("worker_options['ten_minute_test'] = True", inspect.getsource(gpu.run_session))
        self.assertIn("**({'ten_minute_test': True} if ten_minute_test else {})", inspect.getsource(launcher.run))

    def test_two_answers_do_not_end_explicit_endurance(self):
        from research.h3_personal.live_relay_worker import _should_stop_listening
        self.assertTrue(_should_stop_listening(2, 2, False, False))
        self.assertFalse(_should_stop_listening(2, 2, False, True))
        self.assertTrue(_should_stop_listening(2, 2, True, True))
        self.assertFalse(_should_stop_listening(1, 2, True, True))
        self.assertIn("worker_options['hold_listening'] = True", inspect.getsource(gpu.run_session))

    def test_late_both_rank_manifest_accepted_with_finite_bound(self):
        c = dict(schema=SCHEMA, id='conversation', profile_sha256='a'*64,
                 sequence=141, logical_source_id='idle', logical_index=141,
                 role='listening', source_final=False, conversation_final=False,
                 committed_before=124+140*102, max_windows=142, audio_offer=None)
        m = dict(streaming_audio=c, source_index=0, index=0, total_frames=124,
                 expected_windows=1, final=True,
                 candidate={'audio_context':'window-context-canonical-v1'})
        self.assertIs(validate_manifest(m), c)
        c['max_windows'] = 145
        with self.assertRaises(ValueError):
            validate_manifest(m)


if __name__ == '__main__':
    unittest.main()
