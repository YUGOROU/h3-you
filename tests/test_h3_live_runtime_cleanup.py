"""No GPU/network/device calls: ownership cleanup under initialization races."""
from concurrent.futures import Future
from unittest import TestCase, main
from unittest.mock import Mock, patch
from research.hf_h3_live_voice import close_tts_future


class CleanupTests(TestCase):
    def test_pending_initialization_is_cancelled(self):
        future=Future()
        close_tts_future(future)
        self.assertTrue(future.cancelled())

    def test_running_initializer_returned_late_is_closed(self):
        future=Future();future.set_running_or_notify_cancel()
        owner=Mock()
        close_tts_future(future)
        owner.close.assert_not_called()
        future.set_result(owner)
        owner.close.assert_called_once_with()

    def test_completed_initializer_is_closed_immediately(self):
        future=Future();owner=Mock();future.set_result(owner)
        close_tts_future(future)
        owner.close.assert_called_once_with()

    def test_owned_child_is_closed_once(self):
        future=Future();owner=Mock();future.set_result(owner)
        close_tts_future(future,owner)
        owner.close.assert_called_once_with()

    def test_failed_initializer_has_no_child_to_close(self):
        future=Future();future.set_running_or_notify_cancel()
        close_tts_future(future)
        future.set_exception(RuntimeError('initialization failed'))
        self.assertIsInstance(future.exception(),RuntimeError)

    def test_already_exited_owned_process_is_not_a_cleanup_failure(self):
        future=Future();owner=Mock();owner.close.side_effect=ProcessLookupError()
        future.set_result(owner)
        close_tts_future(future)
        owner.close.assert_called_once_with()

    def test_generator_shutdown_failure_still_closes_late_tts(self):
        import os
        import sys
        import types
        from pathlib import Path
        from research.hf_h3_live_voice import run_session
        future=Future();future.set_running_or_notify_cancel()
        backend=Mock();backend.start.side_effect=RuntimeError('model initialization failed')
        backend.close.side_effect=RuntimeError('generator shutdown failed')
        paths={key:Path('/not-read') for key in ('base','personal','smd','reference')}
        spec={'identity_prefix':'original64','live_voice_session':{'id':'session'}}
        hub=types.SimpleNamespace(HfApi=Mock(),hf_hub_download=Mock(),CommitOperationAdd=Mock())
        with patch.dict(os.environ,HF_TOKEN='unit-test-placeholder'),patch.dict(sys.modules,huggingface_hub=hub),patch('research.h3_personal.live_voice_runtime.LiveVoiceRuntime',return_value=backend):
            with self.assertRaisesRegex(RuntimeError,'generator shutdown failed'):
                run_session(Path('/not-read'),paths,spec,'not-accessed',future,0)
        owner=Mock();future.set_result(owner)
        owner.close.assert_called_once_with()
        hub.hf_hub_download.assert_not_called()


if __name__=='__main__':main()
