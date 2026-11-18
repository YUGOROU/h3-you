"""CPU-only worker ownership/exception regressions; no native runtime launch."""
import tempfile
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research.h3_personal.live_relay_worker import run_worker

MODULE = 'research.h3_personal.live_relay_worker.'


class CleanupLifecycleTests(unittest.TestCase):
    def exercise(self, *, native=False, cleanup_failure=False, retire_failure=False):
        primary = RuntimeError('primary failure')
        backend = Mock()
        if cleanup_failure:
            backend.close_unused.side_effect = ValueError('cleanup failure')
        adapter = Mock()
        if cleanup_failure:
            adapter.close.side_effect = OSError('adapter cleanup failure')
        driver = Mock()
        driver._stream = None
        driver.inspect.return_value = SimpleNamespace(status='waiting')
        driver.step.side_effect = primary if not retire_failure else None
        if retire_failure:
            driver.step.return_value = SimpleNamespace(envelope=SimpleNamespace(
                conversation_final=True, regions=()))
            backend.retire.side_effect = primary
        thread = Mock(ident=None)
        thread.is_alive.return_value = False
        pump = Mock()
        if native:
            def consume(packets):
                next(packets)
                return {}
            pump.run.side_effect = consume
        def ready():
            if not native:
                raise primary
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            for name, value in (
                ('LiveNativeBackend', backend), ('_conversation', driver),
                ('ProducerPump', pump), ('threading.Thread', thread),
                ('Commands', Mock()),
            ):
                stack.enter_context(patch(MODULE + name, return_value=value))
            stack.enter_context(patch(MODULE + 'role_profile_digest', return_value='a'*64))
            with self.assertRaises(RuntimeError) as caught:
                run_worker(SimpleNamespace(quality_profile={}), Mock(), adapter,
                    binding=SimpleNamespace(profile_sha256='a'*64, session='test'),
                    context=None, root=root, deadline=time.monotonic()+10,
                    on_ready=ready)
            self.assertIs(caught.exception, primary)
        backend.abandon.assert_not_called()
        return backend

    def test_cleanup_failures_do_not_mask_primary_and_unused_cleanup_is_attempted(self):
        backend = self.exercise(cleanup_failure=True)
        backend.close_unused.assert_called_once_with()

    def test_unused_owner_is_closed_without_native_release(self):
        backend = self.exercise()
        backend.close_unused.assert_called_once_with()
        backend.retire.assert_not_called()

    def test_native_uncertainty_retains_nonreusable_owner(self):
        backend = self.exercise(native=True, cleanup_failure=True)
        backend.close_unused.assert_not_called()
        backend.retire.assert_not_called()

    def test_retirement_uncertainty_is_not_downgraded_to_local_abandonment(self):
        backend = self.exercise(native=True, retire_failure=True)
        backend.retire.assert_called_once_with()
        backend.close_unused.assert_not_called()


if __name__ == '__main__':
    unittest.main()
