import io
import json
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research.h3_personal import live_relay_worker as worker


class ProducerFailureEvidenceTests(unittest.TestCase):
    def test_timing_is_bounded_and_sink_ack_not_discarded(self):
        pump=SimpleNamespace(timing=lambda:dict(windows=[dict(index=i,ack_sink_accepted=.5,
            ack_sink_consumed=99,token='SECRET') for i in range(200)]))
        out=io.StringIO()
        with redirect_stdout(out):worker._emit_producer_evidence(pump,Mock(),failed=False)
        receipt=json.loads(out.getvalue())
        self.assertEqual(len(receipt['windows']),144)
        self.assertEqual(receipt['windows'][0]['ack_sink_accepted'],.5)
        self.assertNotIn('ack_sink_consumed',receipt['windows'][0])
        self.assertNotIn('SECRET',out.getvalue())

    def test_pump_failure_evidence_precedes_cleanup_and_logging_cannot_mask(self):
        for broken_logging in (False, True):
            with self.subTest(broken_logging=broken_logging), tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                primary = OSError('SECRET token payload text')
                events = []
                backend = Mock()
                driver = Mock()
                driver.close.side_effect = lambda: events.append('cleanup')
                adapter = SimpleNamespace(close=lambda: events.append('adapter_close'),
                    last_failure={'http_status': 503, 'error_type': 'SECRET', 'payload': 'SECRET'})
                pump = SimpleNamespace(
                    sender=SimpleNamespace(snapshot=lambda: dict(reserved=2, verified=1,
                        acked=0, pending=2, inflight_bytes=42, final=False,
                        terminal='SECRET', token='SECRET')),
                    highwater=2, _send_queue=[object()], _error=primary,
                    timing=lambda: dict(windows=[dict(index=0, send_started=.1,
                        token='SECRET')], payload='SECRET'))
                def fail(packets):
                    raise primary
                pump.run = fail
                thread = Mock(ident=None)
                thread.is_alive.return_value = False
                for name, value in [('role_profile_digest', Mock(return_value='a'*64)),
                                    ('LiveNativeBackend', Mock(return_value=backend)),
                                    ('_conversation', Mock(return_value=driver)),
                                    ('ProducerPump', Mock(return_value=pump)),
                                    ('Commands', Mock(return_value=Mock())),
                                    ('threading.Thread', Mock(return_value=thread))]:
                    stack.enter_context(patch.object(worker, name.split('.')[0], value) if '.' not in name
                        else patch('research.h3_personal.live_relay_worker.'+name, value))
                output = io.StringIO()
                stack.enter_context(redirect_stdout(output))
                dumps = json.dumps
                def serialize(value, **kwargs):
                    if value.get('type') in ('h3_producer_timing', 'h3_producer_failure'):
                        self.assertNotIn('cleanup', events)
                        events.append(value['type'])
                        if broken_logging:
                            raise ValueError('serialization failed')
                    return dumps(value, **kwargs)
                stack.enter_context(patch.object(worker.json, 'dumps', side_effect=serialize))
                with self.assertRaises(OSError) as caught:
                    worker.run_worker(SimpleNamespace(quality_profile={}), Mock(), adapter,
                        binding=SimpleNamespace(profile_sha256='a'*64, session='test'),
                        context=None, root=root, deadline=time.monotonic()+10)
                self.assertIs(caught.exception, primary)
                driver.close.assert_called_once()
                backend.close_unused.assert_called_once()
                self.assertIn('h3_producer_timing', events)
                self.assertIn('h3_producer_failure', events)
                self.assertIn('adapter_close', events)
                self.assertNotIn('SECRET', output.getvalue())
                if not broken_logging:
                    rows = [json.loads(line) for line in output.getvalue().splitlines()]
                    evidence = next(r for r in rows if r['type'] == 'h3_producer_failure')
                    self.assertEqual(evidence['sender']['pending'], 2)
                    self.assertEqual(evidence['queue_depth'], 1)
                    self.assertTrue(evidence['transport_failed'])
                    self.assertEqual(evidence['http_status'], 503)


if __name__ == '__main__':
    unittest.main()
