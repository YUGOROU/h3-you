"""Offline payload/plan checks only, never make paid requests."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from research import probe_h3_asr_providers as probe


class ProbeTests(unittest.TestCase):
    def test_offline_default(self):
        out = io.StringIO()
        with patch.object(sys, 'argv', ['probe', '--directory', '/nonexistent']), contextlib.redirect_stdout(out):
            probe.main()
        plan = json.loads(out.getvalue())
        self.assertFalse(plan['true_input_streaming_tested'])
        self.assertEqual(plan['max_calls'], 6)

    def test_together_offline_plan(self):
        out = io.StringIO()
        with patch.object(sys, 'argv', ['probe', '--directory', '/nonexistent', '--provider', 'together']), contextlib.redirect_stdout(out):
            probe.main()
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['max_calls'], 4)
        self.assertEqual(plan['route'], '/together/v1/audio/transcriptions')
        self.assertEqual(plan['allocation_usd'], '0.20')

    def test_budget_reconciliation_distinguishes_router_gate(self):
        for results, expected in (([], '0.3'), ([{'http_status': 200}], '0.3'),
                                  ([{'http_status': 402, 'body': 'Pay-as-you go is not enabled for provider together yet.'}], '0')):
            with self.subTest(results=results), tempfile.TemporaryDirectory() as tmp:
                ledger = Path(tmp)/'budget.json'
                ledger.write_text(json.dumps(dict(spent_usd=0, reservations={})))
                key = probe.reserve(ledger)
                probe.reconcile(ledger, key, results)
                value = json.loads(ledger.read_text())
                self.assertFalse(value['reservations'])
                self.assertEqual(value['terminal_reconciliations'][key]['charge_bound_usd'], expected)

    def test_correct_adapter_route_and_models(self):
        self.assertEqual(probe.ROUTE, '/deepinfra/v1/openai/audio/transcriptions')
        self.assertEqual(len(probe.MODELS), 3)
        self.assertEqual(probe.MODELS[1][1], 'nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b')

    def test_multipart_stream_is_explicit_not_an_input_stream_claim(self):
        body, kind = probe.multipart(b'public-wav', 'Qwen/Qwen3-ASR-1.7B', True)
        self.assertIn('multipart/form-data', kind)
        self.assertIn(b'name="stream"\r\n\r\ntrue', body)
        self.assertIn(b'filename="jfk.wav"', body)
        self.assertLess(body.index(b'name="model"'), body.index(b'name="file"'))
        normal, _ = probe.multipart(b'audio', 'model')
        self.assertNotIn(b'name="stream"', normal)


if __name__ == '__main__':
    unittest.main()
