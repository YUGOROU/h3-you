"""Offline resident lease guards; no API or audio-device execution."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.h3_personal import resident_endpoint as r


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def lease(self):
        lease = r.ResidentEndpoint()
        lease.root = self.root
        lease.closed = False
        lease.owned = True
        lease._token = 'test-token'
        lease._key = 'memory-only'
        lease._path = '/owned'
        lease.lease_id = 'fresh-id'
        lease.deadline = time.time() + 1800
        lease.receipt = {'terminal_state': 'unconfirmed'}
        return lease

    def test_invalid_accounting_precedes_all_operations(self):
        for accounting in ('', 'automatic', None, [], {}):
            with self.subTest(accounting=accounting):
                # Non-path inputs prove validation precedes path or cloud operations.
                with self.assertRaisesRegex(ValueError, 'accounting must be'):
                    r.ResidentEndpoint.open(None, None, accounting=accounting)

    def test_default_accounting_is_legacy(self):
        import inspect
        parameter = inspect.signature(r.ResidentEndpoint.open).parameters['accounting']
        self.assertEqual(parameter.default, 'legacy-reservation')
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_legacy_accounting_reserves_local_ledger(self):
        lease = self.lease()
        lease.receipt.update(r._accounting_receipt('legacy-reservation'))
        with patch.object(r, 'reserve') as reserve:
            lease._reserve_accounting()
        reserve.assert_called_once_with(
            r.PRIVATE / 'compute-budget.json', lease.lease_id,
            scope='hf-endpoint-ASR-T4-attended-comet')
        self.assertEqual(lease.receipt['reservation'], lease.lease_id)
        self.assertEqual(lease.receipt['reservation_usd'], '.75')

    def test_operator_managed_has_no_ledger_reservation(self):
        lease = self.lease()
        lease.owned = False
        lease.receipt.update(r._accounting_receipt('operator-managed'))
        with patch.object(r, 'reserve') as reserve:
            lease._reserve_accounting()
            lease.close()
        reserve.assert_not_called()
        receipt = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(receipt['accounting'], 'operator-managed')
        self.assertIsNone(receipt['reservation'])
        self.assertIsNone(receipt['reservation_usd'])

    def test_close_once_and_no_secret(self):
        lease = self.lease()
        with patch.object(r, 'pause_owned', return_value='paused') as pause:
            lease.close()
            lease.close()
        pause.assert_called_once_with('test-token', '/owned')
        self.assertIsNone(lease._key)
        self.assertNotIn('memory-only', (self.root / 'result.json').read_text())
        self.assertEqual(json.loads((self.root / 'result.json').read_text())['terminal_state'], 'paused')

    def test_baseexception_context_cleanup(self):
        lease = self.lease()
        with patch.object(r, 'pause_owned', return_value='paused') as pause:
            with self.assertRaises(KeyboardInterrupt):
                with lease:
                    raise KeyboardInterrupt()
        pause.assert_called_once()
        self.assertEqual(lease.receipt['error_type'], 'KeyboardInterrupt')

    def test_capture_deadline_guard_closes_without_api_or_device(self):
        lease = self.lease()
        lease.deadline = time.time() + 100
        with patch.object(r, 'pause_owned', return_value='paused'), patch.object(r, 'control') as api, patch.object(r.subprocess, 'Popen') as capture:
            with self.assertRaises(ValueError):
                lease.capture(self.root / 'turn', time.time() + 50)
        api.assert_not_called()
        capture.assert_not_called()
        self.assertTrue(lease.closed)

    def test_clean_silent_turn_keeps_resident_lease_running(self):
        lease=self.lease();lease.root=self.root.resolve();lease._binary=self.root/'binary';lease._pins={'capture_binary':'pin'}
        lease._intent={};lease._url='https://example.endpoints.huggingface.cloud'
        transcript=dict(final=False, errors=['no_native_final'],cleanup_complete=True,
                        input_committed=True,finalized_audio_seconds=16,audio_sent_seconds=16)
        process=SimpleNamespace(stdout=SimpleNamespace(close=lambda:None))
        def spawn(*args,**kwargs):
            kwargs['stderr'].write((json.dumps(dict(type='stopped',cleanup=True,frames=256000,nonzero_samples=0))+'\n').encode())
            return process
        def transcribe(*args,**kwargs):kwargs['pcm']();return dict(transcript)
        with patch.object(r,'control',return_value={'status':{'state':'running'}}), patch.object(r,'assert_deployment'), patch.object(r,'wait_for_capture',return_value={}), patch.object(r,'sha',return_value='pin'), patch.object(r.subprocess,'Popen',side_effect=spawn), patch.object(r,'stop_capture'), patch.object(r,'transcribe',side_effect=transcribe), patch.object(r,'pause_owned') as pause:
            with self.assertRaises(r.NoSpeech):lease.capture(self.root/'turn',time.time()+5)
        pause.assert_not_called();self.assertFalse(lease.closed)
        self.assertEqual(json.loads((self.root/'turn/result.json').read_text())['skipped'],'no_speech')

    def test_local_vad_no_speech_rejects_without_pausing(self):
        import io
        lease = self.lease(); lease.root = self.root.resolve()
        lease._binary = self.root/'binary'; lease._pins = {'capture_binary': 'pin'}
        lease._intent = {}; lease._url = 'https://example.endpoints.huggingface.cloud'
        def spawn(*args, **kwargs):
            kwargs['stderr'].write((json.dumps(dict(type='stopped', cleanup=True,
                frames=128000, nonzero_samples=100))+'\n').encode())
            return SimpleNamespace(stdout=io.BytesIO(bytes(256000)))
        def transcribe(*args, **kwargs):
            boundary = kwargs['pcm']()
            self.assertEqual(boundary.read(5120), b'')
            kwargs['on_input_stop']()
            return dict(final=False, errors=['no_input_audio'], cleanup_complete=True,
                        input_committed=False, finalized_audio_seconds=0, audio_sent_seconds=0)
        with patch.object(r, 'control', return_value={'status': {'state': 'running'}}), \
             patch.object(r, 'assert_deployment'), patch.object(r, 'sha', return_value='pin'), \
             patch.object(r.subprocess, 'Popen', side_effect=spawn), \
             patch.object(r, 'stop_capture'), patch.object(r, 'transcribe', side_effect=transcribe), \
             patch.object(r, 'pause_owned') as pause:
            with self.assertRaises(r.NoSpeech):
                lease.capture(self.root/'vad-turn', None)
        pause.assert_not_called()
        self.assertFalse(lease.closed)
        saved = json.loads((self.root/'vad-turn/result.json').read_text())
        self.assertEqual(saved['vad']['reason'], 'no_speech_window')
        self.assertFalse(saved['input_committed'])
        self.assertEqual(saved['timing']['wait_status'], 'not_needed')

    def test_local_vad_retains_native_coverage_requirement(self):
        import io
        import math
        lease = self.lease(); lease.root = self.root.resolve()
        lease._binary = self.root/'binary'; lease._pins = {'capture_binary': 'pin'}
        lease._intent = {}; lease._url = 'https://example.endpoints.huggingface.cloud'
        voice = b''.join(int(2000*math.sin(2*math.pi*200*i/16000)).to_bytes(
            2, 'little', signed=True) for i in range(320))*20 + bytes(32000)
        def spawn(*args, **kwargs):
            kwargs['stderr'].write((json.dumps(dict(type='stopped', cleanup=True,
                frames=len(voice)//2, nonzero_samples=1))+'\n').encode())
            return SimpleNamespace(stdout=io.BytesIO(voice))
        def transcribe(*args, **kwargs):
            boundary = kwargs['pcm'](); sent = b''.join(iter(lambda: boundary.read(5120), b''))
            self.assertEqual(len(sent), int(.9*32000))
            kwargs['on_input_stop']()
            return dict(text='fixture only', final=True, errors=[], cleanup_complete=True,
                        input_committed=True, completion_policy='input-finalized',
                        finalized_audio_seconds=.1, audio_sent_seconds=len(sent)/32000)
        with patch.object(r, 'control', return_value={'status': {'state': 'running'}}), \
             patch.object(r, 'assert_deployment'), patch.object(r, 'sha', return_value='pin'), \
             patch.object(r.subprocess, 'Popen', side_effect=spawn), \
             patch.object(r, 'stop_capture'), patch.object(r, 'transcribe', side_effect=transcribe), \
             patch.object(r, 'pause_owned') as pause:
            with self.assertRaisesRegex(r.TurnRejected, 'full_native_finalized_coverage_required'):
                lease.capture(self.root/'vad-turn', None)
        pause.assert_not_called()
        self.assertFalse(lease.closed)

    def test_pause_failure_still_receipts_and_clears_key(self):
        lease = self.lease()
        with patch.object(r, 'pause_owned', side_effect=RuntimeError('unconfirmed')):
            with self.assertRaises(RuntimeError):
                lease.close()
        self.assertTrue((self.root / 'result.json').exists())
        self.assertIsNone(lease._key)

    def test_intent_is_defensive_copy(self):
        lease = self.lease()
        lease._intent = {'model': {'revision': 'immutable'}}
        lease.intent['model']['revision'] = 'changed'
        self.assertEqual(lease.intent['model']['revision'], 'immutable')

    def parent(self):
        spec = r.deployment('owner/model', 'reviewed', 'owned', 1)  # expired parent is not resumed
        (self.root / 'intent.json').write_text(json.dumps(spec))
        (self.root / 'source-pins.json').write_text('{}')
        return spec

    def test_explicit_external_cleanup_missing_result(self):
        spec = self.parent()
        receipt = self.root / 'external-cleanup.json'
        receipt.write_text(json.dumps(dict(terminal_state='paused', original_result_exists=False)))
        self.assertEqual(r._parent(spec, self.root, SimpleNamespace(private=True, sha='reviewed')), spec)
        for invalid in ({'terminal_state': 'running', 'original_result_exists': False}, {'terminal_state': 'paused'}, {'terminal_state': 'paused', 'original_result_exists': True}):
            receipt.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                r._parent(spec, self.root, SimpleNamespace(private=True, sha='reviewed'))

    def test_parent_config_and_private_branch_guards(self):
        spec = self.parent()
        (self.root / 'result.json').write_text('{"terminal_state":"paused"}')
        for info in (SimpleNamespace(private=False, sha='reviewed'), SimpleNamespace(private=True, sha='other')):
            with self.assertRaises(ValueError):
                r._parent(spec, self.root, info)
        altered = json.loads(json.dumps(spec))
        altered['compute']['scaling']['maxReplica'] = 4
        (self.root / 'intent.json').write_text(json.dumps(altered))
        with self.assertRaises(ValueError):
            r._parent(spec, self.root, SimpleNamespace(private=True, sha='reviewed'))


if __name__ == '__main__':
    unittest.main()
