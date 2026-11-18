"""Mocked Zoom control-chain wiring: no app, device, model, network or capture.

Covers the seams added after the v2 offline result: DirectASRInput pipeline
binding, ResidentEndpoint prepare/spawn/receipt ordering, InputBlocked lease
health, and build_native hash gates.
"""
import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from research.h3_personal import resident_endpoint as r
from research.h3_personal.direct_asr_input import DirectASRInput
from research.h3_personal.endpoint_asr import FINAL
from research.h3_personal.zoom_input import (CaptureInterrupted, InputBlocked,
                                             source_contract)
from research.h3_personal.zoom_turn_input import ZoomTurnInput, build_native

SOURCE = source_contract(reference_sha256='a' * 64, capture_source_sha256='b' * 64)


def zoom_capture(profile_sha256, samples=128000):
    return dict(schema='h3-zoom-asr-capture-v1', profile_sha256=profile_sha256, input_epoch=1,
                captured_samples=samples, extracted_samples=samples, submitted_samples=samples,
                capture_frames=samples + 10, capture_stopped=True, capture_cleanup=True,
                capture_app='zoom', mute_behavior='unmuted', eof=True, pending_samples=0,
                discarded_pending_samples=0, raw_audio_fallback=False,
                operator_confirmed_remote_idle=True)


def base_receipt():
    return dict(text='こんにちは', final=True, errors=[], capture_stopped=True,
                cleanup_complete=True, input_committed=True,
                completion_policy='input-finalized', audio_sent_seconds=8,
                finalized_audio_seconds=8,
                events=[dict(text='こんにちは', final=True,
                             endpoint_source='native_completed',
                             native_event_type=FINAL)])


class Pipeline:
    """Stub pipeline; a real ZoomTurnInput is used where only the contract matters."""
    capture_argv = ['--stream', '--app', 'zoom']

    def __init__(self, source=SOURCE, prepare=None):
        self.source_contract = copy.deepcopy(source)
        self.prepare = prepare
        self.calls = []
        self.reader = SimpleNamespace(receipt=lambda: None)

    def prepare_capture(self, *, deadline):
        self.calls.append(('prepare', deadline))
        if self.prepare is not None:
            self.prepare(deadline)

    def wrap_capture(self, raw, *, capture_seconds):
        self.calls.append(('wrap', capture_seconds, raw))
        return self.reader

    def finish_capture(self, last, transcript, *, capture_seconds):
        self.calls.append(('finish', last, capture_seconds))
        return zoom_capture(self.source_contract['profile_sha256'])


class Lease:
    name = 'h3-asr-t4-7c926aae18'
    closed = False

    def __init__(self, root, results, *, capture_source_sha256='b' * 64):
        self.root = Path(root)
        self.deadline = time.time() + 1800
        self.intent = dict(name=self.name, model=dict(revision='a' * 40))
        self._pins = dict(capture_source=capture_source_sha256, capture_binary='pin')
        self.results = iter(results)
        self.calls = []

    def capture(self, directory, planned_start, *, capture_seconds, input_pipeline=None):
        self.calls.append((directory, planned_start, capture_seconds, input_pipeline))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class DirectInputZoomTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pipeline = ZoomTurnInput(object(), copy.deepcopy(SOURCE),
                                      confirmation=lambda **kw: True,
                                      all_participants_consented=True)

    def source(self, results, **kwargs):
        lease = Lease(self.tmp.name, results, **kwargs)
        return lease, DirectASRInput(lease, capture_seconds=8,
                                     input_pipeline=self.pipeline)

    def call(self, source, remaining=1000):
        with patch('builtins.print'):
            return source(context=dict(session='private', epoch=1),
                          deadline=time.monotonic() + remaining)

    def test_expected_source_and_zoom_receipt_published(self):
        receipt = base_receipt()
        receipt['input_source'] = copy.deepcopy(self.pipeline.source_contract)
        receipt['zoom_capture'] = zoom_capture(
            self.pipeline.source_contract['profile_sha256'], samples=8 * 16000)
        lease, source = self.source([receipt])
        self.assertEqual(source.expected_asr['input_source'],
                         self.pipeline.source_contract)
        item = self.call(source)
        self.assertEqual(item['text'], 'こんにちは')
        self.assertEqual(item['asr']['zoom_capture'], receipt['zoom_capture'])
        self.assertEqual(self.pipeline.source_contract['origin'],
                         'zoom_process_voxlock_endpoint_asr')
        from research.h3_personal.asr_source_contract import asr_origin
        self.assertEqual(asr_origin(item['asr']), 'zoom_process_voxlock_endpoint_asr')
        directory, _start, seconds, pipeline = lease.calls[0]
        self.assertEqual(seconds, 8)
        self.assertIs(pipeline, self.pipeline)
        self.assertEqual(directory.parent, lease.root.resolve())

    def test_operator_decline_returns_none_and_keeps_lease(self):
        lease, source = self.source([InputBlocked('declined'), base_receipt()])
        self.assertIsNone(self.call(source))
        self.assertEqual((source.accepted, source.attempts), (0, 1))
        self.assertFalse(lease.closed)
        self.assertEqual(len(lease.calls), 1)

    def test_capture_interruption_retries_within_budget_then_recovers(self):
        receipt = base_receipt()
        receipt['input_source'] = copy.deepcopy(self.pipeline.source_contract)
        receipt['zoom_capture'] = zoom_capture(
            self.pipeline.source_contract['profile_sha256'], samples=8 * 16000)
        lease, source = self.source([CaptureInterrupted('process churn'), receipt])
        item = self.call(source)
        self.assertEqual(item['text'], 'こんにちは')
        self.assertEqual((source.accepted, source.attempts), (1, 2))
        self.assertFalse(lease.closed)
        self.assertEqual(len(lease.calls), 2)

    def test_capture_interruption_budget_exhaustion_returns_none(self):
        lease, source = self.source([CaptureInterrupted('churn') for _ in range(3)])
        self.assertIsNone(self.call(source))
        self.assertEqual((source.accepted, source.attempts), (0, 3))
        self.assertFalse(lease.closed)

    def test_zoom_receipt_drift_and_missing_capture_rejected(self):
        drift = base_receipt()
        drift['input_source'] = copy.deepcopy(self.pipeline.source_contract)
        drift['input_source']['profile_sha256'] = 'c' * 64
        drift['zoom_capture'] = zoom_capture('c' * 64)
        missing = base_receipt()
        missing['input_source'] = copy.deepcopy(self.pipeline.source_contract)
        for receipt in (drift, missing):
            with self.subTest(missing='zoom_capture' not in receipt):
                lease, source = self.source([receipt])
                with self.assertRaises(ValueError):
                    self.call(source)
                self.assertFalse(lease.closed)
                self.assertEqual(source.accepted, 1)  # Counted, never published.

    def test_capture_seconds_bounds_and_comet_default(self):
        lease = Lease(self.tmp.name, [])
        for bad in (7, 46, True, 8.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DirectASRInput(lease, capture_seconds=bad,
                               input_pipeline=self.pipeline)
        comet = DirectASRInput(lease)
        self.assertNotIn('input_source', comet.expected_asr)
        comet.note_reply_published()  # Comet keeps its existing no-op behaviour.


class ResidentZoomTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lease = r.ResidentEndpoint()
        self.lease.root = self.root.resolve()
        self.lease.closed = False
        self.lease._token = 'test-token'
        self.lease._key = 'memory-only'
        self.lease._path = '/owned'
        self.lease._binary = self.root / 'binary'
        self.lease.lease_id = 'fresh-id'
        self.lease.deadline = time.time() + 1800
        self.lease.receipt = {'terminal_state': 'unconfirmed'}
        self.lease._intent = {}
        self.lease._url = 'https://example.endpoints.huggingface.cloud'
        self.lease.owned = True
        self.lease._pins = {'capture_binary': 'pin', 'capture_source': 'b' * 64}

    def patches(self, *, spawn=None, transcribe=None):
        return [patch.object(r, 'control', return_value={'status': {'state': 'running'}}),
                patch.object(r, 'assert_deployment'),
                patch.object(r, 'wait_for_capture', return_value={}),
                patch.object(r, 'sha', return_value='pin'),
                patch.object(r.subprocess, 'Popen', side_effect=spawn or (lambda *a, **k: None)),
                patch.object(r, 'stop_capture'),
                patch.object(r, 'transcribe', side_effect=transcribe),
                patch.object(r, 'pause_owned', return_value='paused')]

    def test_source_pin_drift_fails_before_prepare_or_spawn(self):
        pipeline = Pipeline(source=source_contract(reference_sha256='a' * 64,
                                                   capture_source_sha256='d' * 64))
        with patch.object(r, 'control', return_value={'status': {'state': 'running'}}), \
                patch.object(r, 'assert_deployment'), \
                patch.object(r, 'sha', return_value='pin'), \
                patch.object(r.subprocess, 'Popen') as capture, \
                patch.object(r, 'pause_owned', return_value='paused') as pause:
            with self.assertRaisesRegex(ValueError, 'capture source pin drift'):
                self.lease.capture(self.root / 'turn', time.time() + 5,
                                   capture_seconds=8, input_pipeline=pipeline)
        capture.assert_not_called()
        pause.assert_called_once()  # Unexpected failure keeps existing close semantics.
        self.assertEqual(pipeline.calls, [])

    def test_capture_interruption_rejects_turn_keeps_lease(self):
        pipeline = Pipeline(prepare=lambda deadline: (_ for _ in ()).throw(
            CaptureInterrupted('tap process list changed')))
        with patch.object(r, 'control', return_value={'status': {'state': 'running'}}), \
                patch.object(r, 'assert_deployment'), \
                patch.object(r, 'wait_for_capture', return_value={}), \
                patch.object(r, 'sha', return_value='pin'), \
                patch.object(r.subprocess, 'Popen') as capture, \
                patch.object(r, 'pause_owned', return_value='paused') as pause:
            with self.assertRaises(CaptureInterrupted):
                self.lease.capture(self.root / 'turn', time.time() + 5,
                                   capture_seconds=8, input_pipeline=pipeline)
        capture.assert_not_called()
        pause.assert_not_called()
        self.assertFalse(self.lease.closed)
        self.assertEqual(self.lease.receipt['last_rejected_turn_code'],
                         'zoom_capture_interrupted')

    def test_operator_decline_keeps_resident_lease_running(self):
        pipeline = Pipeline(prepare=lambda deadline: (_ for _ in ()).throw(InputBlocked('declined')))
        with patch.object(r, 'control', return_value={'status': {'state': 'running'}}), \
                patch.object(r, 'assert_deployment'), \
                patch.object(r, 'wait_for_capture', return_value={}), \
                patch.object(r, 'sha', return_value='pin'), \
                patch.object(r.subprocess, 'Popen') as capture, \
                patch.object(r, 'pause_owned', return_value='paused') as pause:
            with self.assertRaises(InputBlocked):
                self.lease.capture(self.root / 'turn', time.time() + 5,
                                   capture_seconds=8, input_pipeline=pipeline)
        capture.assert_not_called()
        pause.assert_not_called()
        self.assertFalse(self.lease.closed)
        self.assertEqual(self.lease.receipt['last_rejected_turn_code'],
                         'zoom_input_not_eligible')

    def test_zoom_success_binds_explicit_argv_and_source_receipt(self):
        pipeline = Pipeline()
        transcript = dict(base_receipt(), text='こんにちは')
        process = SimpleNamespace(stdout=SimpleNamespace(close=lambda: None))
        seen = {}

        def spawn(argv, **kwargs):
            seen['argv'] = argv
            kwargs['stderr'].write((json.dumps(dict(
                type='stopped', cleanup=True, frames=128010,
                app='zoom', mute_behavior='unmuted')) + '\n').encode())
            return process

        def transcribe(url, **kwargs):
            seen['policy'] = kwargs['completion_policy']
            kwargs['pcm']()
            return dict(transcript)

        patchers = self.patches(spawn=spawn, transcribe=transcribe)
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        result = self.lease.capture(self.root / 'turn', time.time() + 5,
                                    capture_seconds=8, input_pipeline=pipeline)
        self.assertEqual(seen['argv'][1:], ['--stream', '--app', 'zoom'])
        self.assertEqual(seen['policy'], 'input-finalized')
        wrap = [c for c in pipeline.calls if c[0] == 'wrap']
        self.assertEqual(len(wrap), 1)
        self.assertEqual(wrap[0][1], 8)
        self.assertIs(wrap[0][2], process.stdout)
        self.assertEqual(result['input_source'], pipeline.source_contract)
        self.assertEqual(result['zoom_capture'],
                         zoom_capture(pipeline.source_contract['profile_sha256']))
        self.assertNotIn('input_boundary_policy', result)
        self.assertTrue((self.root / 'turn' / 'result.json').exists())


class BuildNativeTests(unittest.TestCase):
    PCM = struct.pack('<48000f', *([0.25] * 48000))
    FILE_SHA = 'c' * 64
    CAPTURE_SHA = 'd' * 64

    def file_sha(self, path):
        return self.FILE_SHA if str(path).endswith('ref.wav') else self.CAPTURE_SHA

    def call(self, **over):
        kwargs = dict(model_dir='model', reference='ref.wav',
                      reference_file_sha256=self.FILE_SHA,
                      reference_pcm_sha256=hashlib.sha256(self.PCM).hexdigest(),
                      capture_source='capture.swift',
                      capture_source_sha256=self.CAPTURE_SHA)
        kwargs.update(over)
        return build_native(**kwargs)

    def test_file_hash_gates_precede_convert_and_model_load(self):
        from research.h3_personal import voxlock_streaming
        with patch('research.run_h3_zoom_preflight.file_sha',
                   side_effect=self.file_sha) as sha, \
                patch('research.run_h3_zoom_preflight.convert') as convert, \
                patch.object(voxlock_streaming, 'load_cpu') as load, \
                patch.object(voxlock_streaming, 'VoxLockStream') as stream:
            for over in (dict(reference_file_sha256='e' * 64),
                         dict(capture_source_sha256='e' * 64)):
                with self.subTest(over=over), \
                        self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    self.call(**over)
            convert.assert_not_called()
            load.assert_not_called()
            stream.assert_not_called()
            self.assertTrue(sha.called)

    def test_pcm_hash_and_offset_gates_precede_model_load(self):
        from research.h3_personal import voxlock_streaming
        with patch('research.run_h3_zoom_preflight.file_sha',
                   side_effect=self.file_sha), \
                patch('research.run_h3_zoom_preflight.convert',
                      return_value=self.PCM) as convert, \
                patch.object(voxlock_streaming, 'load_cpu') as load, \
                patch.object(voxlock_streaming, 'VoxLockStream') as stream:
            for over in (dict(reference_pcm_sha256='f' * 64),
                         dict(reference_offset=28),
                         dict(reference_offset=float('nan'))):
                with self.subTest(over=over), self.assertRaises(ValueError):
                    self.call(**over)
            load.assert_not_called()
            stream.assert_not_called()
            self.assertEqual(convert.call_count, 1)  # Offset gate precedes convert.

    def test_success_uses_explicit_cpu4_batch8_and_unconsented_gate(self):
        from research.h3_personal import voxlock_streaming
        with patch('research.run_h3_zoom_preflight.file_sha',
                   side_effect=self.file_sha), \
                patch('research.run_h3_zoom_preflight.convert', return_value=self.PCM), \
                patch.object(voxlock_streaming, 'load_cpu',
                             return_value=('enroll', 'chunk')) as load, \
                patch.object(voxlock_streaming, 'VoxLockStream',
                             return_value='extractor') as stream:
            pipeline = self.call()
        load.assert_called_once_with('model', threads=4)
        self.assertEqual(stream.call_args.kwargs['batch_frames'], 8)
        reference = stream.call_args.args[2]
        self.assertEqual(len(reference), 48000)
        self.assertEqual(reference[:3], [0.25, 0.25, 0.25])
        self.assertEqual(pipeline.capture_argv, ['--stream', '--app', 'zoom'])
        self.assertEqual(pipeline.source_contract['capture_source_sha256'], self.CAPTURE_SHA)
        self.assertEqual(pipeline.source_contract['reference_sha256'],
                         hashlib.sha256(self.PCM).hexdigest())
        self.assertEqual(pipeline.source_contract['cpu_threads'], 4)
        self.assertFalse(pipeline.consented)
        self.assertFalse(pipeline.source_contract['speaker_identity_verified'])


if __name__ == '__main__':
    unittest.main()
