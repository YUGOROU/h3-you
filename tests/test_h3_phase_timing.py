import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research.h3_personal.phase_timing import PhaseTiming
from research.h3_personal.direct_asr_input import DirectASRInput
from research.run_h3_relay_conversation import run_conversation
from test_h3_relay_conversation import Transport, Director, Sink, Pump, LiveAVBinding


class TimingTests(unittest.TestCase):
    def test_actual_controller_success_and_error(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as root:
                transport, director = Transport(), Director(fail=fail)
                binding = LiveAVBinding('fresh', 'a'*64, 1, max_total_frames=248)
                kwargs = dict(ready=dict(status='ready', session='fresh',
                              profile_sha256='a'*64, epoch=1, expires_at=time.time()+60),
                              max_seconds=2, sink_factory=Sink, pump_factory=Pump)
                if fail:
                    with self.assertRaises(RuntimeError):
                        run_conversation(transport, binding, root, ['SECRET'], director, **kwargs)
                else:
                    run_conversation(transport, binding, root, ['SECRET'], director, **kwargs)
                raw = (Path(root)/'stage-timing.json').read_text()
                row = json.loads(raw)
                self.assertEqual(row['status'], 'failed' if fail else 'completed')
                self.assertEqual([e['stage'] for e in row['events']],
                                 ['input', 'pi'] if fail else ['input', 'pi', 'publication'])
                for secret in ('SECRET', 'private details', '新しい返事', 'fresh'):
                    self.assertNotIn(secret, raw)
                self.assertTrue(director.closed)
                self.assertTrue(transport._closed)
                with self.assertRaises(FileExistsError):
                    PhaseTiming().persist(root)

    def test_direct_capture_callback_is_passed_and_called(self):
        class Endpoint:
            name = 'h3-asr-t4-7c926aae18'
            intent = dict(name=name, model=dict(revision='a'*40))
            closed = False
            deadline = time.time()+1000
            def capture(self, directory, planned_start, *, capture_seconds, on_asr_event):
                for event in ('partial', 'native_completed', 'input_finalized'):
                    on_asr_event(dict(event=event, text='SECRET', token='SECRET',
                                      timestamp=-99999, url='SECRET'))
                return dict(capture_stopped=True, cleanup_complete=True)
        with tempfile.TemporaryDirectory() as root:
            endpoint = Endpoint()
            endpoint.root = root
            source = DirectASRInput(endpoint)
            timing = source.phase_timing = PhaseTiming()
            with patch('research.h3_personal.direct_asr_input._validate', return_value='SECRET'):
                source(context={}, deadline=time.monotonic()+1000)
            self.assertEqual([e['stage'] for e in timing.events],
                             ['asr_partial', 'asr_final', 'asr_input_finalized', 'asr_capture'])
            self.assertNotIn('SECRET', json.dumps(timing.events))

    def test_bound_and_allowlist(self):
        timing = PhaseTiming()
        for _ in range(300):
            timing.on_asr_event(dict(event='partial', text='SECRET'))
        self.assertEqual(len(timing.events), 1)  # preserve space for Pi/EOS stages
        self.assertEqual(timing._asr_counts['partial'], 300)
        for _ in range(300):
            timing.record('pi')
        self.assertEqual(len(timing.events), 256)
        self.assertEqual(timing.dropped, 45)
        with self.assertRaises(ValueError):
            timing.record('SECRET')

    def test_repeated_capture_keeps_first_event_per_attempt(self):
        timing = PhaseTiming()
        for _ in range(2):
            with timing.stage('asr_capture'):
                for _ in range(300):
                    timing.on_asr_event(dict(event='partial', text='SECRET'))
                timing.on_asr_event(dict(event='input_finalized'))
        self.assertEqual([e['stage'] for e in timing.events],
                         ['asr_partial', 'asr_input_finalized', 'asr_capture'] * 2)
        self.assertEqual(timing._asr_counts['partial'], 600)

    def test_persistence_failure_does_not_mask_original_pi_failure(self):
        with tempfile.TemporaryDirectory() as root:
            transport, director = Transport(), Director(fail=True)
            binding = LiveAVBinding('fresh', 'a'*64, 1, max_total_frames=248)
            with patch.object(PhaseTiming, 'persist', side_effect=OSError('SECRET')):
                with self.assertRaises(RuntimeError) as error:
                    run_conversation(transport, binding, root, ['SECRET'], director,
                        ready=dict(status='ready', session='fresh', profile_sha256='a'*64,
                                   epoch=1, expires_at=time.time()+60),
                        max_seconds=2, sink_factory=Sink, pump_factory=Pump)
            self.assertIn('OSError', str(error.exception.__notes__))
            self.assertNotIn('SECRET', str(error.exception.__notes__))
            self.assertTrue(director.closed)
            self.assertTrue(transport._closed)

    def test_gpu_stdout_is_allowlisted(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            timing = PhaseTiming(component='gpu_worker', emit=True)
            with timing.stage('tts_sentence'):
                pass
        event = json.loads(out.getvalue())
        self.assertEqual(event['component'], 'gpu_worker')
        self.assertEqual(event['stage'], 'tts_sentence')
        self.assertEqual(set(event), {'type', 'component', 'clock', 'stage',
                                    'elapsed_seconds', 'duration_seconds', 'status','origin_monotonic'})
        self.assertEqual(event['origin_monotonic'],timing.started)
        self.assertIsInstance(event['origin_monotonic'],float)
