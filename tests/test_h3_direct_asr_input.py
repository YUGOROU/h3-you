import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research.h3_personal.direct_asr_input import DirectASRInput, metadata
from research.h3_personal.endpoint_asr import FINAL
from research.h3_personal.resident_endpoint import NoSpeech, TurnRejected


def receipt():
    return dict(text='こんにちは', final=True, errors=[], capture_stopped=True,
                cleanup_complete=True, input_committed=True,
                completion_policy='input-finalized', audio_sent_seconds=2,
                finalized_audio_seconds=2,
                events=[dict(text='こんにちは', final=True,
                             endpoint_source='native_completed', native_event_type=FINAL)])


class Lease:
    name = 'h3-asr-t4-7c926aae18'
    closed = False

    def __init__(self, root, results):
        self.root = Path(root)
        self.deadline = time.time() + 1000
        self.intent = dict(name=self.name, model=dict(revision='a' * 40))
        self.results = iter(results)
        self.calls = []

    def capture(self, directory, planned_start, *, capture_seconds):
        self.calls.append((directory, planned_start, capture_seconds))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class DirectInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def source(self, results):
        lease = Lease(self.tmp.name, results)
        return lease, DirectASRInput(lease)

    def call(self, source, remaining=1000):
        with patch('builtins.print'):
            return source(context=dict(session='private', epoch=1),
                          deadline=time.monotonic() + remaining)

    def test_actual_callback_and_worker_contract(self):
        from research.run_h3_relay_conversation import _asr_input
        lease, source = self.source([receipt(), receipt()])
        with patch('builtins.print') as notify:
            item = source(context=dict(session='secret', epoch=1),
                          deadline=time.monotonic() + 1000)
        self.assertEqual(_asr_input(item), (item['text'], metadata(lease)))
        notify.assert_called_once_with(
            'ASR capture starts now (VAD boundary, maximum 45 seconds).', flush=True)
        directory, start, seconds = lease.calls[0]
        self.assertEqual(directory.parent, lease.root.resolve())
        self.assertIsNone(start)  # Explicit immediate capture, not a stale timestamp.
        self.assertEqual(seconds, 45)
        self.assertIsNotNone(self.call(source))
        self.assertIsNone(self.call(source))
        self.assertEqual(len(lease.calls), 2)
        self.assertEqual(list(lease.root.iterdir()), [])

    def rtc_receipt(self):
        row=receipt()
        row.update(media_transport='webrtc-exact-v1',rtc_carrier=dict(
            schema='h3-asr-rtc-ipc-complete-v1',transport_complete=True,
            native_eos_verified=False,bytes=64000,sha256='a'*64,local_eof_already_closed=False))
        return row

    def test_rtc_requires_actual_carrier_before_accepting_turn(self):
        invalid=self.rtc_receipt();invalid['rtc_carrier']['transport_complete']=False
        lease=Lease(self.tmp.name,[receipt(),invalid,self.rtc_receipt()])
        lease.intent['model']['env']={'H3_ASR_MEDIA_TRANSPORT':'webrtc-exact-v1'}
        source=DirectASRInput(lease)
        self.assertIsNotNone(self.call(source))
        self.assertEqual(source.attempts,3);self.assertEqual(source.accepted,1)
        self.assertFalse(lease.closed)

    def test_rtc_transport_drift_refuses_before_capture(self):
        lease,source=self.source([])
        lease.intent['model']['env']={'H3_ASR_MEDIA_TRANSPORT':'webrtc-exact-v1'}
        with self.assertRaises(ValueError):self.call(source)
        self.assertEqual(lease.calls,[])

    def test_rtc_bad_carrier_fields_rejected(self):
        from research.h3_personal.direct_asr_input import _validate
        for field,value in [('schema','other'),('bytes',True),('bytes',0),('sha256','x'*64),
                            ('native_eos_verified',True),('local_eof_already_closed',None)]:
            row=self.rtc_receipt();row['rtc_carrier'][field]=value
            with self.subTest(field=field),self.assertRaises(TurnRejected):_validate(row,require_rtc=True)

    def test_bounded_recoverable_attempts(self):
        lease, source = self.source([NoSpeech('silent'), TurnRejected('reject'), receipt()])
        self.assertIsNotNone(self.call(source))
        self.assertIsNone(self.call(source))
        self.assertEqual(len(lease.calls), 3)
        self.assertEqual(len({c[0] for c in lease.calls}), 3)

    def test_no_speech_exhaustion(self):
        lease, source = self.source([NoSpeech('silent')] * 3)
        self.assertIsNone(self.call(source))
        self.assertFalse(lease.closed)

    def test_time_and_closed_gates(self):
        lease, source = self.source([])
        self.assertIsNone(self.call(source, 79))
        lease.deadline = time.time() + 79
        self.assertIsNone(self.call(source))
        lease.closed = True
        self.assertIsNone(self.call(source))
        self.assertEqual(lease.calls, [])

    def test_no_fabricated_capture_or_native_evidence(self):
        for field, value in [('capture_stopped', False), ('cleanup_complete', False),
                             ('input_committed', False), ('final', False),
                             ('finalized_audio_seconds', 1), ('events', []),
                             ('completion_policy', 'first-final')]:
            with self.subTest(field=field):
                bad = receipt()
                bad[field] = value
                lease, source = self.source([bad] * 3)
                self.assertIsNone(self.call(source))
                self.assertEqual(source.accepted, 0)

    def test_pin_drift_and_unexpected_failure(self):
        lease, source = self.source([RuntimeError('transport')])
        with self.assertRaises(RuntimeError):
            self.call(source)
        self.assertFalse(lease.closed)
        lease.intent['model']['revision'] = 'b' * 40
        with self.assertRaises(ValueError):
            self.call(source)


if __name__ == '__main__':
    unittest.main()
