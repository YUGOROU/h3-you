"""Bounded ASR RTC acceptance harness: bounds, receipt validation, no implicit ops."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from research import run_h3_asr_rtc_acceptance as harness


def wav(path, seconds=1.0, rate=16000, channels=1, width=2):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(b'\x01\x00'*int(rate*seconds)*channels)


def transcript(**over):
    row = dict(text='おはようございます', final=True, errors=[], input_committed=True,
               completion_policy='input-finalized', audio_sent_seconds=8.0, finalized_audio_seconds=8.0,
               media_transport='webrtc-exact-v1',
               events=[dict(final=True, native_event_type='final', endpoint_source='native_completed',
                            text='おはようございます')],
               rtc_carrier=dict(transport_complete=True, native_eos_verified=False, bytes=256000, sha256='a'*64))
    row.update(over)
    return row


class PcmTests(unittest.TestCase):
    def test_bounded_mono_16k_pcm_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'s.wav'
            wav(path, seconds=30)
            stream = harness.pcm_source(path, 8)()
            self.assertEqual(len(stream.read()), 8*16000*2)
            wav(Path(directory)/'bad.wav', rate=44100)
            with self.assertRaises(ValueError):
                harness.pcm_source(Path(directory)/'bad.wav', 8)
            wav(Path(directory)/'st.wav', channels=2)
            with self.assertRaises(ValueError):
                harness.pcm_source(Path(directory)/'st.wav', 8)


class ReceiptTests(unittest.TestCase):
    def test_accepts_native_finalized_rtc_receipt(self):
        row = harness.validate_transcript(transcript())
        self.assertEqual(row['text'], 'おはようございます')
        self.assertEqual(row['events'], 1)
        self.assertTrue(row['carrier']['transport_complete'])

    def test_rejects_partial_legacy_or_incomplete_carrier(self):
        for over in (dict(final=False), dict(errors=['no_native_final']), dict(input_committed=False),
                     dict(completion_policy='first-final'), dict(audio_sent_seconds=0),
                     dict(finalized_audio_seconds=1.0), dict(text='   '),
                     dict(media_transport='http-paired-v1'), dict(rtc_carrier=dict(complete=False)),
                     dict(rtc_carrier=dict(complete=True, sha256='a'*64)),
                     dict(rtc_carrier=dict(transport_complete=True,native_eos_verified=False,bytes=1,sha256='x'*64)),
                     dict(rtc_carrier=dict(transport_complete=True,native_eos_verified=True,bytes=1,sha256='a'*64)),
                     dict(events=[dict(final=False)]),
                     dict(events=[dict(final=True, native_event_type='final', endpoint_source='native_completed', text='ちがう')])):
            with self.subTest(over=over):
                with self.assertRaises(ValueError):
                    harness.validate_transcript(transcript(**over))


class GateTests(unittest.TestCase):
    def test_refuses_before_any_resource_without_valid_bounds(self):
        with tempfile.TemporaryDirectory() as parent:
            sample = Path(parent)/'s.wav'
            wav(sample)
            root = Path(parent)/'fresh'
            with patch('research.h3_personal.resident_endpoint.PRIVATE', Path(parent)):
                # existing directory
                root.mkdir()
                with self.assertRaises(ValueError):
                    harness.run(root, parent, sample, 8)
                # bytes must match the reviewed public sample pin
                root2 = Path(parent)/'fresh2'
                with self.assertRaises(ValueError):
                    harness.run(root2, parent, sample, 8)
                # out-of-bounds window
                with self.assertRaises(ValueError):
                    harness.run(root2, parent, sample, 60)
                for bad in (0, 7, 21, 8.0, True):
                    with self.assertRaises(ValueError):
                        harness.run(root2, parent, sample, bad)
                self.assertFalse(root2.exists())

    def test_offline_default_starts_nothing(self):
        with patch('builtins.print') as printed:
            harness.main([])
        payload = json.loads(printed.call_args.args[0])
        self.assertFalse(payload['gpu_started'] or payload['endpoint_updated'] or payload['device_access'])
        self.assertEqual(payload['min_max_replica'], 1)

    def test_run_requires_explicit_arguments(self):
        for argv in (['--run'], ['--run', '--directory', 'x'], ['--run', '--directory', 'x', '--parent', 'y']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                harness.main(argv)


if __name__ == '__main__':
    unittest.main()


class OrchestrationTests(unittest.TestCase):
    """One serial turn, exact RTC transport, lease paused on close."""

    def test_single_serial_turn_writes_receipt_and_closes_lease(self):
        import time
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as parent:
            sample = Path(parent)/'s.wav'
            wav(sample)
            root = Path(parent)/'fresh'
            lease = Mock()
            lease.deadline = time.time()+600
            lease.lease_id = 'a'*32
            lease._url = 'https://x.endpoints.huggingface.cloud'
            lease._token = 'token'
            lease._key = 'k'*40
            lease._rtc_epoch = 0
            turn = Mock()
            turn.transport = object()
            turn._closed = False
            turn.receipt=transcript()['rtc_carrier']
            factory = Mock()
            factory.open_turn.return_value = turn
            calls = {}

            def transcribe(endpoint, **kwargs):
                calls.update(endpoint=endpoint, kwargs=kwargs)
                row=transcript()
                del row['media_transport'];del row['rtc_carrier']
                return row
            import hashlib
            with patch.object(harness, 'PUBLIC_SAMPLE_SHA256', hashlib.sha256(sample.read_bytes()).hexdigest()), \
                 patch.object(harness, 'PRIVATE', Path(parent)), \
                 patch.object(harness.ResidentEndpoint, 'open', return_value=lease) as opened, \
                 patch.object(harness, 'ASRRTCFactory', return_value=factory), \
                 patch('research.h3_personal.endpoint_asr.transcribe', side_effect=transcribe):
                receipt = harness.run(root, parent, sample, 8)
            opened.assert_called_once()
            self.assertEqual(opened.call_args.kwargs['accounting'], 'operator-managed')
            self.assertEqual(factory.open_turn.call_count, 1)
            self.assertEqual(factory.open_turn.call_args.kwargs['epoch'], 1)
            self.assertIs(calls['kwargs']['_transport'], turn.transport)
            self.assertEqual(calls['kwargs']['completion_policy'], 'input-finalized')
            self.assertEqual(calls['kwargs']['max_audio_seconds'], 8)
            turn.close.assert_called_once_with(require_complete=True)
            lease.close.assert_called_once()
            saved = json.loads((root/'result.json').read_text())
            self.assertTrue(saved['complete'])
            self.assertEqual(saved['transcript']['text'], 'おはようございます')
            for flag in ('mic', 'camera', 'screen', 'browser', 'h3', 'tts'):
                self.assertFalse(saved[flag])
