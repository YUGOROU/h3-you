"""CPU-only receipt validation; no devices, transport, or audio source."""
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest

from research.run_h3_relay_conversation import (
    _persist_audio_evidence, _persist_output_timing,
)


def timing():
    return dict(video_epoch=10., audio_epoch=10.2, audio_delay_seconds=.2,
                first_av_submission_monotonic=None, desired_first_video_write=None,
                actual_first_video_write=None, first_video_write_skew=None,
                max_video_lateness=0., queue_highwater=2,
                max_decoded_outstanding=2, video_frames_written=0, ack_count=1,
                schedule_alignment_only=True, remote_verified=False,
                ack_scope='local_raw_queue_and_device_audio_queue',
                accepted_monotonic=(10.1,))


def audio():
    return dict.fromkeys(('source_pcm_samples', 'padding_pcm_samples',
                         'source_pcm16_samples', 'nonzero_pcm16_samples',
                         'peak_pcm16', 'sum_squares_pcm16', 'ack_count'), 0) | dict(
        rms_pcm16=0., encoder_input_only=True, remote_verified=False)


class TimingReceiptTests(unittest.TestCase):
    def test_metrics_sanitized_permissions_and_replay(self):
        row = timing() | dict(source_audio='PRIVATE', path='/arbitrary', text='PRIVATE')
        with tempfile.TemporaryDirectory() as root:
            sink = SimpleNamespace(timing=MappingProxyType(row))
            _persist_output_timing(root, sink)
            path = Path(root) / 'split-av-timing.json'
            saved = json.loads(path.read_text())
            self.assertEqual(saved, timing() | dict(accepted_monotonic=[10.1]))
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                _persist_output_timing(root, sink)
            self.assertEqual(path.read_bytes(), before)

    def test_legacy_skip(self):
        with tempfile.TemporaryDirectory() as root:
            for sink in (SimpleNamespace(), SimpleNamespace(timing={'ack_count': 1})):
                _persist_output_timing(root, sink)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_invalid_metrics_leave_no_receipt(self):
        bad = [('video_epoch', float('nan')), ('audio_epoch', float('inf')),
               ('queue_highwater', True), ('ack_count', -1),
               ('remote_verified', True), ('remote_verified', 0),
               ('schedule_alignment_only', 'true'),
               ('ack_scope', 'remote'), ('accepted_monotonic', [0.] * 145),
               ('accepted_monotonic', [float('nan')]),
               ('accepted_monotonic', [True])]
        for key, value in bad:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    _persist_output_timing(root, SimpleNamespace(timing=timing() | {key: value}))
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_audio_optional_flags(self):
        flags = ('device_queue_only', 'obs_audio_disabled',
                 'device_playback_completed', 'schedule_alignment_only')
        with tempfile.TemporaryDirectory() as root:
            row = audio() | dict.fromkeys(flags, True) | dict(source_audio='PRIVATE')
            _persist_audio_evidence(root, SimpleNamespace(audio_evidence=row))
            saved = json.loads((Path(root) / 'encoder-audio-evidence.json').read_text())
            for key in flags:
                self.assertIs(saved[key], True)
            self.assertNotIn('source_audio', saved)
        for key in flags:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    _persist_audio_evidence(root, SimpleNamespace(audio_evidence=audio() | {key: 1}))


if __name__ == '__main__':
    unittest.main()
