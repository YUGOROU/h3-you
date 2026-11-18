import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from research.run_h3_relay_conversation import _persist_audio_evidence


class AudioReceiptTests(unittest.TestCase):
    def row(self):
        return dict(source_pcm_samples=2000,padding_pcm_samples=0,
            source_pcm16_samples=2000,nonzero_pcm16_samples=1000,
            peak_pcm16=1234,sum_squares_pcm16=20000,ack_count=1,
            rms_pcm16=3.162,encoder_input_only=True,remote_verified=False)

    def test_allowlisted_private_receipt_and_no_replay(self):
        with tempfile.TemporaryDirectory() as root:
            row=self.row();row['transcript']='must-not-save';row['raw_pcm']='private'
            sink=SimpleNamespace(audio_evidence=row)
            _persist_audio_evidence(root,sink)
            path=Path(root)/'encoder-audio-evidence.json';body=path.read_text()
            self.assertNotIn('private',body);self.assertNotIn('must-not-save',body)
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertFalse(json.loads(body)['remote_verified'])
            with self.assertRaises(FileExistsError):_persist_audio_evidence(root,sink)

    def test_invalid_metrics_or_remote_claim_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for key,value in [('ack_count',True),('source_pcm_samples',-1),
                    ('rms_pcm16',float('nan')),('remote_verified',True)]:
                row=self.row();row[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    _persist_audio_evidence(root,SimpleNamespace(audio_evidence=row))
            self.assertFalse(list(Path(root).iterdir()))
            _persist_audio_evidence(root,object())


if __name__=='__main__':unittest.main()
