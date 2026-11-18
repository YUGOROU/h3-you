import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
from research.hf_irodori_tts_diagnose import cases,checked_snapshot


class IrodoriDiagnosticTests(unittest.TestCase):
    def test_three_explicit_conditions(self):
        c=cases();self.assertEqual(len(c),3)
        self.assertEqual([x[2] for x in c],[False,True,True])
        self.assertEqual(c[0][1],c[1][1]);self.assertEqual(c[2][1],'おはようございます。声と口の動きを確認します。')

    def test_asset_bytes_and_revision_verified(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/('a'*40);p.mkdir();(p/'weights').write_bytes(b'model')
            s={'repo':'test/model','revision':'a'*40,'files':{'weights':{'bytes':5,'lfs_sha256':hashlib.sha256(b'model').hexdigest(),'git_blob_id':None}}}
            with patch.dict('sys.modules',huggingface_hub=N(snapshot_download=lambda *a,**k:str(p))):
                self.assertEqual(checked_snapshot(None,s,'synthetic'),p)
                (p/'weights').write_bytes(b'wrong')
                with self.assertRaises(ValueError):checked_snapshot(None,s,'synthetic')

    def test_native_sony_main_cannot_drift(self):
        with tempfile.TemporaryDirectory() as d:
            s={'repo':'sony/silentcipher','revision':'a'*40,'files':{}}
            with patch.dict('sys.modules',huggingface_hub=N(snapshot_download=lambda *a,**k:d)):
                with self.assertRaises(ValueError):checked_snapshot(None,s,'synthetic')
