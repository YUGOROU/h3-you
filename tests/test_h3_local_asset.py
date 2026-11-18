import errno
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research.h3_personal.local_asset import stage


class LocalAssetTests(unittest.TestCase):
    def test_byte_identical_copy_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';dst=Path(d)/'copy';src.write_bytes(b'weights')
            self.assertEqual(stage(src,dst,hashlib.sha256(b'weights').hexdigest()),dst)
            self.assertEqual(dst.read_bytes(),src.read_bytes())
            with self.assertRaises(FileExistsError):stage(src,dst,hashlib.sha256(b'weights').hexdigest())

    def test_hash_mismatch_is_not_retried_or_published(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';dst=Path(d)/'copy';src.write_bytes(b'wrong')
            with self.assertRaises(ValueError):stage(src,dst,'0'*64)
            self.assertFalse(dst.exists());self.assertFalse(dst.with_suffix('.partial').exists())

    def test_remote_eio_reopened_bounded_before_publish(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';dst=Path(d)/'copy';src.write_bytes(b'weights');original=Path.open;calls=[]
            def opening(path,*args,**kwargs):
                if path==src:
                    calls.append(path)
                    if len(calls)==1:raise OSError(errno.EIO,'remote mount read failed')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',opening),patch('research.h3_personal.local_asset.time.sleep'):
                stage(src,dst,hashlib.sha256(b'weights').hexdigest())
            self.assertEqual(len(calls),2);self.assertEqual(dst.read_bytes(),b'weights')

    def test_stale_partial_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)/'copy';partial=dst.with_suffix('.partial');partial.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):stage('unused',dst,'0'*64)
            self.assertEqual(partial.read_bytes(),b'existing')
