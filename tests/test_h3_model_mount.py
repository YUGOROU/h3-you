import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch
from research import hf_identity_pipeline as pipe
from research.hf_curate_h3_assets import hash_file


class ModelMountTests(unittest.TestCase):
    def test_selected_read_only_mount_verifies_original_bytes_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'tensor').write_bytes(b'synthetic')
            sha=hashlib.sha256(b'synthetic').hexdigest(); pin='a'*40
            source=NS(rfilename='tensor',size=9,lfs=NS(sha256=sha))
            replica=NS(rfilename='assets/tensor',size=9,lfs=NS(sha256=sha))
            api=Mock();api.model_info.side_effect=[NS(sha=pin,siblings=[source]),NS(sha=pin,private=True,siblings=[replica])]
            r={'model_repo':'MiniMaxAI/MiniMax-H3','model_lock':{'path':'unused','sha256':'unused'},
               'transformer_files':[], 'model_mount':{'repo':'YUGOROU/synthetic','revision':pin,'path':tmp}}
            download=Mock()
            with patch.object(pipe,'approvals'),patch.object(pipe,'private_repositories'), \
                 patch.object(pipe,'pinned',return_value={'revision':pin,'files':{'tensor':sha}}), \
                 patch.object(pipe,'encoder_inventory'),patch.object(pipe.encoder,'private_root') as mutation:
                self.assertEqual(pipe.provision(r,root,api,download),{'tensor':sha})
                download.assert_not_called();mutation.assert_not_called()
                (root/'extra').write_bytes(b'extra')
                api.model_info.side_effect=[NS(sha=pin,siblings=[source]),NS(sha=pin,private=True,siblings=[replica])]
                with self.assertRaisesRegex(ValueError,'unexpected asset inventory'):
                    pipe.provision(r,root,api,download)
                download.assert_not_called()

    def test_cpu_worker_streaming_hash_includes_git_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'small';path.write_bytes(b'abc')
            self.assertEqual(hash_file(path),hashlib.sha256(b'abc').hexdigest())
            self.assertEqual(hash_file(path,git=True),hashlib.sha1(b'blob 3\0abc').hexdigest())


if __name__=='__main__':unittest.main()
