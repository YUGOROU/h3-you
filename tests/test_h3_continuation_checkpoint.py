from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from research.h3_personal.continuation_checkpoint import rank_checkpoint
from research.h3_personal.identity_checkpoint import WEIGHTS,CONFIG,RECEIPT,EXPORTED,VERIFIED


class ContinuationCheckpointTests(unittest.TestCase):
    def test_two_ranks_have_exclusive_marker_paths_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'canonical';source.mkdir()
            names=(WEIGHTS,CONFIG,RECEIPT,EXPORTED)
            for n in names:(source/n).write_bytes(n.encode())
            (source/VERIFIED).write_text('parent verification')
            def reload(rank):
                local=rank_checkpoint(source,root,rank)
                marker=local/('.'+VERIFIED+'.tmp')
                with marker.open('xb') as f:f.write(str(rank).encode())
                marker.replace(local/VERIFIED)
                return local
            with ThreadPoolExecutor(max_workers=2) as pool:paths=list(pool.map(reload,(0,1)))
            self.assertNotEqual(paths[0],paths[1])
            self.assertEqual((source/VERIFIED).read_text(),'parent verification')
            for rank,p in enumerate(paths):
                self.assertEqual((p/VERIFIED).read_text(),str(rank))
                for n in names:self.assertEqual((p/n).read_bytes(),(source/n).read_bytes())
            with self.assertRaises(FileExistsError):rank_checkpoint(source,root,0)
            with self.assertRaises(ValueError):rank_checkpoint(source,root,2)

    def test_symlink_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'canonical';source.mkdir();outside=root/'outside';outside.write_bytes(b'x')
            (source/WEIGHTS).symlink_to(outside)
            with self.assertRaises(ValueError):rank_checkpoint(source,root,0)
