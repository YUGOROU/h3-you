import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from research.h3_personal.freeze_dataset import build,freeze

class FreezeTests(unittest.TestCase):
    def selected(self,path):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        return [{'id':f'clip_{i:04d}','source':str(path),'source_sha256':digest,'session_id':'s',
                 'start_seconds':i*124/24,'end_seconds':(i+1)*124/24,
                 'pool':'identity' if i<590 else 'motion','human_review':'approved_sampled_images'} for i in range(715)]

    def test_fixed_count_pools_source_hash_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'input.mov';source.write_bytes(b'camera fixture')
            selected=self.selected(source);manifest=p/'selected.json';manifest.write_text(json.dumps(selected));out=p/'frozen.json'
            result=freeze(manifest,out)
            self.assertEqual(result['sample_count'],715)
            self.assertAlmostEqual(result['clip_seconds_with_tail_overlap'],3694.166667,places=5)
            self.assertEqual(freeze(manifest,out),result)
            bad=copy.deepcopy(selected);bad[0]['pool']='motion'
            with self.assertRaises(ValueError):build(bad)
            with self.assertRaises(ValueError):build(selected[:-1])
            bad=copy.deepcopy(selected);bad[0]['id']=bad[1]['id']
            with self.assertRaises(ValueError):build(bad)
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):build(selected)

    def test_portable_semantic_digest_and_forbidden_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'a.mov';source.write_bytes(b'x');other=p/'b.mov';other.write_bytes(b'x')
            selected=self.selected(source);a=build(selected)
            moved=[dict(c,source=str(other)) for c in selected];self.assertEqual(a['dataset_sha256'],build(moved)['dataset_sha256'])
            output=p/'frozen.json';output.write_text('not the frozen dataset')
            manifest=p/'selected.json';manifest.write_text(json.dumps(selected))
            with self.assertRaises(ValueError):freeze(manifest,output)

if __name__=='__main__':unittest.main()
