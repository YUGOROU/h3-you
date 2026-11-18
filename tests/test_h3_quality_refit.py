import json
from pathlib import Path
import tempfile
import unittest
from research.launch_hf_quality_refit import stage

class RefitStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.c={'id':'clip','source_sha256':'a'*64,'session_id':'s','start_seconds':0.,'end_seconds':5.166667}
        self.features={'feature_names':['detection_fraction'],'rows':[{**self.c,'features':[1.],'label':None,'detection_fraction':1.}]}
        self.write('candidate_manifest.json',[self.c])
        self.write('image_labels.json',{'labels':{'clip':{'source_sha256':'a'*64,'label':'identity'},'initial':{'source_sha256':'a'*64,'label':'motion'}}})
        self.write('calibration_queue.json',{'ids':['clip']})
        self.write('quality-job-latest.json',{'run_id':'old','repo':'private/old'})
        self.write('quality_proposals.json',{'run_id':'old','cpu_only':True,'output_revision':'b'*40,'model_sha256':'c'*64,'feature_names':['detection_fraction']})
        self.write('quality-jobs/old/outputs/features.json',self.features)

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def test_stage_reuses_features_does_not_copy_images_or_mutate_labels(self):
        before=(self.root/'image_labels.json').read_bytes()
        target,request=stage(self.root,'new')
        self.assertEqual(request['samples']['clip']['label'],'identity')
        self.assertEqual(request['labels_without_cached_features'],1)
        self.assertEqual(request['feature_origin']['output_revision'],'b'*40)
        self.assertEqual((target/'features.json').read_bytes(),(self.root/'quality-jobs/old/outputs/features.json').read_bytes())
        self.assertEqual((self.root/'image_labels.json').read_bytes(),before)
        self.assertFalse(any(p.suffix in ('.jpg','.png','.mov') for p in target.iterdir()))

    def test_changed_range_refuses_reuse(self):
        self.write('candidate_manifest.json',[{**self.c,'start_seconds':1.}])
        with self.assertRaises(ValueError):stage(self.root,'new')

    def test_incomplete_teacher_and_unpinned_prior_refused(self):
        self.write('calibration_queue.json',{'ids':['missing']})
        with self.assertRaises(ValueError):stage(self.root,'new')
        self.write('quality_proposals.json',{'run_id':'wrong','cpu_only':True})
        with self.assertRaises(ValueError):stage(self.root,'new2')

if __name__=='__main__':unittest.main()
