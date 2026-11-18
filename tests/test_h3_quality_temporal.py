import json
from pathlib import Path
import tempfile
import unittest
from research.h3_personal.quality_temporal import FEATURE_NAMES,summarize
from research.launch_hf_quality import stage

class TemporalQualityTests(unittest.TestCase):
    def metric(self,x=0):
        return {'area':.2,'sharp':100.,'eyes_sharp':30.,'mouth_sharp':40.,'clipped':.1,
                'inside':1.,'eye_open':.2,'mouth_open':.1,'yaw':10.,'pitch':20.,'center':[x,.5]}

    def test_no_fabricated_motion_over_detection_gap(self):
        result=dict(zip(FEATURE_NAMES,summarize([self.metric(0),None,self.metric(1),None])))
        self.assertEqual(result['adjacent_center_motion_max'],0.)
        self.assertEqual(result['valid_longest_run_fraction'],.25)
        self.assertEqual(result['detection_transition_fraction'],1.)

    def test_median_preserves_majority_quality_despite_one_poor_sample(self):
        frames=[self.metric() for _ in range(5)];frames[2]['eyes_sharp']=0.
        result=dict(zip(FEATURE_NAMES,summarize(frames)))
        self.assertGreater(result['eyes_sharp_median_log'],3.)
        self.assertEqual(len(result),21)
        self.assertEqual(summarize([None]*5),[0.]*21)

    def test_invalid_frames_fail(self):
        with self.assertRaises(ValueError):summarize([])
        bad=self.metric();bad['sharp']=float('nan')
        with self.assertRaises(ValueError):summarize([bad])

    def test_v2_stage_includes_all_pools_and_preserves_human_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'previews').mkdir()
            cs=[{'id':pool,'pool':pool,'source_sha256':'a'*64,'session_id':'s','start_seconds':0.,'end_seconds':5.} for pool in ('review','identity','motion')]
            (p/'candidate_manifest.json').write_text(json.dumps(cs))
            labels={'identity':{'source_sha256':'a'*64,'label':'identity'}}
            labelbytes=json.dumps({'labels':labels});(p/'image_labels.json').write_text(labelbytes)
            (p/'calibration_queue.json').write_text(json.dumps({'ids':['identity']}))
            for c in cs:(p/'previews'/f"{c['id']}.jpg").write_bytes(b'test')
            target,request=stage(p,'test',feature_version=2)
            self.assertEqual(len(request['samples']),3)
            self.assertEqual(request['feature_version'],2)
            self.assertTrue((target/'quality_temporal.py').exists())
            self.assertTrue((target/'hf_quality_v2_worker.py').exists())
            self.assertEqual((p/'image_labels.json').read_text(),labelbytes)
            self.assertEqual(sum(s['label'] is None for s in request['samples']),2)

if __name__=='__main__':unittest.main()
