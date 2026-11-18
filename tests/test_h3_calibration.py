import unittest
from research.h3_personal.calibration import choose

class CalibrationTests(unittest.TestCase):
    def fixture(self):
        return [{'id':f'{s}_{i}','session_id':s,'pool':'review',
                 'start_seconds':i*5,'score':i%4} for s,n in [('a',23),('b',372)] for i in range(n)]

    def test_balanced_temporal_queue_and_no_relabelling(self):
        candidates=self.fixture(); labels={'a_0':{'label':'identity'},'b_0':{'label':'reject'}}
        ids=choose(candidates,labels)
        self.assertEqual(len(ids),30);self.assertEqual(len(set(ids)),30)
        self.assertEqual(sum(i.startswith('a_') for i in ids),6)
        self.assertEqual(sum(i.startswith('b_') for i in ids),24)
        self.assertFalse(set(ids)&set(labels))
        self.assertEqual(ids,choose(candidates,labels))
        self.assertGreater(max(int(i.split('_')[1]) for i in ids if i.startswith('b_')),350)

    def test_small_empty_and_invalid_count(self):
        self.assertEqual(choose([],{}),[])
        self.assertEqual(len(choose(self.fixture()[:2],{})),2)
        with self.assertRaises(ValueError):choose([],{},0)
        self.assertEqual(choose([{**c,'pool':'identity'} for c in self.fixture()],{}),[])

    def feature_fixture(self):
        cs=[{**c,'source_sha256':'a'*64,'end_seconds':c['start_seconds']+5.166667} for c in self.fixture()]
        rows={c['id']:{**c,'features':[c['score'],(c['start_seconds']/5)%9],'error':None} for c in cs}
        return cs,rows

    def test_feature_diversity_stays_temporally_stratified(self):
        cs,rows=self.feature_fixture();labels={'a_0':{}}
        ids=choose(cs,labels,40,rows)
        self.assertEqual(len(ids),40);self.assertEqual(len(set(ids)),40)
        self.assertEqual(sum(i.startswith('a_') for i in ids),6)
        self.assertNotIn('a_0',ids)
        self.assertEqual(ids,choose(cs,labels,40,rows))
        long_times=[next(c['start_seconds'] for c in cs if c['id']==i) for i in ids if i.startswith('b_')]
        self.assertGreater(max(long_times),1750)

    def test_invalid_cached_features_refused_not_auto_rejected(self):
        cs,rows=self.feature_fixture()
        rows['a_0']['source_sha256']='b'*64
        with self.assertRaises(ValueError):choose(cs,{},40,rows)
        rows['a_0']['source_sha256']='a'*64;rows['a_0']['features']=[float('nan'),0]
        with self.assertRaises(ValueError):choose(cs,{},40,rows)
        rows.pop('a_0')
        with self.assertRaises(ValueError):choose(cs,{},40,rows)

    def test_full_labels_leave_empty_queue(self):
        cs=self.fixture()
        self.assertEqual(choose(cs,{c['id']:{} for c in cs}),[])

if __name__=='__main__':unittest.main()
