import unittest
from research.h3_personal.image_selection import build

class ImageSelectionTests(unittest.TestCase):
    def fixture(self):
        cs=[{'id':str(i),'source_sha256':'a'*64,'pool':'review','start_seconds':i*5.,'end_seconds':i*5.+5.166667} for i in range(5)]
        labels={str(i):{'label':l,'source_sha256':'a'*64} for i,l in enumerate(['identity','motion','reject','hold'])}
        return cs,labels

    def test_only_explicit_human_accepts_with_limited_scope(self):
        cs,labels=self.fixture();selected,s=build(cs,labels)
        self.assertEqual([x['id'] for x in selected],['0','1'])
        self.assertEqual(s['human_rejected'],1);self.assertEqual(s['unlabelled_candidates'],1)
        for x in selected:
            self.assertFalse(x['training_approved']);self.assertFalse(x['full_clip_review_complete'])
            self.assertEqual(x['audio_stability'],'user_reported_verified')
            self.assertFalse(x['audio_speaker_and_sync_independently_verified'])
        self.assertEqual(cs[0]['pool'],'review')

    def test_hash_and_duplicates_fail_closed(self):
        cs,labels=self.fixture();labels['0']['source_sha256']='b'*64
        with self.assertRaises(ValueError):build(cs,labels)
        with self.assertRaises(ValueError):build(cs+[cs[0]],{})

if __name__=='__main__':unittest.main()
