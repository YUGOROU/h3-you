import unittest
from research.h3_personal.identity_select import assess
from research.h3_personal.identity_review import apply


def frame(area=.12, sharp=100, clipped=.05):
    return {'face_count': 1, 'face_area_ratio':area,
            'face_laplacian_variance':sharp,'face_clipped_fraction':clipped}


class IdentityTriageTests(unittest.TestCase):
    def test_identity_requires_temporal_consistency(self):
        self.assertEqual(assess([frame()]*5)['pool'],'identity')
        self.assertEqual(assess([frame(), {'face_count':0}])['pool'],'review')

    def test_small_face_motion_and_soft_review(self):
        self.assertEqual(assess([frame(area=.04)]*5)['pool'],'motion')
        self.assertEqual(assess([frame(sharp=5)]*5)['pool'],'review')
        self.assertEqual(assess([frame(clipped=.4)]*5)['pool'],'review')

    def test_missing_decode_or_multiple_faces_not_auto_excluded(self):
        for frames in ([], [{'decode_error':True}], [{'face_count':2}]):
            self.assertEqual(assess(frames)['pool'],'review')

    def test_worst_frame_not_average(self):
        result=assess([frame()]*4+[frame(sharp=2)])
        self.assertIn('soft_face_proxy',result['reasons'])

class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.candidates=[{'id':'s_0000','source_sha256':'a'*64}]
        self.decision={'id':'s_0000','source_sha256':'a'*64,'action':'accept',
            'pool':'identity','face_reviewed':True,'audio_reviewed':True,'reason':'reviewed'}

    def test_approval_requires_face_audio_and_source_match(self):
        self.assertTrue(apply(self.candidates,[self.decision])[0]['training_approved'])
        for key,value in [('audio_reviewed',False),('face_reviewed',False),('source_sha256','b'*64)]:
            with self.assertRaises(ValueError):
                apply(self.candidates,[{**self.decision,key:value}])

    def test_unreviewed_not_selected_duplicates_rejected(self):
        self.assertEqual(apply(self.candidates,[]),[])
        with self.assertRaises(ValueError):
            apply(self.candidates,[self.decision,self.decision])
        self.assertEqual(apply(self.candidates,[{**self.decision,'action':'reject'}]),[])


if __name__=='__main__':
    unittest.main()
