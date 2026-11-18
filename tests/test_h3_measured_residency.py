import unittest
from research.h3_personal.resident_session import ResidentSession


class MeasuredResidencyTests(unittest.TestCase):
    def session(self):
        calls=[];s=ResidentSession(lambda:calls.append('release'),allow_measured_auxiliary=True)
        return s,calls

    def load(self,s):
        with s.turn():return s.get({'model':'frozen'},object)

    def test_unknown_peak_releases_without_losing_binding(self):
        s,calls=self.session();self.load(s)
        s.auxiliary_available('video_vae',{'frames':124},128*1024**3)
        self.assertFalse(s.resident);self.assertEqual(calls,['release'])
        with self.assertRaises(ValueError):
            with s.turn():s.get({'model':'different'},object)

    def test_measured_shape_can_retain_exact_same_model(self):
        s,calls=self.session();s.record_auxiliary_peak('video_vae',{'frames':124},10*1024**3)
        original=self.load(s)
        s.auxiliary_available('video_vae',{'frames':124},18*1024**3)
        with s.turn():self.assertIs(s.get({'model':'frozen'},object),original)
        self.assertEqual(s.hits,1);self.assertEqual(calls,[])
        s.close()

    def test_shape_change_or_insufficient_headroom_releases(self):
        for shape,free in [({'frames':120},100*1024**3),({'frames':124},17*1024**3)]:
            s,calls=self.session();s.record_auxiliary_peak('audio_vae',{'frames':124},10*1024**3)
            self.load(s);s.auxiliary_available('audio_vae',shape,free)
            self.assertFalse(s.resident);self.assertEqual(len(calls),1)

    def test_cannot_record_combined_peak_as_isolated(self):
        s,_=self.session();self.load(s)
        with self.assertRaises(ValueError):s.record_auxiliary_peak('audio_vae',{},1)
        s.close()

    def test_default_still_refuses_overlap(self):
        s=ResidentSession(lambda:None);self.load(s)
        with self.assertRaises(ValueError):s.auxiliary_available('video_vae',{},100*1024**3)
        s.close()

    def test_qwen_miss_releases_even_when_aux_fit_known(self):
        s,calls=self.session();self.load(s);s.require_conditioner_available()
        self.assertFalse(s.resident);self.assertEqual(calls,['release'])
