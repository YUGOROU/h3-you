from array import array
from io import BytesIO
import unittest
from research.h3_personal.pcm_boundary import SilenceBoundary

class BoundaryTests(unittest.TestCase):
    def test_greeting_gap_does_not_end_before_minimum(self):
        data=array('h',[1000]*16000+[0]*(16000*4)+[1000]*(16000*8)+[0]*(16000*4)).tobytes()
        reader=SilenceBoundary(BytesIO(data))
        for _ in range(13):self.assertTrue(reader.read(32000))
        self.assertFalse(reader.done)
        for _ in range(3):self.assertTrue(reader.read(32000))
        self.assertTrue(reader.done);self.assertEqual(reader.read(32000),b'')
        self.assertEqual(reader.reason,'application_energy_silence')

    def test_no_voice_is_bounded(self):
        reader=SilenceBoundary(BytesIO(bytes(32000*30)))
        for _ in range(16):self.assertTrue(reader.read(32000))
        self.assertEqual(reader.read(32000),b'')
        self.assertEqual(reader.reason,'no_voice_window')

    def test_malformed_pcm_rejected(self):
        with self.assertRaises(ValueError):SilenceBoundary(BytesIO(b'x')).read(1)

if __name__=='__main__':unittest.main()
