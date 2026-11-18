import unittest
from research.hf_tts_japanese_diagnose import case_specs


class JapaneseDiagnosticCases(unittest.TestCase):
    def test_no_reference_is_explicitly_diagnostic_not_final_identity(self):
        specs=case_specs('こんにちは！')
        self.assertEqual(specs[0],('no-reference','こんにちは！',False))
        self.assertEqual(len(specs),2)

    def test_style_keeps_same_spoken_content_and_existing_reference(self):
        name,text,ref=case_specs('こんにちは！')[1]
        self.assertEqual(name,'reference-japanese-style')
        self.assertTrue(ref)
        self.assertEqual(text,'(in Japanese, clear natural pronunciation)こんにちは！')
