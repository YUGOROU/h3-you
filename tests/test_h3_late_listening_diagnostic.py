import json
import os
from pathlib import Path
import tempfile
import unittest

from unittest.mock import patch
from research.h3_personal.late_listening_diagnostic import ENV, read_config, position_policy, capture


class DiagnosticTests(unittest.TestCase):
    def config(self, phase=0):
        return dict(schema='h3-late-listening-diagnostic-v1', phase=phase,
                    position_policy=('global-v1', 'window-local-v1')[phase],
                    conversation_id=('late-listening-global', 'late-listening-local')[phase],
                    expected_frames=634, max_windows=8)

    def test_disabled_needs_no_context_or_tensor(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(read_config())
            self.assertEqual(position_policy(None), 'global-v1')
            capture(None, None, None, None, None, None, None)

    def test_strict_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            with patch.dict(os.environ, {ENV: str(path)}):
                for phase in (0, 1):
                    c = self.config(phase)
                    path.write_text(json.dumps(c))
                    self.assertEqual(read_config(), c)
                for key, value in [('phase', True), ('expected_frames', 635),
                                   ('max_windows', 9), ('position_policy', 'global-v1'),
                                   ('conversation_id', 'other'), ('extra', 1)]:
                    c = self.config(1)
                    c[key] = value
                    path.write_text(json.dumps(c))
                    with self.assertRaises(ValueError):
                        read_config()
                path.write_text(' ' * 4097)
                with self.assertRaises(ValueError):
                    read_config()
                path.unlink()
                target = Path(directory) / 'target'
                target.write_text(json.dumps(self.config()))
                path.symlink_to(target)
                with self.assertRaises(OSError):
                    read_config()

    def test_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config'
            path.write_text('{"phase":0,"phase":1}')
            with patch.dict(os.environ, {ENV: str(path)}), self.assertRaises(ValueError):
                read_config()

    def test_private_receipts_use_bound_coverage_and_are_immutable(self):
        from research.h3_personal.late_listening_diagnostic import _save
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ,H3_REF2VA_COVERAGE_DIR=d):
                _save(self.config(),0,dict(exact=True))
                p=Path(d)/'late-listening-0-rank-0.json'
                self.assertEqual(p.stat().st_mode & 0o777,0o400)
                self.assertTrue(json.loads(p.read_text())['exact'])
                with self.assertRaises(FileExistsError):_save(self.config(),0,{})

    def test_matched_phases_have_unique_native_session_receipts(self):
        from research.h3_personal.streaming_admission import StreamingAdmission
        from research.h3_personal.streaming_windows import ResamplingContext
        from research.h3_personal.streaming_trial import manifest_for,write_wav
        from research.hf_h3_late_listening_diagnostic import config,SECTION,validate_section
        control=StreamingAdmission('c',context=ResamplingContext('same-rate-32k'))
        t=control.reserve(control.inspect())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'idle.wav';write_wav(p,b'\x00'*96000)
            manifests=[manifest_for(t,conversation_id=config(phase)['conversation_id'],
                profile_sha256='a'*64,wav=p,native_window_prefix=f'late_phase_{phase}_window')
                for phase in (0,1)]
            self.assertNotEqual(manifests[0]['session_id'],manifests[1]['session_id'])
            self.assertNotEqual(manifests[0]['streaming_audio']['id'],manifests[1]['streaming_audio']['id'])
        validate_section({'late_listening_diagnostic':dict(SECTION)})
        for key,value in (('phases',True),('decode_parity',1),('expected_frames',532)):
            with self.assertRaises(ValueError):
                validate_section({'late_listening_diagnostic':dict(SECTION,**{key:value})})


if __name__ == '__main__':
    unittest.main()
