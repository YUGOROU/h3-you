import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from research.prepare_h3_obs_audio_route import main, prepare, repair_profile


PROFILE = ('[General]\r\nSecret=private-token\r\n[Video]\r\nBaseCX=1920\r\n'
           'BaseCY=1080\r\nOutputCX=1280\r\nOutputCY=720\r\n[Audio]\r\n'
           ' MonitoringDeviceId = default\r\n monitoringdeviceid=BlackHole2ch_UID\r\n'
           ' MonitoringDeviceName : Default\r\n monitoringdevicename=BlackHole 2ch\r\n'
           'SampleRate=48000\r\n[Other]\r\nGeometry=secret-geometry\r\n')


class AudioRouteTests(unittest.TestCase):
    def test_preserves_noncritical_bytes_and_geometry(self):
        result = repair_profile(PROFILE)
        self.assertTrue(result.startswith(PROFILE.split('[Audio]')[0]))
        self.assertIn('SampleRate=48000\r\n', result)
        self.assertTrue(result.endswith('[Other]\r\nGeometry=secret-geometry\r\n'))
        self.assertEqual(result.count('MonitoringDeviceId=BlackHole2ch_UID\r\n'), 1)
        self.assertEqual(result.count('MonitoringDeviceName=BlackHole 2ch\r\n'), 1)
        self.assertEqual(sum('monitoringdevice' in x.lower() for x in result.splitlines()), 2)
        self.assertEqual(repair_profile(result), result)

    def test_localized_default_and_unrelated_duplicates_preserved(self):
        text='[Video]\nOutputCX = 1920\nOutputCX=1280\n[Audio]\nMonitoringDeviceId=default\nMonitoringDeviceName=既定\n'
        result=repair_profile(text)
        self.assertTrue(result.startswith(text.split('[Audio]')[0]))
        self.assertIn('MonitoringDeviceId=BlackHole2ch_UID\n',result)
        self.assertIn('MonitoringDeviceName=BlackHole 2ch\n',result)
        self.assertNotIn('既定',result)

    def test_reject_ambiguous_or_other_routes(self):
        for text in ('[Audio]\nMonitoringDeviceId=Speakers\n',
                     '[DEFAULT]\nMonitoringDeviceId=default\n[Audio]\n',
                     '[Other]\nMonitoringDeviceName=Default\n[Audio]\n',
                     '[Audio] garbage\nMonitoringDeviceId=default\n',
                     '[Audio]\nMonitoringDeviceId=default\n  continued\n',
                     '[Audio]\nOther=value\n  MonitoringDeviceId=default\n',
                     '[Audio]\nMonitoringDeviceName=Headphones\n',
                     '[Audio]\n[Audio]\n',
                     '[audio]\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                repair_profile(text)

    def test_prepare_private_review_source_untouched_and_console_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inputs = root/'inputs'
            inputs.mkdir()
            profile, scene = inputs/'basic.ini', inputs/'scene.json'
            original = PROFILE.encode()
            profile.write_bytes(original)
            scene.write_text(json.dumps({'sources': [], 'secret': 'scene-secret'}))
            target = root/'review'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(['--profile', str(profile), '--scene', str(scene),
                      '--source', 'private-source', '--directory', str(target)])
            plan = json.loads((target/'plan.json').read_text())
            self.assertEqual(profile.read_bytes(), original)
            self.assertEqual(plan['source_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(plan['candidate_sha256'], hashlib.sha256((target/'basic.ini.candidate').read_bytes()).hexdigest())
            self.assertFalse(plan['applied'])
            self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            self.assertEqual((target/'plan.json').stat().st_mode & 0o777, 0o600)
            for secret in ('private-token', 'secret-geometry', 'scene-secret', 'private-source', str(root)):
                self.assertNotIn(secret, output.getvalue() + json.dumps(plan))
            with self.assertRaises(FileExistsError):
                prepare(profile, scene, 'source', target)
            self.assertEqual(profile.read_bytes(), original)

    def test_reject_symlinks_and_internal_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inputs = root/'inputs'
            inputs.mkdir()
            profile, scene = inputs/'basic.ini', inputs/'scene.json'
            profile.write_text(PROFILE)
            scene.write_text('{}')
            link = root/'link'
            link.symlink_to(profile)
            with self.assertRaises(ValueError):
                prepare(link, scene, 'source', root/'review')
            with self.assertRaises(ValueError):
                prepare(profile, scene, 'source', inputs/'review')
            outlink = root/'outlink'
            outlink.symlink_to(inputs, target_is_directory=True)
            with self.assertRaises(ValueError):
                prepare(profile, scene, 'source', outlink/'review')


if __name__ == '__main__':
    unittest.main()
