import json
from pathlib import Path
import tempfile
import unittest
from research.prepare_h3_obs_standby import collection,prepare,launch_arguments,AUDIO_KEYS,PROFILE_ID


class StandbyTests(unittest.TestCase):
    def test_no_media_capture_audio_or_virtual_start(self):
        c=collection();self.assertEqual([s['id'] for s in c['sources']],['scene'])
        self.assertFalse(c['sources'][0]['settings']['items'])
        self.assertTrue(all(c[k] is None for k in AUDIO_KEYS))
        self.assertFalse(any(x.startswith('--start') for x in launch_arguments()))
        self.assertIn('--safe-mode',launch_arguments())

    def test_new_files_only_preserve_existing_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'basic/scenes').mkdir(parents=True);(root/'basic/profiles').mkdir()
            old=root/'user.ini';old.write_bytes(b'unchanged-private-settings')
            result=prepare(root);self.assertFalse(result['launched']);self.assertFalse(result['media_playback'])
            self.assertEqual(old.read_bytes(),b'unchanged-private-settings')
            scene=root/'basic/scenes'/f'{PROFILE_ID}.json'
            self.assertEqual(scene.stat().st_mode & 0o777,0o600)
            self.assertEqual(json.loads(scene.read_text())['sources'][0]['id'],'scene')
            with self.assertRaises(ValueError):prepare(root)
