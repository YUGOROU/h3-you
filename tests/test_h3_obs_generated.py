import tempfile
from pathlib import Path
import unittest
from research.prepare_h3_obs_generated import collection,prepare,AUDIO_KEYS


class OBSGeneratedTests(unittest.TestCase):
    def test_no_sensor_or_global_audio_sources(self):
        data=collection('/private/accepted-review.mp4')
        self.assertEqual({s['id'] for s in data['sources']},{'ffmpeg_source','scene'})
        self.assertTrue(all(data[k] is None for k in AUDIO_KEYS))
        media=data['sources'][0]
        self.assertTrue(media['muted']);self.assertEqual(media['mixers'],0)
        self.assertFalse(media['settings']['looping'])
        self.assertEqual(data['modules'],{})

    def test_capture_modules_disabled_without_bundle_edits(self):
        import json
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from research.prepare_h3_obs_generated import disable_capture_modules,BLOCKED_MODULES
        with tempfile.TemporaryDirectory() as folder:
            with patch('subprocess.run',return_value=NS(returncode=1)):disable_capture_modules(folder)
            rows=json.loads((Path(folder)/'plugin_manager/modules.json').read_text())
            self.assertEqual({r['module_name'] for r in rows},set(BLOCKED_MODULES))
            self.assertTrue(all(r['enabled'] is False for r in rows))
            self.assertTrue((Path(folder)/'plugin_manager/modules.before-sensors-off.json').is_file())

    def test_never_overwrites_existing_obs_config(self):
        with tempfile.TemporaryDirectory() as folder:
            media=Path(folder)/'video';media.write_bytes(b'not a generated clip')
            with self.assertRaisesRegex(ValueError,'overwrite'):prepare(media,folder)

if __name__=='__main__':unittest.main()
