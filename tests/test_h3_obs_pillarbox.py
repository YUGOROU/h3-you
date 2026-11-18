import json
import re
from pathlib import Path
import tempfile
import unittest
from research.prepare_h3_obs_live_relay import prepare as original
from research.prepare_h3_obs_pillarbox import prepare, ID, OLD


class PillarboxTests(unittest.TestCase):
    def test_geometry_routes_original_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'basic/profiles').mkdir(parents=True)
            (root/'basic/scenes').mkdir()
            original(root)
            before = (root/'basic/scenes'/f'{OLD}.json').read_bytes()
            prepare(root)
            written = (root/'basic/profiles'/ID/'basic.ini').read_text()
            for key, value in (("MonitoringDeviceId", "BlackHole2ch_UID"),
                               ("MonitoringDeviceName", "BlackHole 2ch")):
                entries = re.findall(r'^\s*' + key + r'\s*[=:]\s*(.*?)\s*$',
                                     written, flags=re.IGNORECASE | re.MULTILINE)
                self.assertEqual(entries, [value])
            self.assertEqual(before, (root/'basic/scenes'/f'{OLD}.json').read_bytes())
            data = json.loads((root/'basic/scenes'/f'{ID}.json').read_text())
            self.assertEqual(data['resolution'], {'x':1920,'y':1080})
            scene = next(s for s in data['sources'] if s['id']=='scene')
            item = scene['settings']['items'][0]
            self.assertEqual(item['bounds'], {'x':607.5,'y':1080.0})
            self.assertEqual(item['pos']['x']*2+item['bounds']['x'],1920)
            self.assertEqual(item['scale']['x'],item['scale']['y'])
            self.assertEqual(item['bounds']['x']/item['bounds']['y'],252/448)
            media=next(s for s in data['sources'] if s['id']=='ffmpeg_source')
            self.assertEqual(media['monitoring_type'],1)
            self.assertEqual(media['mixers'],0)
            with self.assertRaises(ValueError):prepare(root)

    def test_rejects_sensor_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'basic/profiles').mkdir(parents=True)
            (root/'basic/scenes').mkdir();original(root)
            p=root/'basic/scenes'/f'{OLD}.json'
            data=json.loads(p.read_text());data['sources'].append({'id':'av_capture_input'})
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):prepare(root)
            self.assertFalse((root/'basic/profiles'/ID).exists())
