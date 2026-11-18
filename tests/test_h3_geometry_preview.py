import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from research.h3_personal.reference_geometry import prepare_rgb_reference,LEGACY,CORRECTED
from research.h3_personal.geometry_preview import filters,render,digest
from research.preview_h3_geometry import main


class GeometryPreviewTests(unittest.TestCase):
    def test_reference_hw_regression_portrait_and_landscape(self):
        for width,height,h,w in [(1737,3088,3648,2048),(3088,1737,2048,3648),(2048,2048,2048,2048)]:
            image=SimpleNamespace(width=width,height=height);image.convert=lambda mode:image
            resolver=lambda x,y:(h,w)
            prepare=lambda image,hh,ww:(ww,hh)
            self.assertEqual(prepare_rgb_reference(image,resolver,prepare,policy=CORRECTED),(w,h))
            self.assertEqual(prepare_rgb_reference(image,resolver,prepare,policy=LEGACY),(h,w))
            with self.assertRaises(ValueError):prepare_rgb_reference(image,resolver,prepare,policy='guess')

    def test_crop_pad_same_anisotropy_no_silent_canvas_change(self):
        modes=filters(768,1344,.8)
        self.assertAlmostEqual(modes['pad']['actual_ratio'],.8,places=2)
        self.assertAlmostEqual(modes['crop']['actual_ratio'],.8,places=2)
        self.assertEqual(modes['pad']['source_crop_fraction'],0)
        self.assertAlmostEqual(modes['crop']['source_crop_fraction'],.2,places=2)
        self.assertIn('setsar=1',modes['pad']['filter'])
        for ratio in [float('nan'),float('inf'),0,.49,1.01]:
            with self.assertRaises(ValueError):filters(768,1344,ratio)
        with self.assertRaises(ValueError):filters(767,1344,.8)

    def test_plan_only_no_input_access(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--input','/not/read','--output','/not/created','--ratio','.8']),0)
        self.assertTrue(json.loads(output.getvalue())['plan_only'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_cpu_render_preserves_audio_frames_and_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x96:rate=4:duration=1',
                            '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264',
                            '-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True,timeout=20)
            before=digest(source);r=render(source,root/'previews',[.8])
            self.assertEqual(digest(source),before);self.assertEqual(len(r['candidates']),2)
            self.assertFalse(r['ratio_verified']);self.assertEqual(r['frames'],'4')
            self.assertTrue(all(c['decoded'] and c['audio_payload_unchanged'] for c in r['candidates']))
            with self.assertRaises(FileExistsError):render(source,root/'previews',[.8])
