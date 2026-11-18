import copy
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import importlib.util
from research.h3_personal.continuation_contract import validate,SCOPE,AUDIO_SHA
from research.h3_personal.compute_price_approval import request_limit,RATE
from research.h3_personal.continuation_export import mux, recorded_wav_info


def contract():
    return dict(schema=SCOPE,width=768,height=1344,total_frames=685,window_frames=277,overlap_frames=22,
        video_shift=12,reference_policy='correct-hw-v2',schedule_points=9,seed=314159,
        turbo=dict(repo='lightx2v/Minimax-h3-Turbo',revision='3ec17a324ced54151364f24f8b5fb6bf7e26414f',
            path='/mounted-h3-turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',
            file='minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',bytes=1383677808,
            sha256='9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2'),
        audio=dict(repo='private/recorded',revision='fadd78ca615a06949fcad8ebab489b166d6fa6f1',
            path='/mounted-h3-native-audio/reference.wav',file='reference.wav',bytes=4093038,sha256=AUDIO_SHA))


class NativeWorkerTests(unittest.TestCase):
    def test_legacy_cache_producer_sources_remain_byte_frozen(self):
        expected={'hf_identity_encode.py':'6f8fd77bb924cf81e4c19e6ff64ca7f629c28282835403a22cfc9b6ec4743d7b',
            'h3_personal/component_encode.py':'2114f05051974013db7b538ff38dfaa654398109dbc94bbbbb4ed311511afd6c',
            'h3_personal/component_cache.py':'14513e8e11ad1788c999c8113c3aeaee14dac578183d3b513642955d87c8c5fa'}
        for name,sha in expected.items():
            self.assertEqual(hashlib.sha256((Path('research')/name).read_bytes()).hexdigest(),sha,name)

    @unittest.skipUnless(importlib.util.find_spec('PIL'),'Pillow required')
    def test_corrected_backend_is_separate_and_legacy_geometry_is_unchanged(self):
        from PIL import Image
        from research.h3_personal.component_encode import PinnedBackend
        from research.h3_personal.continuation_backend import CorrectedBackend
        fake=SimpleNamespace(resolve_reference_image_size=lambda w,h:(96,64),
                             prepare_reference_image=lambda im,h,w:im.resize((w,h)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.png';Image.new('RGB',(32,48)).save(path)
            unit={'sources':{'image':{'path':str(path)}}}
            with patch.dict('sys.modules',{'diffusers.modular_pipelines.minimax_h3.packing_ref2va':fake}):
                self.assertEqual(object.__new__(PinnedBackend).image(unit).size,(96,64))
                self.assertEqual(object.__new__(CorrectedBackend).image(unit).size,(64,96))
                with self.assertRaises(ValueError):object.__new__(CorrectedBackend).image(unit,geometry_policy='legacy-wh-v1')

    def test_finite_explicit_contract_and_no_matrix_price_replay(self):
        c=contract();self.assertEqual(validate(c),c)
        for key,value in [('video_shift',6),('total_frames',7213),('schedule_points',5),('reference_policy','legacy-wh-v1')]:
            with self.assertRaises(ValueError):validate(dict(c,**{key:value}))
        from test_h3_compute_price_approval import proof
        p=proof();r=dict(hf_price_precision_approval=p,rate_usd_hour=RATE,preview_only=True,
            overall_seconds=3600,territory_approval={'provider':'huggingface_jobs'},inference={'native_continuation':c})
        with self.assertRaises(ValueError):request_limit(r)
        p['scope']=SCOPE;self.assertEqual(request_limit(r),RATE)
        with self.assertRaises(ValueError):request_limit(dict(r,inference={}))

    def test_generated_upload_allowlist_excludes_private_inputs(self):
        source=Path('research/run_h3_native_continuation.py').read_text()
        self.assertIn("names=['review.mp4','continuation-profile.json','decode-profile.json']",source)
        self.assertNotIn('original.wav',source)
        self.assertIn("'audio.safetensors'",source)
        self.assertIn('private_conditioning_do_not_upload',source)

    @unittest.skipUnless(shutil.which('ffprobe'),'FFprobe required')
    def test_extensible_pcm_wav_does_not_depend_on_python_wave_version(self):
        import struct
        pcm_guid=bytes.fromhex('0100000000001000800000aa00389b71')
        fmt=struct.pack('<HHIIHHHHI16s',65534,1,48000,144000,3,24,22,24,4,pcm_guid)
        body=b'WAVE'+b'fmt '+struct.pack('<I',len(fmt))+fmt+b'data'+struct.pack('<I',144)+bytes(144)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'extensible.wav';p.write_bytes(b'RIFF'+struct.pack('<I',len(body))+body)
            before=p.read_bytes()
            self.assertEqual(recorded_wav_info(p),(48,48000,1,3))
            self.assertEqual(p.read_bytes(),before)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_lossless_pcm_original_audio_mux_no_trimming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'video.mp4';audio=root/'source.wav';out=root/'review.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x96:rate=24:duration=1',
                '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True,timeout=20)
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1',
                '-c:a','pcm_s24le',str(audio)],check=True,timeout=20)
            sha=hashlib.sha256(audio.read_bytes()).hexdigest()
            r=mux(video,audio,out,expected_sha=sha,samples=48000,sample_rate=48000,frames=24)
            self.assertTrue(r['full_original_pcm_preserved']);self.assertEqual(r['audio_codec'],'alac')
            self.assertTrue(r['original_wav_unchanged'])
