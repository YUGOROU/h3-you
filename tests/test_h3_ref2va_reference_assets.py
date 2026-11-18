import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from research.h3_personal.ref2va_reference_assets import validate_box,prepare,verify,sha
from research.h3_personal.fast_ref2va import condition_key,FastRef2VAConfig
from research.h3_personal.ref2va_runtime import request

class ReferenceAssetsTests(unittest.TestCase):
    def test_box_is_explicit_square_and_bounded(self):
        self.assertEqual(validate_box([10,20,110,120],200,300),(10,20,110,120))
        for box in ([0,0,20,20],[0,0,100,90],[-1,0,99,100],[0,0,400,400],[False,0,100,100]):
            with self.assertRaises(ValueError):validate_box(box,200,300)

    def test_cache_is_bound_to_all_ordered_images(self):
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/'original';crop=Path(folder)/'crop';original.write_bytes(b'own');crop.write_bytes(b'face')
            refs=[NS(media_type='image',source=original),NS(media_type='image',source=crop)]
            b=NS(references=refs,prompt='own',height=256,width=448,num_frames=124,fps=24)
            first=condition_key(b);b.references=refs[::-1];self.assertNotEqual(first,condition_key(b))
            b.references=refs;crop.write_bytes(b'changed');self.assertNotEqual(first,condition_key(b))
            b.references=refs+[refs[0]]
            with self.assertRaises(ValueError):condition_key(b)

    def test_native_request_preserves_original_first(self):
        api=NS(**{key:NS for key in ('GenerationRequest','InputConfig','SamplingConfig','OutputConfig')})
        single=request(api,NS,'original',FastRef2VAConfig())
        dual=request(api,NS,'original',FastRef2VAConfig(),face_crop='crop')
        self.assertEqual([r.source for r in single.inputs.references],['original'])
        self.assertEqual([r.source for r in dual.inputs.references],['original','crop'])

    @unittest.skipUnless(importlib.util.find_spec('PIL'),'existing Pillow required; no install')
    def test_native_crop_does_not_downsample_eye_pixels(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'own.png';Image.new('RGB',(300,400),(10,20,30)).save(source)
            output=Path(folder)/'crop';prepare(source,output,sha(source),[20,30,220,230],preserve_resolution=True)
            proof=verify(source,output/'face-crop.png',output/'crop-receipt.json')
            self.assertEqual(proof['crop_size'],[200,200]);self.assertEqual(proof['resize_policy'],'native-crop')

    @unittest.skipUnless(importlib.util.find_spec('PIL'),'existing Pillow required; no install')
    def test_crop_provenance_and_pixels_recomputed(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'own.png';Image.new('RGB',(200,300),(30,90,160)).save(source)
            output=Path(folder)/'crop';digest=sha(source);prepare(source,output,digest,[10,20,110,120])
            receipt=output/'crop-receipt.json';verify(source,output/'face-crop.png',receipt)
            data=json.loads(receipt.read_text());data['source_sha256']='0'*64;receipt.write_text(json.dumps(data))
            with self.assertRaises(ValueError):verify(source,output/'face-crop.png',receipt)
            self.assertEqual(sha(source),digest)
