"""Execute pinned native image-control source with fake pixel objects (CPU).

No Pillow/Torch install or private pixels. This checks source branch/geometry,
NOT actual pixel decoding, native neural conditioning or visual acceptance.
"""
import ast
import hashlib
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from research.h3_personal.conditioning_experiment import experimental_resolver
from research.h3_personal.fast_continuation import REFERENCE_SHA


@unittest.skipUnless(os.environ.get('H3_FASTVIDEO_ARCHIVE'),'existing pinned archive opt-in')
class Tests(unittest.TestCase):
    def test_exact_native_image_preparation_preserves_two_aspects_without_crop(self):
        archive=Path(os.environ['H3_FASTVIDEO_ARCHIVE'])
        with tarfile.open(archive) as files:
            member=next(m for m in files.getmembers() if m.name.endswith('/minimax_h3/reference.py'))
            raw=files.extractfile(member).read()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),REFERENCE_SHA)
        tree=ast.parse(raw);names={'prepare_reference','prepare_reference_image'}
        methods=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        class Image:
            def __init__(self,size):self.size=size;self.resizes=[];self.conversions=[]
            def convert(self,mode):self.conversions.append(mode);return self
            def resize(self,size,filter):
                self.resizes.append((size,filter));return Image(size)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*methods],type_ignores=[])
        globals=dict(os=os,Image=NS(Image=Image,Resampling=NS(LANCZOS='native-lanczos')),
            ImageOps=NS(exif_transpose=lambda image:image),MiniMaxH3PreparedReference=lambda **kw:NS(**kw),
            resolve_reference_image_size=experimental_resolver)
        exec(compile(ast.fix_missing_locations(module),'<exact pinned native image branch>','exec'),globals)
        originals=[Image((9,16)),Image((14,11))]
        with patch('research.h3_personal.conditioning_experiment.reference_short_edge',return_value=1024):
            outputs=[globals['prepare_reference'](NS(media_type='image',source=image),124,32000) for image in originals]
        self.assertEqual([x.image.size for x in outputs],[(1024,1824),(1312,1024)])
        self.assertEqual([x.conversions for x in originals],[['RGB'],['RGB']])
        self.assertEqual([x.resizes for x in originals],[
            [((1024,1824),'native-lanczos')],[((1312,1024),'native-lanczos')]])


if __name__=='__main__':unittest.main()
