"""Private, operator-selected face crops. No face detector or identity scoring."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_box(box,width,height):
    if len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('four integer oriented-image coordinates required')
    left,top,right,bottom=box
    if not (0<=left<right<=width and 0<=top<bottom<=height):raise ValueError('crop outside image')
    if right-left!=bottom-top or right-left<64:raise ValueError('square crop at least64px; no aspect distortion')
    return tuple(box)


def prepare(source,output,expected_sha256,box,*,preserve_resolution=False):
    from PIL import Image,ImageOps
    source=Path(source).resolve();output=Path(output).resolve()
    if output.is_relative_to(ROOT):raise ValueError('derived personal image must remain outside repository')
    if sha(source)!=expected_sha256:raise ValueError('original reference SHA mismatch')
    with Image.open(source) as original:
        image=ImageOps.exif_transpose(original).convert('RGB')
        box=validate_box(box,*image.size);crop=image.crop(box)
        if not preserve_resolution:crop=crop.resize((512,512),Image.Resampling.LANCZOS)
        output.mkdir(mode=0o700,parents=True,exist_ok=False)
        dest=output/'face-crop.png';crop.save(dest);dest.chmod(0o600)
        report=dict(schema='h3-ref2va-operator-face-crop-v1',source_sha256=expected_sha256,
            oriented_source_size=list(image.size),box=list(box),crop_size=list(crop.size),crop_sha256=sha(dest),
            resize_policy='native-crop' if preserve_resolution else 'lanczos512',
            source_modified=False,automatic_face_detection=False,identity_quality_verified=False)
    if sha(source)!=expected_sha256:raise RuntimeError('original reference changed')
    path=output/'crop-receipt.json';path.write_text(json.dumps(report,indent=2)+'\n');path.chmod(0o600)
    return report


def verify(original,crop,receipt):
    data=json.loads(Path(receipt).read_text())
    if data['schema']!='h3-ref2va-operator-face-crop-v1' or data['source_sha256']!=sha(original) or data['crop_sha256']!=sha(crop):
        raise ValueError('crop binding differs')
    from PIL import Image,ImageOps
    with Image.open(original) as image:
        oriented=ImageOps.exif_transpose(image)
        if list(oriented.size)!=data['oriented_source_size']:raise ValueError('source geometry differs')
        validate_box(data['box'],*oriented.size)
        # Recompute exact pixels, not merely trust a sidecar's claim.
        expected=oriented.convert('RGB').crop(tuple(data['box']))
        policy=data.get('resize_policy','lanczos512')
        if policy=='lanczos512':expected=expected.resize((512,512),Image.Resampling.LANCZOS)
        elif policy!='native-crop':raise ValueError('unsupported crop resize policy')
    with Image.open(crop) as image:
        if list(image.size)!=data['crop_size'] or image.size!=expected.size or image.convert('RGB').tobytes()!=expected.tobytes():raise ValueError('crop pixels differ')
    return data
