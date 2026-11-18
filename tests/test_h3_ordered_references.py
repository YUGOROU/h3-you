"""Synthetic PNG chunk/metadata fixtures; no pixel/native-model decoding proof."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from research.h3_personal import ordered_references as refs


def png_chunk(kind, data=b''):
    return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data))


def png_header(width, height, *, color=2):
    # Historical helper name retained; now complete PNG framing for orientation
    # absence proof. These tests never invoke a pixel decoder.
    header=b'\x89PNG\r\n\x1a\n'+png_chunk(b'IHDR',struct.pack('!IIBBBBB',width,height,8,color,0,0,0))
    pixels=b''.join(b'\x00'+b'\x00'*(width*3) for _ in range(height))
    return header+png_chunk(b'IDAT',zlib.compress(pixels))+png_chunk(b'IEND')


def tiff_orientation(value=1, *, endian='<', kind=3, length=1, duplicate=False):
    entry=struct.pack(endian+'HHI',0x0112,kind,length)+struct.pack(endian+'H',value)+b'\x00\x00'
    return ((b'II' if endian=='<' else b'MM')+struct.pack(endian+'HI',42,8)
            +struct.pack(endian+'H',2 if duplicate else 1)+entry*(2 if duplicate else 1)+struct.pack(endian+'I',0))


class Tests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.value=dict(schema=refs.SCHEMA,references=[])
        for i,(role,width,height) in enumerate((('person',9,16),('surroundings',14,11))):
            path=self.root/f'{i}.png';data=png_header(width,height);path.write_bytes(data)
            self.value['references'].append(dict(role=role,path=str(path),width=width,height=height,
                                                sha256=hashlib.sha256(data).hexdigest()))

    def test_detached_ordered_config_and_deterministic_digest(self):
        validated=refs.verify_local_files(self.value)
        self.assertEqual(validated,self.value)
        digest=refs.config_sha256(self.value)
        reordered_keys=json.loads(json.dumps(self.value,sort_keys=True))
        self.assertEqual(refs.config_sha256(reordered_keys),digest)
        validated['references'][0]['width']=1
        self.assertEqual(self.value['references'][0]['width'],9)
        self.assertEqual(refs.verify_local_files(self.value,expected_sha256=digest),self.value)
        self.assertIn('Reference image 1',refs.role_prompt(self.value))
        self.assertIn('same physical place',refs.role_prompt(self.value))

    def test_exact_count_fields_role_order_and_types(self):
        mutations=(lambda v:v['references'].reverse(),lambda v:v['references'].pop(),
            lambda v:v['references'].append(copy.deepcopy(v['references'][0])),
            lambda v:v.update(extra=True),lambda v:v['references'][0].update(extra=True),
            lambda v:v['references'][0].update(width=True),
            lambda v:v['references'][0].update(width=65,height=16),
            lambda v:v['references'][0].update(sha256='A'*64),
            lambda v:v['references'][0].update(width=32769),
            lambda v:v['references'][0].update(width=32768,height=32768))
        for mutation in mutations:
            bad=copy.deepcopy(self.value);mutation(bad)
            with self.subTest(value=bad),self.assertRaises(ValueError):refs.validate_ordered_references(bad)
        with self.assertRaises(ValueError):refs.validate_ordered_references(self.value,check_files=1)

    def test_structural_validation_has_no_file_requirement_but_fresh_check_does(self):
        Path(self.value['references'][0]['path']).unlink()
        refs.validate_ordered_references(self.value)
        with self.assertRaises(ValueError):refs.verify_local_files(self.value)

    def test_changed_same_name_hash_and_geometry_rejected(self):
        path=Path(self.value['references'][0]['path'])
        for data in (png_header(9,16)+b'changed',png_header(10,16)):
            path.write_bytes(data)
            with self.assertRaises(ValueError):refs.verify_local_files(self.value)
        bad=copy.deepcopy(self.value);bad['references'][0]['width']=8
        path.write_bytes(png_header(9,16))
        with self.assertRaises(ValueError):refs.verify_local_files(bad)

    def test_changed_declaration_cannot_match_original_expected_digest(self):
        expected=refs.config_sha256(self.value)
        bad=copy.deepcopy(self.value)
        a,b=bad['references']
        for key in ('path','width','height','sha256'):a[key],b[key]=b[key],a[key]
        self.assertNotEqual(refs.config_sha256(bad),expected)
        with self.assertRaises(ValueError):refs.verify_local_files(bad,expected_sha256=expected)

    def test_png_crc_and_rgb_header_not_full_decode(self):
        path=Path(self.value['references'][0]['path'])
        for data in (b'x'*33,png_header(9,16,color=6),png_header(9,16)[:-1]+b'x'):
            path.write_bytes(data)
            bad=copy.deepcopy(self.value);bad['references'][0]['sha256']=hashlib.sha256(data).hexdigest()
            with self.assertRaises(ValueError):refs.verify_local_files(bad)
        # Complete chunk framing is required to prove metadata absence; pixel
        # decode and native processing are still NOT performed.
        truncated=png_header(9,16)[:33];path.write_bytes(truncated)
        bad=copy.deepcopy(self.value);bad['references'][0]['sha256']=hashlib.sha256(truncated).hexdigest()
        with self.assertRaises(ValueError):refs.verify_local_files(bad)
        path.write_bytes(png_header(9,16));refs.verify_local_files(self.value)

    def metadata_config(self, *chunks):
        value=copy.deepcopy(self.value);record=value['references'][0]
        data=png_header(record['width'],record['height'])
        data=data[:33]+b''.join(chunks)+data[33:]
        Path(record['path']).write_bytes(data);record['sha256']=hashlib.sha256(data).hexdigest()
        return value

    def test_orientation_absent_identity_and_policy_binding(self):
        evidence=refs.verify_orientation(self.value)
        self.assertEqual(evidence['orientation_policy'],refs.ORIENTATION_POLICY)
        self.assertFalse(evidence['images'][0]['exif_present'])
        self.assertIsNone(evidence['images'][0]['exif_orientation'])
        self.assertEqual(evidence['images'][0]['stored_size'],[9,16])
        self.assertEqual(evidence['images'][0]['oriented_size'],[9,16])
        identity=refs.content_identity(self.value)
        self.assertEqual(identity['preprocessing']['orientation_policy'],refs.ORIENTATION_POLICY)
        for endian in ('<','>'):
            value=self.metadata_config(png_chunk(b'eXIf',tiff_orientation(endian=endian)))
            refs.verify_local_files(value)
            proof=refs.verify_orientation(value)['images'][0]
            self.assertTrue(proof['exif_present']);self.assertEqual(proof['exif_orientation'],1)
            self.assertEqual(proof['stored_size'],proof['oriented_size'])
        bad=copy.deepcopy(identity);bad['preprocessing']['orientation_policy']='ignore'
        with self.assertRaises(ValueError):refs.validate_content_identity(bad)

    def test_orientation_nonidentity_malformed_and_ambiguous_fail_closed(self):
        payloads=[tiff_orientation(v) for v in (0,2,3,4,5,6,7,8,9)]
        payloads += [b'',b'Exif\x00\x00'+tiff_orientation(),tiff_orientation(kind=4),
                     tiff_orientation(length=2),tiff_orientation(duplicate=True),
                     b'II'+struct.pack('<HI',42,1000),
                     b'II'+struct.pack('<HI',42,8)+struct.pack('<HI',0,8)]
        for payload in payloads:
            value=self.metadata_config(png_chunk(b'eXIf',payload))
            with self.subTest(payload=payload),self.assertRaises(ValueError):refs.verify_local_files(value)
        value=self.metadata_config(png_chunk(b'eXIf',tiff_orientation()),png_chunk(b'eXIf',tiff_orientation()))
        with self.assertRaises(ValueError):refs.verify_local_files(value)

    def test_png_chunk_scan_not_substring_and_alternate_orientation_rejected(self):
        # TIFF-looking bytes in unrelated text are not an eXIf chunk.
        value=self.metadata_config(png_chunk(b'tEXt',b'Description\x00eXIf'+tiff_orientation(6)))
        self.assertFalse(refs.verify_orientation(value)['images'][0]['exif_present'])
        for key in (b'Raw profile type exif',b'XML:com.adobe.xmp'):
            for kind in (b'tEXt',b'zTXt',b'iTXt'):
                value=self.metadata_config(png_chunk(kind,key+b'\x00arbitrary'))
                with self.assertRaises(ValueError):refs.verify_local_files(value)
        for chunk in (png_chunk(b'eXIf',tiff_orientation())[:-1]+b'x',
                      struct.pack('!I',refs.MAX_FILE_BYTES)+b'eXIf'):
            value=self.metadata_config(chunk)
            with self.assertRaises(ValueError):refs.verify_local_files(value)

    def test_empty_valid_exif_without_orientation_is_allowed(self):
        value=self.metadata_config(png_chunk(b'eXIf',b'II'+struct.pack('<HIHI',42,8,0,0)))
        proof=refs.verify_orientation(value)['images'][0]
        self.assertTrue(proof['exif_present']);self.assertIsNone(proof['exif_orientation'])
        self.assertEqual(proof['oriented_size'],[9,16])

    def test_file_and_ancestor_symlink_refused(self):
        path=Path(self.value['references'][0]['path'])
        moved=path.with_suffix('.real');path.rename(moved);path.symlink_to(moved)
        with self.assertRaises(ValueError):refs.verify_local_files(self.value)
        path.unlink();moved.rename(path)
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        bad=copy.deepcopy(self.value);bad['references'][0]['path']=str(alias/path.name)
        with self.assertRaises(ValueError):refs.verify_local_files(bad)

    def test_relative_noncanonical_repo_and_duplicate_paths_refused(self):
        for path in ('relative.png',str(self.root)+'/../0.png',str(Path(__file__).resolve().parent/'private.png')):
            bad=copy.deepcopy(self.value);bad['references'][0]['path']=path
            with self.assertRaises(ValueError):refs.validate_ordered_references(bad)
        for key in ('path','sha256'):
            bad=copy.deepcopy(self.value);bad['references'][1][key]=bad['references'][0][key]
            with self.assertRaises(ValueError):refs.validate_ordered_references(bad)

    def test_file_byte_bound_and_nonregular_file(self):
        with patch.object(refs,'MAX_FILE_BYTES',32):
            with self.assertRaises(ValueError):refs.verify_local_files(self.value)
        path=Path(self.value['references'][0]['path']);path.unlink();path.mkdir()
        with self.assertRaises(ValueError):refs.verify_local_files(self.value)

    def test_portable_content_identity_survives_relocation_but_not_policy_or_content(self):
        portable=refs.config_digest(self.value)
        local=refs.config_sha256(self.value)
        relocated=copy.deepcopy(self.value)
        directory=self.root/'relocated';directory.mkdir()
        for record in relocated['references']:
            old=Path(record['path']);new=directory/old.name
            new.write_bytes(old.read_bytes());record['path']=str(new)
        self.assertEqual(refs.content_sha256(relocated),portable)
        self.assertNotEqual(refs.config_sha256(relocated),local)
        refs.verify_local_files(relocated,expected_content_sha256=portable)
        with self.assertRaises(ValueError):refs.verify_local_files(relocated,expected_sha256=local)
        self.assertNotIn('path',str(refs.content_identity(self.value)))
        with self.assertRaises(ValueError):
            refs.verify_local_files(relocated,expected_content_sha256=portable,short_edge=512)
        bad=copy.deepcopy(self.value);bad['references'][1]['sha256']='a'*64
        self.assertNotEqual(refs.config_digest(bad),portable)
        for edge in (True,256,2048):
            with self.assertRaises(ValueError):refs.config_digest(self.value,short_edge=edge)

    def test_expected_native_no_crop_round32_geometry_not_token_proof(self):
        a=refs.expected_prepared_sizes(self.value,short_edge=1024)
        b=refs.expected_prepared_sizes(self.value,short_edge=512)
        self.assertEqual([(r['width'],r['height']) for r in a['references']],[(1024,1824),(1312,1024)])
        self.assertEqual([(r['width'],r['height']) for r in b['references']],[(512,896),(640,512)])
        self.assertIsNone(a['R']);self.assertIsNone(a['T'])
        self.assertEqual(a['preprocessing']['crop'],'none')
        self.assertNotEqual(refs.config_digest(self.value,short_edge=1024),refs.config_digest(self.value,short_edge=512))

    def test_prepared_geometry_rejects_unsupported_aspect_without_cropping(self):
        bad=copy.deepcopy(self.value)
        bad['references'][1].update(width=1,height=8)
        # Reject at selection BEFORE permission-gated staging/network as well.
        with self.assertRaises(ValueError):refs.validate_ordered_references(bad)
        with self.assertRaises(ValueError):refs.expected_prepared_sizes(bad)
        self.assertEqual((bad['references'][1]['width'],bad['references'][1]['height']),(1,8))


if __name__=='__main__':unittest.main()
