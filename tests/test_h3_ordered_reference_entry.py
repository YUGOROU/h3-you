"""Synthetic local contract tests; no provider/model/device operations."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

from research import launch_h3_direct_conversation as entry


def png(width,height):
    from test_h3_ordered_references import png_header
    return png_header(width,height)


class OrderedEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        refs=[]
        for role,w,h in [('person',9,16),('surroundings',14,11)]:
            data=png(w,h);path=self.root/(role+'.png');path.write_bytes(data)
            refs.append(dict(role=role,path=str(path),sha256=hashlib.sha256(data).hexdigest(),width=w,height=h))
        self.config=dict(schema='h3-ordered-references-v1',references=refs)
        self.cfg=self.root/'config.json';self.cfg.write_text(json.dumps(self.config))
        self.args=['--directory',str(self.root/'campaign'),'--headless-interrupt','--visual-intents',
                   '--vox-parent',str(self.root/'unused-parent'),'--ordered-references-config',str(self.cfg)]

    def test_offline_prints_portable_identity_no_resources(self):
        output=io.StringIO()
        with patch.object(entry,'run',side_effect=AssertionError('no run')),patch.object(entry,'prepare',side_effect=AssertionError('no stage')),contextlib.redirect_stdout(output):
            entry.main(self.args)
        row=json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(row['type'],'h3_ordered_references_offline')
        self.assertIs(row['cloud_operation'],False)
        self.assertEqual(row['visual_acceptance'],'unreviewed')
        self.assertNotIn(str(self.root),output.getvalue())
        self.assertFalse((self.root/'campaign').exists())

    def test_cli_run_requires_scoped_confirmation_before_input_or_lease(self):
        with patch('builtins.input',side_effect=AssertionError('no input')),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):entry.main(self.args+['--run'])
        self.assertFalse((self.root/'campaign').exists())

    def test_cli_requires_visual_selection_and_confirmation_requires_config(self):
        for args in (['--directory',str(self.root/'out'),'--ordered-references-config',str(self.cfg)],
                     ['--directory',str(self.root/'out'),'--confirm-private-reference-upload']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):entry.main(args)

    def test_after_typed_input_files_are_rechecked_before_lease(self):
        calls=[]
        def typed(*_):
            calls.append(True)
            Path(self.config['references'][1]['path']).write_bytes(png(14,12))
            return 'synthetic turn'
        with patch('builtins.input',side_effect=typed),patch.object(entry,'run',side_effect=AssertionError('no run')):
            with self.assertRaises(ValueError):
                entry.main(self.args+['--run','--confirm-private-reference-upload'])
        self.assertEqual(len(calls),2)
        self.assertFalse((self.root/'campaign').exists())

    def test_run_guard_precedes_resources(self):
        with self.assertRaises(ValueError):entry.run(self.root/'out',ordered_references=self.config)
        with self.assertRaises(ValueError):entry.run(self.root/'out',confirm_private_reference_upload=True)
        with self.assertRaises(ValueError):entry.run(self.root/'out',ordered_references=self.config,
            confirm_private_reference_upload=True,visual_intents=False)
        self.assertFalse((self.root/'out').exists())

    def test_prepare_rejects_before_api_or_directory_creation(self):
        _,metadata=entry.ordered_selection(self.config)
        for row,kwargs in ((dict(visual_intents=True,**metadata),{}),
                           (dict(visual_intents=True),dict(confirm_private_reference_upload=True)),
                           (dict(visual_intents=True,**dict(metadata,ordered_reference_sha256='f'*64)),dict(confirm_private_reference_upload=True))):
            with self.assertRaises(ValueError):entry.prepare(self.root/'out',row,object(),ordered_references=self.config,**kwargs)
        self.assertFalse((self.root/'out').exists())

    def test_staging_exact_original_synthetic_bytes_and_modes(self):
        stage=self.root/'stage';stage.mkdir()
        entry.stage_ordered_inputs(stage,self.config)
        for ref,name in zip(self.config['references'],('person.png','surroundings.png')):
            dest=stage/'ordered-inputs'/name
            self.assertEqual(dest.read_bytes(),Path(ref['path']).read_bytes())
            self.assertEqual(dest.stat().st_mode&0o777,0o600)
        with self.assertRaises(FileExistsError):entry.stage_ordered_inputs(stage,self.config)

    def test_fresh_file_change_rejected_no_fallback(self):
        Path(self.config['references'][1]['path']).write_bytes(png(14,12))
        with self.assertRaises(ValueError):entry.ordered_selection(self.config)

    def test_ready_hash_bound_and_missing_or_unexpected_rejected(self):
        row=dict(type='h3_direct_ready',status='ready',session='a'*32,profile_sha256='b'*64,
                 epoch=1,max_segments=16,max_total_frames=1654,port=7860,expires_at=time.time()+300,
                 turn_interruptions=True,visual_intents=True,ordered_reference_sha256='c'*64)
        kwargs=dict(turn_interruptions=True,visual_intents=True,ordered_reference_sha256='c'*64)
        self.assertEqual(entry.validate_ready(row,'a'*32,**kwargs),row)
        for change in ({'ordered_reference_sha256':'d'*64},{'ordered_reference_sha256':None}):
            with self.assertRaises(ValueError):entry.validate_ready(row,'a'*32,**dict(kwargs,**change))
        missing=dict(row);del missing['ordered_reference_sha256']
        with self.assertRaises(ValueError):entry.validate_ready(missing,'a'*32,**kwargs)


if __name__=='__main__':unittest.main()
